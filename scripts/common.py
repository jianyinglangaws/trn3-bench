"""Shared setup and timing helpers for the micro-benchmarks (`from common import *`).

Process setup (runs at import, before the Neuron torch plugin loads):
  - pins one logical NeuronCore per rank: NEURON_RT_VISIBLE_CORES = LOCAL_RANK * CORE_STRIDE
  - optional NUMA_BIND=1: pins the rank's host threads to the CPUs local to its device
  - forces the NKI trace cache off (see below)
  - initializes a gloo process group when WORLD_SIZE > 1 (host-side barriers / reductions only)

Timing:
  - bench():       best-of-reps windows; each window starts after a barrier and its global time is
                   the slowest rank's, so WORLD * work / t_global is a true concurrent aggregate.
  - bench_slope(): difference between two in-launch repeat counts, so launch and sync overhead
                   cancel (host-side stand-in for device-profile timing of NKI microkernels).
  - emit():        one jsonl record per rank plus one {"rank": "agg"} record (rank 0) to $OUT.
"""
import os, sys, time, json, warnings
warnings.filterwarnings("ignore")

LR = int(os.environ.get("LOCAL_RANK", 0))
RANK = int(os.environ.get("RANK", 0))
WORLD = int(os.environ.get("WORLD_SIZE", 1))
STRIDE = int(os.environ.get("CORE_STRIDE", 1))
SLOPE_WARM_S = float(os.environ.get("SLOPE_WARM_S", "1.0"))

os.environ.setdefault("NEURON_RT_VISIBLE_CORES", str(LR * STRIDE))
# torch-neuronx 2.14 persists traced NKI kernels across processes keyed on the arguments only, so a
# kernel whose body depends on closure values (mode, repeat count, ...) silently reuses a stale
# NEFF. Forced off rather than setdefault: the torchrun launcher imports torch_neuronx, which
# exports NKI_ENABLE_TRACE_CACHE=1 to every worker. BENCH_NKI_TRACE_CACHE=1 opts back in.
os.environ["NKI_ENABLE_TRACE_CACHE"] = os.environ.get("BENCH_NKI_TRACE_CACHE", "0")


def _bind_numa():
    """Pin this process to the CPUs local to its Neuron device (as reported by neuron-ls), so its
    first-touch host memory lands on the device's NUMA node."""
    import subprocess
    core = int(os.environ["NEURON_RT_VISIBLE_CORES"].split(",")[0].split("-")[0])
    devices = json.loads(subprocess.run(["neuron-ls", "--json-output"], capture_output=True,
                                        text=True, check=True).stdout)
    for d in devices:
        if core in d["neuroncore_ids"]:
            cpus = set()
            for r in d["cpu_affinity"].split(","):
                lo, _, hi = r.partition("-")
                cpus.update(range(int(lo), int(hi or lo) + 1))
            os.sched_setaffinity(0, cpus)
            return


if os.environ.get("NUMA_BIND", "0") == "1":
    _bind_numa()

import torch
import torch.distributed as dist
if WORLD > 1:   # before the Neuron plugin registers its PrivateUse1 device
    dist.init_process_group("gloo")
try:
    import torch_neuronx  # noqa: F401  (torch-neuronx native `neuron` device)
    DEFAULT_BACKEND = "neuron"
except ImportError:       # vllm-neuron images ship libtorch-neuronx-lite (eager paths only tested)
    import libtorch_neuronx_lite  # noqa: F401
    DEFAULT_BACKEND = "neuron_libtorch"
dev = torch.device("neuron:0")


def wait():
    torch.neuron.synchronize()


def barrier():
    # CPU all-reduce rather than dist.barrier(): the latter probes the current accelerator
    # (PrivateUse1) for its barrier tensor, which some Neuron torch plugins don't implement.
    if WORLD > 1:
        dist.all_reduce(torch.zeros(1))


def allreduce_max(x):
    if WORLD == 1:
        return x
    t = torch.tensor([x], dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def maybe_compile(fn):
    """torch.compile fn unless COMPILE=0. Returns (fn, mode), mode in {"compiled", "eager"}."""
    if os.environ.get("COMPILE", "1") == "0":
        return fn, "eager"
    # dynamic=False: Neuron needs static shapes; otherwise Dynamo re-traces the 2nd size as [?,?]
    backend = os.environ.get("BACKEND", DEFAULT_BACKEND)
    return torch.compile(fn, backend=backend, dynamic=False), "compiled"


def warm(fn):
    """First call (compilation) on local rank 0 first, so the other ranks hit the compile cache
    instead of all compiling the same graph concurrently."""
    if LR == 0:
        fn(); wait()
    barrier()
    if LR != 0:
        fn(); wait()
    barrier()


def bench(fn, iters, warmup=2, reps=3, sync_each=False):
    """Best-of-reps seconds per window of `iters` calls: (this rank, slowest rank)."""
    for _ in range(warmup):
        fn()
    wait()
    best_l = best_g = float("inf")
    for _ in range(reps):
        barrier()
        t = time.perf_counter()
        for _ in range(iters):
            fn()
            if sync_each:
                wait()
        wait()
        el = time.perf_counter() - t
        best_l, best_g = min(best_l, el), min(best_g, allreduce_max(el))
    return best_l, best_g


def bench_slope(make_fn, r_lo, r_hi, iters, reps=3):
    """Overhead-free timing for kernels with an in-launch repeat count.

    make_fn(R) returns a callable doing R repeats of the unit of work in ONE launch. Returns
    ((dt_local, dt_global), t_hi_local): dt = t(r_hi) - t(r_lo) per window of `iters` calls, i.e.
    the time of (r_hi - r_lo) * iters units with launch/sync overhead cancelled (dt_global = the
    slowest rank's dt), plus the raw r_hi window for an overhead-inclusive comparison.
    """
    lo, hi = make_fn(r_lo), make_fn(r_hi)
    warm(lo); warm(hi)
    # A freshly loaded NEFF runs several times slower for its first executions: warm each kernel
    # for a fixed wall time (not a fixed call count), then alternate lo/hi windows and keep the
    # best of each, so a slow phase cannot land on one side of the difference only.
    for f in (lo, hi):
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < SLOPE_WARM_S:
            f(); wait()
    t_lo = t_hi = float("inf")
    for _ in range(reps):
        t_lo = min(t_lo, bench(lo, iters, warmup=0, reps=1)[0])
        t_hi = min(t_hi, bench(hi, iters, warmup=0, reps=1)[0])
    dt = t_hi - t_lo
    return (dt, allreduce_max(dt)), t_hi


def emit(rec, work, t_local, t_global, unit, scale):
    """Append this rank's record (and, on rank 0, the aggregate record) to $OUT.

    rec: identifying fields; work: per-rank work units per window; unit/scale: rate field name and
    divisor (e.g. "tflops", 1e12). Aggregate rate = WORLD * work / t_global.
    """
    rate, agg_rate = work / t_local / scale, WORLD * work / t_global / scale
    r = {"rank": RANK, "world": WORLD, **rec, "time_s": t_local, unit: rate}
    with open(os.environ.get("OUT", "/tmp/bench.jsonl"), "a") as f:
        f.write(json.dumps(r) + "\n")
        if RANK == 0:
            f.write(json.dumps({**r, "rank": "agg", "time_s": t_global, unit: agg_rate}) + "\n")
    if RANK == 0:
        print(json.dumps({**r, "agg_" + unit: agg_rate}), flush=True)
