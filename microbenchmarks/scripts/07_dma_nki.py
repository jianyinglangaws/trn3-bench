# DMA bandwidth at the NKI level (nisa.dma_copy), per descriptor-generation mode:
#   hbm2hbm (HBM -> HBM, bytes counted read+write, comparable to 05_hbm_copy),
#   load    (HBM -> SBUF), store (SBUF -> HBM)   [bytes moved]
# x dge in {none (runtime-built static descriptors), hwdge (DGE hardware), swdge (GpSimd)}.
# The kernel runs on every physical core of the logical core (nl.num_programs) and splits the tiles.
# Timing: bench_slope over in-kernel repeat counts, so launch overhead cancels. Every repeat writes
# a distinct destination and every load feeds a dependent reduction, so the compiler cannot drop
# repeats as dead work; a slope rate far above the raw rate flags that it did anyway.
import argparse
from common import *
import nki, nki.isa as nisa, nki.language as nl
from nki.isa.constants import dge_mode as DGE

p = argparse.ArgumentParser()
p.add_argument("--kinds", default="hbm2hbm,load,store")
p.add_argument("--dge", default="none,hwdge,swdge")
p.add_argument("--fdim", type=int, default=8192, help="tile free size (bf16); tile = [pdim, fdim]")
p.add_argument("--pdim", type=int, default=128)
p.add_argument("--mb", type=int, default=128, help="MiB per pass (split across physical cores)")
p.add_argument("--r_lo", type=int, default=2); p.add_argument("--r_hi", type=int, default=10)
p.add_argument("--iters", type=int, default=5); p.add_argument("--reps", type=int, default=3)
a = p.parse_args()
P, F = a.pdim, a.fdim
T = max(2, a.mb * 2**20 // (P * F * 2)); T -= T % 2     # tiles per pass, even (2 physical cores)


def make_kernel(kind, dge, R):
    @nki.jit
    def k(src):                                           # src [T, P, F] HBM
        out = nl.ndarray((R * T, P, F) if kind != "load" else (T, P, F), dtype=src.dtype, buffer=nl.shared_hbm)
        npg = nl.num_programs(0); pid = nl.program_id(0); per = T // npg
        NB = 4                                            # SBUF ring so consecutive loads overlap
        sbs = [nl.ndarray((P, F), dtype=src.dtype, buffer=nl.sbuf) for _ in range(NB)]
        if kind == "store":
            nisa.dma_copy(dst=sbs[0], src=src[0])
        acc = nl.ndarray((P, 8), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(acc, 0.0)
        for r in nl.sequential_range(R):
            for t in nl.affine_range(per):
                i = pid * per + t
                if kind == "hbm2hbm":
                    nisa.dma_copy(dst=out[r * T + i], src=src[i], dge_mode=dge)
                elif kind == "load":
                    b = sbs[t % NB]
                    nisa.dma_copy(dst=b, src=src[i], dge_mode=dge)
                    nisa.tensor_tensor(dst=acc, data1=acc, data2=b[:, 0:8], op=nl.add)
                else:
                    nisa.dma_copy(dst=out[r * T + i], src=sbs[0], dge_mode=dge)
        if kind == "load":
            nisa.dma_copy(dst=out[pid][:, 0:8], src=acc)
        return out
    return k


src = torch.randn(T, P, F).bfloat16().to(dev)
for kind in a.kinds.split(","):
    for dname in a.dge.split(","):
        dge = getattr(DGE, dname)
        launch = lambda R: (lambda k=make_kernel(kind, dge, R): k(src))   # bind a fresh kernel per R
        (dt_l, dt_g), t_hi = bench_slope(launch, a.r_lo, a.r_hi, a.iters, a.reps)
        per_pass = T * P * F * 2 * (2 if kind == "hbm2hbm" else 1)      # bytes per repeat
        work = per_pass * (a.r_hi - a.r_lo) * a.iters
        raw = per_pass * a.r_hi * a.iters / t_hi / 1e9                  # overhead-inclusive rate
        # Slope above raw is expected (launch overhead removed) but only by a few %; far above
        # means repeats were optimized away or the repeat count changed something else.
        suspect = not (0 < dt_l and work / dt_l / 1e9 < 1.25 * raw)
        if suspect and RANK == 0:
            print(f"WARNING {kind}/{dname}: slope rate inconsistent with raw rate {raw:.0f} GB/s", flush=True)
        emit({"bench": "dma_nki", "kind": kind, "dge": dname, "pdim": P, "fdim": F, "mb": a.mb,
              "raw_GBps": raw, "suspect": suspect}, work, dt_l, dt_g, "GBps", 1e9)
