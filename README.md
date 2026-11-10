# trn3-bench — AWS Trainium3 micro-benchmark suite

Single-node benchmarks for AWS Trainium3: compute (bf16 / fp32 / MXFP8 GEMM and the Tensor Engine
ceiling), HBM bandwidth (framework copy, STREAM, NKI DMA modes), host↔device transfers, and
NeuronLink collectives. Each runs at 1 logical core, one chip, and the full instance, and the
results are compared against the AWS published Trainium3 specs.

## Benchmarks

| script | measures | how |
|---|---|---|
| `08_engine_peak.py` | Tensor Engine ceiling, bf16 and MXFP8 | NKI back-to-back `nc_matmul` / `nc_matmul_mx` on on-chip (SBUF) operands; result checked against `A^T B` |
| `01_gemm.py` | bf16 / fp32 GEMM | `torch.compile`d chain of square matmuls `y = y @ w` in one graph; checked vs CPU fp32 for N ≤ 2048 |
| `06_mxfp8_gemm.py` | MXFP8 GEMM | nki-library `matmul_mxfp8`; `--mode prequant` (operands already MXFP8) or `--mode bf16` (quantized in-kernel); accuracy checked with nki-library's golden quantizer |
| `07_dma_nki.py` | DMA bandwidth per descriptor mode | NKI `dma_copy` HBM→HBM, HBM→SBUF, SBUF→HBM × `dge_mode` none / hwdge / swdge |
| `05_hbm_copy.py` | HBM bandwidth through the framework | device `copy_` (read + write), chunked |
| `02_stream.py` | STREAM copy / scale / add / triad | one torch op per kernel into a preallocated output, so counted bytes = moved bytes |
| `03_h2d.py` | host↔device bandwidth | `copy_` into preallocated buffers; `--pin` for page-locked host memory |
| `04_collectives.sh` | all-reduce / all-gather / reduce-scatter / all-to-all | `nccom-test`, 32 B – 4 GiB, bf16 |

Supporting files: `run_micro.sh` (suite runner), `common.py` (process setup and timing),
`aggregate.py` (markdown tables).

## Quick start

Requirements: a Trainium3 instance with the Neuron driver/runtime, a Python environment with
`torch-neuronx` (`nki` comes with it), and for the MXFP8 GEMM a checkout of
[nki-library](https://github.com/aws-neuron/nki-library).

Keep the venv, the compile cache, and `HOME`/`TMPDIR` on local disk. On an NFS home directory,
imports and compiles are many times slower.

```bash
python3 -m venv /local/nvme/torch-neuronx-venv
source /local/nvme/torch-neuronx-venv/bin/activate
pip install "torch-neuronx==2.14.*" "neuronx-cc==2.*" nki \
  --extra-index-url https://pip.repos.neuron.amazonaws.com \
  --extra-index-url https://download.pytorch.org/whl/cpu

VENV=/local/nvme/torch-neuronx-venv CACHE_DIR=/local/nvme/neuron_cache NKILIB=~/nki-library \
  bash scripts/run_micro.sh                    # full suite, ~1 h on a 64-core instance
python3 scripts/aggregate.py results/<instance-type> > results/<instance-type>/summary_tables.md
```

Run a subset or other rank counts:

```bash
ONLY="engine gemm" NPS="1 4" bash scripts/run_micro.sh
OUT=/tmp/x.jsonl torchrun --nproc_per_node=4 scripts/07_dma_nki.py --kinds load --dge hwdge
```

### `run_micro.sh` settings

| variable | default | meaning |
|---|---|---|
| `ONLY` | `engine gemm mxfp8 stream dma hbm h2d coll` | suites to run |
| `NPS` | `1 4 64` | rank counts for per-core sweeps: 1 core, one chip, the instance |
| `VENV` | newest `/opt/aws_neuronx_venv_torch_neuronx*` | environment to activate |
| `R` | `results/<instance-type>` (from `neuron-ls`) | output directory |
| `CACHE_DIR` | `/tmp/neuron_cache` | Neuron compile cache |
| `NKILIB` | `~/nki-library` | nki-library checkout (`06_mxfp8_gemm.py`) |
| `TIMEOUT` | `5400` | seconds per step |

Per-process settings read by `common.py`:

| variable | meaning |
|---|---|
| `CORE_STRIDE` | rank *i* uses logical core *i × stride*; spreads ranks across chips (16 ranks with stride 4 = one core per chip) |
| `NUMA_BIND=1` | pin each rank to the CPUs local to its device |
| `COMPILE=0` | eager mode instead of `torch.compile` (`01_gemm.py`; `02_stream.py` is eager by default) |
| `BENCH_NKI_TRACE_CACHE=1` | re-enable the NKI trace cache (off by default, see pitfalls) |

## Output

Each step writes `<name>.jsonl` and `<name>.log` to `$R`, plus `$R/collectives/<op>_r<ranks>.txt`.
Every measurement produces one record per rank and one aggregate record (`"rank": "agg"`), e.g.:

```json
{"rank": "agg", "world": 64, "bench": "gemm", "dtype": "bf16", "n": 8192, "chain": 1,
 "mode": "compiled", "relL2": null, "time_s": 0.1456, "tflops": 9667.8}
```

`aggregate.py` turns a results directory into markdown tables (per-core mean and aggregate).

## Methodology

- **One logical core per rank.** `NEURON_RT_VISIBLE_CORES` is set before the Neuron plugin loads.
  With LNC=2 a logical core is two physical NeuronCore-v4s. NKI kernels run on both
  (`nl.num_programs(0) == 2`) and split the work.
- **Honest aggregates.** Each timing window starts after a barrier, and the aggregate uses the
  slowest rank's time: `world × work / t_slowest`. Summing per-rank rates measured in different
  windows overstates concurrent throughput.
- **Best of 3 windows** after warm-up; compilation happens on local rank 0 first so the other
  ranks reuse its cache.
- **Overhead-free timing for NKI microkernels** (`bench_slope`). The kernel repeats its work R
  times inside one launch; timing two values of R and taking the difference cancels launch and
  sync overhead (a host-side stand-in for device-profile timing). `07_dma_nki.py` marks a result
  `suspect` if that rate is more than 25 % above the plain timed rate.
- **FLOP counting.** 2 per multiply-add. An `nc_matmul_mx` contracts over the partitions *and*
  the 4 values packed in each x4 element: stationary `[128, M]`, moving `[128, N]` →
  `2·M·N·512` FLOPs.
- **Correctness.** GEMM (N ≤ 2048), MXFP8 GEMM, and engine-peak results are checked against
  references and the script exits on failure; relL2 is recorded.

## Results (trn3-dev1.48xlarge, 16 chips, LNC=2, 64 logical cores)

torch-neuronx 2.14.3, neuronx-cc 2.28, nki 0.7, nki-library `ad2f1b9`, 2026-11-10.
Spec ratios use the per-chip figures (671 TF bf16, 183 TF fp32, 2,517 TF MXFP8, 4.9 TB/s
HBM/DMA, 2.56 TB/s NeuronLink) × 16.

| metric | 1 logical core | one chip | instance | % of spec |
|---|---|---|---|---|
| Tensor Engine ceiling, bf16 | 155.6 TF | 622 TF | 9.97 PF | 93 % |
| GEMM bf16 (N = 8192) | 151.5 TF | 606 TF | 9.67 PF | 90 % |
| GEMM fp32 (N = 2048) | 38.8 TF | 155 TF | 2.48 PF | 85 % |
| Tensor Engine ceiling, MXFP8 | 540–574 TF | 1.69 PF | 25.0 PF | 62 % at full load (see below) |
| GEMM MXFP8, pre-quantized (N = 8192) | 491 TF | 1.91 PF | 29.6 PF | 74 % |
| DMA load HBM→SBUF (hwdge) | 1.12 TB/s | 4.48 TB/s | 71.0 TB/s | 91 % |
| DMA store SBUF→HBM (hwdge) | 1.20 TB/s | 4.80 TB/s | 77.6 TB/s | 99 % |
| DMA HBM→HBM (hwdge, read + write) | 1.00 TB/s | 3.99 TB/s | 63.5 TB/s | 81 % |
| STREAM triad, fp32 | 0.93 TB/s | 3.72 TB/s | 59.4 TB/s | — |
| H2D / D2H, 1 GiB | 13.8 / 14.3 GB/s | 14 / 14 GB/s | 113 / 114 GB/s | — |
| all-reduce, 64 ranks | — | — | 7.2 µs (32 B), 446 GB/s busbw | — |
| all-to-all, 64 ranks | — | — | 16.6 µs (32 B), 33 GB/s busbw | — |

Observations:

- **Compute scales almost perfectly.** bf16 per core is unchanged from 1 to 64 cores, and the
  framework GEMM reaches 97 % of the measured engine ceiling.
- **MXFP8 drops ~25 % per core when all 4 cores of a chip run gap-free MX matmuls** (engine
  ceiling 540–574 → ~420 TF per core), reproducibly. The real GEMM, which has gaps, holds ~485
  TF per core, so a power or activity throttle fits. The cause is unconfirmed.
- **One-directional DMA reaches 91–99 % of the 4.9 TB/s per chip.** HBM→HBM copies top out at
  81 % (counting read + write) in every descriptor mode and from both physical cores, so a copy is
  inherently more expensive than a one-way transfer.
- **STREAM is memory-bound, not engine-bound:** the same ~0.9 TB/s per core in fp32 and bf16.
- **Host transfers are limited per device pair.** Devices *2i* and *2i + 1* share ~14 GB/s per
  direction, so the instance caps at ~113 GB/s regardless of process count. Pinned memory and
  NUMA binding make no difference.
- **All-to-all collapses beyond one chip** (294 GB/s busbw at 4 ranks → 33 GB/s at 64), while
  the other collectives hold ~440 GB/s at 64 ranks.
- **Small MXFP8 GEMMs are launch-bound:** N = 2048 runs one ~100 µs kernel per call (~170–220 TF).

## Pitfalls

These each produced silently wrong numbers or crashes while building the suite:

1. **NKI trace cache serves stale kernels.** torch-neuronx 2.14 caches traced NKI kernels across
   processes keyed on the arguments only, so kernels that differ in Python values outside their
   arguments reuse the wrong compiled kernel. The `torchrun` launcher itself enables it for every
   worker, so `common.py` forces it off.
2. **Fresh kernels start slow.** A newly loaded kernel runs ~4× slower for its first executions;
   `bench_slope` warms each kernel for a fixed time and alternates the two measurements.
3. **Repeated identical work gets optimized away** (same data to the same destination).
   Microkernels write a distinct destination per repeat and feed every load into a dependent
   reduction.
4. **`torch.compile` needs `dynamic=False`.** Otherwise the second matrix size is re-traced with
   symbolic shapes, which the Neuron compiler rejects.
5. **Large compiled graphs fail at run time** ("Unable to find output tensor
   intermediate0_sg0000", torch-neuronx 2.14.3). `01_gemm.py` caps work per graph with
   `--graph_macs` (default 2·4096³ multiply-adds), which shortens the chain for large N.
6. **nki-library's `matmul_mxfp8` can't be called several times inside one NKI kernel**: the
   outputs come out wrong. One launch = one GEMM.
7. **Static-descriptor stores** (`07_dma_nki.py --dge none`, kind `store`) give an unreliable
   repeat-count slope and are flagged `suspect`.
8. **Ranks fill cores in order.** 16 ranks land on cores 0–15 = 4 chips; use `CORE_STRIDE` to
   spread them.
9. **`dist.barrier()` fails** on Neuron torch plugins that don't implement the accelerator hooks;
   `common.py` uses a CPU all-reduce instead.
10. **A single 4 GB DMA can hit the 30 s runtime execution timeout**; `05_hbm_copy.py` copies in
    chunks.

## Other instance sizes

Check `neuron-ls` for the chip count and LNC configuration, then set `NPS` to 1 / cores per chip /
total cores, `RANKS` in `04_collectives.sh`, and the `CORE_STRIDE` runs in the H2D section of
`run_micro.sh`.

## Acknowledgements

`01`–`05`, `run_micro.sh` and `aggregate.py` are reworked from the micro-benchmark scripts in
[tonyhuangai2026/trn2-bench](https://github.com/tonyhuangai2026/trn2-bench). The rework fixed
STREAM byte counting (copy and triad ran as two kernels but were counted as one pass), compiled
the GEMM chain (it ran eagerly), replaced summed per-rank aggregates with barrier-synced ones, and
removed per-iteration allocation from the host-transfer test. Ideas for `07_dma_nki.py` and
`08_engine_peak.py` (descriptor-mode sweep, engine-level ceilings) came from a separate Trainium3
NKI microbenchmark suite; the kernels here are independent implementations.

## License

MIT, see [LICENSE](LICENSE).
