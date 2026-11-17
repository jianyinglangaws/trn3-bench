# Hy3-FP8 serving on one trn3 node: 85k shared prefix under a latency SLO

This directory holds a vllm-neuron patch series and the results it gives. The series serves tencent/Hy3-FP8 on one
trn3 node, with prefill and decode on the same server, under the following SLO:

| metric | SLO |
|---|---|
| TTFT P50 | < 7 s |
| TPOT P50 | < 20 ms |
| prefix-cache hit rate | > 92% |

The workload is 85,000-token prompts with a 78,336-token shared prefix and 1024 output tokens, sampled at temperature
0.9 in a closed loop at concurrency C.

**Result:** C = 20 gives about 800 output tok/s per node with every SLO met. MATH-500 is 97.0% and needle 12/12.

## The patch series

`patches/` contains three commits for
[vllm-neuron](https://github.com/qingzwang/vllm-neuron) branch `hy3-fp8-contrib`. They apply on commit `e891054`
(PR #17):

| patch | contents |
|---|---|
| `0001` | **Runtime** (`vllm_neuron/`): the trn3 spec-decode slot-mapping compile fix (neuronx-cc `NCC_ILSM901`); gang prefill scheduling (`HY3_PREFILL_GANG`); profiler options from `$VLLM_NEURON_PROFILER_JSON` |
| `0002` | **Model** (`contrib/models/hy_v3/src`): trn3 prefill / sparse-attention / decode optimizations, each behind an `HY3_*` flag that is off by default; module tests |
| `0003` | **Config**: `scripts/serve_85k_trn3.sh`, `scripts/bench_85k_slo.sh`, the trn3 FP8 KV-scale profile (448 range), `docs/TRN3_85K_SLO.md` |

`docs/TRN3_85K_SLO.md` (added by `0003`) lists every flag of the configuration, the decode profile, and the switches
that were measured and left off.

## Apply, serve, benchmark

```bash
git clone https://github.com/qingzwang/vllm-neuron.git && cd vllm-neuron
git checkout -b hy3-trn3-85k-slo e891054
git am /path/to/trn3-bench/hunyuan3-vllm/patches/*.patch

cd contrib/models/hy_v3 && pip install --no-deps -e .
export PYTHONPATH=$PWD/../../..          # use this checkout's vllm_neuron runtime
MODEL=/local/nvme/Hy3-FP8 bash scripts/serve_85k_trn3.sh          # fresh compile ~50 min
bash scripts/bench_85k_slo.sh results/trn3_85k "20 24"             # in another shell
```

Keep the model, the compile caches and `HOME`/`TMPDIR` on local NVMe, as for the [micro-benchmarks](../microbenchmarks/).
`GANG_WAIT=5.5` trades TTFT for TPOT at C = 24.

## Results

One trn3 node, vllm-neuron 0.29.0.1.2.0 image (Python 3.14, vLLM 0.29.0), neuronx-cc 2.28, TP=64 / EP=64, LNC=2,
FP8 KV, MTP k=3, sparse attention 0.0625. These come from the patched tree with a fresh compile
(`validation.txt`):

| C | output tok/s | TPOT P50 | TTFT P50 | prefix hit | MTP tok/step | SLO |
|---|---|---|---|---|---|---|
| 20 | 807 | 18.80 ms | 5.27 s | 92.1% | 2.15 | pass |
| 20 | 802 | 19.28 ms | 5.01 s | 92.1% | 2.16 | pass |
| 24 | 888 | 21.36 ms | 5.25 s | 92.1% | 2.15 | fail (TPOT) |
| 24 | 898 | 20.06 ms | 5.63 s | 92.1% | 2.16 | fail (TPOT) |

Accuracy: MATH-500 no_think (temperature 0.9) 97.0% (485/500); needle at 85k, 12 depths: 12/12.

Repeat runs of the same configuration, before the rebase onto `e891054`:
- C = 20: all 71 runs pass, mean 776 tok/s (gang wait 5.5).
- C = 24: 7 of 13 runs pass, TPOT P50 19.5-20.5 ms (gang wait 5.5).

C = 24 is on the TPOT limit.

### Where the time goes (decode, batch 24)

One target step takes 28.1 ms, plus 3.7 ms for the MTP draft. On TP0 that is attention 10.5 ms, routed experts
11.8 ms and collectives 4.2 ms. On one physical core, 161 BF16 all-reduces of 786 KB take 3.95 ms per step.

At C = 24, TPOT P50 is about 14.8 ms of decode plus 6-7 ms of prefill stalls per token. The next levers are the
per-layer all-reduces, the 1.3 ms of torch ops in the sparse selection, and the prefill stalls.

### Context

The trn2 1P1D chat end-to-end result in the vllm-neuron Hy3 README is TPOT 19.38 ms at C = 8 on two trn2 nodes, about
192 output tok/s per node.
