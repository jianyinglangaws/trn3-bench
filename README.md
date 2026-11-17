# trn3-bench: AWS Trainium3 benchmarks

| directory | contents |
|---|---|
| [`microbenchmarks/`](microbenchmarks/) | Single-node micro-benchmarks: Tensor Engine ceiling, bf16 / fp32 / MXFP8 GEMM, HBM bandwidth (framework copy, STREAM, NKI DMA modes), host-device transfers and NeuronLink collectives. Each runs at 1 logical core, one chip and the full instance, and the results are compared with the published Trainium3 specs |
| [`hunyuan3-vllm/`](hunyuan3-vllm/) | Hunyuan 3 (tencent/Hy3-FP8) served with vllm-neuron on one trn3 node, for 85k shared-prefix prompts under a TTFT / TPOT SLO. It holds a vllm-neuron patch series, how to reproduce, and the results (about 800 output tok/s at C = 20 with every SLO met) |

## License

MIT, see [LICENSE](LICENSE).
