# GEMM micro-benchmark: chained square matmuls (y = y @ w) fused into one compiled graph
# (COMPILE=0 falls back to eager, one dispatch per matmul).
import argparse
from common import *
p = argparse.ArgumentParser()
p.add_argument("--sizes", default="1024,2048,4096,8192,16384")
p.add_argument("--dtype", default="bf16")
p.add_argument("--chain", type=int, default=16)   # matmuls per graph
p.add_argument("--iters", type=int, default=20)
p.add_argument("--reps", type=int, default=3)
# Cap matmul work per compiled graph (in units of N^3 multiply-adds). Above ~2*4096^3 the Neuron
# compiler splits the graph into subgraphs whose intermediate the torch-neuronx 2.14 executor
# fails to bind ("Unable to find output tensor intermediate0_sg0000" on trn3).
p.add_argument("--graph_macs", type=float, default=2 * 4096 ** 3)
p.add_argument("--check_max_n", type=int, default=2048, help="correctness-check sizes <= this; 0=off")
a = p.parse_args()
dt = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp16": torch.float16}[a.dtype]
for n in map(int, a.sizes.split(",")):
    chain = max(1, min(a.chain, int(a.graph_macs // n ** 3)))
    iters = a.iters if n <= 8192 else max(5, a.iters // 2)
    x_h = (torch.randn(n, n) * 0.01).to(dt)
    # near-identity so values stay bounded across thousands of chained matmuls
    w_h = (torch.eye(n) + torch.randn(n, n) * (0.01 / n ** 0.5)).to(dt)
    x, w = x_h.to(dev), w_h.to(dev)
    def chain_fn(y, w):
        for _ in range(chain): y = y @ w
        return y
    step, mode = maybe_compile(chain_fn)
    state = {"y": x}
    def it(): state["y"] = step(state["y"], w)
    warm(it)   # compiles (local rank 0 first); the check below reuses the compiled graph
    rel = None
    if n <= a.check_max_n:   # one chained call vs an fp32 CPU reference on the same rounded inputs
        ref = chain_fn(x_h.float(), w_h.float())
        rel = ((step(x, w).float().cpu() - ref).norm() / ref.norm()).item()
        if not rel < 0.05:
            sys.exit(f"GEMM correctness check failed at n={n}: relL2={rel:.4g}")
    t_l, t_g = bench(it, iters, reps=a.reps)
    emit({"bench": "gemm", "dtype": a.dtype, "n": n, "chain": chain, "mode": mode, "relL2": rel},
         2 * n ** 3 * chain * iters, t_l, t_g, "tflops", 1e12)
    del x, w, state
