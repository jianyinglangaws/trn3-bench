# Tensor Engine peak: back-to-back nc_matmul (bf16) / nc_matmul_mx (MXFP8 e4m3, NeuronCore-v4+)
# on SBUF-resident operands, rotating 4 PSUM banks. This is the instruction-throughput ceiling that
# real GEMMs (01_gemm.py, 06_mxfp8_gemm.py) should be judged against.
# FLOPs per instruction (2 per MAC):
#   bf16: stationary [K=128, M=128], moving [K=128, N]          -> 2*M*N*K
#   mx  : stationary [128, M=128] x4, moving [128, N] x4        -> 2*M*N*(4*128)
#         (contraction runs over the partitions AND the 4 values packed in each x4 element)
# Operands are quantized on device (nisa.quantize_mx), so the MX layout is exactly what the
# hardware expects. Correctness: the accumulated result / G must match A^T B.
# Runs on every physical core of the logical core; timing = bench_slope over G.
import argparse
from common import *
import nki, nki.isa as nisa, nki.language as nl

p = argparse.ArgumentParser()
p.add_argument("--kinds", default="bf16,mxfp8")
p.add_argument("--n", type=int, default=512, help="moving free size (512 = one fp32 PSUM bank)")
p.add_argument("--g_lo", type=int, default=2000); p.add_argument("--g_hi", type=int, default=10000)
p.add_argument("--iters", type=int, default=5); p.add_argument("--reps", type=int, default=3)
a = p.parse_args()
P = M = 128; N = a.n; NBK = 4


def make_kernel(kind, G):
    @nki.jit
    def k(x, y):   # bf16: x [P, M], y [P, N];  mxfp8: x [P, 4M], y [P, 4N] (4 packed K values per column)
        npg = nl.num_programs(0); pid = nl.program_id(0)
        out = nl.ndarray((npg, M, N), dtype=nl.float32, buffer=nl.shared_hbm)
        xs = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf); nisa.dma_copy(dst=xs, src=x)
        ys = nl.ndarray(y.shape, dtype=y.dtype, buffer=nl.sbuf); nisa.dma_copy(dst=ys, src=y)
        if kind == "mxfp8":
            xq = nl.ndarray((P, M), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
            xsc = nl.ndarray((P, M), dtype=nl.uint8, buffer=nl.sbuf)
            yq = nl.ndarray((P, N), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
            ysc = nl.ndarray((P, N), dtype=nl.uint8, buffer=nl.sbuf)
            nisa.quantize_mx(dst=xq, src=xs, dst_scale=xsc)
            nisa.quantize_mx(dst=yq, src=ys, dst_scale=ysc)
        ps = [nl.ndarray((M, N), dtype=nl.float32, buffer=nl.psum) for _ in range(NBK)]
        for j in range(NBK):
            nisa.memset(ps[j], 0.0)
        for i in nl.affine_range(G // NBK):
            for j in range(NBK):
                if kind == "mxfp8":
                    nisa.nc_matmul_mx(ps[j], stationary=xq, moving=yq, stationary_scale=xsc,
                                      moving_scale=ysc, accumulate=True)
                else:
                    nisa.nc_matmul(ps[j], stationary=xs, moving=ys, accumulate=True)
        res = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=res, src=ps[0])
        for j in range(1, NBK):
            nisa.tensor_tensor(dst=res, data1=res, data2=ps[j], op=nl.add)
        nisa.dma_copy(dst=out[pid], src=res)
        return out
    return k


for kind in a.kinds.split(","):
    pk = 4 if kind == "mxfp8" else 1
    x = torch.randn(P, pk * M) * 0.5; y = torch.randn(P, pk * N) * 0.5
    unpack = lambda t, f: t.bfloat16().float().reshape(P, f, pk).permute(0, 2, 1).reshape(pk * P, f)
    ref = unpack(x, M).T @ unpack(y, N)
    xd, yd = x.bfloat16().to(dev), y.bfloat16().to(dev)
    # correctness at g_lo (also tells us how many physical cores the kernel ran on)
    o = make_kernel(kind, a.g_lo)(xd, yd).cpu()
    npg = o.shape[0]
    rel = max(((o[i] / a.g_lo - ref).norm() / ref.norm()).item() for i in range(npg))
    tol = 0.1 if kind == "mxfp8" else 1e-2
    if not rel < tol:
        sys.exit(f"{kind} engine-peak correctness failed: relL2={rel:.4f} (tol {tol})")
    launch = lambda G: (lambda k=make_kernel(kind, G): k(xd, yd))      # bind a fresh kernel per G
    (dt_l, dt_g), _ = bench_slope(launch, a.g_lo, a.g_hi, a.iters, a.reps)
    flop = 2 * M * N * pk * P * npg            # per instruction-slot, summed over physical cores
    emit({"bench": "engine_peak", "kind": kind, "n": N, "physical_cores": npg, "relL2": rel},
         flop * (a.g_hi - a.g_lo) * a.iters, dt_l, dt_g, "tflops", 1e12)
