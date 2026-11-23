# MXFP8 GEMM (OCP microscaling, e4m3 data + e8m0 scale per 32-element block) on NeuronCore-v4
# (Trainium3+), using the nki-library experimental kernel nkilib.experimental.matmul_mxfp8.
#   --mode prequant : operands already MXFP8 in HBM -> measures the MX matmul itself
#   --mode bf16     : bf16 [M,K] x [N,K] inputs, quantized + swizzled inside the kernel each call
#   --mode fp4      : MXFP4 (e2m1 data + e8m0 scale) operands already in HBM, same kernel: the data
#                     tensors are viewed as float4_e2m1fn_x4 (4 nibbles per 16-bit element), so the
#                     Tensor Engine runs nc_matmul_mx on FP4. No in-kernel FP4 quantizer exists
#                     (nisa.quantize_mx emits FP8 only), so there is no bf16-input FP4 mode.
# Needs nki-library: NKILIB=/path/to/nki-library (default ~/nki-library).
# FLOPs counted as 2*M*N*K. One kernel call = one full GEMM (no chaining).
import argparse
from common import *
NKILIB = os.path.expanduser(os.environ.get("NKILIB", "~/nki-library"))
sys.path[:0] = [NKILIB, os.path.join(NKILIB, "src")]   # nkilib_src.* and test.utils.* (golden quantizer)
import numpy as np
import nki, nki.language as nl
from nkilib_src.nkilib.experimental.matmul_mxfp8.matmul_mxfp8_generic_kernel import matmul_mxfp8

p = argparse.ArgumentParser()
p.add_argument("--sizes", default="2048,4096,8192,16384")   # square M=K=N
p.add_argument("--mode", default="prequant", choices=["prequant", "bf16", "fp4"])
p.add_argument("--iters", type=int, default=20)
p.add_argument("--reps", type=int, default=3)
p.add_argument("--check_n", type=int, default=1024, help="accuracy check size (golden quantizer); 0=skip")
a = p.parse_args()
LNC2 = os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2") == "2"   # kernel shards over both physical cores


@nki.jit
def mx_prequant(lq, ls, rq, rs):
    return matmul_mxfp8(lq, rq, lhs_scales=ls, rhs_scales=rs, run_with_lnc2=LNC2, output_dtype=nl.bfloat16)


@nki.jit
def mx_fp4(lq, ls, rq, rs):   # lq/rq: int16 [K/4, F] holding packed e2m1 codes
    return matmul_mxfp8(lq.view(nl.float4_e2m1fn_x4), rq.view(nl.float4_e2m1fn_x4), lhs_scales=ls,
                        rhs_scales=rs, run_with_lnc2=LNC2, output_dtype=nl.bfloat16)


@nki.jit
def mx_bf16(lhs, rhs):   # lhs [M, K], rhs [N, K]
    return matmul_mxfp8(lhs, rhs, lhs_is_swizzled=False, rhs_is_swizzled=False, run_with_lnc2=LNC2,
                        output_dtype=nl.bfloat16)


def golden_quant(x_kf):
    """[K, F] fp32 -> (fp8 data [K/4, 4F] (wrapX swizzled), e8m0 scales [K/32, F]) via nki-library golden."""
    from test.integration.nkilib.experimental.matmul_mxfp8.utils import swizzle_tensor
    from test.utils.mx_utils import quantize_mx_golden
    q, s = quantize_mx_golden(swizzle_tensor(x_kf.astype(nl.bfloat16)), nl.float8_e4m3fn_x4)
    q = torch.from_numpy(np.ascontiguousarray(q).view(np.uint8)).view(torch.float8_e4m3fn)
    s = torch.from_numpy(np.ascontiguousarray(s).view(np.uint8)).view(torch.float8_e8m0fnu)
    return q.reshape(x_kf.shape[0] // 4, -1).to(dev), s.to(dev)


def random_mx(k, f):
    """Valid MXFP8 operand without the (slow) host quantizer: e4m3 values, unit (2^0) scales."""
    q = (torch.randn(k // 4, 4 * f) * 0.5).to(torch.float8_e4m3fn)
    s = torch.full((k // 32, f), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    return q.to(dev), s.to(dev)


E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])


def random_fp4(k, f):
    """MXFP4 operand: random e2m1 codes packed 4 per int16 (element j in bits 4j..4j+3), unit
    scales. Returns device (data [K/4, F], scales [K/32, F]) and the values [K, F] (K = 4*row + j;
    any K order common to both operands gives the same product)."""
    codes = torch.randint(0, 16, (k // 4, f, 4))
    packed = sum(codes[..., j] << (4 * j) for j in range(4))
    packed = torch.where(packed >= 32768, packed - 65536, packed).to(torch.int16)
    s = torch.full((k // 32, f), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    return (packed.to(dev), s.to(dev)), E2M1[codes].permute(0, 2, 1).reshape(k, f)


def check(n):
    """relL2 of the kernel vs a bf16 matmul of the unquantized inputs (MXFP8 e4m3 ~0.04 expected).
    fp4: vs the exact fp32 product of the e2m1 values (only bf16 output rounding, ~2e-3)."""
    if a.mode == "fp4":
        (l, lv), (r, rv) = random_fp4(n, n), random_fp4(n, n)
        ref = lv.T @ rv
        return ((mx_fp4(*l, *r).float().cpu() - ref).norm() / ref.norm()).item()
    A = np.random.randn(n, n).astype(np.float32) * 0.5; B = np.random.randn(n, n).astype(np.float32) * 0.5
    if a.mode == "prequant":
        out = mx_prequant(*golden_quant(np.ascontiguousarray(A.T)), *golden_quant(B))
    else:
        out = mx_bf16(torch.from_numpy(A).bfloat16().to(dev), torch.from_numpy(np.ascontiguousarray(B.T)).bfloat16().to(dev))
    ref = torch.from_numpy(A).bfloat16().float() @ torch.from_numpy(B).bfloat16().float()
    o = out.float().cpu()
    return ((o - ref).norm() / ref.norm()).item()


rel = None
if a.check_n and LR == 0:
    rel = check(a.check_n)
    print(f"[mx {a.mode}] accuracy check n={a.check_n}: relL2={rel:.4f}", flush=True)
    if not rel < (0.01 if a.mode == "fp4" else 0.1):
        sys.exit(f"MX {a.mode} accuracy check failed (relL2={rel})")
barrier()

for n in map(int, a.sizes.split(",")):
    if a.mode == "prequant":
        (lq, ls), (rq, rs) = random_mx(n, n), random_mx(n, n)
        fn = lambda: mx_prequant(lq, ls, rq, rs)
    elif a.mode == "fp4":
        ((lq, ls), _), ((rq, rs), _) = random_fp4(n, n), random_fp4(n, n)
        fn = lambda: mx_fp4(lq, ls, rq, rs)
    else:
        lhs = (torch.randn(n, n) * 0.5).bfloat16().to(dev); rhs = (torch.randn(n, n) * 0.5).bfloat16().to(dev)
        fn = lambda: mx_bf16(lhs, rhs)
    iters = a.iters if n <= 8192 else max(5, a.iters // 2)
    warm(fn)
    t_l, t_g = bench(fn, iters, reps=a.reps)
    emit({"bench": "gemm", "dtype": {"prequant": "mxfp8", "fp4": "mxfp4"}.get(a.mode, "mxfp8_bf16in"), "n": n, "chain": 1,
          "mode": f"nki_{a.mode}", "relL2": rel}, 2 * n ** 3 * iters, t_l, t_g, "tflops", 1e12)
