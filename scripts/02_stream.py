# STREAM copy/scale/add/triad on device-resident tensors (framework elementwise path).
# Every kernel is a single op writing into a preallocated C, so the bytes counted (STREAM
# convention: 2N copy/scale, 3N add/triad) match the bytes actually moved even in eager mode.
# (The old `A*1.0+0.0` / `A + s*B` forms ran as 2 kernels and under-reported copy/triad.)
import argparse
from common import *
p = argparse.ArgumentParser()
p.add_argument("--mb", type=int, default=1024)     # MB per array
p.add_argument("--iters", type=int, default=30)
p.add_argument("--reps", type=int, default=3)
p.add_argument("--dtype", default="fp32")
a = p.parse_args()
dt = {"bf16": torch.bfloat16, "fp32": torch.float32}[a.dtype]
esz = torch.tensor([], dtype=dt).element_size()
N = a.mb * 2**20 // esz
A = torch.ones(N, dtype=dt, device=dev); B = torch.full((N,), 2.0, dtype=dt, device=dev)
C = torch.zeros(N, dtype=dt, device=dev); s = 3.0
one = torch.ones((), dtype=dt, device=dev)   # runtime 0-d tensor: multiply can't be folded away
wait()
kernels = {  # name: (fn, arrays touched per call)
    "copy":  (lambda: torch.mul(A, one, out=C), 2),
    "scale": (lambda: torch.mul(B, s, out=C), 2),
    "add":   (lambda: torch.add(A, B, out=C), 3),
    "triad": (lambda: torch.add(A, B, alpha=s, out=C), 3),   # C = A + s*B, one kernel
}
mode = "eager" if os.environ.get("COMPILE", "0") == "0" else "compiled"
for name, (fn, nb) in kernels.items():
    if mode == "compiled":
        fn = maybe_compile(fn)[0]
    warm(fn)
    t_l, t_g = bench(fn, a.iters, reps=a.reps)
    emit({"bench": "stream", "kernel": name, "dtype": a.dtype, "mb": a.mb, "mode": mode},
         nb * N * esz * a.iters, t_l, t_g, "GBps", 1e9)
