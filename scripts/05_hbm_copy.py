# Device-to-device copy_ bandwidth (DMA path) as HBM bandwidth proxy.
# Copies are chunked (--chunk_mb) so a single DMA never hits the NRT 30 s execution timeout
# (a monolithic 4 GB copy did: FATAL-RT-UNDEFINED-STATE).
import argparse
from common import *
p = argparse.ArgumentParser()
p.add_argument("--mb", type=int, default=1024)
p.add_argument("--chunk_mb", type=int, default=512)
p.add_argument("--iters", type=int, default=20)
p.add_argument("--reps", type=int, default=3)
a = p.parse_args()
N = a.mb * 2**20 // 4; CH = max(1, a.chunk_mb * 2**20 // 4)
# independent per-chunk buffers (copy_ between slice views is not supported by every runtime)
sizes = [min(CH, N - i) for i in range(0, N, CH)]
pairs = [(torch.zeros(n, device=dev), torch.randn(n, device=dev)) for n in sizes]
def it():
    for dst, src in pairs: dst.copy_(src)
warm(it)
t_l, t_g = bench(it, a.iters, reps=a.reps)
emit({"bench": "hbm_copy", "mb": a.mb, "chunk_mb": a.chunk_mb},
     2 * N * 4 * a.iters, t_l, t_g, "GBps_rw", 1e9)
