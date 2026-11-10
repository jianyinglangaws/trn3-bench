# Host<->device transfer bandwidth into preallocated buffers (no per-iteration device
# allocation, no fresh pageable host tensor per D2H). --pin uses page-locked host memory.
import argparse
from common import *
p = argparse.ArgumentParser()
p.add_argument("--sizes_mb", default="1,4,16,64,256,1024,4096")
p.add_argument("--iters", type=int, default=10)
p.add_argument("--reps", type=int, default=3)
p.add_argument("--pin", action="store_true")
a = p.parse_args()
pinned = False
for mb in map(int, a.sizes_mb.split(",")):
    nbytes = mb * 2**20
    h = torch.randn(nbytes // 4)                  # fp32 source
    ho = torch.empty_like(h)                      # D2H destination
    if a.pin:
        try:
            h, ho = h.pin_memory(), ho.pin_memory(); pinned = True
        except Exception as e:
            if RANK == 0: print(f"pin_memory unavailable ({e!r}); using pageable", flush=True)
    d = torch.empty(h.shape, dtype=h.dtype, device=dev)
    h2d_l, h2d_g = bench(lambda: d.copy_(h), a.iters, reps=a.reps, sync_each=True)
    d2h_l, d2h_g = bench(lambda: ho.copy_(d), a.iters, reps=a.reps, sync_each=True)
    rec = {"bench": "h2d_d2h", "mb": mb, "pinned": pinned}
    emit({**rec, "dir": "h2d"}, nbytes * a.iters, h2d_l, h2d_g, "GBps", 1e9)
    emit({**rec, "dir": "d2h"}, nbytes * a.iters, d2h_l, d2h_g, "GBps", 1e9)
    del h, ho, d
