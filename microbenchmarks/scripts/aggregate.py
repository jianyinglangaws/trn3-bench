#!/usr/bin/env python3
"""Summarize a results directory (jsonl from the micro-benchmarks + nccom-test output) as markdown.

    python3 aggregate.py [RESULTS_DIR] > RESULTS_DIR/summary_tables.md

Aggregate columns use the {"rank": "agg"} record each measurement writes (WORLD * work / slowest
rank's window). Result files from before that record existed fall back to summing per-rank rates.
"""
import collections, glob, json, os, re, sys

RD = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "../results"))
out = []


def section(title, *cols):
    out.append(f"\n## {title}\n")
    row(*cols)
    row(*["---"] * len(cols))


def row(*cells):
    out.append("| " + " | ".join(str(c) for c in cells) + " |")


def records(f):
    return [json.loads(l) for l in open(f) if l.strip()]


def load(f, key, val):
    """{key(r): (mean over ranks, aggregate)} for records carrying field `val`."""
    per, agg = collections.defaultdict(list), {}
    for r in records(f):
        if val not in r or key(r) is None:
            continue
        (agg.__setitem__(key(r), r[val]) if r["rank"] == "agg" else per[key(r)].append(r[val]))
    return {k: (sum(v) / len(v), agg.get(k, sum(v))) for k, v in per.items()}


def first(f, key, field):
    """{key(r): r[field]} from the first record per key (fields identical across ranks)."""
    d = {}
    for r in records(f):
        d.setdefault(key(r), r.get(field))
    return d


def files(pattern):
    """Matching files sorted by their _np<N> rank count."""
    fs = glob.glob(f"{RD}/{pattern}")
    return sorted(fs, key=lambda f: (re.sub(r"_np\d+", "", f), nprocs(f)))


def nprocs(f):
    return int(re.search(r"_np(\d+)", f).group(1))


def ranks_label(f):
    st = re.search(r"_stride(\d+)_", f)
    return f"{nprocs(f)} (core stride {st.group(1)})" if st else nprocs(f)


section("Tensor Engine peak (NKI back-to-back matmul, SBUF-resident)",
        "kind", "ranks", "per-core TFLOPs", "aggregate TFLOPs", "relL2")
for f in files("engine_peak_np*.jsonl"):
    rel = first(f, lambda r: r["kind"], "relL2")
    for k, (m, ag) in sorted(load(f, lambda r: r["kind"], "tflops").items()):
        row(k, nprocs(f), f"{m:.1f}", f"{ag:.0f}", f"{rel[k]:.4f}")

section("GEMM (bf16/fp32: torch chained square matmul; mxfp8: nki-library kernel)",
        "dtype", "ranks", "N", "per-core TFLOPs", "aggregate TFLOPs")
for f in files("gemm_*np*.jsonl"):
    dt = re.search(r"gemm_(\w+?)_np\d+", f).group(1)
    for n, (m, ag) in sorted(load(f, lambda r: r["n"], "tflops").items()):
        row(dt, nprocs(f), n, f"{m:.1f}", f"{ag:.0f}")

section("DMA (NKI dma_copy; hbm2hbm counts read+write)",
        "kind", "dge", "ranks", "per-core GB/s", "aggregate TB/s", "suspect")
for f in files("dma_nki_np*.jsonl"):
    key = lambda r: (r["kind"], r["dge"])
    sus = first(f, key, "suspect")
    for k, (m, ag) in sorted(load(f, key, "GBps").items()):
        row(*k, nprocs(f), f"{m:.0f}", f"{ag / 1000:.2f}", "YES" if sus[k] else "")

section("HBM copy_ (torch DMA path, read+write)", "ranks", "MB", "per-core GB/s", "aggregate TB/s")
for f in files("hbm_copy_np*.jsonl"):
    for mb, (m, ag) in sorted(load(f, lambda r: r["mb"], "GBps_rw").items()):
        row(nprocs(f), mb, f"{m:.0f}", f"{ag / 1000:.2f}")

section("STREAM (torch elementwise, 1 kernel per op)", "dtype", "ranks", "kernel", "per-core GB/s", "aggregate GB/s")
for f in files("stream_*np*.jsonl"):
    dt = first(f, lambda r: 0, "dtype").get(0) or "fp32"
    d = load(f, lambda r: r["kernel"], "GBps")
    for k in ("copy", "scale", "add", "triad"):
        if k in d:
            row(dt, nprocs(f), k, f"{d[k][0]:.0f}", f"{d[k][1]:.0f}")

section("Host<->Device bandwidth (fp32 tensors)",
        "ranks", "MB", "host mem", "H2D GB/s (per-proc / agg)", "D2H GB/s (per-proc / agg)")
for f in files("h2d_*np*.jsonl"):
    if any("h2d_GBps" in r for r in records(f)):   # old format: both directions in one record
        h, d = load(f, lambda r: r["mb"], "h2d_GBps"), load(f, lambda r: r["mb"], "d2h_GBps")
    else:
        h = load(f, lambda r: r["mb"] if r["dir"] == "h2d" else None, "GBps")
        d = load(f, lambda r: r["mb"] if r["dir"] == "d2h" else None, "GBps")
    for mb in sorted(h):
        row(ranks_label(f), mb, "pinned" if "_pin" in f else "pageable",
            f"{h[mb][0]:.1f} / {h[mb][1]:.0f}", f"{d[mb][0]:.1f} / {d[mb][1]:.0f}")

section("Collectives (nccom-test, bf16)", "op", "ranks", "size B", "time us", "algbw GB/s", "busbw GB/s")
for f in sorted(glob.glob(f"{RD}/collectives/*.txt")):
    op, r = re.search(r"(\w+)_r(\d+)", os.path.basename(f)).groups()
    text = open(f, errors="ignore").read().replace("\r", "\n")   # nccom-test progress uses \r
    for ln in text.splitlines():
        m = re.match(r"\s*(\d+)\s+\d+\s+\w+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", ln)
        if m:
            row(op, r, *m.groups())

print("\n".join(out))
