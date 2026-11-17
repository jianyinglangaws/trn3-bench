#!/bin/bash
# nccom-test sweep (32 B .. 4 GiB, bf16) of all-reduce / all-gather / reduce-scatter / all-to-all.
# RANKS (default "4 16 64"), OUT (default <repo>/results/collectives).
OUT=${OUT:-$(cd "$(dirname "$0")" && pwd)/../results/collectives}; mkdir -p "$OUT"
export PATH=/opt/aws/neuron/bin:$PATH
for R in ${RANKS:-4 16 64}; do
  for op in allr allg redsct alltoall; do
    echo "=== $(date -u +%T) ranks=$R op=$op"
    timeout 1200 nccom-test -r $R -b 32 -e 4G -f 4 -n 20 -w 5 -d bf16 $op > "$OUT/${op}_r${R}.txt" 2>&1
    tail -5 "$OUT/${op}_r${R}.txt"
  done
done
echo "=== collectives done"
