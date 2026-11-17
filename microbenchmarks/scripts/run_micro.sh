#!/bin/bash
# Micro-benchmark suite runner. Each step is one torchrun launch; results go to $R/<name>.jsonl
# (+ <name>.log), collectives to $R/collectives/. Summarize with: python3 aggregate.py $R
#
#   bash run_micro.sh                          # everything, at 1 / 4 (one chip) / 64 ranks
#   ONLY="engine gemm" NPS="1 4" bash run_micro.sh
#
# ONLY  subset of: engine gemm mxfp8 stream dma hbm h2d coll   (mxfp8 and engine/mxfp8 need trn3+)
# NPS   rank counts for the per-core sweeps (default "1 4 64"; set to your cores-per-chip / total)
# VENV  torch-neuronx venv to activate (default: newest /opt/aws_neuronx_venv_torch_neuronx*)
# R     results dir (default: <repo>/results/<instance-type>)
# CACHE_DIR  Neuron compile cache (default /tmp/neuron_cache; keep it on local disk, not NFS)
# NKILIB     nki-library checkout for 06_mxfp8_gemm.py (default ~/nki-library)
# TIMEOUT    seconds per step (default 5400)
set -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
VENV=${VENV:-$(ls -d /opt/aws_neuronx_venv_torch_neuronx* 2>/dev/null | sort -V | tail -1)}
[ -n "$VENV" ] && source "$VENV/bin/activate"
ITYPE=$(neuron-ls 2>/dev/null | awk '/instance-type:/ {print $2; exit}')
R=${R:-$HERE/../results/${ITYPE:-unknown-instance}}; mkdir -p "$R"
export NEURON_COMPILE_CACHE_URL=${CACHE_DIR:-${NEURON_COMPILE_CACHE_URL:-/tmp/neuron_cache}}
NPS=${NPS:-"1 4 64"}; ONLY=${ONLY:-"engine gemm mxfp8 stream dma hbm h2d coll"}
want() { [[ " $ONLY " == *" $1 "* ]]; }
cd "$HERE"
run() { # name nproc script args...
  local name=$1 np=$2; shift 2
  echo "=== $(date -u +%T) $name (np=$np)"
  rm -f "$R/$name.jsonl"
  OUT=$R/$name.jsonl timeout ${TIMEOUT:-5400} torchrun --nproc_per_node=$np "$@" > "$R/$name.log" 2>&1 \
    || echo "!!! $name failed (exit $?), see $R/$name.log"
  grep -hE '^\{' "$R/$name.log" | tail -8
}
if want engine; then   # Tensor Engine ceiling (NKI, SBUF-resident back-to-back matmuls)
  for np in $NPS; do run engine_peak_np$np $np 08_engine_peak.py; done
fi
if want gemm; then
  for np in $NPS; do run gemm_bf16_np$np $np 01_gemm.py --dtype bf16 --sizes 1024,2048,4096,8192,16384; done
  for np in $NPS; do run gemm_fp32_np$np $np 01_gemm.py --dtype fp32 --sizes 2048,4096,8192; done
fi
if want mxfp8; then
  for np in $NPS; do run gemm_mxfp8_np$np $np 06_mxfp8_gemm.py --mode prequant; done
  run gemm_mxfp8_bf16in_np1 1 06_mxfp8_gemm.py --mode bf16
fi
if want stream; then
  for np in $NPS; do run stream_np$np $np 02_stream.py --mb 1024; done
  run stream_bf16_np1 1 02_stream.py --mb 1024 --dtype bf16   # engine- vs memory-bound check
fi
if want h2d; then
  run h2d_np1 1 03_h2d.py
  run h2d_pin_np1 1 03_h2d.py --pin
  # Aggregate PCIe: one chip (4 cores), one core on each of 16 chips, one per device pair, all
  # 64 cores. CORE_STRIDE spreads ranks: 16 ranks without it land on cores 0-15 = only 4 chips.
  # (On trn3-dev1.48xlarge, devices 2i/2i+1 share ~14 GB/s per direction.)
  run h2d_np4 4 03_h2d.py --sizes_mb 16,64,256,1024
  CORE_STRIDE=4 run h2d_stride4_np16 16 03_h2d.py --sizes_mb 16,64,256,1024
  CORE_STRIDE=8 run h2d_stride8_np8 8 03_h2d.py --sizes_mb 16,64,256,1024   # 1 core per device pair
  run h2d_np64 64 03_h2d.py --sizes_mb 16,64,256,1024
fi
if want dma; then      # NKI dma_copy per descriptor mode: HBM->HBM, HBM->SBUF, SBUF->HBM
  for np in $NPS; do run dma_nki_np$np $np 07_dma_nki.py; done
fi
if want hbm; then
  run hbm_copy_np1 1 05_hbm_copy.py --mb 4096
  run hbm_copy_np4 4 05_hbm_copy.py --mb 1024   # one chip
  run hbm_copy_np64 64 05_hbm_copy.py --mb 1024
fi
want coll && OUT=$R/collectives bash "$HERE/04_collectives.sh"
echo "=== $(date -u +%T) micro done"
