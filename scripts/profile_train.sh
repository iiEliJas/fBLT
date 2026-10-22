#!/usr/bin/env bash
set -euo pipefail

build_dir="${1:-build-cuda}"
shift || true
out_dir="${OUT_DIR:-runs/profiles}"
mkdir -p "$out_dir"
stamp="$(date +%Y%m%d-%H%M%S)"

if ! command -v nsys >/dev/null 2>&1; then
    echo "nsys not found; install Nsight Systems or run the binary directly" >&2
    exit 1
fi

nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  -o "$out_dir/train_$stamp" \
  "$build_dir/train_blt_d" "$@"

echo "report: $out_dir/train_$stamp.nsys-rep"
echo "stats:  nsys stats $out_dir/train_$stamp.nsys-rep"