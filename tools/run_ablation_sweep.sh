#!/usr/bin/env bash
# run_ablation_sweep.sh — Run all toy-scale ablation configs × 3 seeds on CUDA.
#
# Usage (from repo root):
#   tools/run_ablation_sweep.sh [--backend cpu|cuda]
#
# Runs all 28 configs in configs/ablations/ with seeds 7, 42, 123.
# Total: 84 runs. Results appended to bench/results.jsonl.
# Per-run logs go to logs/<tag>_s<seed>.log.
#
# Launch detached:
#   mkdir -p logs && nohup tools/run_ablation_sweep.sh --backend cuda \
#     > logs/ablation_sweep.log 2>&1 & echo $! > logs/ablation_sweep.pid

set -u

BACKEND="cuda"
while [ $# -gt 0 ]; do
    case "$1" in
        --backend) shift; BACKEND="$1"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

SEEDS="7 42 123"
CONFIGS=$(ls configs/ablations/*.json | sort)
N=$(echo "$CONFIGS" | wc -w)
TOTAL=$((N * $(echo "$SEEDS" | wc -w)))

echo "[ablation] backend=$BACKEND configs=$N seeds=3 total_runs=$TOTAL"
echo "[ablation] started $(date -Is)"

run=0
for seed in $SEEDS; do
    for cfg in $CONFIGS; do
        run=$((run + 1))
        echo "[ablation] [$run/$TOTAL] seed=$seed config=$cfg ($(date -Is))"
        tools/run_sweep.sh --seed "$seed" --backend "$BACKEND" "$cfg"
    done
done

echo "[ablation] all done ($(date -Is))"
