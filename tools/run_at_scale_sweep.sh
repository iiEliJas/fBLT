#!/usr/bin/env bash
# run_at_scale_sweep.sh — Run all 9 at-scale arms × 3 seeds on CUDA.
#
# Usage (from repo root):
#   tools/run_at_scale_sweep.sh [--backend cpu|cuda]
#
# Runs the 9 arms from BENCHMARKS.md §9 using train_blt_d with inline flags.
# Each arm × seed produces one log in logs/s6_<arm>_s<seed>.log.
# BPB results are appended to bench/results.jsonl as JSONL entries.
#
# Launch detached:
#   mkdir -p logs && nohup tools/run_at_scale_sweep.sh --backend cuda \
#     > logs/at_scale_sweep.log 2>&1 & echo $! > logs/at_scale_sweep.pid

set -u

BACKEND="cuda"
while [ $# -gt 0 ]; do
    case "$1" in
        --backend) shift; BACKEND="$1"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

SEEDS="7 42 123"

# Base flags shared by all arms (except e256)
BASE="--corpus data/train.bin --eval-corpus data/heldout.bin --eval-windows 300"
BASE="$BASE --embed 128 --hidden 256 --layers 2 --window 96 --block-size 4"
BASE="$BASE --steps 20000 --lr 0.05 --lr-decay 1 --backend $BACKEND"

# Diffusion defaults (arms that use diffusion)
DIFF="--diffusion 1 --mask-warmup 10000 --mask-scale 0.3 --t-min 0.1"

# Write arm definitions: name|extra_flags
ARMS=(
    "plain|--diffusion 0"
    "base|$DIFF"
    "dec1|$DIFF --dec-layers 1"
    "dec3|$DIFF --dec-layers 3"
    "xlast|$DIFF --cross-attn last"
    "late|$DIFF --mask-late-step 14000 --mask-late-scale 0.5"
    "hit|$DIFF --t-warmup-hi 0.25 --t-hi-start 0.8"
    "entp|$DIFF --entropy-patches --entropy-lm runs/entlm.bin"
    "e256|--diffusion 1 --mask-warmup 10000 --mask-scale 0.3 --t-min 0.1 --embed 256 --hidden 512 --window 192 --steps 10000"
)

N_ARMS=${#ARMS[@]}
TOTAL=$((N_ARMS * $(echo "$SEEDS" | wc -w)))

echo "[at-scale] backend=$BACKEND arms=$N_ARMS seeds=3 total_runs=$TOTAL"
echo "[at-scale] started $(date -Is)"

run=0
for seed in $SEEDS; do
    for arm_def in "${ARMS[@]}"; do
        arm="${arm_def%%|*}"
        extra="${arm_def#*|}"
        run=$((run + 1))

        log="logs/s6_${arm}_s${seed}.log"
        echo "[at-scale] [$run/$TOTAL] arm=$arm seed=$seed ($(date -Is))"

        if [ "$BACKEND" = "cuda" ] && [ -x ./bin-cuda/train_blt_d ]; then
            TRAIN_BIN="./bin-cuda/train_blt_d"
        else
            TRAIN_BIN="./bin/train_blt_d"
        fi
        # shellcheck disable=SC2086
        $TRAIN_BIN $BASE $extra --seed "$seed" \
            --save-weights "runs/s6_${arm}_s${seed}.fblt" \
            > "$log" 2>&1
        rc=$?

        # Extract RESULT line and write to results.jsonl
        result_line=$(grep "^RESULT" "$log")
        if [ $rc -eq 0 ] && [ -n "$result_line" ]; then
            # Parse values from RESULT line
            causal_bpb=$(echo "$result_line" | sed -n 's/.*causal_bpb=\([^ ]*\).*/\1/p')
            masked_acc=$(echo "$result_line" | sed -n 's/.*masked_acc=\([^ ]*\).*/\1/p')
            eval_windows=$(echo "$result_line" | sed -n 's/.*eval_windows=\([^ ]*\).*/\1/p')

            # Build JSONL entry
            ts=$(date +%s)
            json="{\"timestamp\":$ts,\"phase\":\"at_scale\",\"name\":\"train_eval\",\"tag\":\"s6_${arm}_s${seed}\","
            json="${json}\"metrics\":{\"bpb\":$causal_bpb,\"masked_acc\":$masked_acc,"
            json="${json}\"arm\":\"$arm\",\"seed\":$seed,\"eval_windows\":$eval_windows},"
            json="${json}\"latency\":{\"n\":1}}"
            echo "$json" >> bench/results.jsonl
            echo "[at-scale]   -> bpb=$causal_bpb masked_acc=$masked_acc"
        else
            echo "[at-scale]   -> FAILED (exit $rc)"
        fi
    done
done

echo "[at-scale] all done ($(date -Is))"
