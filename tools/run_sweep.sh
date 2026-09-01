#!/usr/bin/env bash
# run_sweep.sh — sequential sweep queue runner.
#
# Usage (from repo root):
#   tools/run_sweep.sh [--seed N] [--backend cpu|cuda] <config.json> [config2.json ...]
#
# - Skips a config whose effective tag already appears in bench/results.jsonl
#   (set SWEEP_FORCE=1 to disable the skip, e.g. for the determinism check).
# - When --seed N is given, the effective tag becomes <tag>_s<N> (so the same
#   config with different seeds are tracked as separate entries).
# - Appends one JSON status line per config to configs/sweep_state.json.
# - Per-run stdout/stderr goes to logs/<eff_tag>.log; queue messages go to the
#   caller's redirect. Launch detached:
#     mkdir -p logs && nohup setsid tools/run_sweep.sh --seed 7 --backend cuda \
#       configs/ablations/a.json > logs/queue.log 2>&1 & echo $! > logs/queue.pid

set -u
mkdir -p logs bench/ckpts configs

SEED=""
BACKEND=""
CONFIGS=()

# Parse flags from positional args
while [ $# -gt 0 ]; do
    case "$1" in
        --seed)    shift; SEED="$1"; shift ;;
        --backend) shift; BACKEND="$1"; shift ;;
        *)         CONFIGS+=("$1"); shift ;;
    esac
done

for cfg in "${CONFIGS[@]}"; do
    base_tag=$(python3 - "$cfg" << 'PYEOF'
import json, sys
print(json.load(open(sys.argv[1]))["tag"])
PYEOF
)
    if [ -n "$SEED" ]; then
        eff_tag="${base_tag}_s${SEED}"
    else
        eff_tag="$base_tag"
    fi

    if [ "${SWEEP_FORCE:-0}" != "1" ] && [ -f bench/results.jsonl ] && \
        grep -q "\"tag\": *\"$eff_tag\"" bench/results.jsonl; then
        echo "[queue] skip $eff_tag (already in results.jsonl)"
        continue
    fi

    echo "{\"tag\":\"$eff_tag\",\"status\":\"running\",\"config\":\"$cfg\",\"ts\":\"$(date -Is)\"}" \
        >> configs/sweep_state.json
    echo "[queue] start $eff_tag ($(date -Is))"

    SEED_ARG=""
    [ -n "$SEED" ] && SEED_ARG="--seed $SEED"
    if [ "$BACKEND" = "cuda" ] && [ -x ./bin-cuda/train_sweep ]; then
        SWEEP_BIN="./bin-cuda/train_sweep"
    else
        SWEEP_BIN="./bin/train_sweep"
    fi
    # shellcheck disable=SC2086
    $SWEEP_BIN --config "$cfg" --results bench/results.jsonl \
        --backend "$BACKEND" $SEED_ARG \
        > "logs/$eff_tag.log" 2>&1
    rc=$?

    # When --seed is used, train_sweep writes the base tag. Rename to eff_tag
    # so the skip-check works on re-runs.
    if [ $rc -eq 0 ] && [ -n "$SEED" ] && [ -f bench/results.jsonl ]; then
        python3 -c "
import json, sys
with open('bench/results.jsonl') as f: lines = f.readlines()
for i in range(len(lines)-1, -1, -1):
    try:
        d = json.loads(lines[i])
        if d.get('tag') == '$base_tag':
            d['tag'] = '$eff_tag'
            lines[i] = json.dumps(d, separators=(',',':')) + '\n'
            break
    except: pass
with open('bench/results.jsonl','w') as f: f.writelines(lines)
"
    fi

    if [ $rc -eq 0 ]; then
        st=done
    elif [ $rc -eq 124 ]; then
        st=timeout
    elif [ $rc -eq 130 ]; then
        st=signalled_checkpointed
    else
        st="exit$rc"
    fi
    echo "{\"tag\":\"$eff_tag\",\"status\":\"$st\",\"config\":\"$cfg\",\"ts\":\"$(date -Is)\"}" \
        >> configs/sweep_state.json
    echo "[queue] end $eff_tag -> $st ($(date -Is))"
done
echo "[queue] all configs processed"
