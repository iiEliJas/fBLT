#!/usr/bin/env bash
# run_sweep.sh — sequential sweep queue runner.
#
# Usage (from repo root):
#   tools/run_sweep.sh <config.json> [config2.json ...]
#
# - Skips a config whose tag already appears in bench/results.jsonl
#   (set SWEEP_FORCE=1 to disable the skip, e.g. for the determinism check).
# - Appends one JSON status line per config to configs/sweep_state.json.
# - Per-run stdout/stderr goes to logs/<tag>.log; queue messages go to the
#   caller's redirect. Launch detached:
#     mkdir -p logs && nohup setsid tools/run_sweep.sh configs/ablations/a.json \
#       configs/ablations/b.json > logs/queue.log 2>&1 & echo $! > logs/queue.pid

set -u
mkdir -p logs bench/ckpts configs

for cfg in "$@"; do
    tag=$(python3 - "$cfg" << 'PYEOF'
import json, sys
print(json.load(open(sys.argv[1]))["tag"])
PYEOF
)
    if [ "${SWEEP_FORCE:-0}" != "1" ] && [ -f bench/results.jsonl ] && \
        grep -q "\"tag\": *\"$tag\"" bench/results.jsonl; then
        echo "[queue] skip $tag (already in results.jsonl)"
        continue
    fi

    echo "{\"tag\":\"$tag\",\"status\":\"running\",\"config\":\"$cfg\",\"ts\":\"$(date -Is)\"}" \
        >> configs/sweep_state.json
    echo "[queue] start $tag ($(date -Is))"

    ./bin/train_sweep --config "$cfg" --results bench/results.jsonl \
        > "logs/$tag.log" 2>&1
    rc=$?

    if [ $rc -eq 0 ]; then
        st=done
    elif [ $rc -eq 124 ]; then
        st=timeout
    elif [ $rc -eq 130 ]; then
        st=signalled_checkpointed
    else
        st="exit$rc"
    fi
    echo "{\"tag\":\"$tag\",\"status\":\"$st\",\"config\":\"$cfg\",\"ts\":\"$(date -Is)\"}" \
        >> configs/sweep_state.json
    echo "[queue] end $tag -> $st ($(date -Is))"
done
echo "[queue] all configs processed"
