# A/B Test: SGD vs AdamW noise floor comparison
# Base arm (xattn=all) + xlast arm (dec-layers=1, xattn=last)
# Each group: 5 reruns, seed 7, 40k steps, lr-decay on
#
# SGD: lr=0.05 (matches Section 9 recipe)
# AdamW: lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight-decay=0.01
#
# Usage:
#   bash tools/run_ab_test.sh          # run all 20
#   bash tools/run_ab_test.sh 5        # run first 5
#   bash tools/run_ab_test.sh 6 20     # run from 6 to 20

set -euo pipefail
cd "$(dirname "$0")/.."

BIN="bin-cuda/train_blt_d"
CORPUS="data/train.bin"
STEPS=40000
REPORT=500
BACKEND="cuda"
CLIP=5.0
SEED=7

mkdir -p runs logs

START=${1:-1}
END=${2:-20}

RUNS=(
  "sgd_base"      "--optimizer sgd --lr 0.05"
  "sgd_base2"     "--optimizer sgd --lr 0.05"
  "sgd_base3"     "--optimizer sgd --lr 0.05"
  "sgd_base4"     "--optimizer sgd --lr 0.05"
  "sgd_base5"     "--optimizer sgd --lr 0.05"
  "adamw_base"    "--optimizer adamw --lr 0.001 --beta1 0.9 --beta2 0.999 --eps 1e-8 --weight-decay 0.01"
  "adamw_base2"   "--optimizer adamw --lr 0.001 --beta1 0.9 --beta2 0.999 --eps 1e-8 --weight-decay 0.01"
  "adamw_base3"   "--optimizer adamw --lr 0.001 --beta1 0.9 --beta2 0.999 --eps 1e-8 --weight-decay 0.01"
  "adamw_base4"   "--optimizer adamw --lr 0.001 --beta1 0.9 --beta2 0.999 --eps 1e-8 --weight-decay 0.01"
  "adamw_base5"   "--optimizer adamw --lr 0.001 --beta1 0.9 --beta2 0.999 --eps 1e-8 --weight-decay 0.01"
  "sgd_xlast"     "--optimizer sgd --lr 0.05 --dec-layers 1 --cross-attn last"
  "sgd_xlast2"    "--optimizer sgd --lr 0.05 --dec-layers 1 --cross-attn last"
  "sgd_xlast3"    "--optimizer sgd --lr 0.05 --dec-layers 1 --cross-attn last"
  "sgd_xlast4"    "--optimizer sgd --lr 0.05 --dec-layers 1 --cross-attn last"
  "sgd_xlast5"    "--optimizer sgd --lr 0.05 --dec-layers 1 --cross-attn last"
  "adamw_xlast"   "--optimizer adamw --lr 0.001 --beta1 0.9 --beta2 0.999 --eps 1e-8 --weight-decay 0.01 --dec-layers 1 --cross-attn last"
  "adamw_xlast2"  "--optimizer adamw --lr 0.001 --beta1 0.9 --beta2 0.999 --eps 1e-8 --weight-decay 0.01 --dec-layers 1 --cross-attn last"
  "adamw_xlast3"  "--optimizer adamw --lr 0.001 --beta1 0.9 --beta2 0.999 --eps 1e-8 --weight-decay 0.01 --dec-layers 1 --cross-attn last"
  "adamw_xlast4"  "--optimizer adamw --lr 0.001 --beta1 0.9 --beta2 0.999 --eps 1e-8 --weight-decay 0.01 --dec-layers 1 --cross-attn last"
  "adamw_xlast5"  "--optimizer adamw --lr 0.001 --beta1 0.9 --beta2 0.999 --eps 1e-8 --weight-decay 0.01 --dec-layers 1 --cross-attn last"
)

idx=0
total=${#RUNS[@]}
for ((i=0; i<total; i+=2)); do
  idx=$((idx + 1))
  if (( idx < START )); then continue; fi
  if (( idx > END )); then break; fi

  label="${RUNS[$i]}"
  flags="${RUNS[$((i+1))]}"

  echo "=== [$idx/20] $label ==="
  $BIN --corpus "$CORPUS" --steps $STEPS --seed $SEED --backend "$BACKEND" \
    $flags --lr-decay 1 --report-every $REPORT \
    --eval-corpus "$CORPUS" --eval-windows 300 \
    --save-weights "runs/ab_${label}.fblt" \
    2>&1 | tee "logs/ab_${label}.log"
  echo ""
done

echo "Done. Results in logs/ab_*.log"
