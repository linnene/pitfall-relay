#!/bin/bash
# Frozen latent probe on final Actors.  One policy at a time.
set -u
D=/data/amgg_flex_demo/flex-warehouse-safe-20261118-95e0c41a
FLEX_ROOT=${FLEX_ROOT:-$D/FLEX_eval}
EVAL_ROOT=${EVAL_ROOT:-$D/runs/eval}
TRAIN_COUNT=${TRAIN_COUNT:-60}
CAL_COUNT=${CAL_COUNT:-40}
cd "$FLEX_ROOT/experiments/warehouse_safe" || exit 1
FLEX_SHA=$(cd "$FLEX_ROOT" && git rev-parse HEAD)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$D/logs/probe" "$D/runs/probe"
for seed in 11 22 33; do
  eval_dir="$EVAL_ROOT/seed${seed}_final"
  [ -f "$eval_dir/summary.json" ] || { echo "missing $eval_dir/summary.json"; continue; }
  out="$D/runs/probe/seed${seed}_final"
  echo "=== probe seed=$seed $(date -Is) flex=$FLEX_SHA"
  PYTHONPATH="$D/MacroMARL/src" "$D/envs/macro/bin/python" probe_latent.py \
    --eval-dir "$eval_dir" \
    --out-dir "$out" \
    --train-count "$TRAIN_COUNT" \
    --cal-count "$CAL_COUNT" \
    --seed "$seed" \
    > "$D/logs/probe/seed${seed}_final.log" 2>&1
  echo "rc=$? $(date -Is)"
done
echo "PROBE_DONE $(date -Is)"
