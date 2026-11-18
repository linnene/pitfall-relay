#!/bin/bash
# Formal Warehouse-A reproduction: one seed at a time, no concurrency.
set -u
D=/data/amgg_flex_demo/flex-warehouse-safe-20261118-95e0c41a
cd "$D/FLEX/experiments/warehouse_safe" || exit 1
FLEX_SHA=$(cd "$D/FLEX" && git rev-parse HEAD)
if [ -z "$(cd "$D/FLEX" && git status --porcelain)" ]; then FLEX_DIRTY=false; else FLEX_DIRTY=true; fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$D/logs"
for seed in 11 22 33; do
  echo "=== seed $seed start $(date -Is) flex=$FLEX_SHA dirty=$FLEX_DIRTY"
  PYTHONPATH="$D/MacroMARL/src" \
    "$D/envs/macro/bin/python" run_train.py \
      --upstream-root "$D/MacroMARL" \
      --config "$D/FLEX/experiments/warehouse_safe/configs/warehouse_a_maciaicc.json" \
      --seed "$seed" \
      --out-root "$D/runs" \
      --flex-sha "$FLEX_SHA" \
      --flex-dirty "$FLEX_DIRTY" \
      --torch-threads 1 \
      > "$D/logs/train_seed_${seed}.log" 2>&1
  rc=$?
  echo "=== seed $seed end rc=$rc $(date -Is)"
  if [ "$rc" -ne 0 ]; then
    echo "stopping after seed $seed failure; see logs/train_seed_${seed}.log"
    exit "$rc"
  fi
done
echo "ALL_SEEDS_DONE $(date -Is)"
