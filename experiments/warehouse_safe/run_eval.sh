#!/bin/bash
# Fixed-actor evaluation for completed Warehouse-A seeds.
set -u
D=/data/amgg_flex_demo/flex-warehouse-safe-20261118-95e0c41a
cd "$D/FLEX/experiments/warehouse_safe" || exit 1
FLEX_SHA=$(cd "$D/FLEX" && git rev-parse HEAD)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$D/logs/eval" "$D/runs/eval"
for seed in 11 22 33; do
  root="$D/runs/seed_${seed}/checkpoints"
  [ -d "$root" ] || { echo "missing $root"; continue; }
  final_dir=$(find "$root" -maxdepth 1 -type d -name 'epi_*' | sort | tail -1)
  init_dir="$root/epi_000000"
  eval_seed=$((7100 + seed))
  for phase in initial final; do
    ckpt="$init_dir"
    [ "$phase" = final ] && ckpt="$final_dir"
    out="$D/runs/eval/seed${seed}_${phase}"
    echo "=== $phase seed=$seed ckpt=$ckpt $(date -Is)"
    PYTHONPATH="$D/MacroMARL/src" "$D/envs/macro/bin/python" eval_policy.py \
      --upstream-root "$D/MacroMARL" \
      --config configs/warehouse_a_maciaicc.json \
      --checkpoint-dir "$ckpt" --label "seed${seed}_${phase}" \
      --seed "$seed" --eval-seed "$eval_seed" --episodes 200 \
      --out-dir "$out" --hooks on --flex-sha "$FLEX_SHA" \
      > "$D/logs/eval/seed${seed}_${phase}_on.log" 2>&1
    echo "rc=$? $(date -Is)"
  done
  # Hook-off paired subset on the same final checkpoint and first 20 seeds.
  PYTHONPATH="$D/MacroMARL/src" "$D/envs/macro/bin/python" eval_policy.py \
    --upstream-root "$D/MacroMARL" \
    --config configs/warehouse_a_maciaicc.json \
    --checkpoint-dir "$final_dir" --label "seed${seed}_final" \
    --seed "$seed" --eval-seed "$eval_seed" --episodes 20 \
    --out-dir "$D/runs/eval/seed${seed}_final_off" --hooks off --flex-sha "$FLEX_SHA" \
    > "$D/logs/eval/seed${seed}_final_off.log" 2>&1
  "$D/envs/macro/bin/python" compare_hooks.py \
    --on-dir "$D/runs/eval/seed${seed}_final" \
    --off-dir "$D/runs/eval/seed${seed}_final_off" \
    --out "$D/runs/eval/seed${seed}_hook_compare.json" \
    >> "$D/logs/eval/seed${seed}_final_off.log" 2>&1
  echo "hook compare rc=$? $(date -Is)"
done
echo "EVAL_DONE $(date -Is)"
