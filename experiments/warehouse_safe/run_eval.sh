#!/bin/bash
# Fixed-actor evaluation for completed Warehouse-A seeds.
set -u
D=/data/amgg_flex_demo/flex-warehouse-safe-20261118-95e0c41a
FLEX_ROOT=${FLEX_ROOT:-$D/FLEX_eval}
cd "$FLEX_ROOT/experiments/warehouse_safe" || exit 1
FLEX_SHA=$(cd "$FLEX_ROOT" && git rev-parse HEAD)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$D/logs/eval" "$D/runs/eval"
for seed in 11 22 33; do
  root="$D/runs/seed_${seed}/checkpoints"
  [ -d "$root" ] || { echo "missing $root"; continue; }
  final_dir="$root/epi_040000"
  init_dir="$root/epi_000000"
  [ -f "$final_dir/manifest.json" ] || { echo "missing final checkpoint $final_dir"; continue; }
  [ -f "$init_dir/manifest.json" ] || { echo "missing initial checkpoint $init_dir"; continue; }
  eval_seed=$((7100 + seed))
  for phase in initial final; do
    ckpt="$init_dir"
    [ "$phase" = final ] && ckpt="$final_dir"
    out="$D/runs/eval/seed${seed}_${phase}"
    if [ -f "$out/summary.json" ]; then
      echo "=== $phase seed=$seed already complete: $out"
      continue
    fi
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
  if [ -f "$D/runs/eval/seed${seed}_final_off/summary.json" ]; then
    echo "=== hook-off seed=$seed already complete"
    continue
  fi
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
