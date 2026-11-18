#!/bin/bash
# Additional natural trajectories for the latent probe only.
# These are not the 200-episode task-evaluation set.
set -u
D=/data/amgg_flex_demo/flex-warehouse-safe-20261118-95e0c41a
FLEX_ROOT=${FLEX_ROOT:-$D/FLEX_eval}
EPISODES=${EPISODES:-600}
EVAL_BASE=${EVAL_BASE:-8100}
cd "$FLEX_ROOT/experiments/warehouse_safe" || exit 1
FLEX_SHA=$(cd "$FLEX_ROOT" && git rev-parse HEAD)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$D/logs/probe" "$D/runs/probe_eval"
for seed in 11 22 33; do
  ckpt="$D/runs/seed_${seed}/checkpoints/epi_040000"
  [ -f "$ckpt/manifest.json" ] || { echo "missing $ckpt"; continue; }
  out="$D/runs/probe_eval/seed${seed}_final"
  echo "=== probe collect seed=$seed episodes=$EPISODES $(date -Is) flex=$FLEX_SHA"
  PYTHONPATH="$D/MacroMARL/src" "$D/envs/macro/bin/python" eval_policy.py \
    --upstream-root "$D/MacroMARL" \
    --config configs/warehouse_a_maciaicc.json \
    --checkpoint-dir "$ckpt" --label "seed${seed}_probe_final" \
    --seed "$seed" --eval-seed "$((EVAL_BASE + seed))" --episodes "$EPISODES" \
    --out-dir "$out" --hooks on --flex-sha "$FLEX_SHA" \
    > "$D/logs/probe/collect_seed${seed}_final.log" 2>&1
  echo "rc=$? $(date -Is)"
done
echo "PROBE_COLLECT_DONE $(date -Is)"
