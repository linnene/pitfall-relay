# Warehouse-A handoff variant: index

Task: `flex-warehouse-handoff-variant-20261118-a92b6f31`
Base reproduction: `flex-warehouse-safe-20261118-95e0c41a` (`ed774c5`)
Variant implementation: `fe3067f`; stratified split: `2211e33`

## Result in one paragraph

The interface-stall variant creates real T1 decision windows without forcing
actions: a 300-episode frozen-Actor gate had 289 stall failures and 27 windows
(9.34%); 43/43 controlled same-seed pairs matched in action/observation prefix,
with 7 windowed pairs. The original Actors retained task ability (seed 11
300/300; seed 22 192/200; seed 33 193/200), so no retraining was used. In
3,000-episode natural holdouts per policy, seeds 11/22 had 340/189 windows.
The frozen 5% threshold produced test episode FPR 1.000/0.996; baseline and
real latent both had recall 1.0, with paired difference exactly 0, and real
was not better than shuffled. The formal SAFE support condition is **not met
and the calibration is invalid**, so the result is **undetermined**, not proof
that the latent is universally uninformative. Recommendation: do not integrate
SAFE on this variant without a better task distribution/representation.

See `/tmp/flex-warehouse-handoff-variant-20261118-a92b6f31.md` for full tables
and caveats.

## Commits

- `fe3067f` variant, recorder, immutable run manifest
- `67beac7` pair check and window gate
- `893a5f9` windowed-event counting
- `2211e33` stratified split by stall duration/window
- `f19cdaf` docs and direct local rule

## Key remote paths

```
/data/amgg_flex_demo/flex-warehouse-handoff-variant-20261118-a92b6f31/
  checkpoints/seed{11,22,33}_{initial,final}/
  runs/gate_natural_seed11/      300 natural gate episodes
  runs/gate_control_seed11/      100 stall-off control episodes
  runs/gate_pairs_seed11.json    43 valid pairs, 7 windowed
  runs/gate_decision_seed11.json gate passed
  runs/task_seed{22,33}/         200-episode variant task checks
  runs/probe_seed{11,22,33}/     3,000 natural trajectories each
  runs/probe_split_seed11_22.json
  runs/probe_seed{11,22}_latent_strat/
  runs/local_rule_seed11_22.json
  logs/gate/
```

No raw data, checkpoints, or logs are in Git; they remain on `amgg`.

## Reproduction

```bash
D=/data/amgg_flex_demo/flex-warehouse-handoff-variant-20261118-a92b6f31
cd "$D/FLEX_eval/experiments/warehouse_handoff_variant"
# Gate (already run; outputs included)
# python eval_variant.py ... --episodes 300 --out-dir .../gate_natural_seed11
# python eval_variant.py ... --episodes 100 --stall-control --out-dir .../gate_control_seed11
# python pair_check.py ...
# python gate_decision.py ...
# 3000-episode natural collections
# python eval_variant.py ... --episodes 3000 --out-dir .../probe_seed11
# Stratified split and frozen probe
python make_split.py --eval-dir "$D/runs/probe_seed11" \
  --eval-dir "$D/runs/probe_seed22" --train-count 1000 --cal-count 500 \
  --test-count 1500 --out "$D/runs/probe_split_seed11_22.json"
cd ../warehouse_safe
python probe_latent.py --eval-dir "$D/runs/probe_seed11" \
  --event interface_handoff_failure:1 --train-count 1000 --cal-count 500 \
  --split-file "$D/runs/probe_split_seed11_22.json" --seed 11 \
  --out-dir "$D/runs/probe_seed11_latent_strat"
```

The variant is an abstract interface condition, not a physical fault model.
Seed 33 had only 15 windows in 3,367 failures (0.45%) and was not used for the
formal two-policy probe.
