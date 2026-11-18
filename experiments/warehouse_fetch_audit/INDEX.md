# Fetch sender audit: index

Task: `flex-warehouse-fetch-audit-20261118-6e28c1af`
Existing archives only; no new rollouts, retraining, task changes, or downloads.
Audit code: `545f732` (`fetch_audit.py`); view builder: `e4b4929`
(`build_fetch_view.py`). Final index commit: this file.

## Result

Timing and samples pass: seed 11 has 336 legal Fetch windows (T1 two options +
interface-relevant choice + Fetch valid new latent strictly after failure and
before T1's choice), seed 22 has 161; normal episodes are 1,630/1,499. All 43
existing controlled pairs have identical action/observation prefixes.

The Fetch probe does **not** meet the SAFE support condition:

- seed 11: baseline/real/shuffled recall 1.000; real-baseline paired difference
  0 (95% CI [0,0]); actual test episode FPR 1.000/1.000/0.093.
- seed 22: baseline/real/shuffled recall 0.963/0.976/0.963; real-baseline
  difference +1.2 pp (95% CI [0, 3.8]); actual FPR 0.997/0.999/0.061.
- Calibration is invalid (actual FPR far above 5%), so no valid same-FPR
  comparison; real is not better than shuffled and does not reach +10 pp.
  Conclusion: undetermined/no evidence for Fetch latent increment, not a
  universal “latent undetectable” result.

## Remote paths

```
/data/amgg_flex_demo/flex-warehouse-fetch-audit-20261118-6e28c1af/
  code/fetch_audit.py, build_fetch_view.py
  runs/fetch_audit.json
  runs/fetch_view_seed{11,22}/       read-only JSON views + NPZ symlinks
  runs/fetch_split_seed11_22.json
  runs/fetch_probe_seed{11,22}/
  runs/fetch_probe_combined.json
  logs/fetch_audit.log, fetch_probe_seed*.log
```

Source archives:
`/data/amgg_flex_demo/flex-warehouse-handoff-variant-20261118-a92b6f31/runs/`
(`probe_seed{11,22,33}`); checkpoints/manifest anchors remain in the source
task and were not modified.

## Reproduction

```bash
D=/data/amgg_flex_demo/flex-warehouse-fetch-audit-20261118-6e28c1af
S=/data/amgg_flex_demo/flex-warehouse-handoff-variant-20261118-a92b6f31
python "$D/code/fetch_audit.py" --eval-dir "$S/runs/probe_seed11" \
  --eval-dir "$S/runs/probe_seed22" --eval-dir "$S/runs/probe_seed33" \
  --out "$D/runs/fetch_audit.json"
python "$D/code/build_fetch_view.py" --source "$S/runs/probe_seed11" \
  --audit "$D/runs/fetch_audit.json" --out "$D/runs/fetch_view_seed11"
# split file copied from the variant task's original stratified split
cd "$S/FLEX_eval/experiments/warehouse_safe"
python probe_latent.py --eval-dir "$D/runs/fetch_view_seed11" \
  --event fetch_handoff_failure:2 --split-file "$D/runs/fetch_split_seed11_22.json" \
  --train-count 1000 --cal-count 500 --seed 11 \
  --out-dir "$D/runs/fetch_probe_seed11"
```
