# Fetch calibration repair and one-shot holdout: index

Task: `flex-fetch-cal-final-20261118-4a70d21c`
Date: 2026-11-18
Host: `amgg` only. No retraining, no refitting on holdout, no new probes.

## Verdict

Calibration repair and internal checks pass. The one-shot fresh holdout does
**not** meet the SAFE support/acceptance condition: the real Fetch latent does
not add the required +10 percentage points over the local-history baseline on
either policy, and on seed 11 it is exactly equal to both baseline and shuffled.
The variant is closed: no additional collection, no new probe, no automatic
next round.

## 1. Root cause of the old ~100% test FPR

The old threshold rule used `numpy.quantile(..., interpolation="higher")` plus
a `>=` alarm. The score distribution has a large exact tie mass at the
threshold, so `>=` alarmed on almost every normal episode.

Old calibration/test episode maxima (existing scores only; no old test used to
choose a new threshold):

| seed | arm | n cal | k | threshold | cal below/equal/above | old `>=` cal FPR | strict `>` cal FPR | test equal/above | old `>=` test FPR | strict `>` test FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 11 | baseline | 269 | 257 | 0.0049555772 | 0/263/6 | 1.0000 | 0.02230 | 787/809, 22 | 1.0000 | 0.02719 |
| 11 | real | 269 | 257 | 0.0029979406 | 0/263/6 | 1.0000 | 0.02230 | 787/809, 22 | 1.0000 | 0.02719 |
| 11 | shuffled | 269 | 257 | 0.0092514115 | 239/24/6 | 0.11152 | 0.02230 | 52/809 equal, 23 above (734 below) | 0.09271 | 0.02843 |
| 22 | baseline | 265 | 253 | 0.0030270917 | 1/263/1 | 0.99623 | 0.00377 | 725/732, 5 | 0.99727 | 0.00683 |
| 22 | real | 265 | 253 | 0.0006464126 | 1/263/1 | 0.99623 | 0.00377 | 724/732, 7 | 0.99863 | 0.00956 |
| 22 | shuffled | 265 | 253 | 0.0098602012 | 248/16/1 | 0.06415 | 0.00377 | 43/732 equal, 2 above (687 below) | 0.06148 | 0.00273 |

Scores were not probability-saturated: no score at 0 or 1 in any arm. The
diagnostic's `normal_episode_overlap: true` is a local-index artifact: each
split re-enumerates samples from zero. The original split lists are disjoint
(`split_disjoint: true`); the normal calibration and test sets are separate
original episodes. Root cause is ties plus the `>=` comparator, not a true
distribution shift.

## 2. Repair and internal self-check

Repair in `experiments/warehouse_safe/probe_latent.py` (`ad31c06`):

- threshold: sort normal-episode maxima, `k = ceil((n+1)*0.95)`, threshold =
  `values[k-1]`; if `k > n`, threshold = `inf` (no alarm);
- alarm: strict `score > threshold` everywhere (events, early alarms, normal
  FPR, paired event hits).

Three same-architecture probes were fit on the original fitting set only
(1,000 train episodes per seed); the original independent calibration set
(500 episodes, 269/265 normal) supplied only the thresholds. No old test data
entered threshold selection. Frozen calibration results:

| seed | arm | threshold | n/k | cal equal | cal FPR strict `>` |
|---|---:|---:|---:|---:|---:|
| 11 | baseline | 0.004955577218125768 | 269/257 | 263 | 0.02230 |
| 11 | real | 0.0029979405550890166 | 269/257 | 263 | 0.02230 |
| 11 | shuffled | 0.009251411458680641 | 269/257 | 24 | 0.02230 |
| 22 | baseline | 0.0030270916586056123 | 265/253 | 263 | 0.00377 |
| 22 | real | 0.0006464125608062028 | 265/253 | 263 | 0.00377 |
| 22 | shuffled | 0.009860201227503355 | 265/253 | 16 | 0.00377 |

`self_check.py` passed: exact-tie quantiles (0% strict FPR vs 50% with `>=`),
constant scores (0%), `k > n` (no alarm point), unequal-length episode
aggregation, strict `(failure, deadline)` boundaries, later-event window
truncation, strict comparator at the threshold, deterministic shuffled
latents, and a manual reconstruction of the 382-dimensional feature vector
from local observations/actions only. A frozen-artifact verifier recomputed
every threshold and FPR exactly from the original calibration split; all
matched and all were <= 5%.

## 3. Frozen holdout

Frozen before collection: policies (checkpoint aggregate hashes), variant and
base configs, three probe models, thresholds, data contract, and new seed
ranges. Full manifest: `experiments/warehouse_fetch_final/frozen_manifest.json`.

- New eval seeds: seed 11 `20011..22010`, seed 22 `24022..26021`; 2,000
  episodes each, 4,000 total.
- Disjoint from prior eval ranges: old observed maximum end 12432; overlap with
  old episode-seed sets = 0; the two new ranges do not overlap each other.
- Same variant, same frozen Actor checkpoints, `hooks=on`, no stall control,
  CPU, no retraining.
- Raw data: `/data/amgg_flex_demo/flex-fetch-cal-final-20261118-4a70d21c/runs/holdout_seed{11,22}`.

Holdout timing/sample gate (Fetch legal new latent strictly after failure and
before T1's choice):

| seed | episodes | all failures | T1 two options | interface relevant | Fetch legal new latent | normal episodes |
|---|---:|---:|---:|---:|---:|---:|
| 11 | 2,000 | 2,011 | 936 | 240 | 237 | 1,060 |
| 22 | 2,000 | 2,307 | 865 | 143 | 118 | 997 |

Both gates pass (>= 50 legal windows and >= 100 complete normal episodes per
policy).

## 4. Holdout results

All alarms use the frozen strict `>` thresholds; no refit. Delay is steps after
the failure. "Early" alarms are before the failure and are **not** counted as
timely recall.

| seed | arm | legal windows | legal recall | all-failure timely coverage | early legal events | median delay | episode FPR [95% CI] | decision FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 11 | baseline | 237 | 0.9958 | 0.1174 (236/2011) | 0 | 1 | 0.02453 [0.01601, 0.03396] | 0.00376 |
| 11 | real | 237 | 0.9958 | 0.1174 (236/2011) | 0 | 1 | 0.02453 [0.01601, 0.03396] | 0.00188 |
| 11 | shuffled | 237 | 0.9958 | 0.1174 (236/2011) | 0 | 1 | 0.02547 [0.01604, 0.03491] | 0.00224 |
| 22 | baseline | 118 | 0.9661 | 0.04941 (114/2307) | 95 | 1 | 0.00802 [0.00301, 0.01404] | 0.00114 |
| 22 | real | 118 | 0.9915 | 0.05072 (117/2307) | 95 | 1 | 0.00802 [0.00301, 0.01404] | 0.00086 |
| 22 | shuffled | 118 | 0.9576 | 0.04898 (113/2307) | 5 | 1 | 0.00100 [0.00000, 0.00301] | 0.00021 |

Episode-clustered paired differences (real minus control; 1,000 bootstrap
draws resampling episodes):

| seed | contrast | event recall diff [95% CI] | episode recall diff [95% CI] | FPR diff [95% CI] |
|---|---:|---:|---:|---:|
| 11 | real - baseline | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.00000 [0.00000, 0.00000] |
| 11 | real - shuffled | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | -0.00094 [-0.00283, 0.00000] |
| 22 | real - baseline | +0.02542 [0.00000, 0.05358] | +0.04615 [0.00000, 0.10769] | 0.00000 [0.00000, 0.00000] |
| 22 | real - shuffled | +0.03390 [0.00800, 0.06957] | +0.05385 [0.00769, 0.11538] | +0.00702 [0.00301, 0.01304] |

## 5. Acceptance

| condition | seed 11 | seed 22 | overall |
|---|---:|---:|---:|
| 2,000 fresh episodes, >=50 legal windows, >=100 normal episodes | pass | pass | pass |
| every arm episode FPR <= 7.5% | pass | pass | pass |
| real FPR not > 2.5 pp above baseline | pass (0) | pass (0) | pass |
| real FPR not > 2.5 pp above shuffled | pass (-0.09 pp) | pass (+0.70 pp) | pass |
| real - baseline >= +10 pp | fail (0) | fail (+2.54 pp) | **fail** |
| paired recall interval lower bound > 0 | fail ([0,0]) | fail ([0,0.0536] event; [0,0.108] episode) | **fail** |
| real better than shuffled | fail (equal) | pass (+3.39 pp; CI +0.80..+6.96) | **fail (both policies required)** |

## 6. Interpretation and stop

- Seed 11 is at a ceiling: local-history baseline already recalls 0.9958 of
  legal windows; real and shuffled are identical. The real latent adds nothing
  measurable there.
- Seed 22 shows a small real-latent gain over baseline (+2.54 pp event-level,
  +4.62 pp episode-level) and a small gain over shuffled (+3.39 pp event,
  +5.39 pp episode), with FPR close to baseline and only 0.70 pp above shuffled.
  It is well below the pre-registered +10 pp requirement, and the paired
  baseline interval includes zero.
- All-failure timely coverage is low in both policies (11.7% / 5.1%) because
  only 237/2011 and 118/2307 failures have a legal T1 two-option interface
  window; the denominator is reported so the low coverage is not mistaken for
  a detector failure. Legal-window recall is the primary recall metric.
- Seed 22 has many early pre-failure alarms (95 legal events) in both baseline
  and real, but early alarms are not timely and are excluded from recall.
- Result is negative/undetermined for the required SAFE increment, not a
  universal claim that Fetch latents are undetectable. The variant should be
  closed under the pre-registered stop rule. No extra collection, no new probe,
  no automatic next round.

## 7. Anchors and remote paths

- Calibration diagnostic: `experiments/warehouse_fetch_audit/calibration_diag.py`
  (`a7b9b62`).
- Probe repair: `experiments/warehouse_safe/probe_latent.py` (`ad31c06`).
- Frozen fit: `experiments/warehouse_fetch_final/fit_frozen_fetch.py`
  (`306bd6e`, `16ca07e`).
- Self-check/evaluator: `experiments/warehouse_fetch_final/self_check.py`,
  `eval_frozen_fetch.py` (`d2a2017`, fix `51ccd62`).
- Frozen manifest: `experiments/warehouse_fetch_final/frozen_manifest.json`
  (`f5ea1e6`).
- Execution code commit: `51ccd6223332d3c93dc5186de32bffb3c497df0a`.
- Archived collection code: variant repo commit `d0b3152`; `eval_variant.py`
  sha256 `4aa5b572...`; `variant.py` sha256 `e015635e...`.
- Policy checkpoints: seed 11 aggregate `6c45b0d7...`; seed 22 aggregate
  `ef469c37...` (full per-file hashes in each checkpoint `manifest.json`).
- Frozen probe models: seed 11 `839b9b0e...`, seed 22 `4e242565...`.
- Frozen threshold files: seed 11 `7900e22f...`, seed 22 `b4e3c333...`.
- Result hashes: audit `6cecba56...`, eval seed 11 `16bd666b...`, eval seed 22
  `90e3d976...`, self-check seed 11 `23151868...`, seed 22 `4d73bd67...`.

Remote task directory:

```
/data/amgg_flex_demo/flex-fetch-cal-final-20261118-4a70d21c/
  code/                         fit, self-check, evaluator, archived helpers
  frozen/seed{11,22}/           models.pkl, thresholds.json, thresholds.sha256
  runs/holdout_seed{11,22}/     2,000 fresh episodes each (raw arrays + manifests)
  runs/fetch_audit_holdout.json
  runs/fetch_view_holdout_seed{11,22}/
  runs/eval_holdout_seed{11,22}.json
  runs/self_check_seed{11,22}.json
  logs/
```

Budget: one collection of 4,000 episodes, no retraining, no refitting; the
task stayed within the two-hour hard limit. Raw trajectories and logs remain
on `amgg`; this index and the report are the only local records.
