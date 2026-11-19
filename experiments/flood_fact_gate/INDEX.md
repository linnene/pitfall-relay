# Flood fact gate: index

Task: `flex-flood-fact-gate-4d926eb3`
Date: 2026-11-19
Host: `amgg` only, single-CPU Python. No training, SAFE/FLEX, Isaac, or new
sensors. The old 93-group holdout is sealed; its 13-item difference is not
used as evidence.

## Verdict

**Fact gate passes on the frozen 72-group abstract holdout. Recommend only an
evaluation of whether a high-level RL interface is worthwhile; do not train,
do not claim FLEX is validated, and do not claim failure specificity.**

The result is evidence for a local execution-record value in this abstract
task, not evidence of real flood transfer or of an Actor having learned to
use it.

## Protocol

- B chooses at its own fixed inspection node `T_B=10`; the trigger depends
  only on B's clock/task progress and is identical across arms. A success,
  missing message, or B already safe does not force a different trigger.
- Normal telemetry: A water trace, pump/isolation status, A failure notice
  when present, B trace and damage. The execution fact: A's locally measured
  actual pump outflow, isolation closure, and before/after levels.
- One `decide` function for all arms. Baseline gets no fact object; real gets
  the real fact; permuted gets a fixed cross-scenario fact. The result file
  records `fact_visible=false` for baseline and true only for the other arms.
- No-timely groups have no fact in any arm and are step-equivalent.

## Development gate (16 groups)

8 relevant + 8 non-applicable with identical normal A/B traces but different
facts. Relevant baseline loses 8/8, real fact loses 0/8, permuted loses 8/8;
non-applicable baseline/fact preserve 8/8, permuted makes 8/8 unnecessary
isolations. A succeeds in 4/8 and fails in 4/8; the fact works in both.

## Frozen holdout (72 new seeds)

24 relevant (9001–9024), 24 non-applicable (9101–9124), 24 no-timely
(9201–9224); one run, no exclusion or re-draw.

| Group/metric | baseline | fact | permuted |
| --- | ---: | ---: | ---: |
| relevant B loss | 24/24 | **0/24** | 24/24 |
| relevant A preserved | 6/24 | 6/24 | 6/24 |
| relevant A+B team items (max 48) | 6 | **30** | 6 |
| relevant all-preserved | 0/24 | **6/24** | 0/24 |
| nonapp B loss | 0/24 | 0/24 | 0/24 |
| nonapp A+B team items (max 48) | 30 | 30 | 30 |
| nonapp unnecessary isolation | 0/24 | **0/24** | 16/24 |
| notimely B loss | 14/24 | 14/24 | 14/24 |
| notimely A+B team items (max 48) | 16 | 16 | 16 |

Acceptance: relevant `fact - baseline` B loss difference = -24, paired
bootstrap 95% CI [-1.0, -1.0], excludes zero; fact better than permutation;
A+B team 30 vs 6, not lower. Non-applicable adds 0/24 unnecessary isolation
(threshold <=2), team 30 vs 30. No-timely arms are identical.

## Fact specificity and cost

- A succeeds in 6/24 relevant groups; the fact preserves B in those as well
  as in A-failure groups. Conclusion is **work-record/execution-fact value,
  not failure-specific**.
- Normal telemetry already contains the full water trace and pump/isolation
  status; the exclusive fact is the actual applied work (pump outflow and
  closure), not a hidden standard state field. The baseline cannot see it.
- Relevant fact requests isolation in 24/24 groups; baseline/permuted 0/24.
  Non-applicable fact 0, permuted 16 unnecessary isolations.
- Waiting/queue time is not modeled in this abstract outcome; only isolation
  request occupancy is reported. Window coverage: all relevant/non-applicable
  A facts fall before B's node; no-timely has no fact.

## Anchors and data

- Execution SHA: `8c23522`; script sha256:
  `88ecce47c63b4c6277bb2f3834607bb2b743a36b1457e8efab769cc74d56c699`;
  `holdout.json` sha256:
  `3d9200d200ef9442e37af322e314950c0056853fe9c8dc1f04e43f52a58dd4bd`.
- Python 3.12.3 on `amgg-Super-Server`, single CPU process, no external
  packages, no training checkpoint.
- Holdout seeds as above; development 8 relevant + 8 non-applicable.
- Raw data: `/data/amgg_flex_demo/flex-flood-fact-gate-4d926eb3/`
  (`anchor.json`, `holdout.json`, `holdout.time`, `dev4.json`), code under
  `.../code/`. Local report: `/tmp/flex-flood-fact-gate-4d926eb3.md`.
- Unit tests on `amgg`: 4/4; deterministic replay; baseline field-isolation
  test; no-message equivalence test.
- Wall time about 10 minutes, under the two-hour cap.

## Recommendation

Use this task only to evaluate a high-level RL interface; do not start
training, do not connect SAFE/FLEX, and do not claim FLEX or real-world
transfer. An independent review should test the abstract fact/model against a
different generator before any training decision.
