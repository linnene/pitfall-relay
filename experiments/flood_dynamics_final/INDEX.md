# Flood dynamics final: index

Task: `flex-flood-dynamics-final-5a17c9e2`
Date: 2026-11-19
Host: `amgg` only, single-CPU Python. No training, SAFE/FLEX, Isaac, or new
sensors. The old 93-group holdout was not rerun.

## Verdict

**The native-evolution development gate and the frozen 72-group holdout pass.
The execution record changes B's high-level choice in this abstract task.
Recommend only an evaluation of a high-level RL interface; do not train, do
not claim FLEX is validated, and do not claim real-flood transfer.**

## Native dynamics

A and B water volumes are evolved every minute from the pre-sampled inflow,
actual pump drainage and area-scoped isolation. Equipment damage is integrated
from the evolved level. B's transport/isolate outcomes are computed by the
same evolution, not by an outcome table. B decides at its own fixed inspection
node t=10, independent of A.

Normal telemetry contains A/B water traces, pump/isolation status, A failure
notice and B damage. The execution fact contains A's actual pump outflow,
isolation closure and before/after levels. A and B normal traces are identical
in the paired relevant/non-applicable cases; actual A work differs. The fact
estimator uses the observable rates and the applied work:
`ql=(r0-r1)/closure`, `qs=r0+pump-ql`.

## Development gate (16 groups)

8 relevant + 8 non-applicable. Relevant baseline/real/permuted B losses are
8/0/8; non-applicable 0/0/0 with 8 permuted unnecessary isolations. A succeeds
6/8 and fails 2/8; the fact works for both. The same normal state with a
different legal action gives an evolved consequence.

## Frozen holdout (72 new seeds)

24 relevant, 24 non-applicable, 24 no-timely; one run, no exclusion/redraw.

| Group/metric | baseline | fact | permuted |
| --- | ---: | ---: | ---: |
| relevant B loss | 24/24 | **0/24** | 24/24 |
| relevant A preserved | 18/24 | 18/24 | 18/24 |
| relevant A+B team items (max 48) | 18 | **42** | 18 |
| relevant all-preserved | 0/24 | **18/24** | 0/24 |
| nonapp B loss | 0/24 | 0/24 | 0/24 |
| nonapp A+B team items | 42 | 42 | 42 |
| nonapp unnecessary isolation | 0/24 | **0/24** | 14/24 |
| notimely B loss | 0/24 | 0/24 | 0/24 |
| notimely A+B team items | 42 | 42 | 42 |

Relevant `fact - baseline` B loss = -24, paired bootstrap 95% CI [-1.0, -1.0],
excludes zero; fact beats permutation; team 42 vs 18, not lower. Non-applicable
adds 0/24 unnecessary isolation (threshold <=2), team 42 vs 42. No-timely arms
are step-equivalent.

A succeeds in 18/24 relevant groups and fails in 6; the fact preserves B in
both. The conclusion is therefore execution-record value, not failure
specificity. Normal failure notices do not separately change the baseline
choice.

## Resources and limits

Relevant fact requests isolation in 24/24 groups; baseline/permuted 0. The
isolation action includes 2 minutes of request/travel plus 5 minutes transport,
mean completion 17 vs 15 minutes. Non-applicable fact requests 0; the permuted
control makes 14 unnecessary requests (mean 16.17 minutes). No queue/waiting is
modeled; only isolation occupancy is reported. All relevant/non-applicable
facts arrive before B's t=10 node; no-timely groups have no fact.

The dynamics are a simplified abstract model; the pass is not evidence of real
flood transfer.

## Anchors and data

- Execution SHA: `de4d15a`; script sha256:
  `aa0383e5e5c3de20526b341b45c16d64bf44918f2c6fe147a26efe118f5de09d`.
- Python 3.12.3 on `amgg-Super-Server`, single CPU, no external packages, no
  training checkpoint.
- Holdout seeds: relevant 9001–9024, non-applicable 9101–9124, no-timely
  9201–9224.
- Raw data: `/data/amgg_flex_demo/flood-dynamics-final-5a17c9e2/`
  (`anchor.json`, `holdout.json`, `holdout.time`, `dev_final2.json`), code
  under `.../code/`. Local report:
  `/tmp/flex-flood-dynamics-final-5a17c9e2.md`.
- Unit tests on `amgg`: 4/4; deterministic replay, baseline field isolation,
  no-timely equivalence.
- Wall time about 15 minutes, under the two-hour cap.

## Recommendation

Evaluate whether this abstract task is worth a high-level RL interface. Do not
start training or connect SAFE/FLEX. An independent review should recheck the
native parameters, the exclusive fact field and the permutation control before
any training decision.
