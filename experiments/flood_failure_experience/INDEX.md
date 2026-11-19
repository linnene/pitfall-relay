# Flood failure experience: index

Task: `flex-flood-failure-experience-946b2e71`
Date: 2026-11-19
Host: `amgg` only, single-CPU Python. No training, SAFE/FLEX, Isaac, or new
sensors. Development gate only; no 72-group holdout.

## Verdict

**Development gate fails: with full normal telemetry, actual pump/isolation
measurements and the ordinary failure notice, A's task-failure experience has
no independent increment. Stop; do not run the 72-group holdout.**

A's order failure is real: pump-first loses A's equipment in 8/8 seeds while
transport-first preserves it in 8/8, with both skills executing normally. The
failure is caused by the order squeezing the transport window, not by a
failed pump, isolation or transport action.

## Why the gate fails

The normal view already contains A's full water trace, pump on/off timeline,
transport on/off timeline, actual pump rate, failure notice, and B's
water/damage. The normal arm reconstructs:

- `observed_order` from the legal pump timeline (8/8 equal to the experience
  arm's `attempt.order`);
- `a_failure_notice` from the ordinary notice (8/8 equal to the experience
  outcome);
- the same B choice as the experience arm in every development pair.

| Group | normal vs experience choice | B loss normal/experience/permuted |
| --- | ---: | ---: |
| similar (8 seeds) | 8/8 identical | 0/0/0 |
| non-applicable (8 seeds) | 8/8 identical | 0/0/0 |

The permuted arm uses a sibling-seed experience; it does not create a real
advantage over normal telemetry. There is no pair where the normal arm makes
a wrong order choice and the experience arm repairs it.

A success/transport-first arm was not separately run: the failure gate had
already failed, and adding a success arm after seeing the result would not be
a valid repair.

## Development details

- 16 paired seeds: 8 similar-condition, 8 non-applicable; the same seed and
  exogenous process with only A's order changed.
- A pump-first: pump set-up 0–2, pump 2–10, transport 10–15; lost 8/8.
- A transport-first: transport 0–5, pump 5–13; preserved 8/8.
- B has an independent node and its own order choice; normal and experience
  choices agree; B loss is zero for all arms in this development set.
- The normal view is not missing pump, isolation or transport fields; the
  experience summary is a compression of already visible facts.

## Limits

The probe is an abstract native water/damage evolution with a deterministic
decision rule; it does not establish real-flood transfer or that a trained
Actor could use failure experience. No holdout was run; no parameters were
re-tuned after observing the gate result.

## Anchors and data

- Code commit: `176ef45`; script sha256:
  `2538cdd267f2211c94c59e216585b6466ae07d88f4ae1883da1b978e38e216a7`;
  `dev3.json` sha256:
  `7ccc585446002d7bf1c834f4abf5a57a6ca797325280d7dd54e9d2338168abaf`.
- Python 3.12.3 on `amgg-Super-Server`, single CPU, no external packages;
  no training checkpoint.
- Raw data:
  `/data/amgg_flex_demo/flex-flood-failure-experience-946b2e71/`
  (`results/dev3.json`, `dev3.err`, `dev3.stdout`); code under `.../code/`.
- Unit tests on `amgg`: 4/4 passed (order-caused failure, normal order
  recovery, normal/experience choice equality, deterministic replay).
- Local report: `/tmp/flex-flood-failure-experience-946b2e71.md`.
- Wall time about 10 minutes, under the two-hour cap.

## Recommendation

Close this failure-experience candidate; keep the flood prototype as a
control. Do not run the 72-group holdout, start training, or connect
SAFE/FLEX. If reopened, first demonstrate a legitimate normal-telemetry gap
that actually prevents recovery of the attempt/order; do not hide fields or
invent a new clock.
