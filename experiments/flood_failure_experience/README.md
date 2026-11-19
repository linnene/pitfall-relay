# Flood failure-experience gate

Task: `flex-flood-failure-experience-946b2e71`
Date: 2026-11-19
Host: `amgg` only, single-CPU Python. No training, SAFE/FLEX, Isaac, or new
sensors. This is a development gate only; no holdout was run.

## Question

Can A's task-level failure experience (condition, attempted order, outcome)
add a B decision beyond full normal telemetry, actual pump/isolation
measurements, reconstructible derived quantities, and an ordinary failure
notice?

## Probe

A has two legal orders for the same task: pump-first (transport delayed until
after pump set-up) and transport-first. Water and damage evolve minute by
minute; both skills execute normally. Pump-first loses A's equipment, while
transport-first preserves it; the difference is order, not a failed skill.

The normal telemetry view contains A's full water trace, pump timeline,
transport timeline, actual pump rate, failure notice, B water/damage. The
experience view contains Condition/Attempt/Outcome. The normal arm recovers
A's attempted order from the legal pump timeline and the outcome from the
notice; the experience arm reads the summary. Both use the same choice rule:
avoid pump-first when the observed A attempt was pump-first and failed.

B has its own independent node and the same order choice. Similar-condition
pairs should choose transport-first; non-applicable pairs should keep the
normal choice. The permuted arm uses a fixed sibling-seed experience.

## Development gate (16 pairs)

8 similar + 8 non-applicable seeds. All A pump-first runs lost and all A
transport-first runs preserved. In both families, normal and experience
choices are identical for every seed (8/8 and 8/8), and B loss is 0 for all
arms. The normal view's `observed_order` equals the experience
`attempt.order`, and its failure notice equals the experience outcome.

Therefore the normal telemetry already recovers the execution facts and the
choice; the failure-experience summary has no independent increment. Under
the task stop rule the gate is closed: no 72-group holdout, no re-tuning, no
new sensors or hidden fields.

## Limits

The probe uses an abstract native water/damage evolution and a deterministic
decision rule; it does not establish real-flood transfer, and it does not
show that a trained Actor could use failure experience. No holdout was run.

## Run

```
python3 failure_gate.py --out results/dev.json
python3 -m unittest -v test_failure_gate.py
```
