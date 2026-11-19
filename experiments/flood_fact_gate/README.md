# Flood fact gate: independent B node and local execution fact

Task: `flex-flood-fact-gate-4d926eb3`
Date: 2026-11-19
Host: `amgg` only, single-CPU Python. No training, SAFE/FLEX, Isaac, or new
sensors.

## Protocol

B decides at its own fixed inspection node (`T_B=10`), independent of A's
result and messages. A has an independent work interval with pump outflow and
isolation closure; normal telemetry contains A's water trace, pump/isolation
status, B's trace and damage, and an A-failure notice when A failed. The
execution fact contains A's locally measured actual pump outflow, isolation
closure and the before/after water samples.

The normal A/B traces are identical for the relevant and non-applicable
families. The fact separates them: a pump-accounting water balance estimates
the residual shared inflow. The same `decide` function is used in all arms;
only the payload object differs. Baseline receives no fact object; real and
permuted arms receive their respective fact object. The permuted arm uses a
fixed cross-scenario fact (never B truth).

- `relevant`: shared inflow high; baseline (normal telemetry) transports and
  loses B; real fact isolates and preserves B; permuted fact transports and
  loses.
- `nonapp`: shared low; baseline transports and preserves; real fact
  transports and preserves; permuted fact isolates unnecessarily.
- `notimely`: no timely fact; all arms use the same baseline input and must
  be step-equivalent.

The abstract outcome model uses nonnegative water levels; A/B equipment
damage and preservation are computed from the traces. A outcome is mixed
(success and failure) and does not change B's node.

## Development gate (16 groups)

8 relevant + 8 non-applicable, paired normal telemetry. Gate requires a
same-normal-telemetry pair with different legal execution facts and opposite
B choices, plus real replay consequences. No holdout is run unless this
passes.

## Holdout (72 new seeds)

24 relevant, 24 non-applicable, 24 no-timely. Three arms as above. Relevant
acceptance: real fact at least 4/24 fewer B losses than baseline, paired 95%
bootstrap interval excluding zero, real better than permutation, team A+B
not lower. Non-applicable: at most 2/24 extra unnecessary isolation, team not
lower. No-timely: arms identical. The holdout is run once after the freeze
commit; no re-draw or exclusion by A outcome.

## Run

```
python3 fact_gate.py dev --out results/dev.json
python3 fact_gate.py holdout --out results/holdout.json
```
