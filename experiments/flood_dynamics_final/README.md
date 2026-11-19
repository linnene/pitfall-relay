# Flood dynamics final: native evolution fact gate

Task: `flex-flood-dynamics-final-5a17c9e2`
Date: 2026-11-19
Host: `amgg` only, single-CPU Python. No training, SAFE/FLEX, Isaac, or new
sensors.

## Native evolution

A and B water volumes are evolved every minute from pre-sampled shared and
local inflow, actual pump drainage and area-scoped isolation. Equipment damage
is integrated from the evolved level; B can transport or request shared
isolation from its independent inspection node `T_B=10`. No outcome table
assigns B damage, winning actions or A outcomes. The only added logic connects
the external inflow process and the executed pump/isolation actions to the
same evolution.

## Information arms

- normal: A water trace, pump/isolation status, A failure notice (when
  present), B trace and damage.
- real fact: normal plus A's locally measured actual pump outflow, isolation
  closure and before/after levels.
- permuted fact: same format, fixed cross-scenario fact.

All arms use the same `decide` function and estimator. Baseline receives no
fact object; the fact arm receives the real object; the permuted arm receives
the fixed permutation. The result records `fact_visible` so baseline field
isolation is checkable. No-timely groups have no fact in any arm.

A and B normal traces are parameterized to be identical across the paired
relevant/non-applicable families, while the actual A work records differ. The
fact estimator uses the observable A rate change and the actual applied work:
`ql = (r0 - r1) / closure`, `qs = r0 + pump - ql`.

## Gates

Development: 16 groups (8 relevant + 8 non-applicable). A credible pair must
have the same normal trace, different legal work facts, opposite B choices and
real replay consequences. Otherwise stop without holdout.

Holdout (only if development passes): 24 relevant, 24 non-applicable and 24
no-timely new seeds, three arms, one run. Relevant acceptance: real fact at
least 4/24 fewer B losses than baseline, paired 95% bootstrap CI excluding
zero, better than permutation, team A+B not lower. Non-applicable: at most
2/24 extra unnecessary isolation, team not lower. No-timely: step-equivalent
arms. Report A/B/all-preserved, successful vs failed A work, resource
occupancy and waiting.

## Run

```
python3 dynamics.py dev --out results/dev.json
python3 dynamics.py holdout --out results/holdout.json
```
