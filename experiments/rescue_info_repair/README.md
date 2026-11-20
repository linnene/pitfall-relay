# Rescue information repair

Task: `flex-rescue-info-repair-e61a739b`
Date: 2026-11-20
Host: `amgg` only, single-CPU Python. No training, SAFE/FLEX, Isaac, or new
scenario search.

## Repair

Reuses the task-level actions/resources/delivery rule from `292b924`. A and B
are different cases with independent local seeds; A's experience is one actual
serial execution and its confirmed team outcome. B's legal observation is
limited to current HP, a short recent HP trend, clear/transport ETA,
stage gain and ambulance ETA. It cannot see the full case, true future decay,
category, arm name or counterfactual. The normal estimator uses the observed
trend as its decay estimate and simulates both orders. The experience arm may
only raise that same observed decay estimate after a similar confirmed serial
failure; no optimal action or counterfactual is sent.

## Development gate (12 cases)

6 `parallel_better` + 6 `serial_better` new seeds; A/B cases separate.

| Regime | A confirmed failure | normal vs experience | normal/experience team |
| --- | ---: | ---: | ---: |
| parallel_better (6) | 6/6 | 6/6 same | 2/2 vs 2/2 |
| serial_better (6) | 0/6 | 6/6 same | 2/2 vs 2/2 |

The observed trend already predicts the optimal order in every case; the
experience arm never changes B's choice or team delivery. There is no pair
with near-identical legal observations but different optimal B choices that
needs A's record. The development gate fails; no 24+24 holdout is run.

## Limits

This is a minimal abstract rescue-ordering task. It shows that the legal
trend/ETA estimator is sufficient in this development set; it does not prove
that no possible failure-experience signal exists. The previous 12 cases are
only diagnostic and are not counted as evidence here.

## Run

```
python3 rescue_info_repair.py --out results/dev.json
python3 -m unittest -v test_rescue_info_repair.py
```
