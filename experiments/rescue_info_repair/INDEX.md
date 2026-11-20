# Rescue info repair: index

Task: `flex-rescue-info-repair-e61a739b`
Date: 2026-11-20
Host: `amgg` only, single-CPU Python. No training, SAFE/FLEX, Isaac, or new
scenario search.

## Verdict

**Information increment not supported: a legal recent-trend/ETA estimator
already chooses the same order as A's confirmed failure experience. The
development gate fails; stop before the 24+24 holdout.**

The earlier 12 cases are only reversal diagnostics and are not evidence here.

## Repair and isolation

- A and B are different cases and seeds; correlation only comes from the
  shared task/exogenous mechanism.
- B's legal observation is limited to current HP, a short recent HP trend,
  clear/transport ETA, stage gain and ambulance ETA. It contains no complete
  case, true future decay, category, arm name or counterfactual.
- A's experience is one actual serial attempt, timestamps, condition and the
  confirmed team outcome; no alt-plan counterfactual or suggested action.
- The normal estimator treats the observed trend as decay and simulates both
  orders. The experience arm may only raise that same observed estimate after
  a similar confirmed failure; the decision function is shared.
- Scoring truth and hidden decay remain in the simulator only; replacing them
  does not enter the legal observation or decision.

## Development gate (12 cases)

6 `parallel_better` + 6 `serial_better` new seeds.

| Regime | A confirmed failure | normal vs experience | team delivery |
| --- | ---: | ---: | ---: |
| parallel_better (6) | 6/6 | 6/6 identical | 2/2 vs 2/2 |
| serial_better (6) | 0/6 | 6/6 identical | 2/2 vs 2/2 |

Normal choices are parallel 6/6 and serial 6/6; experience choices are
identical. B team delivery is 2/2 in both arms. There is no legal pair with
near-identical observations, different B optima and a before-decision A
record that explains the change. The existing trend is sufficient, so the
task stop rule applies.

## Limits

This is a minimal abstract rescue-ordering model, not RCRS or a real rescue
system. It shows trend/ETA sufficiency in this development set; it does not
prove that no other failure-experience signal could ever be useful. No
holdout was run and no training was started.

## Anchors and data

- Code commit: `e105412`; script sha256:
  `fcef44bcd514145810930adf2a591844f2890aca4b42f5e2c05e4ca5ffe5a187`;
  `dev.json` sha256:
  `0fe621b66f73f82728fba8c931e654a437e2b859953149496007f7a52a0aeb05`.
- Python 3.12.3 on `amgg-Super-Server`, single CPU, no external packages;
  no training checkpoint.
- Raw data: `/data/amgg_flex_demo/flex-rescue-info-repair-e61a739b/`
  (`results/dev.json`, `dev.err`, code under `.../code/`).
- Unit tests on `amgg`: 5/5 (A/B separation, legal field whitelist, normal/
  experience choice equality, truth not in legal observation, deterministic
  replay).
- Local report: `/tmp/flex-rescue-info-repair-e61a739b.md`.
- Wall time about 10 minutes, under the two-hour cap.

## Recommendation

Close this rescue experience candidate. Do not run 24+24, train an Actor, or
connect SAFE/FLEX. A future reopen needs a legitimate case where legal
observations cannot distinguish the B optimum and A's confirmed record changes
the choice under the same decision function and communication budget.
