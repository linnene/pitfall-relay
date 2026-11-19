# Flood pilot: index

Task: `flex-flood-pilot-b71e8a42`
Date: 2026-11-19
Host: `amgg` only, single-CPU Python job. No Isaac, Mac-IAICC, SAFE, FLEX,
Actor training, or system packages.

## Verdict

**Both gates pass: high-level ordering changes outcomes, and A's full local
execution history adds value over normal state sharing and a plain failure
notice in the relevant frozen holdout.** This is scripted evidence for a
task-level information increment, not evidence that an Actor learned to use
it. No training was started.

## Gate 1: order reversal (16 development seeds)

Same initial state and exogenous inflow sequence per seed; each seed run
twice with identical results. Two fixed schedules:

- `pump_first` (P): local robots pump, then transport; shared isolation robot
  is requested after both pump set-ups.
- `transport_first` (I): local robots transport immediately; shared robot
  requests A's isolation at t=0.

| Regime | P preserved A/B | I preserved A/B | Winner |
| --- | --- | --- | --- |
| `high_a` (A high pressure) | 0/8, 8/8 (8/16) | 8/8, 8/8 (16/16) | I in 8/8 |
| `urgent_b` (B more urgent) | 8/8, 8/8 (16/16) | 8/8, 0/8 (8/16) | P in 8/8 |

High pressure: A's pump set-up delays transport past the damage line; early
transport saves A (P loss at t=6, I preservation at t=5). Urgent B: B's lower
threshold and faster damage make pumping necessary; early isolation of A
occupies the shared robot and B is lost at t=4. The same action set and flows
are paired; only order differs. No single fixed order wins both regimes.

## Gate 2: frozen 48-group holdout

A follows pump-first and fails; B's state continues to evolve and B makes a
natural decision at A's failure. Three paired arms use the same local
forecasting framework:

- `shared`: normal state sharing.
- `notice`: shared + ordinary A-failure notice.
- `history`: notice + A's observable water curve, chosen order, actual pump
  work duration and equipment outcome.

The history arm may only update its inflow estimate from A's observable
pre-pump slope; a consistency guard rejects transfer when B's own recent
trend is not rising. No true inflow, oracle applicability label, or suggested
action is passed.

Relevant groups (n=24, shared pipe with B branch lag):

| Metric | shared | notice | history |
| --- | ---: | ---: | ---: |
| B equipment loss | 23/24 | 23/24 | **0/24** |
| Team preserved | 1/24 | 1/24 | **24/24** |
| Decision changes vs notice | — | — | 24/24 |
| Wrong isolation requests | 0 | 0 | 1 |
| Mean completion time (min) | 13 | 13 | 14 |

Non-applicable groups (n=24, A's rise is local and B's branch is flat):

| Metric | shared | notice | history |
| --- | ---: | ---: | ---: |
| B equipment loss | 0/24 | 0/24 | 0/24 |
| Team preserved | 24/24 | 24/24 | 24/24 |
| Wrong isolation requests | 0 | 0 | 0 |

Paired clustered interval for relevant B-loss difference
(`history - notice`): mean -0.958, bootstrap 95% CI [-1.0, -0.875];
non-applicable difference 0.0.

Thresholds: relevant loss gain 23/24 (>=4), team preservation not lower,
non-applicable extra wrong isolation 0/24 (<=2), team preservation not lower.
All 48 groups valid, no run errors. One relevant history decision over-isolated
although immediate transport would have preserved B; it still preserved B and
cost 1 minute on average, so the gain is not cost-free.

## Interpretation

The relevant groups contain a shared inflow step that A observes before B's
lagged branch. A's pre-pump curve lets the history arm raise its B inflow
estimate and isolate in time; the plain failure notice does not quantify the
shared rise, and normal state does not reveal it early. The guard keeps A's
local change from being transferred in non-applicable groups. This supports a
high-level RL interface candidate, subject to an independent replication with
unseen generator parameters; it does not show learned experience use.

## Anchors and data

- Design/freeze commit: `f157375`; formal holdout execution SHA: `5c4949c`.
- Script sha256: `0a90fd5c25a6ce7d4037a8b5decc4196dc3a200bd08f83ec1de645adb2b0e104`;
  `holdout.json` sha256:
  `ffda9dd5c5c6ea768837b6ade37a2fdbaaa88ef9f0bd6439d0f55909ac075519`.
- Python 3.12.3 on `amgg-Super-Server`; single CPU process; no external
  packages; no training checkpoint.
- Development seeds: `high_a` 1–8, `urgent_b` 1–8. Holdout: relevant
  2001–2024, non-applicable 3001–3024.
- Raw data:
  `/data/amgg_flex_demo/flex-flood-pilot-b71e8a42/results/`
  (`gate1_final.json`, `holdout.json`, `anchor.json`, `holdout.time`, smoke
  records) and code under `.../code/`.
- Unit tests on `amgg`: 3/3 passed.
- Local report: `/tmp/flex-flood-pilot-b71e8a42.md`.
- Wall time about 10 minutes, well under the two-hour cap; Mac used only for
  editing and the report.

## Recommendation

Keep this task as a high-level RL interface candidate. First replicate with a
new generator parameterization and untouched seeds, especially to test the
guard and the cost of unnecessary isolation. Do not start training from this
scripted pass, and do not describe it as evidence that an Actor has learned
to use execution history.
