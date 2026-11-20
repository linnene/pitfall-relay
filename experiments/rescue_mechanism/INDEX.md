# Rescue mechanism precheck: index

Task: `flex-rescue-mechanism-1f83ba62`
Date: 2026-11-20
Host: `amgg` only, single-CPU Python. No Actor training, SAFE/FLEX, Isaac, or
new environment search.

## Verdict

**Mechanism not supported for the FLEX experience increment.** The abstract
rescue-ordering reversal is real, but a strong rule using normal HP/trend/
clear/transport telemetry already chooses the same plan as A's failure
experience. Stop before the formal 24+24 comparison; no holdout, no training.

## Precheck (12 development cases)

Two casualties, one shared ambulance. `serial` serves B first while A clears;
`parallel` stages at A during A's clearing (real work occupying the ambulance),
serves A faster, then B. HP decays; success is alive arrival at the refuge.

| Regime | serial team | parallel team | reversal |
| --- | ---: | ---: | ---: |
| `parallel_better` (6) | 1/2 | 2/2 | 6/6 |
| `serial_better` (6) | 2/2 | 1/2 | 6/6 |

The reversal is actual task outcome: urgent A misses the window under serial;
urgent B misses under parallel because the shared ambulance is staged for A.

## Information check

B has its own case and normal telemetry (HP, trend, clear/transport times,
shared-resource ETA). The strong local rule predicts both plans and chooses
the better team outcome. A's observed serial failure and confirmed outcome
produce the same choice in **12/12** development cases; B's team delivery is
2/2 under both choices; there is no additional choice or resource benefit
from the experience.

Therefore the ordinary state/trend is sufficient. The task stop rule is met;
no 24+24 frozen comparison was run and no new parameters were searched.

## Limits

The model is a minimal abstract rescue-ordering prototype, not RCRS or a real
rescue system. The ordering reversal is a mechanism existence observation; it
does not establish that an Actor can learn from failure experience, and it
does not support entering Actor training.

## Anchors and data

- Code commit: `292b924`; script sha256:
  `d5282ce0b8ad8d158eb13ef46a010bf15a747ad0d58ee62f1a2b823efdda94b8`;
  `dev.json` sha256:
  `0c9663967e4961451616cd347c292520261627db9314b95425c4f28c27ad5fde`.
- Python 3.12.3 on `amgg-Super-Server`, single CPU, no external packages;
  no training checkpoint.
- Raw data: `/data/amgg_flex_demo/flex-rescue-mechanism-1f83ba62/`
  (`results/dev.json`, `dev.err`, code under `.../code/`).
- Unit tests on `amgg`: 4/4 passed.
- Local report: `/tmp/flex-rescue-mechanism-1f83ba62.md`.
- Wall time about 10 minutes, under the two-hour cap.

## Recommendation

Close this FLEX experience candidate; keep the ordering reversal as an
abstract mechanism note. Do not start the 24+24 holdout, train an Actor, or
connect SAFE/FLEX. A future reopen needs a legitimate case where normal
state/trend cannot recover the same choice, with the same decision function
and only message content changed.
