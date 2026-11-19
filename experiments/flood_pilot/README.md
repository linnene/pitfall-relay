# Flood pilot: abstract two-area task

Task: `flex-flood-pilot-b71e8a42`
Date: 2026-11-19
Host: `amgg` only, single-CPU Python job. No Isaac, Mac-IAICC, SAFE, FLEX,
Actor training, or system packages.

## Task

Two work areas A and B in one facility. Each area has a local water level, a
piece of equipment with a damage threshold, a local handling robot, and its
own pump. A single shared isolation robot must travel and work; it can
isolate one area inlet at a time. Area inflows are pre-sampled per seed:
a common pipe contribution plus area-local contributions. A and B have
different areas, thresholds, pump capacities and action times. Isolation
reduces the selected area's total inflow to `iso_effect` of its current value
(the actual area-inlet scope), not the other area.

Macro actions are `pump`, `request+execute isolation`, `transport equipment`
and `wait`. Pump and transport occupy the area's local robot; isolation
occupies the shared robot. Water is nonnegative: drain is capped by available
water, and equipment is lost when accumulated water damage reaches 1 before a
successful transport to the safe point. The simulator is deterministic per
seed; truth is used only by the simulator and scorer, never by a policy.

## Gate 1 (development, 16 seeds)

For each seed the same initial state and exogenous inflow sequence are used.
Two fixed schedules are compared:

- `pump_first`: each local robot deploys its pump, then transports; the
  shared robot requests isolation only after both pump set-ups.
- `transport_first`: each local robot transports immediately; the shared
  robot requests A's isolation at t=0.

`high_a` seeds make A high-pressure: pump-first delays A's transport past
damage, while transport-first saves it. `urgent_b` seeds make B fragile and
more urgent: pump-first protects B, while early isolation of A occupies the
shared robot and B is lost. Eight seeds per regime; no seed is reused as a
holdout. The gate passes only if both contrast directions are reproducible.

## Gate 2 (frozen holdout, 48 groups)

After gate 1, A follows a pump-first schedule and fails. At A's failure, B
(whose state has continued to evolve) has a natural decision. The same local
decision framework is used in three paired arms:

1. `shared`: normal local/team state only.
2. `notice`: shared plus an ordinary A-failure notice.
3. `history`: notice plus A's locally observable water curve, chosen order,
   actual pump work duration and equipment outcome.

The framework forecasts damage for transport, pump+transport and
isolate+transport using current/recent B water, trend, pump capacity, damage
threshold, isolation ETA and predicted loss time. The history arm may only
adjust its inflow estimate from A's observable pre-pump slope; a consistency
guard rejects transfer when B's own recent trend is not rising with A. It
never receives true inflow, an oracle applicability label, or a suggested
action. The holdout has 24 relevant groups (shared pipe, B branch lags A) and
24 non-applicable groups (A's rise is local and B's branch is flat). Group
labels are used only for scoring.

Thresholds for a positive information-value result: in relevant groups,
history loses at least 4 fewer B targets out of 24 than notice, with team
preservation not lower; in non-applicable groups, history adds at most 2/24
wrong isolation requests, with team preservation not lower. Report includes
B loss, team preservation, decision changes, wrong isolation, timing and
paired bootstrap intervals. A pass is evidence for a task-level information
increment only, not that an Actor learned to use it.

## Run

```
python3 flood_pilot.py gate1 --out results/gate1.json
python3 flood_pilot.py holdout --out results/holdout.json
```

The holdout is run once after the code/config/seed freeze commit.
