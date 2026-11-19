# Flood recheck: sensor history versus execution context

Task: `flex-flood-recheck-39c6d8a1`
Date: 2026-11-19
Host: `amgg` only, single CPU. Reuses the frozen water/damage/action mechanics
from `experiments/flood_pilot/flood_pilot.py` (commit `5c4949c`). No Isaac,
SAFE/FLEX, training, or system packages.

## Goal

The previous pilot's "full history" arm actually keyed on A's pre-pump water
slope, so it did not separate ordinary sensor sharing from failure-execution
context. This recheck uses four parameter categories and four information
arms to test whether A's high-level attempt, execution interval and outcome
add value beyond A's raw sensor history.

## Four categories (24 new holdout seeds each)

1. `relevant_pump`: shared inflow is transferable to B; A's pump is active in
   the observation window, so A's raw water slope is nearly flat and hides
   the shared rise. B's branch lags A.
2. `relevant_delay`: another transferable condition with a different
   propagation delay, pump capacity, transport time and isolation ETA.
3. `nonapp_local`: both A and B water levels rise, but A's rise is mainly
   local inflow and B's shared contribution is small; A's history is not
   transferable.
4. `nonapp_handled`: both levels rise and A's shared rise is real, but B has
   already been handled separately at the decision point; isolation would be
   unnecessary.

Category labels are used only by the scorer, never by a policy.

## Four arms (same event stream, initial state and seed)

- `state`: normal state sharing.
- `notice`: normal + ordinary A-failure notice.
- `sensor`: notice + A's same-window raw water samples and standard visible
  pump on/off state; it can reproduce the previous raw-slope estimate.
- `exec`: sensor history + A's high-level order, pump execution interval and
  equipment outcome.

All arms use the same decision framework and prediction model. The `exec`
arm may correct A's raw slope using the observed pump-active interval and
standard pump capacity; it may not receive true inflow, category labels or a
suggested action. A consistency guard rejects transfer when B's own recent
trend is not rising or when A's corrected estimate is more than 5x B's
estimate. The guard does not use "B curve flat" as an applicability label:
both non-applicable categories have rising B water.

The `sensor` arm deliberately gets the raw samples and pump state needed to
reproduce the prior estimate; it is not weakened. Its remaining limitation is
that it lacks the execution interval needed to remove A's pump effect.

## Freeze and decision

Development seeds are 4 per category (16 total); holdout seeds are 24 per
category (96 total, disjoint from development). The holdout is run once after
the code/config/seed freeze commit.

Primary: `exec` vs `sensor` B equipment loss over the 48 relevant groups.
Require at least 8 fewer losses, a paired 95% bootstrap interval excluding
zero, no negative reversal in either relevant subclass, and no decrease in
team preservation. Non-applicable: `exec` vs `sensor` adds at most 4/48
unnecessary isolations or wrong reassignments, with no decrease in team
preservation. Report actual A/B/team counts, all-preserved rate, protocol
invalidity proportions, occupied time, decision changes and worst cases.

If `exec` only beats `notice`, or ties `sensor`, the result supports sensor
sharing only and does not justify a FLEX main task or RL interface. If the
denominator is insufficient or the protocol is invalid, report undetermined
and stop.

## Run

```
python3 flood_recheck.py dev --out results/dev.json
python3 flood_recheck.py holdout --out results/holdout.json
```

All raw data stays under
`/data/amgg_flex_demo/flex-flood-recheck-39c6d8a1/`.
