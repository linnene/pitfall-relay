# Relay-site pilot

Task: `flex-relay-site-pilot-20261118-5be93a27`
Host: `amgg` only. Reuse the accepted Isaac Sim 4.5 / Nav2 / Carter stack;
no Actor training, no SAFE integration, no new assets.

## Question

With the ordinary shared map and failure notices retained, can one applicable
failure experience reduce a receiver's repeated trial-and-error?

This is a task-feasibility and information-value pilot, not a FLEX result.

## Candidate task

- Robots A, B, C are Carters. A and B each hold one logical order; C receives
  the relay and delivers. Payload is task resource state only; no grasping and
  no physical-transfer claim.
- Two transfer stations, near and far. A and B must both reach the selected
  station, be close and slow, then commit a logical handoff. Changing station
  changes both A's and B's docking plans, not just a route to one point.
- Candidate failure: the near station's docking bay has insufficient clear
  width for the loaded Carter's Nav2 docking maneuver, while the far station
  is clear. The condition is scene geometry, independent of policy actions.
  No client cancel, no runtime obstacle flag, no compressed timeout, and no
  hand-authored failure label.

## Gates before any paired holdout

1. **Native failure.** A reaches the near bay and gets a real Nav2 abort
   (planner/controller), not a client cancel or timeout. The far bay succeeds
   under the same controller and budget.
2. **Observability.** Check whether the shared map or B's observation at the
   choice node already determines that the near station is infeasible. If it
   does, the scenario is not applicable and the pilot stops.
3. **Causal chain.** A and B can both hand off under the normal layout; A's
   failure is a local action result and an actual non-handoff; B naturally
   reaches a choice node with both stations executable; C receives and
   cooperates with the selected station; no central allocator or forced wait.
4. **Repeat trial.** Without experience, B's fixed local navigation estimate
   selects the lower-cost feasible station and actually repeats the failure.
   If B would not repeat, stop; do not substitute a weaker baseline.
5. **Condition matching.** The full experience's Condition must be matchable
   with locally visible state at B's choice. If it requires runtime-invisible
   information, stop.

Only after all gates pass: freeze code, layouts, rules, seeds, and the
three-arm message treatment; run 12 persistent-failure and 12 resolved /
not-applicable groups with original-info, ordinary-notice, and full-experience
arms. Gate: ordinary notice >= 6/12 real repeats, full experience at least
4/12 fewer with no completion loss, and at most 1/12 extra wrong reselect in
the not-applicable group.

## Current status

Development probes only. No holdout has been run. Raw data stays under
`/data/amgg_flex_demo/flex-relay-site-pilot-20261118-5be93a27/`.
