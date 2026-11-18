# Warehouse-A handoff-interface stall variant

Task: `flex-warehouse-handoff-variant-20261118-a92b6f31`
Base: `ed774c53762801a0c50096948ecb5a596792ac85` (Warehouse-A reproduction)
Formal implementation commit: `fe3067f` / analysis commits `67beac7`, `893a5f9`.

## What changes

Only the Fetch-T0 handoff interface condition changes.  Two short stall
intervals per episode are sampled from a hash of the episode seed (independent
of the Actor, actions, roles and failure labels).  At the completion step of a
Fetch `Pass_Obj_T0/T1`, if the interface is stalled and the transfer would
otherwise succeed, the upstream not-ready failure branch is used: the object is
not transferred and the original drop penalty applies.  If the interface is not
stalled, the original transfer occurs.  No macro-action is forced, no timer is
slept, no teammate is delayed, and no other reward/task logic changes.

The variant records for every stall failure:

- failure step and target;
- T1's first real Actor forward/decision step after the failure;
- whether T1 had both a tool-station option and a real delivery option;
- the chosen action and whether it uses the shared interface;
- the local T1 basket/ready observation at the decision;
- whether the stall interval was still active at the decision (reported
  separately; the window definition uses interface-relevant choice, not
  stall-active-at-decision).

A window is counted when T1's first decision after the failure has both a valid
tool option and a valid delivery option, and T1 chooses the tool action.  The
probe includes this decision step as the detection point: the Actor forward at
that decision produces the new latent, and the detector can use it before the
chosen macro-action is executed by SAFE.  Non-window failures are kept in the
report so total window coverage is not hidden.

## Gate (frozen original Actor, seed 11)

- Natural short check: 300 episodes, 289 stall failures, 27 windows
  (9.34% coverage); 300/300 task success.
- Controlled pairs: 100 control episodes; 43 episodes with natural stall
  failures and matching no-stall runs.  All 43 share selected/actual action and
  observation prefixes up to the failure; 7 are windowed pairs.
- Criteria in `configs/handoff_variant.json`: >=10 windowed events, >=10 valid
  pairs, >=5 windowed pairs, <=300 natural and <=100 control episodes.
- Gate result: passed (`runs/gate_decision_seed11.json`).

Seed 22 task check under the variant: 192/200 success, 221 stall failures,
9 windows.  Seed 33: 193/200 success, 194 failures, 0 windows.  The original
Actors retain task ability; no retraining or new feedback was needed.  Frozen
seeds 11 and 22 provide the two independent policies for the probe; seed 33 has
no windows and is reported as a policy-specific negative.

## Probe

Pre-registered probe uses `probe_latent.py --event interface_handoff_failure:1`,
with the variant `probe_events` (windowed failures have an exclusive sentinel
deadline `t1_decision_step + 1`; non-window failures have no deadline).  The
normal set is complete episodes with no stall failure.  Split is decided by
episodes and collection seeds; thresholds are frozen on normal calibration
episodes at 5% episode false alarms.  Arms: local history, plus real GRU latent,
plus fixed cross-episode shuffled latent.  A direct local inventory/ready rule
is reported separately (`local_rule.py`).

This variant is an abstract execution condition, not a physical fault model;
the gate and probe do not establish physical failure coverage.
