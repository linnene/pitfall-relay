# RCRS live transport and order pair

Task: `flex-rcrs-transport-pair-0de672a9`
Date: 2026-11-19
Host: `amgg` only. Reuse the locked RCRS `565d6d4`, official ADF Java sample,
and isolated Java 21 / Python environment. No training, SAFE, or FLEX.

## Question

Can an official RCRS component complete a live rescue delivery, and only then
does a high-level order choice (serial prerequisite vs early preparation)
make a real, native difference?

## Gates

1. Build the official ADF Java sample against the locked server and connect
   the required roles.
2. Produce one native live delivery: clear/load/refuge-unload with the
   civilian alive, confirmed by native events and the agent's local feedback.
   Stop if not within 60 minutes or 8 starts.
3. Only if (2) passes: find a local executable order choice that changes
   progress before the route is fully clear; freeze actions/targets/timing
   and run same-seed pairs. Stop if both arms wait or outcomes are equal.
4. Only after a native pair: check B's decision opportunity and ordinary
   information; mark desktop-only value as unverified.

## Current status

Closed. The official ADF sample clears 23 blockades natively on Kobe but
completes no live transport (`AKLoad`/`AKUnload` = 0); its ambulance transport
never emits `ActionRescue` for a buried civilian. Verdict and anchors are in
`INDEX.md`. No formal pair or B check was run. Raw data stays under
`/data/amgg_flex_demo/flex-rcrs-transport-pair-0de672a9/`.
