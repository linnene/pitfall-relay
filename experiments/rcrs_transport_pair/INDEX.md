# RCRS transport pair: index

Task: `flex-rcrs-transport-pair-0de672a9`
Date: 2026-11-19
Host: `amgg` only. No training, no SAFE/FLEX, no dynamics/action edits.

## Verdict

**Simulation reusable; no qualified task (仅能复用仿真、没有合格任务).**

The locked RCRS server, official ADF Java sample and 90 roles connect and run
a complete 270-step Kobe episode. The official police ExtAction clears 23
blockades natively with zero distance rejections. But the official sample
completes no live transport at all: `AKLoad` = 0, `AKUnload` = 0 for the whole
episode, and the selected civilian `1953270408` is dead by kernel time 174.

The blocker is the official component's transport logic, not connectivity,
map, or clearing:

- `SampleHumanDetector.filterRescueTargets` selects only civilians with
  `buriedness > 0`.
- `DefaultExtActionTransport.calcRescue` returns `ActionLoad` only when the
  target is at the ambulance position and `buriedness == 0`; otherwise it
  returns only `ActionMove`. It never returns `ActionRescue`.
- Only the fire-brigade `DefaultExtActionFireRescue` emits `ActionRescue` in
  this stack.

So ambulances keep approaching a buried target that nothing in their own
component can unbury: a component-level deadlock. Per the task stop rule, we
did not write a custom ambulance rescue controller, did not freeze a pair, and
did not interpret this as evidence about ordering.

## Runs

| Run | Map/config | Agents | Result |
| --- | --- | --- | --- |
| `runs/adf1` | official `maps/test` | 3 | connected; 300 steps; 5 civilians are not casualties, so no rescue target |
| `runs/adf_kobe1` | Kobe | 3 attempted | wrapper start did not reach agent phase; no evidence used |
| `runs/kobe_start` | official Kobe | 90 (30 fire / 30 police / 30 ambulance) | complete 270-step episode; 23 blockades cleared; no load/unload |

Kobe final native counts from `kernel.log`:

```
AKMove   4866 = 68125
AKClear  4873 = 157
AKRescue 4872 = 757
AKLoad   4867 = 0
AKUnload 4868 = 0
```

`clear.log`: `Cleared Blockade` = 23, `Rejecting` = 0.

Target `1953270408`: buriedness 60 throughout; last recorded state HP 42,
damage 125; kernel records `Agent Civilian (1953270408) is dead` at
`Time = 174` and ignores its later commands. The ADF ambulance log selected
this civilian at 10:30:29, but no load/unload action followed.

## Why there is no pair or B check

The first-stage gate (one native live delivery with native evidence and local
agent feedback) was not met. Therefore no action contract, target,
destination, timing, role rules, or seeds were frozen; no same-seed serial vs
parallel arm was run; no second scenario seed was run; and B's high-level
choice was not tested. Ordinary notification vs full-experience comparison
was not reached. This is not a negative result for schedule ordering.

## Environment and anchors

- Server: `roborescue/rcrs-server` 2.1, commit
  `565d6d4bf72e4acaed2a3bbacceb9ba29b26e635`; Gradle 9.3.1.
- ADF sample: `roborescue/adf-sample-agent-java`, commit
  `e9339b10a5188ae15747fba836b6ea035b3b570d`.
- ADF core: `roborescue/adf-core-java`, commit
  `350255c676a0dda3ea47a974541dbee2e5fb3f73`.
- Java: OpenJDK 21.0.10 in `envs/rcrs`; no host install, no GPU.
- Kobe `map.gml` sha256
  `b75c7d4124e0733045df39ed2a7e500ad32bfb2ff70d69d05d862e39176d5ee9`;
  `scenario.xml` sha256
  `f7dfdbba53e25ef931850532b14028707b7566b430f120a8b576d550b4b904c3`;
  `kernel.cfg` sha256
  `c27014eee389824146552334cdf9efe3c58e037bc8ad2579bd5d47d92497154d`;
  `civilian.cfg` sha256
  `59890a90886e53962385e326e232bc7c8dd75ddbeff9aece958aa4f031791914`.
- Test-map `map.gml` sha256
  `8d0851611029c2d5049a3d606031c4aa5e833115d0a826d576289a93a6979331`;
  `scenario.xml` sha256
  `55ebf34adb56abf4018dfa835513d2b750d19e3163d7951a1b83cb18b910ee60`;
  `kernel.cfg` sha256
  `8e72bb1aa4eb6e2d1241fa3c239e3e66c02dee68108053bd51533f9aa05877a5`.
- Run code anchor: `ff81e72` (ADF wrapper map selection); task notes
  `0940a7f`; this index is the final documentation commit.
- 3 simulation starts (16 max), about 50 minutes, no training checkpoint.

## Raw data

`/data/amgg_flex_demo/flex-rcrs-transport-pair-0de672a9/runs/`
(`adf1`, `adf_kobe1`, `kobe_start`) and
`src/adf-sample/logs/`. `kernel.log` is about 1.7 GB and `misc.log` about
24 MB; both stay on `amgg`. Local report:
`/tmp/flex-rcrs-transport-pair-0de672a9.md`.

## Recommendation

Treat RCRS as a reusable simulation, not an off-the-shelf qualified task. If
it is revisited, first make a separate minimal ambulance `ActionRescue`
ExtAction and prove one native full chain (clear, unbury, load, refuge unload,
alive) before any ordering pair, B check, MacroMARL adapter, FLEX, or
training. Do not spend more on the current official sample path.
