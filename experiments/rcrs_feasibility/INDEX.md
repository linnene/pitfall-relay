# RoboCup Rescue feasibility: index

Task: `flex-rcrs-feasibility-6a2f1d8c`
Date: 2026-11-18
Host: `amgg` only. No Actor training, no SAFE/FLEX development.

## Verdict

**Original RCRS connectivity: verified. Native task-level paired evidence: not
established within this budget. Recommendation: fill one specified gap (a
police route/clear controller) before judging the task; do not enter full
MacroMARL adaptation yet.**

The official Java server, Python core and sample agents connect and exchange
legal local observations and commands. A scripted police agent can issue a
native `AKClear` and the clear simulator removes a blockade (two blockades
cleared in the last run). But the same script could not move close enough to
the next route blockade: the server rejected it as "not adjacent" at 16.7 m,
and the ambulance remained at its start road. The resulting no-delivery is a
scheduler/navigation defect in our probe, not evidence that RCRS cannot
provide the task or that a schedule ordering fails. No paired arm was run.

## Environment and original link

- Server: `roborescue/rcrs-server` at `565d6d4bf72e4acaed2a3bbacceb9ba29b26e635`
  (release 2.1 build; Gradle 9.3.1).
- Python core: `roborescue/rcrs-core-python` at
  `db980d7e3487be063545fd95b3495fd4acbc0b11`.
- Python sample: `roborescue/rcrs-sample-agent-python` at
  `b67ab77a9cdbc677a5ba49999d8bc4afeddc4a46`.
- Isolated env: `$D/envs/rcrs`, Python 3.10.22, OpenJDK 21.0.10,
  protobuf 3.20.3, rtree 1.4.1. No host install; no GPU.
- Map/config: official `maps/test` (`map.gml`,
  sha256 `8d085161...`; `scenario.xml`, sha256 `55ebf34a...`;
  `kernel.cfg`, sha256 `8e72bb1a...`).
- Roles: one police force, one ambulance team, one fire brigade, five
  civilians, two refuges; no initial fire in the test scenario.
- Verified: all three named roles connect; each receives its own legal local
  world model; commands appear in the kernel command stream; the clear
  simulator executes and removes blockades. A dump at t=10 shows the
  ambulance seeing roads 974/975/976, refuge 255 and building 254, with
  blockades on 975/976; the police sees its own road and route blockades.
- 300-timestep server; `kernel.agents.ignoreuntil: 3`; config and map fixed.

## Last scripted run (`runs/run_early5`)

Schedule: police starts at t=3; ambulance starts immediately at t=3; fire
idles. Police local blockades at t=27 include roads 971/976/266/274/279.
Native clear log:

```
Cleared Blockade (2100749325)      # road 279
Rejecting ... Target = 2100749364: agent is not adjacent ... 16702.5
Cleared Blockade (2100749363)      # road 971
Rejecting ... Target = 2100749363: target does not exist
```

The police then moved to road 971 but could not approach the remaining
blockade on 971/266; the ambulance stayed at road 976 and never reached
building 254. Therefore no civilian was loaded or delivered and no
success/failure pair was produced.

Why this is not a task-level negative:

- The clear simulator did execute real native clear commands; the failure is
  our agent's inability to navigate to within `clear.repair.distance` of the
  next blockade and to keep its local world model synchronized after removal.
- The task's stop rule says failures caused by navigation/control in the
  probe cannot be attributed to ordering. We did not manufacture a failure or
  run the message arms.

## B's decision opportunity (desktop only, not validated)

At t=10 the ambulance's legal local model contains its own position (road
976), blockades on 975/976, building 254 and refuge 255; it does not receive
server truth. Two native executable options exist:

1. start immediately toward 254 and stage until the police clears 976/975;
2. wait at 976 for the route to become passable, then move 976→975→974→254.

Both use the same `AKMove`/`AKRescue`/`AKLoad`/`AKUnload` actions. The
ambulance cannot observe the police's outcome until it moves or receives a
message, so a factual A event could be delivered before B commits. This is a
desktop observation; no B-arm was run, and no claim of information value is
made.

## MacroMARL adaptation estimate

| Interface | Estimate |
| --- | --- |
| Joint training information / team reward | RCRS exposes per-agent local perception; no native team reward. Need a task wrapper and reward definition. |
| Execution local observation | Thin adapter: encode local entities/graph and action mask; no global truth. |
| Action mask | Thin adapter over discrete RCRS commands (move/clear/rescue/load/unload/extinguish); no continuous control. |
| Async completion / `mac_done` | Thin adapter: map action result/sense changes to done; RCRS commands are per-timestep, not vectorized. |
| Observation alignment | Medium: variable-length graph entities and local visibility need a fixed encoder. |
| Reset/seed/server isolation | Medium/heavy: one server is a central simulation; parallel training needs separate ports/processes and seed isolation. |
| Throughput | Single instance: about 0.8 timestep/s observed (145 timesteps in ~180 s) with 8 agents; a 300-step episode is minutes. No two-instance test was run. |

This is not a thin drop-in for MacroMARL; a single-instance task adapter is
feasible, but parallel training would need a wrapper/kernel rewrite. The
immediate bottleneck is the police route/clear controller, not the learner.

## Recommendation

**Only fill one specified gap:** implement a police controller that drives to
within `clear.repair.distance` of each route blockade, verifies native removal
from its local model, and then re-runs the same schedule pair. If that pair
still cannot produce a native success/failure contrast, close the candidate.
Do not enter full MacroMARL adaptation before that pair; if the pair succeeds
but the parallel interface still requires a kernel rewrite, compare another
implementation rather than switching algorithms silently.

## Anchors and data

- Workspace execution SHA: `badd65b` (`relay_agents.py` with clear-attempt
  cap; run_early5 used this code).
- Agent sha256: `0f86002b2a5a721cc246abee985fa269919965ddb20a10e21ec81abdd05b8090`.
- Server/core/sample commits as above; no training checkpoint.
- Raw data: `/data/amgg_flex_demo/flex-rcrs-feasibility-6a2f1d8c/runs/`
  (`dump4`, `run_early1`–`run_early5`), logs under `logs/` and each run.
- 10 simulation starts used (24 max); no holdout, no 72-run phase, no GPU.
- All simulation and analysis on `amgg`; no raw data in Git.
