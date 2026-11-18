# Relay-site pilot: index

Task: `flex-relay-site-pilot-20261118-5be93a27`
Date: 2026-11-18
Host: `amgg` only. No Actor training, no SAFE integration, no new assets.

## Verdict

Stopped at the native-failure gate. The current Isaac/Nav2/Carter stack can
produce a clean `SUCCEEDED` in wide and moderately tight docking bays, and a
planner-infeasible failure in an over-tight bay, but it does not produce a
**native, action-visible failure** in the narrow but planner-feasible band.
The only such band observed was a budget timeout: the action remained
`EXECUTING`, no result was delivered, and the BT abort in the log was induced
by the probe's own shutdown. Counting that as failure would violate the
no-timeout-compression rule.

Because A's failure cannot be confirmed by a local action result/status, the
required causal chain (A fails -> B reaches a choice with two executable
stations -> ordinary notice vs full experience) cannot be established without
modifying the controller/BT or adding a timeout supervisor. The candidate is
closed. No paired arms, no 24-group holdout, no fabricated failure labels.

## Development probes

All probes use the existing `isaacsim4.5.0-nav2:local` image, three or one
Carter, one GPU, and the same Nav2 parameters as the accepted stack. No client
cancellation is issued; the probe exits at the stated budget and the shutdown
SIGINT is not treated as failure evidence.

| run | shape | clear widths (m) | outcome | native action result |
| --- | --- | --- | --- | --- |
| `bay_probe1` | straight | 1.8 / 1.4 / 1.2 | all three `SUCCEEDED` | yes, 0.16-0.18 m xy error |
| `bay_probe2` | straight | 1.4 / 1.18 / 1.10 | all three `SUCCEEDED` | yes, 0.12-0.13 m |
| `bay_probe4` | L turn | 0.75 | planner: `failed to generate a valid path`; robot never moved | no result; shutdown only |
| `bay_probe5` | L turn | 0.90 | `SUCCEEDED` in 47.4 s | yes, 0.18 m |
| `bay_probe6` | L turn | 0.80 | moved, then still executing at 120 s | no result |
| `bay_probe7` | L turn | 0.85 | moved close, still executing at 150 s | no result |
| `bay_probe8` | L turn | 0.85 | action status history `[[2]]` (EXECUTING); no result | no |
| `bay_probe9` | L turn | 0.85 | still executing at 240 s, pose (4.382, 0.440, 2.739) | no |

`bay_probe3` was a three-robot startup attempt that stalled before goals and
is not used as evidence.

Key observations:

- The 0.75 m L-turn is planner-infeasible from the shared static map. B's
  planner uses the same map and robot radius, so B can rule it out before
  choosing; this is exactly the "shared map suffices" stop condition.
- 0.9 m succeeds, so the failure is not a broad "narrow corridor" effect. The
  0.8/0.85 m cases are close to the controller's limit but do not abort; they
  simply fail to finish.
- In `bay_probe8`, the NavigateToPose action status topic published only
  `2` (EXECUTING). The subsequent `Failed to get result for follow_path in
  node halt` / `Aborting handle` lines occur when the probe kills the process
  group, not during the run. There is no local action result or ABORTED
  status to hand to a mission layer.

## Stop conditions

The task's stop rules apply:

- "若现成控制接口不支持，或无法在不改控制器情况下得到真实成功/失败对照，
  停止并报告，不造失败标签。"
- "A 的失败由局部动作结果和实际未交接确认."
- "禁止通过客户端主动取消、硬编码障碍状态或压缩超时制造失败."

The observed 0.8/0.85 m cases are timeouts, not native failures; the 0.75 m
case is map-visible. A mission layer would either see a timeout (forbidden as
failure evidence) or need log scraping (not a local action result). We did not
build the relay, verify C's dependency, or run any paired arm. The result is a
task-feasibility negative, not a test of experience value.

## Anchors

- Workspace execution SHA: `24c458a63b4187c6bbc5a5b8cf7334b3593bace1`
  (`bay_probe.py` includes action-status capture).
- Probe code sha256: `a147243096ad4f55bbe5e912b3c01ab7472009251049e2808567090f21a23d11`.
- Base stack: `experiments/isaac_nav2` at `79d6e29`; Nav2 params sha256
  `b781d916a7f2c449b86d49bc0f649420d625bacf10957bee8f5c447485e272cd`.
- Image: `isaacsim4.5.0-nav2:local`, id
  `sha256:9bcb836eea0f19c5befaf80fd1d8f0cc37d08409b09444c47b44f47bd4108874`.
- Probe-9 map hashes: PGM `66a1ec9353368605300ebdfddd9f921b50f6a44d8fd5d1398026c011845afbd4`,
  YAML `d622fa448d1b77feb0fd87575c8ee96772596de6f3cf6e4369b2e9bc484b2f8e`.
- Probe-9 result: `5a04562a710b599c4a784285c0b969ac6d664c9398e8b2179f6a876b081becda`.
- No training checkpoint was created or used.

Remote data:

```
/data/amgg_flex_demo/flex-relay-site-pilot-20261118-5be93a27/
  code/                 baseline isaac_nav2 copy + bay_probe.py
  runs/bay_probe{1,2,4,5,6,7,8,9}/
  logs/
```

## Reproduction

```bash
D=/data/amgg_flex_demo/flex-relay-site-pilot-20261118-5be93a27
docker run --rm --gpus device=0 \
  -e OMNI_KIT_ALLOW_ROOT=1 -e ROS_DOMAIN_ID=87 -e RUN_OUT=/runs/bay_probe9 \
  --shm-size=2g -v "$D/code":/work -v "$D/runs":/runs \
  --entrypoint bash isaacsim4.5.0-nav2:local \
  -lc "bash /work/run_probe.sh --timeout 240 --single --shape l --widths 0.85"
```
