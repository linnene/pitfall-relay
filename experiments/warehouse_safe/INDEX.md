# Warehouse-A reproduction and SAFE pre-study: index

Task: `flex-warehouse-safe-20261118-95e0c41a`
Status: formal training complete (3 seeds, 120,000 episodes); task evaluation
and natural-event diagnostics complete.

## Result in one paragraph

The official Warehouse-A MacIAICC segment (`OSD-D-v7`, 3 agents, 4 envs,
200 steps, 40,000 episodes/seed) trains successfully on `amgg`. All three
explicit seeds (11/22/33) reach 200/200 task successes on 200 independent
evaluation episodes, versus 7.0%/2.5%/2.5% for the initial Actors. The latent
SAFE pre-study is **undetermined**: the final policies produce almost no
natural handoff failures, and the natural failures that exist in seed 11 have
no post-failure decision opportunity before the teammate's next relevant
decision; tool-wait failures occur in every episode, leaving no event-free
normal episodes for the pre-registered 5% false-alarm calibration. This is a
data-regime result, not evidence that the Actor latent is universally
uninformative. Details and exact numbers are in
`/tmp/flex-warehouse-safe-20261118-95e0c41a.md`.

## Locked protocol

- Upstream `yuchen-x/MacroMARL` commit
  `f5899cd7b96bb3dbf7b265494a3ba8a15b2e3fe2`; tarball SHA-256
  `9d1e71a220d293202f23d8528a8555d64f754ef6e8c3885b61e9c98b64873d11`.
- Official `experiments/Warehouse/mac_iaicc.sh` first block, original
  hyperparameters. Explicit training seeds 11/22/33; separate evaluation
  stream (base 7100 + seed); no exploration, no online update.
- Training launcher/checkpoint contract: `run_train.py`, `run_seeds.sh`,
  `configs/warehouse_a_maciaicc.json`; formal launch at clean commit
  `ad4f5fc20ab6b5696b220c8b6ae28cdaa41460f1`, `flex_dirty=false`.
- Evaluation: `eval_policy.py`; probe: `probe_latent.py`;
  aggregation: `aggregate_eval.py`, `summarize_probe.py`.

## Formal training

| seed | start | end | wall | final eval return | final checkpoint |
|---|---|---:|---:|---:|---|
| 11 | 10:40:09 | 11:28:26 | 2895.7 s | 533.0 | `epi_040000` |
| 22 | 11:28:26 | 12:21:30 | 3182.4 s | 507.0 | `epi_040000` |
| 33 | 12:21:30 | 13:17:01 | 3329.6 s | 501.7 | `epi_040000` |

Total formal training: 120,000 episodes, 2 h 37 min. Every 2,000-episode
milestone has a manifest with parent, upstream SHA, FLEX SHA, config SHA-256,
per-file SHA-256, episode count and wall time. Final actor SHA-256:

- seed 11 `11_agent_01.tar`: `cdded3eb5faa543143715aba0ec4a29991e728852a4b5aacd364c7d6b58ed323`
- seed 22 `22_agent_01.tar`: `a48e0273254e35cdc66010b3ff0456b656dcf67a8fb4d8cc314d9e1f6eaf72d8`
- seed 33 `33_agent_01.tar`: `3e7ff96d015964646d4d10d9e2af77733adea7c924a187a879b7da80fd1deafc`

Full mapping: `runs/seed_<seed>/checkpoint_index.jsonl` and each
`checkpoints/epi_*/manifest.json`.

## Fixed-Actor task evaluation

Initial and final Actors, 200 episodes each, same eval seeds per seed,
decentralized, `eps=0`, `test_mode=True`. Hook-on/off paired subset of 20
episodes is exactly equal in actions, actual macro-actions, success, steps,
return and event counts for all three seeds.

| seed | initial success | final success | final mean steps | paired success diff [95% CI] |
|---|---:|---:|---:|---:|
| 11 | 0.070 | 1.000 | 67.2 | +0.930 [0.895, 0.965] |
| 22 | 0.025 | 1.000 | 73.0 | +0.975 [0.950, 0.995] |
| 33 | 0.025 | 1.000 | 78.4 | +0.975 [0.950, 0.995] |

Paired return differences are positive with intervals above zero; see
`runs/eval/paired_all.json`. The paper reports discounted-return curves
(20 trials, 10 evaluation episodes every 100 training episodes), not a
success/completion-time table; this is task-level behavioral reproduction,
not numerical reproduction.

## Natural-event diagnostics and probe

Additional natural trajectories for the final Actors (not the 200-episode task
evaluation):

| seed | episodes | success | handoff failures | handoff reachable | tool timeouts | event-free normal episodes |
|---|---:|---:|---:|---:|---:|---|
| 11 | 3000 | 3000 | 101 | 0 | 6104 | handoff 2899; tool 0 |
| 22 | 3000 | 2999 | 1 | 1 (window 4) | 9025 | handoff 2999; tool 0/17 by agent |
| 33 | 3000 | 3000 | 0 | 0 | 3003 | handoff 3000; tool 0/3000 by agent |

Pre-registered probe status (3,000-episode natural set, 500/500/2,000
train/cal/test):

- handoff: no positive post-failure samples (seed 11 all windows empty;
  seed 22 one event in test, training positives zero).
- tool wait: no usable normal-episode calibration (0 or 12 normal test
  episodes; agent 0 has no normal samples at all). Exploratory normal-window
  probes are reported separately and do not satisfy the protocol.

## Data paths on amgg

```
/data/amgg_flex_demo/flex-warehouse-safe-20261118-95e0c41a/
  runs/seed_{11,22,33}/              training state, manifests, checkpoints
  runs/eval/seed{11,22,33}_{initial,final}/  episode JSON/NPZ, summaries
  runs/eval/paired_all.json
  runs/probe_eval/seed{11,22,33}_final_3000/ natural probe trajectories
  runs/probe/seed{11,22}_final_3000_{episode,window}/  probe outputs
  logs/                               setup, calibration, resume, training, eval, probe
```

No weights, trajectories or raw logs are in Git; all remain on `amgg`.

## Reproduction commands

```bash
D=/data/amgg_flex_demo/flex-warehouse-safe-20261118-95e0c41a
cd "$D/FLEX/experiments/warehouse_safe"
bash run_seeds.sh

cd "$D/FLEX_eval/experiments/warehouse_safe"
bash run_eval.sh
EVAL_ROOT="$D/runs/probe_eval" EVAL_SUFFIX=_3000 TRAIN_COUNT=500 CAL_COUNT=500 \
  bash run_probe.sh
```

The eval worktree is pinned to the eval code commit; training used the
separate clean `FLEX` checkout at `ad4f5fc`. See the task brief at
`/tmp/flex-warehouse-safe-20261118-95e0c41a.md` for the full mapping and
caveats.
