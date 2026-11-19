# Flood recovery: index

Task: `flex-flood-recover-7c5e29d4`
Date: 2026-11-19
Host: `amgg` only, single CPU, read-only original holdout. No training, no
SAFE/FLEX, no new seeds or dynamics.

## Verdict

**Strict recovery is impossible: there is no frozen B decision clock
independent of A failure. Formal status remains undetermined; do not enter
the high-level RL interface.**

The three original seeds were replayed with the frozen `e1c6fac` code. A was
preserved in all three (`relevant_pump` 7017 at t=9, 7022 at t=8;
`nonapp_handled` 7304 at t=8). The frozen `one()` sets `t0 = A loss time` and
raises when A does not fail, so there is no B decision time to use. No new
clock was selected, no B outcome was fabricated, and the 93-group holdout
was not rerun.

## Three-seed audit

| Category | Seed | A result | A preserved at | Input hash prefix |
| --- | ---: | --- | ---: | --- |
| relevant_pump | 7017 | preserved, no failure | 9 | `e3fa0afe55713d66` |
| relevant_pump | 7022 | preserved, no failure | 8 | `60272c3ca6d21144` |
| nonapp_handled | 7304 | preserved, no failure | 8 | `1daaa3124fa6a777` |

A equipment: 3/3 preserved. Because B has no decision clock, B counts, A+B
counts and the all-preserved rate are **missing**, not zero and not 3/3.

The only provable no-message range is conditional: when A does not fail, no
failure notice, sensor packet or execution context is emitted, so all four
arms would receive the same normal-state inputs and any paired difference
would be zero if a common fallback existed. The frozen code has no such
fallback and no actual arm run; absolute B preservation remains missing.

## Arm-input audit (read-only, no new arms)

The frozen `_choose` receives the same `a_ctx` for every arm; that context
already contains `exec_q`, `pump_work`, `order` and `outcome`. The `sensor`
arm has A's `pump_from`/`pump_work` and could recompute `exec_q` from the
same legal sensor fields. The `exec` arm's only difference is
`exec_q = raw_q + 0.25 * pump_rate` when the pump was active; `order` and
`outcome` are never read by `_choose`.

Therefore there is no exclusive execution-history field. The previous
numerical difference cannot be attributed to independent execution history;
it is a different inflow estimate using sensor-available pump timing. Input
masking is not enforced. This is a protocol inconsistency, so the task cannot
enter RL and no new arm is run to repair it.

## Merge status and original subset

- Original holdout JSON remains read-only:
  sha256 `8d7156f8f36b779dd3ce0e0d4ae0f3c2f204de342a0a9bab5408e23c96d65254`.
- No merged 96-group result. Original valid subset: 93 groups; A preserved
  0/93; relevant B losses state/notice/sensor/exec = 17/15/16/3 over 46
  groups; non-applicable 47 groups all arms 0 losses; all-preserved 0/93.
- Original protocol: 3/96 (3.1%) A-did-not-fail; A-unconfirmed 0/93,
  message-late 0/93, no-executable-choice 0/93.
- Recovery groups: B counts and error-isolation counts missing; no denominator
  can be completed. Formal status undetermined.

## Anchors and data

- Frozen recheck code: `e1c6fac`; audit commit: `13ff670`.
- Audit sha256:
  `136a0543358ff477a74fe425d7d4fd633f0424df31a2d296a02737d2d3f41daa`;
  frozen `flood_recheck.py` sha256:
  `386df3d9301bfb87a814dd86ee3f84563cddafc00a094c23baf0af8b41b915c0`.
- Python 3.12.3 on `amgg-Super-Server`, one CPU process, no external
  packages; no training checkpoint.
- Remote data:
  `/data/amgg_flex_demo/flex-flood-recover-7c5e29d4/`
  (`anchor.json`, `recover_audit.json`, `recover_audit.err`); original
  holdout under
  `/data/amgg_flex_demo/flex-flood-recheck-39c6d8a1/results/holdout.json`.
- Local report: `/tmp/flex-flood-recover-7c5e29d4.md`.
- Wall time about 10 minutes; automatic return was not verified, so no
  resend and no plugin change.

## Next step (not started)

If reopened, define and freeze a B decision clock independent of A's result,
and implement genuinely disjoint arm input masks; then prove which execution
field is exclusive and its effect. Do not treat the three preserved-A cases
as losses or successes, and do not merge them into the old 96 without a
valid clock.
