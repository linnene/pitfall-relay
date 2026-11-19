# Flood recheck: index

Task: `flex-flood-recheck-39c6d8a1`
Date: 2026-11-19
Host: `amgg` only, single-CPU Python job. No Isaac, SAFE/FLEX, Actor training,
or system packages.

## Verdict

**Undetermined: the formal denominator is insufficient; stop under the
protocol.** The numerical direction supports extra value from execution
context over sensor history, but only 46/48 relevant groups and 47/48
non-applicable groups were valid. Three generation cases were excluded
because A did not fail; they are neither arm outcomes nor imputable losses.
The task requires 48 valid groups per side and says to report undetermined
rather than refill or retune. No RL interface and no training follow.

## What differs from the previous pilot

The previous "full history" arm relied on A's raw pre-pump water slope, so it
could not separate ordinary sensor sharing from failure-execution context.
This recheck has four categories and four arms:

- `relevant_pump`: shared inflow transfers to B; A's pump masks its raw slope.
- `relevant_delay`: transferable with different delay/pump/transport/ETA.
- `nonapp_local`: both A and B water rise, but A's rise is mostly local.
- `nonapp_handled`: both rise, but B is already handled; isolation is
  unnecessary.

Arms: `state`, `notice`, `sensor` (A raw samples + standard pump state),
`exec` (sensor + A order, execution interval and outcome). The same decision
framework and input masking are used; `exec` may correct the raw slope with
the observable pump interval, but receives no truth, category label or
suggested action. The consistency guard uses B's own rising trend and a
ratio cap, not "B curve flat".

## Holdout results (valid rows)

Actual A/B/team counts, not just B:

- Valid groups: 93/96. A preserved: 0/93 (A fails by design).
- All-preserved rate (A+B): 0/93, so prior "team preserved" was a B count,
  not a joint A+B rate.
- Relevant B losses (46 valid): state 17, notice 15, sensor 16, exec 3.
- Relevant team counts (A+B): state 29, notice 31, sensor 30, exec 43.
- Relevant wrong/unnecessary isolation: state 0, notice 0, sensor 0, exec 3.
- Non-applicable B losses (47 valid): all arms 0.
- Non-applicable team: all arms 47; wrong isolation: all 0.

`exec - sensor` over 46 valid relevant groups: -13 B losses, mean -0.283,
paired bootstrap 95% CI [-0.413, -0.152]. Both relevant subclasses are
non-negative: pump 0 vs 4, delay 3 vs 12. Numerically this passes the
"at least 8 fewer, CI excludes zero, team not lower" rule, but the required
48-group denominator is not met. The frozen script's
`checks.thresholds_met` does not enforce 48/48; its `true` is not a formal
pass.

Protocol exclusions: 3 generation errors (3.1%): relevant_pump seeds 7017
and 7022, nonapp_handled seed 7304; all are "A did not fail". A-unconfirmed
0/93, message-late 0/93, no-executable-choice 0/93. Attributable subset:
46 relevant valid groups; exec 3 losses, sensor 16.

Costs: exec chose isolation 27 vs sensor 11 in relevant groups; mean
completion 10.54–11.23 min (slightly above baseline), with 3 unnecessary
isolations but no B loss. Worst case: 3 exec losses remain in
`relevant_delay`; they are not explained by message delay or invalid A.

## Why undetermined rather than pass

The task is explicit: primary comparison is over 48 relevant scenarios and
non-applicable over 48; if the denominator is insufficient, protocol error
or independent samples are missing, report undetermined and stop. The 46/48
valid count is not a result to be repaired by re-running the same holdout,
adding seeds, or treating the three A-did-not-fail cases as losses.

## Anchor and data

- Reused mechanics: `experiments/flood_pilot/flood_pilot.py` at `5c4949c`.
- Frozen/holdout execution SHA: `e1c6fac`.
- Script sha256:
  `386df3d9301bfb87a814dd86ee3f84563cddafc00a094c23baf0af8b41b915c0`;
  base `flood_pilot.py` sha256:
  `0a90fd5c25a6ce7d4037a8b5decc4196dc3a200bd08f83ec1de645adb2b0e104`.
- Python 3.12.3 on `amgg-Super-Server`, single CPU process, no external
  packages, no training checkpoint.
- Development seeds: 4/category, 16 total. Holdout: 24/category:
  relevant_pump 7001–7024, relevant_delay 7101–7124,
  nonapp_local 7201–7224, nonapp_handled 7301–7324.
- Raw data:
  `/data/amgg_flex_demo/flex-flood-recheck-39c6d8a1/results/`
  (`anchor.json`, `holdout.json`, `holdout.time`, `devA/devB.json`), code
  under `.../code/`.
- Unit tests on `amgg`: 3/3 passed; two dev runs had identical sha256.
- Wall time about one hour, under the two-hour cap. Local report:
  `/tmp/flex-flood-recheck-39c6d8a1.md`.

## Next step (not started)

If reopened, fix only the A-failure generation so every category has 24
confirmed-failure groups; keep the four arms, parameters, rules and seed
segments otherwise fixed, then run a fresh independent holdout. Do not use
this round's valid subset as a pass, and do not add training or FLEX.
