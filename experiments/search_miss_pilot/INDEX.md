# Search-miss pilot: index

Task: `flex-search-miss-pilot-73bd9a4c`
Date: 2026-11-20
Host: `amgg` only, single-CPU Python. No training, SAFE/FLEX, Isaac, or
Transformer migration. Development gate only; no 72-group holdout.

## Verdict

**Development gate fails: the strong normal coverage/visibility summary already
selects the same plan as A's failure experience. Close the candidate; do not
run the holdout.**

A's fast boundary scan completes normally but misses an occluded target; C's
legal local verification confirms the miss, and A's slow segmented plan finds
it. This is a genuine task-level miss, not a navigation/execution failure.

B searches a different chamber at its own node. Its normal control has the
map, fast coverage/visibility map, uncovered cells, C status and the ordinary
miss notice. A strong coverage rule chooses slow exactly when the chamber has
an uncovered/occluded region. The experience arm has A's
Condition/Attempt/Outcome; in every development pair it makes the same choice.

## Development gate (16 scenarios)

8 relevant (hidden target) + 8 non-applicable (open target); only A's plan is
changed within a seed. A fast misses 8/8 relevant and finds 8/8 nonapp; A slow
finds all 16. B's normal/experience choices:

| Group | normal | experience | B found | C occupancy |
| --- | --- | --- | ---: | ---: |
| relevant (8) | slow 8/8 | slow 8/8 | 8/8 | 8/8 |
| non-applicable (8) | fast 8/8 | fast 8/8 | 8/8 | 0/0 |
| non-applicable permuted | slow 8/8 | — | 8/8 | 8/8 |

The normal strong coverage rule uses `b_uncovered` directly; the experience
summary adds no decision or completion benefit. The permuted control only
adds unnecessary C use in open chambers. Therefore there is no credible pair
with identical normal information, different legitimate experience and
opposite B choices; the task stop rule is met.

## Limits

This is a minimal abstract grid/visibility task, not a robot controller or a
real underground rescue simulation. It shows only that the normal
coverage/visibility summary is sufficient in this prototype; it does not prove
that no other search-miss experience could ever be useful.

## Anchors and data

- Code commit: `712ed34`; script sha256:
  `352c3ec0d86c22336cb4945bf6c127b514fa69769084c2c11fc3292723c9c462`;
  `dev.json` sha256:
  `27e70d8bebfc22249ec0ca11c932a3e1468f3145de2ce5d41838a9f267e2fef5`.
- Python 3.12.3 on `amgg-Super-Server`, single CPU, no external packages;
  no training checkpoint.
- Raw data: `/data/amgg_flex_demo/flex-search-miss-pilot-73bd9a4c/`
  (`results/dev.json`, `dev.err`, code under `.../code/`).
- Unit tests on `amgg`: 4/4 passed.
- Local report: `/tmp/flex-search-miss-pilot-73bd9a4c.md`.
- Wall time under 10 minutes, below the two-hour cap.

## Recommendation

Close this search-miss candidate and keep the flood prototype as the control.
Do not run the 72-group holdout, train an Actor, or connect SAFE/FLEX. A future
reopen would need a legitimate case where the normal coverage/visibility
summary cannot recover the same choice, without hiding fields or inventing a
communication constraint.
