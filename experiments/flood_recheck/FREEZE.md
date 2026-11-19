# Freeze

Task `flex-flood-recheck-39c6d8a1`, frozen 2026-11-19 before the formal
holdout.

- Reused mechanics: `experiments/flood_pilot/flood_pilot.py` at commit
  `5c4949c` (water conservation, macro actions, damage endpoint).
- Development seeds: 4 per category, 16 total:
  `relevant_pump` 5001–5004, `relevant_delay` 5101–5104,
  `nonapp_local` 5201–5204, `nonapp_handled` 5301–5304.
- Holdout seeds: 24 per category, 96 total:
  `relevant_pump` 7001–7024, `relevant_delay` 7101–7124,
  `nonapp_local` 7201–7224, `nonapp_handled` 7301–7324.
- Parameters, category generators, four arm masks and the decision threshold
  are exactly those in `flood_recheck.py` at the commit immediately preceding
  the holdout. No holdout seed or outcome was inspected before this freeze.
- Runtime: Python 3.12.3 on `amgg-Super-Server`, one CPU process, no external
  packages; no training checkpoint; no Isaac/SAFE/FLEX.
- The holdout is run once. The execution SHA, script hash, environment and
  seed segments are recorded in `results/anchor.json` on `amgg`.
