# Freeze

Task `flex-flood-dynamics-final-5a17c9e2`, frozen 2026-11-19 before the
formal holdout.

- Development gate: 16 groups, 8 relevant + 8 non-applicable. No holdout
  seed or outcome inspected before this freeze.
- Holdout seeds: relevant 9001–9024, non-applicable 9101–9124, no-timely
  9201–9224 (24 each, 72 total).
- Native evolution: shared/local inflow, actual pump drainage and area-scoped
  isolation; equipment damage and B transport/isolation outcomes are evolved,
  not tabulated. B node is fixed at t=10, independent of A.
- Three arms: normal telemetry (no fact object), real execution fact, fixed
  cross-scenario permuted fact. Same `decide` function; baseline field
  isolation is recorded via `fact_visible`; no-timely arms receive no fact.
- The execution SHA, script hash, Python version and command are recorded in
  `results/anchor.json` on `amgg` at run time.
- No training checkpoint; no Isaac/SAFE/FLEX; single CPU on `amgg`.
- Holdout is run once; no exclusion, re-draw or threshold change afterward.
