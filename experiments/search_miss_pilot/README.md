# Search-miss pilot: gate

Task: `flex-search-miss-pilot-73bd9a4c`
Date: 2026-11-20
Host: `amgg` only, single-CPU Python. No training, SAFE/FLEX, Isaac, or
Transformer migration.

## Task

Two search robots A/B and one shared C. A fast west-boundary scan covers the
cells visible from the entry; a slow segmented inspection (or C request)
covers the full chamber. Each chamber has a target behind an occluder or in
open view. A fast scan completes normally, and a legal C/local verification
confirms that an occluded target was missed. A slow plan finds it. B searches
a different chamber at its own node.

Normal control: map, fast coverage/visibility map, uncovered cells, C status,
ordinary miss notice, and a strong coverage rule that uses all of these.
Experience arm: A's Condition/Attempt/Outcome. The permuted arm uses a
sibling chamber's experience.

## Gate result

16 development scenarios: 8 relevant (hidden target) + 8 non-applicable
(visible target). A fast misses in 8/8 relevant and finds in 8/8 nonapp;
A slow finds in all 16. B normal and experience choices are identical in
every pair:

| Group | normal choice | experience choice | B found | C occupancy |
| --- | --- | --- | ---: | ---: |
| relevant (8) | slow 8/8 | slow 8/8 | 8/8 | 8/8 |
| non-applicable (8) | fast 8/8 | fast 8/8 | 8/8 | 0/0 |
| non-applicable permuted | slow 8/8 | — | 8/8 | 8/8 |

The strong coverage rule already sees `b_uncovered` and chooses slow exactly
when the chamber has an occluded region. The experience summary adds no
choice or completion benefit, and the permuted control only adds unnecessary
C use in open chambers. The development gate fails; no 72-group holdout is
run.

## Limits

This is a minimal abstract grid/visibility task, not a robot controller or a
real underground rescue simulation. The negative result means the normal
coverage/visibility summary is sufficient in this prototype; it does not
prove that no other search-miss experience could ever be useful.

## Run

```
python3 search_miss.py --out results/dev.json
python3 -m unittest -v test_search_miss.py
```
