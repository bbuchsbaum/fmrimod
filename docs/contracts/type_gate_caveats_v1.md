# Type-Gate Caveats Index

This index lists active **type-governance** caveats: documented,
*temporary* divergences between the authoritative full-strict mypy gate
(decision [`0002`](decisions/0002-full-strict-is-the-authoritative-type-gate.md))
and the current state of the tree, each with an owning bead and a
concrete exit criterion.

This is **not** `docs/contracts/CAVEATS.md`. That file is the *Parity
Caveats Index* and `cross_testing/test_caveats_index.py` cross-checks
its rows against parity-report `caveat_id`s. Type-gate caveats are a
separate concern with a separate lifecycle and deliberately do not
share that index. A type-gate caveat is not a tolerance escape hatch:
it names the affected files, the reason for the divergence, an owning
work item, and the condition under which the row is deleted.

A caveat that outlives its exit criterion is a bug. When an owner bead
closes, the corresponding rows here are removed in the same commit, and
`python3.9 -m mypy fmrimod/` must be correspondingly closer to green.

| Caveat ID | Affected (stable tree) | Owner | Exit criterion |
| --- | --- | --- | --- |
| `scoped-strict-seam-residue` | `design/event_model.py`, `hrf_dispatch.py`, `hrf_integration.py`, and the scattered `[misc]` subclass-base / scoped-required-cast sites in `events/{term,variable,matrix,factor,basis}.py`, `covariate.py`, `group/{dataset,ops}.py`, `utils/misc.py`, `regressor/core.py`, `dispatch.py`, `glm/fmri_lm.py` | `bd-01KRRM1DB1CP8E5CEA2Q1Z49Z5` | The EventModel/HRFProtocol typed protocol-seam is resolved so that, under full-strict, the `[misc]` unused-ignore and redundant-cast sites are removed (not suppressed) and `event_model.py`/`hrf_dispatch.py` no longer emit the Protocol-vs-abstract artifact. Then these rows are deleted. |
| `covariate-eventtype-taxonomy` | `covariate.py` (`CovariateEvent.event_type` returning `"covariate"` outside `EventType = Literal[...]`) | `bd-01KRRM1DZH80R9SPHDH6QG6ZGR` | A taxonomy decision lands: either `EventType` is extended to admit the covariate case, or `CovariateEvent` is formally not an `EventType` member and callers are adjusted. Then this row is deleted. |

## Scope notes

- **In-flight modules are out of scope.** `spec/_compile.py` and
  `dataset/adapters/neuroim_adapter.py` are explicitly in-flight
  ("moving targets") per the `AGENTS.md` Module Map. Full-strict
  errors there are tracked with the in-flight work, not by this
  caveat, and do not block the owner beads above.
- **Count is soft.** `python3.9 -m mypy fmrimod/` reports a
  working-tree-dependent count (~100–130 errors); the snapshot in
  `scoped_vs_strict_gap_v1.md` is point-in-time. The decision and the
  exit criteria are count-independent — they are defined by the
  *seam/taxonomy resolution*, not by a number.
- **Regression boundary.** While these rows are open, full-strict
  errors *within* the affected files/clusters are the documented
  caveat. Full-strict errors *outside* them are regressions and must
  be fixed, not added here.
