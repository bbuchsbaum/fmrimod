# 0002 — Full-strict mypy is the authoritative type gate

Status: Accepted
Date: 2026-05-16
Invariants touched: MISSION non-negotiable #1 (types end-to-end, no
escape hatches); GOVERNANCE §8 (config-change discipline), §10 (steward
approval)
Owner bead: bd-01KRNN0H73CCYGFJSJ30JPVFTW

## Context

The typed-API epic closed ~38 subpackage beads against a *scoped* gate:
`mypy fmrimod/<pkg>/ --follow-imports=skip`, run per subpackage. But
`pyproject.toml [tool.mypy]` is, and has always been, full-tree
`strict = true` with no `follow_imports` override. The two gates
diverge, and the divergence is not cosmetic:

- `--follow-imports=skip` makes every cross-module symbol opaque
  `Any`. That structurally *masks* real cross-module errors. It did:
  full-strict surfaced a completely dead, untested public surface —
  `fmri_rlm()` raised `TypeError` on every call (fixed in `24b992d`),
  and fixing it unblocked a second crash, runwise robust IRLS
  broadcast (fixed in `ff6baa5`). Neither was reachable by any
  `--follow-imports=skip` check.
- The remaining full-strict residue (~100 errors / ~18–19 files,
  count soft and working-tree dependent) is *not* a mechanical
  backlog. After 104 files of mechanical reduction (746 → ~100,
  scoped gate green throughout — see
  `scoped_vs_strict_gap_v1.md`), the residue is an irreducible
  either/or *created by the gate choice itself*: under scoped the
  opaque-`Any` base **requires** a `# type: ignore`/cast; under
  full-strict the resolved base flags that same line
  unused/redundant. No single annotation satisfies both gates.

Keeping scoped as the permanent gate would institutionalize the exact
import-opacity that hid two crashing public bugs, and would leave the
project with two permanent, mutually-exclusive type contracts. A
`CAVEATS.md` row was considered and rejected for the residue:
`docs/contracts/CAVEATS.md` is the **Parity Caveats Index** and
`cross_testing/test_caveats_index.py` cross-checks its rows against
parity-report `caveat_id`s — a type-gate caveat there would fail that
test or silently broaden a single-purpose contract.

This decision was authored by agents across prior sessions, reviewed
by an independent second actor (`Codex-due-diligence`), and approved
by the steward (the user) per GOVERNANCE §10. The steward did not
auto-approve a same-session self-authored decision.

## Decision

1. **Full-strict is authoritative.** `python3.9 -m mypy fmrimod/`
   under the committed `pyproject.toml [tool.mypy] strict = true`
   (no `follow_imports` override) is the gate for any unqualified
   "typed" claim. Adding `follow_imports = skip` to `pyproject` is a
   GOVERNANCE §8 config weakening and remains disallowed.
2. **Scoped is advisory.** `mypy fmrimod/<pkg>/ --follow-imports=skip`
   remains useful as a local development triage / per-subpackage smoke
   check. It MUST NOT be used to close typed-work beads or to claim a
   module is "typed". Scoped-green ≠ strict-green.
3. **The residue is a temporary, exiting caveat — not a second
   permanent contract.** It is documented in
   [`type_gate_caveats_v1.md`](../type_gate_caveats_v1.md), owned by
   two follow-up beads whose closure is the exit criterion:
   - `bd-01KRRM1DB1CP8E5CEA2Q1Z49Z5` — EventModel/HRFProtocol typed
     protocol-seam (the `[misc]` subclass-base unused-ignore /
     scoped-required-cast cluster + the `event_model.py` /
     `hrf_dispatch.py` Protocol-vs-abstract artifact).
   - `bd-01KRRM1DZH80R9SPHDH6QG6ZGR` — `CovariateEvent.event_type`
     vs the `EventType` `Literal` taxonomy question.
4. **No bulk relitigation.** The ~38 scoped-closed beads are not
   reopened. The full-strict residue is already isolated in
   `scoped_vs_strict_gap_v1.md`; any per-bead follow-up is spawned
   only as that residue inventory requires, under the owner beads
   above.

## Consequences

- "Typed" claims are now falsifiable by one command
  (`python3.9 -m mypy fmrimod/`), not by a curated per-subpackage
  procedure that masked dead code.
- Until the two owner beads close, `mypy fmrimod/` is expected to
  report the documented residue. That is the caveat's *open* state,
  not a regression; `type_gate_caveats_v1.md` carries the affected
  files and the exit criterion. New full-strict errors *outside* the
  documented residue ARE regressions.
- The scoped/strict casts and ignores in the residue cluster flip
  from required to redundant once the EventModel/HRFProtocol seam is
  resolved; they are removed under the owner bead, not papered here.
- CI enforcement of full-strict is wired when a runner is available
  (tracked, mirroring the performance-regression "tracked not gated"
  posture); the contractual authority does not wait on CI.
