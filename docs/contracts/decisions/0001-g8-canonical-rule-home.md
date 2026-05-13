# G8 — One canonical home per operational rule

Status: Accepted
Date: 2026-05-13
Invariants touched: G8 (introduced)

## Context

Three governance-coherence bugs of the same shape appeared in a single
working session:

- `Predicate` defined twice with incompatible callable contracts
  (`fmrimod/spec/terms.py:40` vs `fmrimod/contrast/contrast_spec.py:29`).
- `ContrastResult` as a soft-tagged union read in 7+ different ways
  across `combine.py` / `compat.py` / `bids/export.py` / `accessors.py`
  / `group/dataset.py` / parity workflows.
- `AGENTS.md` Session Protocol §4 instructing agents to commit `.mote/`
  ops files while `.gitignore:35` excluded the entire `.mote/` tree.

In each case the underlying failure is identical: one value has two
sources of truth in different documents or modules, and agents (human
or otherwise) silently follow whichever they read first. The
board-to-bead conversion rule itself nearly suffered the same failure
mode — ratified through `mote discuss` history and referenced in
`AGENTS.md` § Coordination and code review + § Work requests, but the
canonical text never landed in `message_board.md` until governance
audit caught it.

These bugs are not random. They are the natural consequence of
recording cross-cutting operational rules in multiple documents
without a discipline that one location is canonical and the rest are
pointers.

## Decision

Add invariant **G8 — One canonical home per operational rule** to
[`GOVERNANCE.md`](../../../GOVERNANCE.md) § 2:

> Every cross-cutting operational rule lives in exactly one document.
> Other documents link *to* that home; they do not restate the rule's
> body. When two documents say things about the same value, exactly
> one is canonical and the other is a one-line pointer.

Falsifier: `grep -rn '<rule-text-fragment>' *.md docs/contracts/`
returns exactly one substantive hit (the canonical home) plus any
number of pointers. Multiple substantive hits is a violation.

Canonical homes registered at decision time:

| Rule | Canonical home |
| --- | --- |
| Board-to-bead conversion | [`message_board.md`](../../../message_board.md) § Board-to-bead Conversion |
| Code-review routing | [`AGENTS.md`](../../../AGENTS.md) § Coordination and code review |
| Work-request tiers and approval | [`AGENTS.md`](../../../AGENTS.md) § Work requests |
| `.mote` tracking policy | [`AGENTS.md`](../../../AGENTS.md) § Session Protocol § 4 |
| Parity divergences | [`docs/contracts/CAVEATS.md`](../CAVEATS.md) |
| Cross-module invariants (G1–G8) | [`GOVERNANCE.md`](../../../GOVERNANCE.md) § 2 |
| Public-API tier table | [`docs/contracts/public_api_policy_v1.md`](../public_api_policy_v1.md) |

Future operational rules added to the project must declare their
canonical home in the commit that introduces them. The decision-note
template (§ 8) should be amended to require a "Canonical home" line
when the decision introduces a new operational rule.

## Consequences

**Enables:**

- A single `grep` becomes the audit for rule duplication.
- New agents joining cold can find each rule's authoritative text in
  exactly one place.
- The "did the docs really land?" failure mode of the board-to-bead
  ratification is now structurally impossible: the canonical home is
  the only thing that exists.

**Costs:**

- One-line pointers in non-canonical documents are slightly less
  self-contained than restatement. Readers may follow links instead of
  reading inline. Net positive against the cost of contradictory
  restatements that drift independently.
- Pre-existing duplication must be cleaned up incrementally; this
  decision does not retroactively fix `Predicate` or `ContrastResult`.
  Those remain as named bead candidates under existing epics.

**Exit criterion:**

This decision is provisional only insofar as the canonical-home
registry above must grow as new operational rules land. The G8
invariant itself is permanent unless explicitly superseded by a later
decision note; G8 is recorded as Accepted, not Provisional.
