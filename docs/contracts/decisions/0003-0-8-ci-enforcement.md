# 0003 - 0.8 CI enforcement gate

Status: Accepted
Date: 2026-05-16
Invariants touched: MISSION non-negotiable #1; GOVERNANCE G5, G7, section 10
Owner bead: bd-01KRRT0ZEM0K909QK4HZCHJ8XH

## Context

The 0.8 release contract requires enforceable evidence, not just local
commands. The release surface is narrower than 1.0: typed first-level modeling,
`covariate(...)` as the identity HRF species, seed-target connectivity helpers,
and caveat discipline.

Decision 0002 reserved CI enforcement of full-strict typing for the moment a
runner is available, while also making clear that type authority does not wait
on CI. The current 0.8 problem is different: PRs to main need a Python floor
and regression gate for tests and lint.

The steward explicitly delegated policy authority for the 0.8 goal on
2026-05-16. Under that delegation, this note resolves the CI policy for 0.8.

There is one important constraint: repo-wide ruff is not yet a clean gate.
At this decision point, `ruff check` reports broad pre-existing debt across
benchmarks, cross-testing, package code, and tests. Making repo-wide ruff
blocking would turn the 0.8 release into a lint-retirement project.

## Decision

1. Add an enforcing GitHub Actions workflow for the 0.8 release gate on PRs to
   `main` and pushes to `main`.
2. Run the documented Python suite on both Python 3.10 and 3.11:

   ```bash
   python -m pytest tests/ -k "not rpy2"
   ```

   Python 3.10 is the floor guard. Python 3.11 matches the current CI norm.
3. Run caveat/proof-artifact discipline checks in the same workflow:

   ```bash
   python -m pytest \
     cross_testing/test_caveats_index.py \
     tests/test_benchmarks/test_parity_proof_artifacts.py -q
   ```

4. Enforce ruff as a scoped 0.8 ratchet, not a repo-wide gate. The scoped gate
   covers the release-surface files that are clean under `F`, `E9`, `I`, and
   `E501`. This catches syntax/import/format drift around the 0.8 seam without
   laundering the existing repo-wide lint backlog.
5. Move the ruff configuration to the current `[tool.ruff.lint]` schema while
   preserving the existing selected and ignored rule sets. This removes the
   deprecated-config warning without changing the intended lint contract.
6. Keep one authoritative 0.8 CI workflow: `Release 0.8 Gate`
   (`.github/workflows/release-0-8.yml`). Do not maintain a second always-on
   full-suite workflow with a different Python matrix or ruff ratchet unless a
   later decision supersedes this one.

## Consequences

- 0.8 gains a real PR gate for the first-level seam and Python floor.
- Repo-wide ruff remains explicitly unclaimed for 0.8. The exit criterion for
  the scoped ratchet is a later lint-retirement bead that makes `ruff check`
  clean across the repository and then replaces the scoped command.
- Duplicate CI workflows are treated as policy drift for 0.8 if their Python
  matrix, install route, or ruff scope differs from this decision.
- This decision does not supersede decision 0002. Full-strict mypy remains the
  authoritative type gate; CI wiring for full-strict follows 0002 and the
  remaining typed-seam owner beads.
