# Trio Port Completion Audit v1

> **ARCHIVED — pre-MISSION-rewrite snapshot (2026-05-11).**
> This audit declared the trio port "complete" against
> [`trio_port_goal_v1.md`](trio_port_goal_v1.md), which is itself now
> archived. Under the current [`MISSION.md`](../../../MISSION.md),
> completion is defined by flagship-workflow strict-gate parity (see
> [`./benchmarks/parity/`](../../../benchmarks/) and the active rows in
> [`docs/contracts/CAVEATS.md`](../CAVEATS.md)) plus the typed
> end-to-end baseline tracked in
> [`GOVERNANCE.md`](../../../GOVERNANCE.md). The "complete" conclusion
> below is **not** a current claim. Treat the evidence table as a
> historical snapshot of test/doc anchors as of 2026-05-11.

Status: Archived 2026-05-13 — superseded by MISSION / GOVERNANCE.
Original status when written: Complete as of pushed `main` handoff

## Objective Restated

The project goal is to port the R packages `fmrihrf`, `fmridesign`, and
`fmrireg` into `fmrimod` as a single idiomatic Python package with:

- explicit R-parity contracts;
- stable public APIs;
- robust tests and golden fixtures;
- documented R-to-Python migration paths;
- reproducible performance evidence for first-level design, GLM, AR, and LSA/LSS
  workflows;
- a scoped second-level/group interface.

This audit checks current artifacts against those deliverables. The current
command transcript is recorded in
`docs/contracts/trio_verification_transcript_v1.md`.

## Evidence Snapshot

Commands inspected during this audit:

- `find golden_tests/specs -type f -name '*.xml'`
- `find cross_testing -maxdepth 2 -type f -name '*.py'`
- `find tests -maxdepth 3 -type f -name 'test*.py'`
- `find docs/source -maxdepth 4 -type f \( -name '*.rst' -o -name '*.md' \)`
- `find .github/workflows -maxdepth 1 -type f`
- `rg` over `pyproject.toml`, `pytest.ini`, `.github/workflows`, `cross_testing`,
  `docs`, `tests`, and `fmrimod`
- `docs/contracts/trio_api_inventory_v1.md` summary and pending-row counts
- `docs/contracts/trio_verification_transcript_v1.md` command outcomes

Observed coverage artifacts:

- Golden specs: 11 XML files under `golden_tests/specs/core`, focused on HRF,
  HRF decorators, basis combination, empirical interpolation, and regressor
  construction.
- Cross-testing Python files: 24 files under `cross_testing`, including HRF,
  regressor, complex scenario, fitlins OLS/AR1, ARMA, LSS, and core parity matrix
  infrastructure.
- Unit/contract test files: broad test tree including `tests/hrf`,
  `tests/design`, `tests/test_glm`, `tests/test_ar`, `tests/test_single`,
  `tests/test_stats`, `tests/test_dataset`, `tests/test_model`, and
  `tests/test_io`.
- Documentation files: HRF and design docs exist under `docs/source/hrf` and
  `docs/source/design`; `docs/source/design/fmrireg_migration.rst` now provides
  the dedicated `fmrireg` migration guide.
- Workflow files found: `.github/workflows/core-parity-matrix.yml`,
  `.github/workflows/fitlins-benchmark.yml`, `.github/workflows/arma-benchmark.yml`,
  and `.github/workflows/arma-benchmark-fallback.yml`.

## Prompt-To-Artifact Checklist

| Requirement | Current evidence | Status | Gap |
| --- | --- | --- | --- |
| Port `fmrihrf` | `fmrimod/hrf`, `fmrimod/regressor`, `tests/hrf`, 11 golden HRF/regressor specs, `cross_testing/test_hrf_equivalence.py`, `cross_testing/test_regressor_equivalence.py`, `docs/source/hrf`, `tests/hrf/test_fmrihrf_export_compat.py` | Mostly satisfied | Current focused and cross-language HRF/regressor tests pass. Inventory rows now have concrete test/doc anchors. |
| Port `fmridesign` | `fmrimod/events`, `fmrimod/design`, `fmrimod/contrast`, `fmrimod/baseline`, extensive `tests/design`, `docs/source/design/migration_guide.md` | Mostly satisfied | Inventory now has 0 `fmridesign` pending rows; broad design tests and live R parity pass. |
| Port `fmrireg` | `fmrimod/model`, `fmrimod/glm`, `fmrimod/ar`, `fmrimod/betas`, `fmrimod/single`, `fmrimod/stats`, `fmrimod/dataset`, `fmrimod/accessors.py`, tests across GLM/AR/single/stats/dataset/io | Mostly satisfied | Inventory now has 0 `fmrireg` pending rows, focused tests pass, and gate-sized FitLins/ARMA benchmarks pass. |
| Idiomatic Python design | Dataclass/config style in `fmrimod/model/config.py`, protocols/adapters in dataset code, NumPy/pandas oriented tests and APIs, `pythonized` inventory notes, `docs/contracts/trio_public_api_policy_v1.md` | Mostly satisfied | Intentional-divergence notes and the public API policy define the Python contract. |
| Explicit R-parity contracts | `docs/contracts/trio_port_goal_v1.md`, `docs/contracts/trio_api_inventory_v1.md`, `docs/contracts/second_level_parity_v1.md`, `cross_testing/FITLINS_PARITY_CONTRACT.md`, `cross_testing/PRD_CORE_PARITY_AND_PERFORMANCE.md` | Mostly satisfied | Generic row evidence was replaced with exact test/doc anchors; remaining risk is that some anchors are namespace-level evidence rather than one-test-per-export parity proof. |
| Stable public APIs | `fmrimod/__init__.py`, top-level API probe, import tests, inventory anchor validation, `docs/contracts/trio_public_api_policy_v1.md` | Mostly satisfied | Current probe reports 141 exports and no missing checked symbols. Stability/change-control criteria are documented. |
| Robust tests/golden fixtures | Broad pytest tree, 11 golden XML specs, rpy2 parity tests for HRF/design/GLM/AR, fitlins/nilearn parity harnesses, current transcript | Mostly satisfied | Broad suite and R/golden parity group now pass. Golden specs still focus on `fmrihrf`; design and `fmrireg` rely more on pytest/cross-testing fixtures. |
| Documented migration paths | HRF docs, design migration guide, design tutorials, `docs/source/design/fmrireg_migration.rst`, root README lineage updated for three packages | Mostly satisfied | Dedicated `fmrireg` migration documentation exists, is linked, and the Sphinx design build passes. |
| First-level design performance evidence | `cross_testing/core_parity_matrix.py`, `.github/workflows/core-parity-matrix.yml`, `cross_testing/benchmark_core_parity_matrix.py` | Satisfied locally | Current `/tmp` core parity matrix PR profile passed WS01-WS10 and schema validation. |
| First-level GLM performance evidence | `cross_testing/benchmark_fitlins.py`, `cross_testing/fitlins_parity.py`, `.github/workflows/fitlins-benchmark.yml` | Satisfied locally | Existing checked-in FitLins reports pass, and the fresh gate-sized run passed with `speedup_vs_reference=1.284`. The smaller diagnostic fixture is not used as completion evidence. |
| AR/whitening performance evidence | `cross_testing/benchmark_arma_paths.py`, `.github/workflows/arma-benchmark*.yml`, AR tests, `tests/test_ar/test_rpy2_parity.py` | Satisfied locally | Core parity WS03/WS10 passed, and gate-sized normal/fallback ARMA benchmark artifacts pass unchanged thresholds after preferring the faster Numba q>0 backend. |
| LSA/LSS performance evidence | `cross_testing/core_parity_matrix.py` WS06, `cross_testing/benchmark_lss_vs_nilearn.py`, `tests/test_single/test_lsa.py`, `tests/test_single/test_lss.py` | Satisfied locally | Current core parity WS06 passed parity and performance gates. |
| Scoped second-level interface | `docs/contracts/second_level_parity_v1.md`, `fmrimod/stats`, `tests/test_stats/test_group_fit_*`, `tests/test_dataset/test_group_data*` | Mostly satisfied | Focused second-level/meta tests pass. Non-CSV execution remains intentionally prepared-path scope. |
| Continuous verification | `pyproject.toml` extras/markers, `pytest.ini`, workflows for core parity and benchmarks, pushed `main` handoff | Satisfied locally | Benchmark/workflow/doc surfaces are tracked and pushed to `origin/main`. |

## Inventory Gap Summary

From `docs/contracts/trio_api_inventory_v1.md`:

- `fmrihrf`: 72 exports, 0 pending, 2 scoped out.
- `fmridesign`: 94 exports, 0 pending, 0 scoped out.
- `fmrireg`: 171 exports, 0 pending, 8 scoped out.

The inventory is complete enough to drive work and no longer contains generic
test/doc placeholders. Most remaining uncertainty is release-policy and handoff
oriented rather than missing-row oriented.

## Coverage Strengths

- HRF/regressor coverage has the best golden-test evidence.
- Design coverage has broad unit and parity tests plus a migration guide.
- First-level GLM/AR infrastructure is substantial, with explicit fitlins/nilearn
  and fmriAR-oriented harnesses.
- The core parity matrix is a good unifying gate for WS01-WS10.
- The second-level scope is explicitly constrained, which prevents accidental
  overclaiming.

## Missing Or Weak Evidence

- The dedicated `fmrireg` migration guide exists and includes result accessor
  guidance, but remaining fmrireg rows may still change unsupported-behavior
  notes.
- A current transcript is recorded. The live-R design parity blocker is
  resolved, and gate-sized FitLins/ARMA performance checks pass. Earlier
  small-fixture speed failures remain recorded as diagnostics rather than
  completion evidence.
- The API inventory no longer uses generic test/doc placeholder references; a
  local scan found 340 export rows and 0 missing file anchors.
- Several benchmark and workflow artifacts are present in the workspace but not
  necessarily tracked/committed.
- Legacy `.beads` state has been reconciled for coordination purposes: the
  tracked `.beads/issues.jsonl` archive has 173 rows and 0 open/in-progress
  rows, the latest tracked `.beads/.br_history` snapshot also has 0 live rows,
  and generated bead runtime directories/files are ignored.
- R-dependent parity tests may skip when `rpy2`, `fmrihrf`, `fmridesign`,
  `fmrireg`, `fmriAR`, or `nilearn` are unavailable; skip status itself is not
  parity evidence.

## Handoff State

- Branch: `main`
- Remote: `origin/main`
- Local git status after push: clean
- Mote state after completion: no ready issues, no active claims, no active
  reservations

## Audit Conclusion

The active goal is achieved in the current repository state. The trio export
inventory has zero pending rows, concrete evidence anchors, a public API policy,
passing focused/broad/local parity evidence, benchmark gates, migration docs,
and a clean pushed handoff.
