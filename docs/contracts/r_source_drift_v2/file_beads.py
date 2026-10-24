"""File the R-source drift v2 beads in the local mote store.

The v2 audit and wave 1 ran in a cloud session without ``mote`` (see
``docs/contracts/r_source_drift_inventory_v2.md``), so the beads are declared
here with fixed ids and filed in one pass on a machine that has mote::

    python docs/contracts/r_source_drift_v2/file_beads.py --dry-run
    python docs/contracts/r_source_drift_v2/file_beads.py [--actor NAME]

Run it from the repository root. New beads go through ``scripts/mote_new.py``
with ``--from-bead`` provenance and the fixed ``--id`` (the wrapper's import
path), so the inventory can cite the ids before they exist. Re-running is
safe: a bead that ``mote show`` already finds is not created again, closing
is idempotent, and notes are only appended on a bead's first filing.

Wave 1 lane beads are filed closed with the PR #21 commits that fixed them.
Open items from v1 already have beads; they get progress notes instead of
duplicates.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

V1_EPIC = "bd-01M0DQSE2QBF4TZ6G9DKYY8G2W"
V1_WAVE4 = "bd-01M0DQSGMSZWFQVKAJ504F4PQM"
V1_WAVE5_S2 = "bd-01M0DQSH4XRHW0Z4G1BXVT710P"
EPIC = "bd-01M5PS5JM4J8A2JKZX9WHN96QS"

INVENTORY = "docs/contracts/r_source_drift_inventory_v2.md"
DETAIL = "docs/contracts/r_source_drift_v2"
WAVE1 = "PR #21 (branch claude/r-source-drift-v2-wave1)"


@dataclass(frozen=True)
class Bead:
    id: str
    title: str
    priority: int
    body: str
    tags: tuple[str, ...] = ()
    source: str = EPIC
    closed: str | None = None


@dataclass(frozen=True)
class Note:
    bead: str
    text: str
    kind: str = "progress"


def _red(detail: str, rows: str) -> str:
    return (
        f"Red checks and cheap-pass disqualifiers: {DETAIL}/{detail} "
        f'"Red checks", rows {rows}. A row closes when its red check fails '
        "on the base commit and passes after the fix."
    )


def _closed(rows: str, commits: str) -> str:
    return f"Fixed in {WAVE1}, commits {commits}: {rows}."


BEADS: list[Bead] = [
    Bead(
        EPIC,
        "R-source drift v2: keep fmrimod in step with the R sources",
        1,
        f"Inventory: {INVENTORY} (merged in PR #20). Audit of every upstream "
        "change since the v1 pins in fmrihrf, fmridesign, fmrireg, fmriAR, "
        "fmrilss, fmrigds, and fmridataset's first audit, against fmrimod "
        "4f14a19: 215 rows (66 wrong-result, 54 contract, 30 feature).\n\n"
        f"Wave 1 (wrong results) landed in {WAVE1}. Remaining: the R-backed "
        "parity run, steward decisions D1-D10, wave 2 contracts, wave 3 "
        "features, and the v1 wave 4 / S2 items (notes on "
        f"{V1_WAVE4} and {V1_WAVE5_S2}).\n\n"
        "Closes when every child bead is closed or routed to a decision note.",
        ("drift-v2", "epic"),
        source=V1_EPIC,
    ),
    # -- Wave 1, filed closed ------------------------------------------------
    Bead(
        "bd-01M5PS5JM4J8A2JKZX9WHN96QT",
        "Port fmriAR 0.4.x noise-model fixes (drift v2 lane 1a)",
        1,
        "BIC penalty and effective-voxel n, exact segment starts, pooled "
        "ARMA, parcel and global pooling, bias-correction solve, censor "
        "masks. " + _red("fmriar.md", "AR-01 to AR-12"),
        ("drift-v2", "wave1", "fmriAR"),
        closed=_closed(
            "AR-01 to AR-12 and the AR part of REG-A-07", "2cc2f60..6f1bff0"
        ),
    ),
    Bead(
        "bd-01M5PS5JM4J8A2JKZX9WHN96QV",
        "Route fmri_lm AR through fit_noise and whiten_apply (drift v2 lane 1b)",
        1,
        "Bias-corrected phi held fixed for GLS, one global phi, per-voxel "
        "GLS, AR on the preprocessed runs, censoring as gaps. "
        + _red("fmrireg_fit_core.md", "REG-A-02, REG-A-03, REG-A-04, REG-A-12"),
        ("drift-v2", "wave1", "fmrireg"),
        closed=_closed(
            "REG-A-02, REG-A-03, REG-A-04, REG-A-12", "fc83a9b..abd5113"
        ),
    ),
    Bead(
        "bd-01M5PS5JM4J8A2JKZX9WHN96QW",
        "Fix first-level composition and numerics (drift v2 lane 1c)",
        1,
        "Robust IRLS, pivoted-QR solve, scale-free t/F, residual df, "
        "factorial rows, sketch inference, smoothing kernel. "
        + _red("fmrireg_fit_core.md", "REG-A-01, REG-A-05 to REG-A-11"),
        ("drift-v2", "wave1", "fmrireg"),
        closed=_closed(
            "REG-A-01, REG-A-05 to REG-A-11, REG-B-06, REG-B-07",
            "a75b60f..8054393",
        ),
    ),
    Bead(
        "bd-01M5PS5JM4J8A2JKZX9WHN96QX",
        "Fix HRF basis and convolution results (drift v2 lane 1d)",
        1,
        "block_hrf widths, clamped B-spline/tent/FIR bases, weighted HRF, "
        "Mexican hat, conv window edges, per-event span truncation, legacy "
        "SPMG undershoot. "
        + _red("fmrihrf.md", "HRF-01 to HRF-05, HRF-07, HRF-08"),
        ("drift-v2", "wave1", "fmrihrf"),
        closed=_closed(
            "HRF-01 to HRF-05, HRF-07, HRF-08 and the FIR end-of-span defect",
            "dbdabea..71a3e3c",
        ),
    ),
    Bead(
        "bd-01M5PS5JM4J8A2JKZX9WHN96QY",
        "Fix design and contrast placement (drift v2 lane 1e)",
        1,
        "Contrasts by column name, covariate columns, where=, factorial and "
        "poly contrasts, hrf(a, b), trialwise options, plain-function bases, "
        "basis_weights. " + _red("fmridesign.md", "DES-01 to DES-09"),
        ("drift-v2", "wave1", "fmridesign"),
        closed=_closed("DES-01 to DES-09", "f7562ce..d932541"),
    ),
    Bead(
        "bd-01M5PS5JM4J8A2JKZX9WHN96QZ",
        "Fix single-trial prewhitening and voxel HRFs (drift v2 lane 1f)",
        1,
        "Aggregate-trial residual noise model, positive-peak voxel HRFs, "
        "frame runs, whitened intercept. "
        + _red("fmrilss.md", "LSS-01 to LSS-04"),
        ("drift-v2", "wave1", "fmrilss"),
        closed=_closed("LSS-01 to LSS-04 and P-1", "537cbcb..6a147d0"),
    ),
    Bead(
        "bd-01M5PS5JM4J8A2JKZX9WHN96R0",
        "Fix group and meta inference (drift v2 lane 1g)",
        1,
        "Relative meta weights, t-based z/p, LMM df, spatial FDR, "
        "meta-regression validity, permutation counting, NaN-safe BH, "
        "regression tau2. "
        + _red("fmrigds.md and fmrireg_group_output.md", "GDS-01 to GDS-07, "
               "REG-B-01 to REG-B-05"),
        ("drift-v2", "wave1", "fmrigds", "fmrireg"),
        closed=_closed(
            "REG-B-01 to REG-B-05 and GDS-01 to GDS-07 (partial rows are "
            "listed on the S2 remainders bead)",
            "fbe6cee..b197d6f",
        ),
    ),
    Bead(
        "bd-01M5PS5JM4J8A2JKZX9WHN96R1",
        "Fix dataset timing, run identity and selectors (drift v2 lane 1h)",
        1,
        "Per-run start_time, BIDS-HDF5 timing, scan-position runs, boolean "
        "selectors, chunk voxel indices, mask-aware study lowering. "
        + _red("fmridataset.md", "DS-01 to DS-06"),
        ("drift-v2", "wave1", "fmridataset"),
        closed=_closed(
            "DS-01 to DS-06 (DS-01 rest is decision D10, DS-02 VoxelSpace is S2)",
            "5bc0bbd..3913893",
        ),
    ),
    Bead(
        "bd-01M5PS5JM4J8A2JKZX9WHN96R2",
        "Register drift v1 in the contract audits; forward matrix_dataset args",
        2,
        "Two integration fixes found during the v2 work: r_source_drift_"
        "inventory_v1.md was missing from both contract audits (two red doc "
        "tests), and the top-level fmrimod.matrix_dataset dropped censor, "
        "start_time and slice_timing_offset.",
        ("drift-v2", "wave1"),
        closed="Fixed in 97c769c (PR #20) and 5e75ccf (PR #21).",
    ),
    # -- Open: verification and decisions -----------------------------------
    Bead(
        "bd-01M5PS5JM4J8A2JKZX9WHN96R3",
        "Run the R-backed parity suite against the v2 R heads",
        1,
        "R was not installed where v2 and wave 1 ran, so every R reference "
        "is a NumPy transcription. Install the R packages at the v2 heads "
        f"({INVENTORY} \"Ranges audited\") and run "
        "FMRIMOD_REQUIRE_R=1 pytest cross_testing/. Re-baseline the "
        "AGENTS.md note on 22 failures from SPMG1 x10 scaling: fmrihrf "
        "daca7ba changed SPMG1's A1 before CRAN 0.4.0 (decision D2).\n\n"
        "Red check: the R PARITY summary line reports that R-backed tests "
        "ran, and each failure is triaged as a port bug (new bead) or a "
        "documented caveat. Cheap pass disqualified: a run whose R tests "
        "skip prints R PARITY: NOT CHECKED and does not close this bead.",
        ("drift-v2", "parity"),
    ),
    Bead(
        "bd-01M5PS5JM4J8A2JKZX9WHN96R4",
        "Steward decision note for drift v2 decisions D1-D10",
        1,
        f"{INVENTORY} \"Steward decisions\": D1 fmridataset direction after "
        "upstream deleted the pre-frame surface; D2 SPMG conventions and the "
        "spmg1-x10-scale caveat; D3 contrast semantics (term contrasts, "
        "subset bases, zero columns); D4 first-level inference (runwise "
        "pooling, censor semantics, se == 0); D5 public surface (bootstrap_glm, "
        "S2 keyword additions); D6 group semantics (REML, weighted "
        "perm_onesample, cov_tri order); D7 single-trial scope (k-means, "
        "R1-GLM, GLMsingle); D8 neural_input semantics; D9 first-level AR "
        "routing scope; D10 run numbering for event-free runs.\n\n"
        "Deliverable: a decision note under docs/contracts/decisions/ that "
        "rules on each, and child beads for the work each ruling creates.",
        ("drift-v2", "steward", "S2"),
    ),
    Bead(
        "bd-01M5PS5JM4J8A2JKZX9WHN96R5",
        "Pool multi-run runwise contrasts by fixed effects or refuse them",
        2,
        "REG-A-13 (S2, waits on D4): fmrimod pools runwise betas by inverse "
        "variance but reports SEs from a common-sigma joint formula, so the "
        "estimate and its SE come from different models. "
        + _red("fmrireg_fit_core.md", "REG-A-13"),
        ("drift-v2", "fmrireg", "S2"),
    ),
    Bead(
        "bd-01M5PS5JM4J8A2JKZX9WHN96R6",
        "Settle subset basis refits, zero columns and zero-event subsets",
        2,
        "DES-10, DES-14 and the v1 zero-event subset row (waits on D3): "
        "fmrimod refits Poly/BSpline bases on a subset, keeps all-zero "
        "columns for excluded levels and raises on zero-event subsets; R "
        "evaluates on all events, drops the columns and keeps canonical "
        "names. Port or caveat. " + _red("fmridesign.md", "DES-10, DES-14"),
        ("drift-v2", "fmridesign", "S2"),
    ),
    Bead(
        "bd-01M5PS5JM4J8A2JKZX9WHN96R7",
        "Close the public-output parity remainders (S2)",
        2,
        "Parts of wave 1 rows held because they change a public output "
        "layout or result contract: REG-B-02 cov_tri packing (lower vs R's "
        "row-major upper triangle); REG-B-05 Fisher statistic as a signed z "
        "instead of chi2; DS-02 study lowering emitting VoxelSpace; REG-B-08 "
        "keep real REML (needs a caveat) or run Paule-Mandel as R does. "
        "Each waits on D5/D6.",
        ("drift-v2", "S2"),
    ),
    Bead(
        "bd-01M5PS5JM4J8A2JKZX9WHN96R8",
        "Make the test suite green in a clean, offline checkout",
        2,
        "Failures that are not code bugs but keep the suite red: "
        "internal_any_audit.json is stale (regenerate with scripts/"
        "api_inventory.py); the Tier A SPM-auditory benchmark deletes its "
        "checked-in report when the osf.io download fails, which makes "
        "test_retirement_receipt_has_expected_rows_and_shape order-dependent; "
        "cross_testing/test_tier_d_showcase expects outdated case ids "
        "(fails at 4f14a19 too). Red check: pytest tests/ passes with no "
        "network except the download-gated benchmarks, which skip.",
        ("drift-v2", "tests"),
    ),
    # -- Wave 2: contracts ----------------------------------------------------
    Bead(
        "bd-01M5PS5JM4J8A2JKZX9WHN96R9",
        "Port fmrihrf contract changes (drift v2 wave 2)",
        2,
        "HRF-06 (CLI run spill), HRF-10 (neural_input, after D8), HRF-11 to "
        "HRF-16 (empty regressors, column names, block lengths, LWU bounds, "
        "CLI options, validation; HRF-16's width part landed in lane 1d). "
        + _red("fmrihrf.md", "HRF-06, HRF-10 to HRF-16"),
        ("drift-v2", "wave2", "fmrihrf"),
    ),
    Bead(
        "bd-01M5PS5JM4J8A2JKZX9WHN96RA",
        "Port fmridesign contract changes (drift v2 wave 2)",
        2,
        "DES-11 to DES-13, DES-15 to DES-21 (Fcontrasts for multi-basis "
        "terms, column patterns, condition maps, colmap metadata, baseline "
        "blocks, covariate naming, nested contrast sets, validate_contrasts, "
        "basis aliases). Check the DES-02 commit for overlap with DES-18. "
        + _red("fmridesign.md", "DES-11 to DES-13, DES-15 to DES-21"),
        ("drift-v2", "wave2", "fmridesign"),
    ),
    Bead(
        "bd-01M5PS5JM4J8A2JKZX9WHN96RB",
        "Port fmrireg contract changes (drift v2 wave 2)",
        2,
        "REG-A-14 (ARMA q>0; small now that fmri_lm routes through "
        "fit_noise), REG-A-16 to REG-A-19 (engine refusals, AR failure "
        "warning, unknown contrasts error, statistic vocabulary), REG-B-09, "
        "REG-B-10, REG-B-12. "
        + _red("fmrireg_fit_core.md and fmrireg_group_output.md",
               "REG-A-14, REG-A-16 to REG-A-19, REG-B-09, REG-B-10, REG-B-12"),
        ("drift-v2", "wave2", "fmrireg"),
    ),
    Bead(
        "bd-01M5PS5JM4J8A2JKZX9WHN96RC",
        "Port fmriAR contract changes (drift v2 wave 2)",
        2,
        "AR-13 to AR-16, AR-18 to AR-22. AR-21 should already hold after "
        "AR-02 (verify and close the row); AR-15's validation belongs in "
        "_normalize_censor; _hr_arma_pooled already supports the AR-13 order "
        "grid. AR-17 is S2. "
        + _red("fmriar.md", "AR-13 to AR-16, AR-18 to AR-22"),
        ("drift-v2", "wave2", "fmriAR"),
    ),
    Bead(
        "bd-01M5PS5JM4J8A2JKZX9WHN96RD",
        "Port fmrigds contract changes (drift v2 wave 2)",
        2,
        "GDS-08 to GDS-12 (LMM converged flag, axis-data round trip, NIfTI "
        "source validation, n_eff assay; GDS-12 affects the R bridge only). "
        + _red("fmrigds.md", "GDS-08 to GDS-12"),
        ("drift-v2", "wave2", "fmrigds"),
    ),
    Bead(
        "bd-01M5PS5JM4J8A2JKZX9WHN96RE",
        "Port fmridataset contract changes (drift v2 wave 2)",
        2,
        "DS-07 to DS-14 (spatial identity, timing validation, TR tolerance, "
        "run_length per file, NIfTI spatial metadata, empty-mask chunking, "
        "subject ids, space restriction). DS-07 may need a neuroim-python "
        "helper; DS-11 suggests routing the NIfTI backend through the "
        "neuroim adapter. " + _red("fmridataset.md", "DS-07 to DS-14"),
        ("drift-v2", "wave2", "fmridataset"),
    ),
    # -- Bugs and follow-ups found during the work ----------------------------
    Bead(
        "bd-01M5PS5JM4J8A2JKZX9WHN96RF",
        "Fix pre-existing bugs P-4 to P-12 from the drift v2 audit",
        2,
        f"{INVENTORY} \"Pre-existing fmrimod bugs\": P-4 no run intercepts on "
        "the single-trial dataset path; P-5 zero SE map when se is None; P-6 "
        "fmri_rlm forces huber; P-7 compress_bids_study drops encoding/"
        "template; P-8 inconsistent mask binarisation; P-9 "
        "SamplingFrame.from_dict start_time 0; P-10 hr_arma ignores burg; "
        "P-11 ContrastSpec misses cond == 'A'; P-12 SPMG2 second column "
        "(with D2). One commit per bug, each with a red check.",
        ("drift-v2", "bugs"),
    ),
    Bead(
        "bd-01M5PS5JM4J8A2JKZX9WHN96RG",
        "Triage drift v2 wave 1 follow-ups",
        3,
        "Found while implementing wave 1 (not inventory rows): "
        "get_hrf('spmg1', n_basis=6) reports nbasis 6 but evaluates one "
        "column; lagged HRFs are truncated at the unlagged span; trialwise() "
        "ignores durations; golden_tests/specs/core/hrf_bspline_basis.xml "
        "describes the old basis; fmriAR defaults parcel pooling with "
        "p='auto' to pacf_weighted multiscale (fmrimod: none); voxelwise AR "
        "with robust fitting is refused (R allows it); fit_noise p='auto' "
        "keeps absolute variance floors; the old public iterative_ar_gls "
        "re-reads raw data (D9); lane 1b's three deliberate differences from "
        "R (zero-weight AR df, global phi from preprocessed residuals, "
        "reestimate_phi with global pooling refused). Split into beads or "
        "close each with a reason.",
        ("drift-v2", "followup"),
    ),
    # -- Wave 3: features -----------------------------------------------------
    *[
        Bead(
            bead_id,
            title,
            3,
            f"{rows}: see the row's Python shape note in {DETAIL}/{detail}. "
            "Red check: port the R tests cited in the row's R ref column.",
            ("drift-v2", "wave3", tag),
        )
        for bead_id, title, rows, detail, tag in [
            ("bd-01M5PS5JM4J8A2JKZX9WHN96RH",
             "Add a continuously sampled feature regressor",
             "HRF-17 and DES-23 (S2 only if promoted to fmrimod.__all__)",
             "fmrihrf.md and fmridesign.md", "fmrihrf"),
            ("bd-01M5PS5JM4J8A2JKZX9WHN96RJ",
             "Add check_estimability for designs and contrasts",
             "DES-22", "fmridesign.md", "fmridesign"),
            ("bd-01M5PS5JM4J8A2JKZX9WHN96RK",
             "Add within-run baseline correlation diagnostics",
             "DES-24", "fmridesign.md", "fmridesign"),
            ("bd-01M5PS5JM4J8A2JKZX9WHN96RM",
             "Add the mean-series shared AR estimator option",
             "REG-A-20", "fmrireg_fit_core.md", "fmrireg"),
            ("bd-01M5PS5JM4J8A2JKZX9WHN96RN",
             "Pin estimate_hrf multi-basis column-to-curve mapping",
             "REG-A-21", "fmrireg_fit_core.md", "fmrireg"),
            ("bd-01M5PS5JM4J8A2JKZX9WHN96RP",
             "Support PM and Huber meta-regression with covariates",
             "REG-B-14", "fmrireg_group_output.md", "fmrireg"),
            ("bd-01M5PS5JM4J8A2JKZX9WHN96RQ",
             "Add fmri_lm fit plots (low priority)",
             "REG-B-17", "fmrireg_group_output.md", "fmrireg"),
            ("bd-01M5PS5JM4J8A2JKZX9WHN96RR",
             "Support the residual-bias correction for parcel pooling",
             "AR-24", "fmriar.md", "fmriAR"),
            ("bd-01M5PS5JM4J8A2JKZX9WHN96RS",
             "Support the AFNI vrt signal-to-total variance ratio",
             "AR-25", "fmriar.md", "fmriAR"),
            ("bd-01M5PS5JM4J8A2JKZX9WHN96RT",
             "Return the censor set on whiten_apply results",
             "AR-27", "fmriar.md", "fmriAR"),
            ("bd-01M5PS5JM4J8A2JKZX9WHN96RV",
             "Add LSS-N trial groups to LSS",
             "LSS-06", "fmrilss.md", "fmrilss"),
            ("bd-01M5PS5JM4J8A2JKZX9WHN96RW",
             "Add opt-in single-trial design diagnostics",
             "LSS-07", "fmrilss.md", "fmrilss"),
            ("bd-01M5PS5JM4J8A2JKZX9WHN96RX",
             "Add coef_contrast for group coefficient contrasts",
             "GDS-13", "fmrigds.md", "fmrigds"),
            ("bd-01M5PS5JM4J8A2JKZX9WHN96RY",
             "Add the ttest:welch group reducer",
             "GDS-14", "fmrigds.md", "fmrigds"),
            ("bd-01M5PS5JM4J8A2JKZX9WHN96RZ",
             "Take Lancaster dfw from the df assay",
             "GDS-15", "fmrigds.md", "fmrigds"),
            ("bd-01M5PS5JM4J8A2JKZX9WHN96S0",
             "Add read_bids_bold, a one-subject fMRIPrep entry point",
             "DS-20 (S1: no top-level promotion)", "fmridataset.md",
             "fmridataset"),
            ("bd-01M5PS5JM4J8A2JKZX9WHN96S1",
             "Add z input, empirical null and fdr:sabha post-hoc to spatial FDR",
             "GDS-06 (feature parts; wrong-result parts landed in lane 1g)",
             "fmrigds.md", "fmrigds"),
        ]
    ],
]

NOTES: list[Note] = [
    Note(
        V1_EPIC,
        f"Superseded by drift v2 epic {EPIC}: waves 1-3 landed in 3c6fc84; "
        f"the open wave 4 and S2 items are carried by {INVENTORY}.",
    ),
    Note(
        V1_WAVE4,
        "Drift v2 update: estimate_hrf smooth FIR can start, since HRF-02's "
        "clamped B-spline basis landed (PR #21); coef_images is partly "
        "covered by accessors.coef_image and ContrastResult.to_nifti; SBHM "
        "alpha_source/whiten_power can start, since LSS-01's aggregate noise "
        "model landed; any lmm:*_knownvar port should use GDS-05's "
        "between/within df rule.",
    ),
    Note(
        V1_WAVE5_S2,
        "Drift v2 update: examine_group and result frames moved upstream "
        "(fmrigds 2f1cc04), so re-baseline on HEAD before any work; the "
        "zero-event subset row is now part of decision D3; the shared-HRF "
        "C++ fast-path row can close, since fmridesign removed that path "
        "(5ca6b9a).",
    ),
]


@dataclass
class Runner:
    dry_run: bool
    actor: str | None
    commands: list[list[str]] = field(default_factory=list)

    def _mote(self, *args: str) -> list[str]:
        prefix = ["mote"] + (["--actor", self.actor] if self.actor else [])
        return prefix + list(args)

    def run(self, command: list[str]) -> int:
        self.commands.append(command)
        if self.dry_run:
            print(shlex.join(command))
            return 0
        return subprocess.run(command, check=False).returncode

    def exists(self, bead_id: str) -> bool:
        if self.dry_run:
            return False
        completed = subprocess.run(
            ["mote", "show", bead_id], check=False, capture_output=True
        )
        return completed.returncode == 0

    def create(self, bead: Bead) -> int:
        command = [
            sys.executable,
            "scripts/mote_new.py",
            bead.title,
            "-p",
            str(bead.priority),
            "--from-bead",
            bead.source,
            "--id",
            bead.id,
            "--body",
            bead.body,
        ]
        for tag in bead.tags:
            command += ["--tag", tag]
        if self.actor:
            command += ["--actor", self.actor]
        return self.run(command)

    def close(self, bead: Bead) -> int:
        assert bead.closed is not None
        return self.run(self._mote("close", bead.id, "--reason", bead.closed))

    def note(self, note: Note) -> int:
        return self.run(self._mote("note", note.bead, "--kind", note.kind, note.text))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="print commands only")
    parser.add_argument("--actor", help="mote actor for every command")
    args = parser.parse_args(argv)

    if not Path("scripts/mote_new.py").is_file():
        parser.error("run from the fmrimod repository root")
    ids = [bead.id for bead in BEADS]
    assert len(ids) == len(set(ids)), "duplicate bead id"

    runner = Runner(dry_run=args.dry_run, actor=args.actor)
    created_epic = False
    for bead in BEADS:
        if runner.exists(bead.id):
            print(f"exists: {bead.id} {bead.title}")
        else:
            if runner.create(bead) != 0:
                print(f"failed to create {bead.id}; stopping", file=sys.stderr)
                return 1
            created_epic |= bead.id == EPIC
        if bead.closed is not None and runner.close(bead) != 0:
            print(f"failed to close {bead.id}; stopping", file=sys.stderr)
            return 1
    if created_epic or args.dry_run:
        for note in NOTES:
            if runner.note(note) != 0:
                print(f"warning: could not add a note to {note.bead}", file=sys.stderr)
    n_closed = sum(bead.closed is not None for bead in BEADS)
    print(f"{len(BEADS)} beads ({n_closed} filed closed), {len(NOTES)} notes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
