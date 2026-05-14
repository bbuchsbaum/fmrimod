"""Burn-down ratchet for the public-API inventory.

Pairs with ``docs/contracts/api_inventory_v1.json`` and the freshness
gate in ``test_api_inventory.py``. The freshness gate catches *new*
public names entering with ``tier=review_pending``; this file pins the
*existing* review_pending set as a baseline and forces it to shrink
monotonically.

Two assertions, both catch real failure modes:

1. **No new review_pending rows.** Tiering a name to spine /
   spine_review / compat / compat_pending_fix / runtime_check is the
   only way ``tier=review_pending`` can change. Adding a new public
   name with ``review_pending`` (rather than tiering it on entry) or
   reverting an already-tiered name back to ``review_pending`` both
   fail the subset assertion below.

2. **Baseline doesn't decay silently.** When a name is tiered or
   removed from ``__all__``, the baseline ``BASELINE_REVIEW_PENDING_NAMES``
   set must be updated in the same commit. Otherwise the burn-down
   ratchet silently widens over time and a re-tiering becomes invisible.

To take a name off the baseline:
- Tier it in ``docs/contracts/api_inventory_v1.json`` (any value other
  than ``"review_pending"``).
- Remove it from ``BASELINE_REVIEW_PENDING_NAMES`` below.
- Both edits land in the same commit, with a board-linked rationale
  for the chosen tier.

Refs: bd-01KRHVVKM80SYC37BQ5HQKRMN1
(Board source: major-issues-lets-talk/post-01KRHVA1NC98BP4KQ1Z29WYXWG,
bullet 4 burn-down framing).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
INVENTORY_PATH = REPO_ROOT / "docs" / "contracts" / "api_inventory_v1.json"


# Public names whose tier is currently ``review_pending`` at the moment
# the burn-down ratchet landed. New entries to this set are forbidden;
# entries leave the set when the corresponding inventory row is tiered
# in the same commit.
BASELINE_REVIEW_PENDING_NAMES = frozenset({
    "Confounds", "Drift", "Fcontrasts", "FieldDiff",
    "HRF", "HRF_BSPLINE", "HRF_FIR", "HRF_FOURIER",
    "HRF_GAMMA", "HRF_GAUSSIAN", "HRF_HALF_COSINE", "HRF_INV_LOGIT",
    "HRF_LWU", "HRF_LWU_BASIS", "HRF_MEXHAT", "HRF_SINE",
    "HRF_SPMG1", "HRF_SPMG2", "HRF_SPMG3", "HRF_TIME",
    "HrfTerm", "Intercept", "Poly", "RobustScale",
    "SPM_CANONICAL", "SPM_WITH_DERIVATIVE", "SPM_WITH_DISPERSION", "SamplingFrame",
    "Scale", "SpecDiff", "SpecSerializationError", "TermDiff",
    "__version__", "acorr_diagnostics", "acquisition_onsets", "afni_restricted_plan",
    "amplitudes", "ar_parameters", "as_hrf", "as_spec",
    "block_hrf", "bootstrap_glm", "build_nuisance_projector", "check_nuisance",
    "clean_nuisance", "coef_image", "coef_names", "column_contrast",
    "combine_contrasts", "combine_runs", "compat", "compute_dvars",
    "condition_basis_list", "condition_map", "confounds", "contrast",
    "contrast_set", "contrast_weights", "data_chunks", "design_matrix",
    "detect_group_data_format", "drift", "dvars_to_weights", "estimate_betas",
    "estimate_hrf", "evaluate", "event_factor", "event_matrix",
    "event_term", "event_variable", "fit_noise", "fitted_hrf",
    "fmri_meta_fit", "fmri_meta_fit_contrasts", "fmri_meta_fit_cov", "fmri_meta_fit_extended",
    "gamma_hrf", "gaussian_hrf", "gen_empirical_hrf", "gen_hrf",
    "gen_hrf_library", "gen_hrf_set", "generate_interaction_contrast", "generate_main_effect_contrast",
    "get_contrasts", "get_covariates", "get_data", "get_data_matrix",
    "get_formula", "get_hrf", "get_mask", "get_rois",
    "get_subjects", "glm_lss", "glm_ols", "global_onsets",
    "group_data", "group_data_from_csv", "group_data_from_h5", "group_data_from_nifti",
    "hrf_blocked", "hrf_bspline_generator", "hrf_daguerre_generator", "hrf_fir_generator",
    "hrf_formula", "hrf_fourier_generator", "hrf_lagged", "hrf_set",
    "hrf_spmg1", "hrf_tent_generator", "interaction_contrast", "intercept",
    "is_spec", "lag_hrf", "latent_dataset", "list_available_hrfs",
    "lsa_single_trial", "lss_single_trial", "matrix_dataset", "meta_effective_n",
    "n_subjects", "null_regressor", "one_against_all_contrast", "oneway_contrast",
    "p_values", "pair_contrast", "pairwise_contrasts", "poly_contrast",
    "pvalues", "r_to_z", "regressor", "regressor_set",
    "samples", "sandwich_from_whitened_resid", "se", "shift",
    "simulate_bold_signal", "simulate_fmri_matrix", "simulate_noise_vector", "simulate_simple_dataset",
    "sliding_window_contrasts", "soft_subspace_options", "spec_diff", "spm_canonical",
    "standard_error", "stats", "t_to_d", "tidy",
    "tidy_fitted_hrf", "unit_contrast", "volume_weights", "voxel_index_chunks",
    "whiten", "whiten_apply", "write_results", "z_to_r",
    "zscores",
})


def _load_inventory() -> dict:
    if not INVENTORY_PATH.exists():
        pytest.fail(
            f"inventory missing at {INVENTORY_PATH}; regenerate with "
            f"`python scripts/api_inventory.py`"
        )
    return json.loads(INVENTORY_PATH.read_text())


def _current_review_pending() -> set[str]:
    return {
        row["name"]
        for row in _load_inventory()["rows"]
        if row["tier"] == "review_pending"
    }


def test_baseline_size_is_known() -> None:
    """Sanity-check the pinned baseline size to catch silent drift."""
    assert len(BASELINE_REVIEW_PENDING_NAMES) == 157, (
        f"BASELINE_REVIEW_PENDING_NAMES size drifted from 157 to "
        f"{len(BASELINE_REVIEW_PENDING_NAMES)}. Update the assertion "
        f"in the same commit that intentionally changed the baseline."
    )


def test_review_pending_set_does_not_grow_beyond_baseline() -> None:
    """No new public names may carry tier=review_pending.

    Failure modes this catches:
    - A new name was added to ``fmrimod.__all__`` with the default
      ``tier=review_pending`` instead of being tiered on entry.
    - An already-tiered name was reverted to ``review_pending``.

    Cheap pass disqualified: silently widening
    ``BASELINE_REVIEW_PENDING_NAMES`` without an inventory tier change.
    The companion ``test_baseline_does_not_decay_silently`` catches that.
    """
    current = _current_review_pending()
    intruders = sorted(current - BASELINE_REVIEW_PENDING_NAMES)
    assert not intruders, (
        f"new review_pending names appeared in the inventory: {intruders}. "
        f"Either tier them in api_inventory_v1.json (the public-API "
        f"contract is no new public names enter without a tier), or — if "
        f"they are genuinely necessary additions — extend "
        f"BASELINE_REVIEW_PENDING_NAMES with a board-linked rationale "
        f"in the same commit (and update test_baseline_size_is_known)."
    )


def test_baseline_does_not_decay_silently() -> None:
    """Names removed from review_pending must come off the baseline.

    Failure modes this catches:
    - A row was tiered (now non-review_pending) but the baseline still
      lists it. The ratchet would silently allow the name back to
      review_pending later without surfacing the regression.
    - A name was removed from ``fmrimod.__all__`` entirely but the
      baseline still references it. Same silent-widening risk.

    To remove a name from the baseline: pair the deletion from
    ``BASELINE_REVIEW_PENDING_NAMES`` with the inventory tier change
    (or the ``__all__`` removal) in the same commit, and decrement the
    expected size in ``test_baseline_size_is_known``.
    """
    declared = {row["name"] for row in _load_inventory()["rows"]}
    current = _current_review_pending()

    # In the baseline but no longer in __all__ → stale (silently widens
    # the ratchet because the row can't appear in any future probe).
    no_longer_declared = sorted(BASELINE_REVIEW_PENDING_NAMES - declared)
    # In the baseline but no longer review_pending → tiered without
    # baseline cleanup.
    no_longer_pending = sorted(
        (BASELINE_REVIEW_PENDING_NAMES & declared) - current
    )

    stale = []
    if no_longer_declared:
        stale.append(f"removed from __all__: {no_longer_declared}")
    if no_longer_pending:
        stale.append(f"already tiered: {no_longer_pending}")
    assert not stale, (
        "BASELINE_REVIEW_PENDING_NAMES has stale entries — pair the "
        "tier or removal with a baseline cleanup in the same commit:\n  "
        + "\n  ".join(stale)
    )
