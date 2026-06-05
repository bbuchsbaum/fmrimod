"""Regression tests for the group-level (second-level) typed surface.

History: the ``tier_a_group_level_t`` parity workflow surfaced four
ergonomic pain points on top of the working ``ols_voxelwise``
numerics. All four are now closed by ``fm.fmri_group_lm`` +
:class:`GroupLmResult`; this file pins the fixed state so the typed
surface can't silently regress.

1. **Top-level ``fm.fmri_group_lm`` entry point.**
2. **Per-subject arrays accepted directly** — the four-call
   construction (``VoxelSpace`` + ``group_dataset`` +
   ``ols_voxelwise`` + key-string accessor) collapses to one call.
3. **``intercept_only=True`` default** — the one-sample test
   doesn't require knowing the R formula idiom ``"~ 1"``.
4. **Typed predictor-name accessors** (``.effect("Intercept")``,
   ``.t_stat("group")``) replace ``"coef:Intercept"`` /
   ``"t_coef:group"`` key strings, and DataFrame column names are
   preserved into the predictor labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import fmrimod as fm
from fmrimod.group import GroupLmResult, group_dataset, ols_voxelwise, VoxelSpace


def _group_inputs(n_subjects: int = 16, n_voxels: int = 32, seed: int = 0):
    rng = np.random.default_rng(seed)
    voxel_drift = np.linspace(-0.4, 0.4, n_voxels)
    true_effect = 0.7 + voxel_drift
    noise = rng.normal(scale=1.0, size=(n_subjects, n_voxels))
    return (true_effect[np.newaxis, :] + noise).astype(np.float64)


# -- Fixed: top-level entry point + one-call construction --------------------


def test_fm_fmri_group_lm_is_top_level() -> None:
    """``fm.fmri_group_lm`` is exposed at the package top level."""
    assert callable(fm.fmri_group_lm), "fm.fmri_group_lm should be callable"
    assert "fmri_group_lm" in fm.__all__


def test_one_call_from_per_subject_array_to_t_map() -> None:
    """``fm.fmri_group_lm(betas)`` returns a typed group result in one call."""
    betas = _group_inputs()
    result = fm.fmri_group_lm(betas)
    assert isinstance(result, GroupLmResult)
    # ``intercept_only=True`` is the documented default; the result
    # carries the canonical predictor name.
    assert result.predictor_names == ("Intercept",)
    # Per-voxel outputs are shape ``(n_voxels,)`` — the typed
    # accessors squeeze the singleton subject / contrast axes.
    assert result.t_stat().shape == (betas.shape[1],)
    assert result.effect().shape == (betas.shape[1],)
    assert result.p_value().shape == (betas.shape[1],)


def test_one_call_matches_scipy_one_sample_t() -> None:
    """The one-call typed path matches scipy's ``ttest_1samp`` bitwise."""
    from scipy import stats as sp_stats

    betas = _group_inputs(n_subjects=20, n_voxels=48, seed=1)
    result = fm.fmri_group_lm(betas)
    np.testing.assert_allclose(
        result.t_stat(),
        sp_stats.ttest_1samp(betas, 0, axis=0).statistic,
        atol=1e-12,
    )


def test_residual_df_equals_n_subjects_minus_predictors() -> None:
    """``residual_df`` is the canonical ``n - p`` for the one-sample case."""
    betas = _group_inputs(n_subjects=24)
    result = fm.fmri_group_lm(betas)
    assert result.residual_df == 23.0  # 24 - 1


# -- Fixed: intercept_only=True default replaces "~ 1" formula --------------


def test_intercept_only_default_replaces_r_formula() -> None:
    """``fm.fmri_group_lm(betas)`` defaults to one-sample t.

    No ``formula="~ 1"`` string required for the common case.
    """
    betas = _group_inputs()
    # Default call works without explicit formula.
    result_default = fm.fmri_group_lm(betas)
    # The legacy ``formula="~ 1"`` path is still accepted for users
    # who want the explicit form.
    result_legacy = fm.fmri_group_lm(
        betas, intercept_only=False, formula="~ 1",
    )
    np.testing.assert_allclose(
        result_default.t_stat(), result_legacy.t_stat(), atol=1e-12,
    )


def test_formula_and_design_matrix_are_mutually_exclusive() -> None:
    """Passing both ``formula=`` and ``design_matrix=`` raises."""
    betas = _group_inputs(n_subjects=10)
    design = pd.DataFrame({"a": [1.0]*10, "b": [0]*5 + [1]*5})
    with pytest.raises(ValueError, match="at most one"):
        fm.fmri_group_lm(
            betas, intercept_only=False,
            formula="~ a + b", design_matrix=design,
        )


# -- Fixed: typed predictor-name accessors ----------------------------------


def test_dataframe_predictor_names_pass_through() -> None:
    """DataFrame column names appear on the typed accessors.

    Before the fix, passing a ``design_matrix=DataFrame`` lost the
    column names — the typed accessors had to use ``x0``, ``x1``,
    ... placeholders. Now the column names propagate cleanly.
    """
    n = 16
    rng = np.random.default_rng(3)
    betas = rng.normal(size=(n, 24))
    betas[8:] += 0.5  # inject a group effect
    design = pd.DataFrame({
        "Intercept": [1.0] * n,
        "group": [0]*8 + [1]*8,
    })
    result = fm.fmri_group_lm(
        betas, intercept_only=False, design_matrix=design,
    )
    assert result.predictor_names == ("Intercept", "group")
    # The typed effect / t_stat accessors resolve by user-visible name.
    effect_group = result.effect("group")
    assert effect_group.shape == (24,)
    assert effect_group.mean() == pytest.approx(0.5, abs=0.5)
    assert result.t_stat("group").shape == (24,)


def test_unknown_predictor_raises_keyerror() -> None:
    """Asking for an undefined predictor raises a clear ``KeyError``."""
    betas = _group_inputs()
    result = fm.fmri_group_lm(betas)
    with pytest.raises(KeyError, match="not in"):
        result.effect("nonexistent")


def test_group_dataset_input_still_works() -> None:
    """A pre-built ``GroupDataset`` can also be passed in."""
    betas = _group_inputs(n_subjects=12, n_voxels=20)
    beta_3d = betas.T[:, :, np.newaxis]
    space = VoxelSpace(shape=(20, 1, 1))
    ds = group_dataset(
        assays={"beta": beta_3d},
        space=space,
        subjects=[f"sub-{i}" for i in range(12)],
        contrasts=["c0"],
    )
    result = fm.fmri_group_lm(ds)
    assert isinstance(result, GroupLmResult)
    assert result.t_stat().shape == (20,)


# -- Positive: raw API still available --------------------------------------


def test_lower_level_ols_voxelwise_still_works() -> None:
    """The lower-level ``ols_voxelwise`` path is preserved."""
    betas = _group_inputs(n_subjects=15, n_voxels=24)
    n_voxels = betas.shape[1]
    beta_3d = betas.T[:, :, np.newaxis]
    space = VoxelSpace(shape=(n_voxels, 1, 1))
    ds = group_dataset(
        assays={"beta": beta_3d},
        space=space,
        subjects=[f"sub-{i}" for i in range(15)],
        contrasts=["c0"],
    )
    result = ols_voxelwise(ds, formula="~ 1")
    # Raw assay keys still work for users who want them.
    assert "coef:Intercept" in result.assays
    assert "t_coef:Intercept" in result.assays
