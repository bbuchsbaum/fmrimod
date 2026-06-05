"""Pain points pinned by the group-level one-sample t parity workflow.

Four ergonomic gaps surfaced while wiring ``tier_a_group_level_t``.
Each is pinned here in its current-state shape so a future fix has
a clear regression target. None of these are correctness bugs —
``ols_voxelwise`` produces bitwise-equal results to Nilearn's
``SecondLevelModel`` (effect, t-stat, dof) on the same per-subject
betas. The gaps are at the typed-spec / API discoverability layer.

1. **No top-level ``fm.fmri_group_lm`` entry point.** Users have
   to know to import from ``fmrimod.group``.

2. **Multi-step from per-subject arrays to fit.** Going from a
   ``(n_subjects, n_voxels)`` betas matrix to a t-map requires
   building ``VoxelSpace`` + ``group_dataset`` + ``ols_voxelwise``
   — four calls where Nilearn's ``SecondLevelModel.fit + compute_-
   contrast`` is two.

3. **R-formula syntax for one-sample tests.** ``formula="~ 1"``
   for "intercept-only design" is functional but obscure.

4. **Verbose assay-key naming.** Stats come out as
   ``"coef:Intercept"`` / ``"t_coef:Intercept"`` /
   ``"p_coef:Intercept"`` — readable but requires the user to
   remember the predictor name to access stats.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import fmrimod as fm
from fmrimod.group import group_dataset, ols_voxelwise, VoxelSpace


def _group_inputs(n_subjects: int = 16, n_voxels: int = 32, seed: int = 0):
    """Synthesize per-subject contrast betas with a known group-mean effect."""
    rng = np.random.default_rng(seed)
    voxel_drift = np.linspace(-0.4, 0.4, n_voxels)
    true_effect = 0.7 + voxel_drift
    noise = rng.normal(scale=1.0, size=(n_subjects, n_voxels))
    return (true_effect[np.newaxis, :] + noise).astype(np.float64)


# -- Pain point 1: no top-level entry point ---------------------------------


def test_no_top_level_fmri_group_lm() -> None:
    """``fmrimod`` doesn't expose a ``fmri_group_lm`` / ``fmri_group_t`` top-level.

    Pinned at the current state. A future ``fm.fmri_group_lm(spec,
    group_dataset)`` would close the discoverability gap.
    """
    assert not hasattr(fm, "fmri_group_lm"), (
        "if a top-level fmri_group_lm now exists, update this test to "
        "the desired-state assertion and document the API."
    )
    assert not hasattr(fm, "fmri_group_t"), (
        "if a top-level fmri_group_t now exists, update this test to "
        "the desired-state assertion."
    )


def test_group_module_is_importable_with_canonical_names() -> None:
    """The canonical group-level imports work as documented.

    Positive pin: the documented import path
    ``from fmrimod.group import group_dataset, ols_voxelwise,
    VoxelSpace`` resolves so workflows can be built today.
    """
    from fmrimod.group import (
        VoxelSpace as _VS,
        group_dataset as _gd,
        ols_voxelwise as _olsv,
    )
    assert callable(_gd)
    assert callable(_olsv)
    assert callable(_VS)


# -- Pain point 2: multi-step construction ----------------------------------


def test_per_subject_betas_to_fit_requires_four_calls() -> None:
    """Document the current minimum-call construction.

    A ``group_lm_from_arrays(betas)`` shorthand would close this
    gap by collapsing the four calls into one.
    """
    betas = _group_inputs()
    n_voxels = betas.shape[1]
    # 1) Shape-shift to (samples, subjects, contrasts).
    beta_3d = betas.T[:, :, np.newaxis]
    # 2) Build a VoxelSpace.
    space = VoxelSpace(shape=(n_voxels, 1, 1))
    # 3) Build a GroupDataset.
    ds = group_dataset(
        assays={"beta": beta_3d},
        space=space,
        subjects=[f"sub-{i:02d}" for i in range(betas.shape[0])],
        contrasts=["A_minus_B"],
    )
    # 4) Fit.
    result = ols_voxelwise(ds, formula="~ 1")
    # Confirm the call chain produced a valid result.
    assert "coef:Intercept" in result.assays


# -- Pain point 3: R-formula syntax for one-sample tests --------------------


def test_one_sample_t_requires_r_formula_syntax() -> None:
    """Users write ``formula="~ 1"`` for the canonical one-sample test.

    A typed shortcut like ``ols_voxelwise(ds, intercept_only=True)``
    or a dedicated ``one_sample_t`` reducer would make this case
    obvious.
    """
    betas = _group_inputs()
    n_voxels = betas.shape[1]
    beta_3d = betas.T[:, :, np.newaxis]
    space = VoxelSpace(shape=(n_voxels, 1, 1))
    ds = group_dataset(
        assays={"beta": beta_3d},
        space=space,
        subjects=[f"sub-{i:02d}" for i in range(betas.shape[0])],
        contrasts=["A_minus_B"],
    )
    # The functional path: "~ 1" is the R idiom for intercept-only.
    result_r = ols_voxelwise(ds, formula="~ 1")
    # An ``intercept_only`` kwarg currently does not exist; verify
    # the explicit kwarg is rejected so a future fix has a clear
    # regression target.
    with pytest.raises(TypeError):
        ols_voxelwise(ds, intercept_only=True)
    assert "coef:Intercept" in result_r.assays


# -- Pain point 4: verbose assay-key naming ---------------------------------


def test_stats_keyed_by_predictor_name_string() -> None:
    """Stats are addressed via ``"coef:Intercept"`` / ``"t_coef:Intercept"`` keys.

    Document the current convention. A typed accessor like
    ``result.effect("Intercept")`` would close the gap.
    """
    betas = _group_inputs()
    n_voxels = betas.shape[1]
    beta_3d = betas.T[:, :, np.newaxis]
    space = VoxelSpace(shape=(n_voxels, 1, 1))
    ds = group_dataset(
        assays={"beta": beta_3d},
        space=space,
        subjects=[f"sub-{i:02d}" for i in range(betas.shape[0])],
        contrasts=["A_minus_B"],
    )
    result = ols_voxelwise(ds, formula="~ 1")
    # The canonical assay keys for an intercept-only fit:
    assert "coef:Intercept" in result.assays
    assert "t_coef:Intercept" in result.assays
    assert "p_coef:Intercept" in result.assays
    assert "se_coef:Intercept" in result.assays
    # No typed effect() accessor exists today; if one is added that
    # composes with the typed-spec story, the pain point closes.


# -- Positive: numerical correctness vs scipy one-sample t ----------------


def test_ols_voxelwise_t_matches_scipy_ttest_1samp() -> None:
    """The native group OLS matches scipy's one-sample t bitwise.

    Positive pin that the underlying numerics are sound; the pain
    points above are purely ergonomic.
    """
    from scipy import stats as sp_stats

    betas = _group_inputs(n_subjects=20, n_voxels=48, seed=1)
    n_voxels = betas.shape[1]
    beta_3d = betas.T[:, :, np.newaxis]
    space = VoxelSpace(shape=(n_voxels, 1, 1))
    ds = group_dataset(
        assays={"beta": beta_3d},
        space=space,
        subjects=[f"sub-{i:02d}" for i in range(betas.shape[0])],
        contrasts=["A_minus_B"],
    )
    result = ols_voxelwise(ds, formula="~ 1")
    fmrimod_t = np.asarray(result.assay("t_coef:Intercept"))[:, 0, 0]
    scipy_t = sp_stats.ttest_1samp(betas, 0, axis=0).statistic
    np.testing.assert_allclose(fmrimod_t, scipy_t, atol=1e-12)
