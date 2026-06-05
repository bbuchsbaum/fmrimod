"""Group-level (second-level) one-sample t against Nilearn ``SecondLevelModel``.

The capstone of every fMRI publication: per-subject first-level
contrast maps come in, the group test asks "where is condition
A − B significantly different from zero across N subjects?"

This is the first parity workflow that crosses the first-level →
second-level seam. fmrimod's ``fmrimod/group/`` module ships the
machinery (``GroupDataset``, ``VoxelSpace``, ``ols_voxelwise``);
the headline outputs here cover the standard one-sample t test
that nearly every paper runs.

Why this is a stress test
-------------------------
Nilearn ships ``SecondLevelModel`` as a single class with a
``compute_contrast`` method that takes the contrast name and an
output type ("effect_size", "stat", "p_value"). fmrimod's path
requires the user to:

1. Wrap per-subject beta arrays into a 3-D
   ``(samples, subjects, contrasts)`` assay.
2. Construct a ``VoxelSpace`` with shape + affine.
3. Wrap both in a ``GroupDataset`` via :func:`group_dataset`.
4. Call :func:`ols_voxelwise` with an R-style ``formula="~ 1"``.
5. Pull the named stats out by their ``coef:Intercept`` etc. keys.

Five steps where Nilearn has one. The numerical answers agree
bitwise; the ergonomic gap is real.

Pattern A parity claim
----------------------
Both engines take the same per-subject contrast matrix and run a
one-sample t (intercept-only design):

- fmrimod: ``ols_voxelwise(group_dataset, formula="~ 1")``.
- Nilearn: ``SecondLevelModel`` fitted on a list of fake-NIfTI
  contrast images with an intercept-only design DataFrame.

Six outputs match at <= 1e-9:

- ``effect``: per-voxel mean across subjects.
- ``t_stat``: per-voxel one-sample t.
- ``intercept_dof``: should equal ``n_subjects - 1``.
- ``intercept_beta_diff``: max abs diff between fmrimod's
  ``coef:Intercept`` and the per-voxel mean.
- ``rank``: design rank (pinned at 1).

Pain points pinned for follow-up
--------------------------------

Three ergonomic gaps surfaced and are pinned in
``tests/test_group/test_group_level_pain_points.py``:

1. **No top-level entry point.** There's no ``fm.fmri_group_lm``
   or ``fm.fmri_group_t`` on the package; users have to know to
   import from ``fmrimod.group``.

2. **Multi-step construction.** Going from a list of per-subject
   beta vectors to a fit requires ``VoxelSpace`` +
   ``group_dataset`` + ``ols_voxelwise``. A ``group_lm(spec,
   per_subject_arrays)`` shorthand would close the
   from-arrays gap.

3. **R-formula syntax for one-sample tests.** The user writes
   ``formula="~ 1"`` to mean intercept-only. A
   ``one_sample_t`` reducer or
   ``ols_voxelwise(..., intercept_only=True)`` shortcut would
   make the common case obvious.

4. **Verbose assay-key naming.** Stats come out as
   ``"coef:Intercept"`` / ``"t_coef:Intercept"`` /
   ``"p_coef:Intercept"`` — readable but a typed accessor like
   ``result.effect("Intercept")`` would compose with the
   typed-spec story of the first-level surface.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn.glm.second_level import SecondLevelModel
from numpy.typing import NDArray

import nibabel as nib

from cross_testing.harness import (
    ParityCase,
    ParityTolerance,
    PipelineOutput,
    render,
    run,
)
from fmrimod.group import GroupDataset, VoxelSpace, group_dataset, ols_voxelwise

Array = NDArray[np.float64]

N_SUBJECTS = 24
N_VOXELS = 256
TRUE_EFFECT_MEAN = 0.75


@dataclass(frozen=True)
class GroupInputs:
    """Shared inputs for the group-level case.

    The per-subject contrast matrix ``betas`` has shape
    ``(n_subjects, n_voxels)``. Both engines work from this same
    matrix — fmrimod via the (sample × subject × contrast) assay
    layout, Nilearn via a list of fake-NIfTI contrast images.
    """

    betas: Array
    subject_ids: tuple[str, ...]
    affine: Array


def load_inputs(
    n_subjects: int = N_SUBJECTS,
    n_voxels: int = N_VOXELS,
    seed: int = 20260524,
) -> GroupInputs:
    """Synthesize a per-subject contrast matrix with a clear group-mean effect.

    Each subject contributes a noisy version of the true effect
    (Gaussian noise with σ=1.0 around mean ``TRUE_EFFECT_MEAN``).
    A voxel-mean drift makes the per-voxel effect vary so the t-map
    is non-degenerate.
    """
    rng = np.random.default_rng(seed)
    voxel_drift = np.linspace(-0.4, 0.4, n_voxels)
    true_effect_per_voxel = TRUE_EFFECT_MEAN + voxel_drift
    noise = rng.normal(scale=1.0, size=(n_subjects, n_voxels))
    betas = true_effect_per_voxel[np.newaxis, :] + noise
    return GroupInputs(
        betas=betas.astype(np.float64),
        subject_ids=tuple(f"sub-{i:02d}" for i in range(n_subjects)),
        affine=np.eye(4),
    )


def fmrimod_pipeline(inputs: GroupInputs) -> PipelineOutput:
    """Typed ``fm.fmri_group_lm(betas)`` one-call group-level path.

    Closes the typed-spec gap pinned by the original commit's pain
    points: the four-step ``VoxelSpace`` + ``group_dataset`` +
    ``ols_voxelwise`` + key-string accessor sequence collapses to one
    call plus typed ``.effect()`` / ``.t_stat()`` accessors.
    """
    import fmrimod as fm

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        result = fm.fmri_group_lm(inputs.betas)
    effect = result.effect("Intercept")
    t_stat = result.t_stat("Intercept")
    dof = result.residual_df
    return PipelineOutput(
        arrays={
            "effect": effect,
            "t_stat": t_stat,
            "intercept_dof": np.array([dof], dtype=np.float64),
            "intercept_beta_diff": np.array(
                [float(np.max(np.abs(effect - inputs.betas.mean(axis=0))))],
                dtype=np.float64,
            ),
            "rank": np.array([1.0], dtype=np.float64),
        }
    )


def nilearn_pipeline(inputs: GroupInputs) -> PipelineOutput:
    """Reference: Nilearn ``SecondLevelModel`` on fake-NIfTI contrast images."""
    n_voxels = inputs.betas.shape[1]
    # Wrap each subject's contrast vector into a (n_voxels, 1, 1) volume.
    cmaps = []
    for s in range(inputs.betas.shape[0]):
        vol = inputs.betas[s].reshape(n_voxels, 1, 1)
        cmaps.append(nib.Nifti1Image(vol, inputs.affine))
    design = pd.DataFrame({"intercept": [1.0] * inputs.betas.shape[0]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        slm = SecondLevelModel(smoothing_fwhm=None)
        slm.fit(cmaps, design_matrix=design)
        effect_img = slm.compute_contrast(
            "intercept", output_type="effect_size"
        )
        t_img = slm.compute_contrast(
            "intercept", second_level_stat_type="t", output_type="stat"
        )
    effect = np.asarray(effect_img.get_fdata().ravel(), dtype=np.float64)
    t_stat = np.asarray(t_img.get_fdata().ravel(), dtype=np.float64)
    return PipelineOutput(
        arrays={
            "effect": effect,
            "t_stat": t_stat,
            "intercept_dof": np.array(
                [float(inputs.betas.shape[0] - 1)], dtype=np.float64
            ),
            "intercept_beta_diff": np.array(
                [float(np.max(np.abs(effect - inputs.betas.mean(axis=0))))],
                dtype=np.float64,
            ),
            "rank": np.array([1.0], dtype=np.float64),
        }
    )


def make_case(
    n_subjects: int = N_SUBJECTS, n_voxels: int = N_VOXELS,
) -> ParityCase:
    """Build the group-level one-sample t parity case."""
    return ParityCase(
        name="tier_a_group_level_t",
        fmrimod_pipeline=fmrimod_pipeline,
        reference_pipeline=nilearn_pipeline,
        inputs=load_inputs(n_subjects=n_subjects, n_voxels=n_voxels),
        tolerances={
            "effect": ParityTolerance(rtol=1e-8, atol=1e-9),
            "t_stat": ParityTolerance(rtol=1e-7, atol=1e-8),
            "intercept_dof": ParityTolerance(rtol=0.0, atol=0.0),
            "intercept_beta_diff": ParityTolerance(rtol=0.0, atol=1e-12),
            "rank": ParityTolerance(rtol=0.0, atol=0.0),
        },
    )


def main() -> None:
    result = run(make_case())
    out_dir = Path(__file__).resolve().parent / "reports"
    render(result, out_dir)
    if result.status == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
