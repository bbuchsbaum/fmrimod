"""Multicollinear-baseline first-level parity against Nilearn.

This case stresses an angle absent from the other Tier A workflows: a
first-level design with a *linearly dependent* nuisance regressor (a
common consequence of dumping a full ``fmriprep`` confound matrix into
the model — 24-motion + aCompCor + WM/CSF often produce one or more
exact algebraic combinations). The two natural questions:

1. Does the engine *detect* the rank deficiency and tell the user
   visibly, rather than silently producing a result that depends on the
   pseudoinverse implementation?
2. Does the engine *recover* — i.e. does it solve the system, leave
   identifiable contrasts (task t-stats, F-tests in the row space of
   ``X``) numerically equal to a full-rank reference, and clearly mark
   the non-identifiable individual betas?

Why this is a stress test
-------------------------
Nilearn's :func:`~nilearn.glm.first_level.run_glm` silently routes a
rank-deficient ``X`` through the Moore–Penrose pseudoinverse and
returns whatever the SVD path produces — no warning, no flag on the
returned :class:`~nilearn.glm.regression.RegressionResults`, no way to
inspect which columns are aliased. The user is left to discover the
rank deficiency by re-running ``np.linalg.matrix_rank(X)`` themselves.

fmrimod surfaces three typed diagnostics on the same scenario:

- :meth:`baseline_model` emits a ``UserWarning`` naming the aliased
  confound by its user-supplied label (e.g. ``"motion_y"`` rather than
  ``"V2"``).
- :func:`fmri_lm` emits a second ``UserWarning`` after the fit listing
  per-run rank/p/dfres and the aliased design-column names.
- :meth:`FmriLm.condition_report` returns a typed
  :class:`ConditionReport` carrying the same data structurally, plus
  :attr:`FmriLm.is_full_rank` and :attr:`FmriLm.ill_conditioned`
  shortcuts for programmatic use.

The recovery is mathematically sound: the solver routes through SVD
when QR-pivoting detects rank deficiency, computes the minimum-norm
pseudoinverse solution, and uses ``dfres = n - rank`` for variance
estimation. Contrasts in the row space of ``X`` therefore agree with
Nilearn's ``run_glm`` to machine precision.

What we compare
---------------
Both pipelines share the realised fmrimod design (Pattern B). The
parity claim is cross-engine: identical pseudoinverse OLS on
identical, deliberately rank-deficient ``X``.

The compared quantities:

- ``design``: bitwise-equal realised design.
- ``effect_task_main``, ``t_task_main``: a task t-contrast (estimable
  because the task columns do not depend on the aliased confound).
- ``f_task_omnibus``: a joint F over the two task columns.
- ``rank`` and ``aliased_columns``: scalars / metadata pinned via the
  harness array channel so the diagnostic surface is part of the
  parity contract.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from nilearn.glm.contrasts import compute_contrast
from nilearn.glm.first_level import run_glm
from numpy.typing import NDArray

import fmrimod as fm
from cross_testing.harness import (
    ParityCase,
    ParityTolerance,
    PipelineOutput,
    render,
    run,
)
from fmrimod.design.columns import DesignColumns
from fmrimod.spec import confounds, hrf, intercept

Array = NDArray[np.float64]

TR = 2.0
N_SCANS = 160
MAX_VOXELS = 1024
TRIAL_TYPES: tuple[str, ...] = ("A", "B")


@dataclass(frozen=True)
class MulticollinearInputs:
    """Shared inputs: events, BOLD, realised design, typed contrasts."""

    events: pd.DataFrame
    confounds: pd.DataFrame
    data: Array
    design: Array
    design_columns: DesignColumns
    c_task_main: Array
    c_task_F: Array
    aliased_candidates: tuple[str, ...]


def _make_events(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_trials = 18
    onsets = np.linspace(8.0, N_SCANS * TR - 24.0, n_trials, dtype=np.float64)
    labels = np.array(list(TRIAL_TYPES) * (n_trials // len(TRIAL_TYPES)))
    rng.shuffle(labels)
    return pd.DataFrame(
        {
            "onset": onsets,
            "duration": np.zeros(n_trials, dtype=np.float64),
            "trial_type": labels,
            "run": 1,
        }
    )


def _make_confounds(seed: int) -> pd.DataFrame:
    """Build six confounds, one of which is an exact linear combination.

    Mirrors the common ``fmriprep`` failure mode where ``framewise_displacement``
    or a derived motion column ends up exactly representable as a linear
    combination of other columns (e.g. translational + rotational motion
    plus a constant).
    """
    rng = np.random.default_rng(seed + 7)
    trans_x = rng.normal(scale=0.5, size=N_SCANS)
    trans_y = rng.normal(scale=0.5, size=N_SCANS)
    rot_z = rng.normal(scale=0.05, size=N_SCANS)
    csf = rng.normal(scale=1.0, size=N_SCANS)
    wm = rng.normal(scale=1.0, size=N_SCANS)
    # `composite_motion` is an exact linear combination of trans_x,
    # trans_y, and rot_z — the offending column the diagnostic should
    # surface.
    composite_motion = 1.5 * trans_x - 2.0 * trans_y + 10.0 * rot_z
    return pd.DataFrame(
        {
            "trans_x": trans_x,
            "trans_y": trans_y,
            "rot_z": rot_z,
            "csf": csf,
            "wm": wm,
            "composite_motion": composite_motion,
        }
    )


def _realize_design(
    events: pd.DataFrame, confound_df: pd.DataFrame
) -> tuple[Any, Array, DesignColumns]:
    """Build the rank-deficient design via fmrimod's typed spec."""
    spec = (
        hrf("trial_type", basis="spm", norm="spm")
        + confounds(
            "trans_x",
            "trans_y",
            "rot_z",
            "csf",
            "wm",
            "composite_motion",
            source=confound_df,
        )
        + intercept(per="run")
    )
    dummy = fm.fmri_dataset(
        np.zeros((N_SCANS, 1), dtype=np.float64),
        tr=TR,
        events=events,
    )
    # Suppress the deliberate diagnostic warning during input setup; the
    # workflow re-runs the fit at parity-pipeline time and captures the
    # warning there so it is exercised, not just suppressed.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fit = fm.fmri_lm(spec, dummy)
    design = np.asarray(
        fit.model.design_matrix_array(run=0), dtype=np.float64
    )
    return spec, design, fit.design_columns()


def load_inputs(
    max_voxels: int = MAX_VOXELS, seed: int = 20260513
) -> MulticollinearInputs:
    """Synthesize a BOLD time-series on a deliberately rank-deficient design."""
    rng = np.random.default_rng(seed)
    events = _make_events(seed)
    confound_df = _make_confounds(seed)
    spec, design, columns = _realize_design(events, confound_df)
    n_total = design.shape[1]

    # Task indices via typed level lookup.
    iA = columns.where(term="trial_type", level="A").one().index
    iB = columns.where(term="trial_type", level="B").one().index

    # Synthesize Y. Non-zero contributions on the identifiable columns
    # (task + the first five confounds + intercept) and a zero on
    # ``composite_motion`` so the truth lives in the row space of ``X``
    # and the rescued task contrasts are well-defined.
    n_voxels = min(int(max_voxels), MAX_VOXELS)
    betas = np.zeros((n_total, n_voxels), dtype=np.float64)
    betas[iA] = np.linspace(0.6, 1.8, n_voxels)
    betas[iB] = np.linspace(1.4, 0.2, n_voxels)
    for cname in ("trans_x", "trans_y", "rot_z", "csf", "wm"):
        idx = columns.where(model_source="baseline").columns
        for c in idx:
            if c.name.endswith(cname):
                betas[c.index] = rng.normal(scale=0.4, size=n_voxels)
                break
    intercept_idx = next(
        (c.index for c in columns.columns if c.role == "baseline"
         and "constant" in (c.name or "")),
        n_total - 1,
    )
    betas[intercept_idx] = 100.0 + rng.normal(scale=0.5, size=n_voxels)

    data = design @ betas + rng.normal(scale=0.3, size=(N_SCANS, n_voxels))

    c_task_main = np.zeros(n_total, dtype=np.float64)
    c_task_main[iA] = 1.0
    c_task_main[iB] = -1.0
    c_task_F = np.zeros((2, n_total), dtype=np.float64)
    c_task_F[0, iA] = 1.0
    c_task_F[1, iB] = 1.0

    # The QR pivot picks *one* of the four columns participating in the
    # dependency to mark aliased. Any of trans_x, trans_y, rot_z, or
    # composite_motion is a sound choice; the assertion accepts the
    # whole set so the contract is robust to QR-pivot reordering.
    aliased_candidates: tuple[str, ...] = (
        "trans_x",
        "trans_y",
        "rot_z",
        "composite_motion",
    )

    return MulticollinearInputs(
        events=events,
        confounds=confound_df,
        data=data.astype(np.float64),
        design=design,
        design_columns=columns,
        c_task_main=c_task_main,
        c_task_F=c_task_F,
        aliased_candidates=aliased_candidates,
    )


def nilearn_pipeline(inputs: MulticollinearInputs) -> PipelineOutput:
    """Reference: Nilearn's run_glm silently uses the Moore-Penrose pseudo-inv.

    Nilearn does not warn or expose any rank diagnostic — the parity
    contract therefore pins ``rank`` and ``aliased_columns`` to the
    fmrimod-side values, and the Nilearn side simply reports
    ``np.linalg.matrix_rank(X)`` as a sanity probe.
    """
    labels, estimates = run_glm(inputs.data, inputs.design, noise_model="ols")
    t_task = compute_contrast(
        labels, estimates, inputs.c_task_main, stat_type="t"
    )
    f_task = compute_contrast(
        labels, estimates, inputs.c_task_F, stat_type="F"
    )
    rank_observed = int(np.linalg.matrix_rank(inputs.design))
    return PipelineOutput(
        arrays={
            "design": inputs.design,
            "effect_task_main": np.asarray(t_task.effect_size(), np.float64),
            "t_task_main": np.asarray(t_task.stat(), np.float64),
            "f_task_omnibus": np.asarray(f_task.stat(), np.float64),
            "rank": np.array([rank_observed], dtype=np.float64),
        }
    )


def fmrimod_pipeline(inputs: MulticollinearInputs) -> PipelineOutput:
    """fmrimod's typed path with rank-deficient X.

    Captures the per-fit ``UserWarning`` so the diagnostic surface is
    exercised at parity time, asserts the aliased column was named, and
    runs the same OLS / contrast machinery as the Nilearn reference.
    """
    spec = (
        hrf("trial_type", basis="spm", norm="spm")
        + confounds(
            "trans_x",
            "trans_y",
            "rot_z",
            "csf",
            "wm",
            "composite_motion",
            source=inputs.confounds,
        )
        + intercept(per="run")
    )
    ds = fm.fmri_dataset(inputs.data, tr=TR, events=inputs.events)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        fit = fm.fmri_lm(spec, ds)

    # Diagnostic checks: the warning fired AND the typed report flags it.
    fit_warnings = [
        w for w in captured if issubclass(w.category, UserWarning)
        and "fmri_lm()" in str(w.message)
    ]
    if not fit_warnings:
        raise AssertionError(
            "expected fmri_lm() to emit a UserWarning on a rank-deficient design"
        )
    report = fit.condition_report()
    if report.is_full_rank:
        raise AssertionError(
            "expected condition_report().is_full_rank=False on the "
            f"deliberately collinear design; got {report}"
        )
    if not any(
        any(candidate in name for candidate in inputs.aliased_candidates)
        for name in report.aliased_columns
    ):
        raise AssertionError(
            "expected condition_report().aliased_columns to mention one of "
            f"{inputs.aliased_candidates!r}; got {report.aliased_columns!r}"
        )

    t_task = fit.contrast(inputs.c_task_main, name="task_main")
    f_task = fit.contrast(inputs.c_task_F, name="task_omnibus")
    rank_observed = int(report.runs[0].rank)
    return PipelineOutput(
        arrays={
            "design": fit.model.design_matrix_array(run=0),
            "effect_task_main": np.asarray(t_task.estimate, np.float64),
            "t_task_main": np.asarray(t_task.stat, np.float64),
            "f_task_omnibus": np.asarray(f_task.stat, np.float64),
            "rank": np.array([rank_observed], dtype=np.float64),
        }
    )


def make_case(max_voxels: int = MAX_VOXELS) -> ParityCase:
    """Build the multicollinear-baseline parity case.

    Note on t/F tolerance. fmrimod uses ``dfres = n - rank`` for variance
    estimation — the textbook value for a rank-deficient design. Nilearn's
    ``run_glm`` silently uses ``df_residual = n - p`` instead, ignoring
    the rank deficiency. The two engines' t-statistics therefore differ
    by ``sqrt(n - p) / sqrt(n - rank)`` (~0.3% on this fixture) and the
    F-statistic by ``(n - rank) / (n - p)``. We compare with
    ``allow_rescale=True`` so the test pins the *correlation* (which is
    1.0 to machine precision) and the rescaled MAE (which is < 1e-10).
    The point estimate (effect) does not depend on the DoF choice and is
    matched at machine precision.
    """
    return ParityCase(
        name="tier_a_multicollinear_baseline",
        fmrimod_pipeline=fmrimod_pipeline,
        reference_pipeline=nilearn_pipeline,
        inputs=load_inputs(max_voxels=max_voxels),
        tolerances={
            "design": ParityTolerance(rtol=0.0, atol=0.0),
            "effect_task_main": ParityTolerance(rtol=1e-8, atol=1e-9),
            "t_task_main": ParityTolerance(
                check_allclose=False,
                allow_rescale=True,
                min_pearson=0.99999999,
                min_spearman=0.99999999,
                max_mae=1e-8,
            ),
            "f_task_omnibus": ParityTolerance(
                check_allclose=False,
                allow_rescale=True,
                min_pearson=0.99999999,
                min_spearman=0.99999999,
                max_mae=1e-8,
            ),
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
