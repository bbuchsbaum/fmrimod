"""BIDS-realistic confounds parity against Nilearn.

Every real fMRI analysis loads a confounds file alongside the BOLD
data: six motion parameters, framewise displacement, CSF and white-
matter signals, aCompCor components. fmriprep and FitLins ship
exactly this column layout (``trans_x``, ``trans_y``, ``trans_z``,
``rot_x``, ``rot_y``, ``rot_z``, ``framewise_displacement``,
``csf``, ``white_matter``, ...) in ``*_desc-confounds_timeseries.tsv``
files. This case stresses how fmrimod's typed ``confounds(...)``
term integrates that real-world signal into the design and how the
resulting contrasts compare to Nilearn's ``add_regs`` /
``confounds`` argument path.

The workflow is structured around the standard motion-omnibus
analysis: in addition to the task contrasts (A−B, joint F over
task), we run the **6-DF motion F-test** that asks "is any of the
six rigid-body motion regressors significantly explaining BOLD
variance?" — the standard QC contrast to flag motion-driven
artifacts at the voxel level.

Pattern B parity claim
----------------------
fmrimod realises the design (task convolutions + drift + confound
passthrough) via the typed spec; both engines solve OLS on that
**same** ``X``. Eight outputs are compared:

- ``effect_A_minus_B`` / ``t_A_minus_B``: condition contrast.
- ``effect_fd`` / ``t_fd``: framewise-displacement nuisance beta
  (single column, useful for QC reports — "how much does FD
  predict the BOLD time series in each voxel?").
- ``f_task_2df``: joint F over the two task columns.
- ``f_motion_6df``: joint F over the six rigid-body motion
  regressors — the motion-omnibus QC contrast.
- ``rank`` and ``design`` (bitwise).

Pattern A side-check: when both engines build the design
independently from the same events + confounds DataFrames, the
**confound passthrough columns are bitwise-equal** (no
convolution, just column attachment). This pins fmrimod's
confound-plumbing correctness against Nilearn's add_regs path
without needing the SPM-canonical shape conventions to align.

Pain points logged for follow-up
--------------------------------

Three ergonomic gaps surfaced while wiring this and are pinned in
``tests/spec/test_confounds_pain_points.py``:

1. **No distinct ``role="confound"``** — confound columns share
   ``role="baseline"`` with intercept and drift. Filtering them
   from the column registry requires name-suffix parsing
   (``c.name.endswith("trans_x")``) rather than typed lookup.
2. **No ``where(name="trans_x")`` direct match** — names get
   prefixed (``"nuis_run1_trans_x"``) so the user's original
   column name from the events table is not addressable in the
   typed lookup.
3. **Multi-run confounds via the typed spec** — ``confounds(
   source=concat_df)`` fails on multi-run designs with
   "Length of nuisance_list (1) must equal number of blocks (N)";
   no public typed-spec path takes a list-of-per-run DataFrames.
   The underlying ``baseline_model(nuisance_list=[df1, df2])``
   supports it; the gap is at the typed Spec surface.
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
from fmrimod.spec import confounds, drift, hrf, intercept

Array = NDArray[np.float64]

TR = 2.0
N_SCANS = 200
MAX_VOXELS = 1024
TRIAL_TYPES: tuple[str, ...] = ("A", "B")

# Rigid-body motion regressors as named in fmriprep / FitLins TSVs.
MOTION_REGRESSORS: tuple[str, ...] = (
    "trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z",
)
# Additional nuisance regressors typically included in real workflows.
OTHER_CONFOUNDS: tuple[str, ...] = (
    "framewise_displacement", "csf", "white_matter",
)
ALL_CONFOUNDS: tuple[str, ...] = MOTION_REGRESSORS + OTHER_CONFOUNDS


@dataclass(frozen=True)
class ConfoundsInputs:
    """Shared inputs for the realistic-confounds parity case."""

    events: pd.DataFrame
    confounds_df: pd.DataFrame
    data: Array
    design: Array
    design_columns: DesignColumns
    task_indices: dict[str, int]
    motion_indices: tuple[int, ...]
    fd_index: int
    c_t_A_minus_B: Array
    c_t_fd: Array
    c_f_task_2df: Array
    c_f_motion_6df: Array


def _make_events(seed: int) -> pd.DataFrame:
    """A/B trials, eight each, jittered onsets."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for cond_idx, trial_type in enumerate(TRIAL_TYPES):
        grid = np.linspace(
            14.0 + 2.0 * cond_idx,
            N_SCANS * TR - 30.0,
            8,
            dtype=np.float64,
        )
        jitter = rng.uniform(-1.0, 1.0, len(grid))
        for onset in grid + jitter:
            rows.append(
                {
                    "onset": float(onset),
                    "duration": 0.0,
                    "trial_type": trial_type,
                    "run": 1,
                }
            )
    return (
        pd.DataFrame(rows).sort_values("onset").reset_index(drop=True)
    )


def _make_confounds_df(seed: int) -> pd.DataFrame:
    """Realistic motion + nuisance time series mirroring fmriprep TSV layout.

    Motion translations in millimetres (typical scanner-frame std ≈ 0.3 mm),
    rotations in radians (typical std ≈ 0.005 rad), framewise displacement
    summary in millimetres (uniformly distributed in [0, 0.5] for the
    healthy-low-motion case). CSF and white-matter mean signals are
    z-scored physiological time courses. All columns are zero-mean so
    the confound regressors do not absorb the GLM intercept.
    """
    rng = np.random.default_rng(seed + 11)
    n = N_SCANS

    # Drift the motion params so they correlate with low-frequency BOLD —
    # makes the F-test non-trivial.
    t = np.arange(n) * TR
    drift_template = np.sin(2 * np.pi * t / 90.0)  # 90-second period

    df = pd.DataFrame({
        "trans_x": 0.2 * drift_template + 0.15 * rng.normal(size=n),
        "trans_y": 0.15 * np.cos(2 * np.pi * t / 75.0) + 0.15 * rng.normal(size=n),
        "trans_z": 0.1 * rng.normal(size=n),
        "rot_x":   0.003 * rng.normal(size=n),
        "rot_y":   0.003 * rng.normal(size=n),
        "rot_z":   0.003 * rng.normal(size=n),
        "framewise_displacement": np.abs(0.1 * rng.normal(size=n)) + 0.05,
        "csf":     rng.normal(size=n),
        "white_matter": rng.normal(size=n),
    })
    # Demean each column — confound regressors should not absorb the
    # intercept, and demeaning is the standard fmriprep convention.
    for col in df.columns:
        df[col] = df[col] - df[col].mean()
    return df


def _realize_design(
    events: pd.DataFrame, confounds_df: pd.DataFrame
) -> tuple[Array, DesignColumns]:
    """Build the design via the typed spec."""
    spec = (
        hrf("trial_type", basis="spm", norm="spm")
        + confounds(*ALL_CONFOUNDS, source=confounds_df)
        + drift("poly", degree=2)
        + intercept(per="run")
    )
    dummy = fm.fmri_dataset(
        np.zeros((N_SCANS, 1), dtype=np.float64),
        tr=TR,
        events=events,
        slice_timing_offset=0.0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        fit = fm.fmri_lm(spec, dummy, engine="concat")
    design = np.asarray(
        fit.model.design_matrix_array(run=None), dtype=np.float64
    )
    return design, fit.design_columns()


def _find_confound_index(columns: DesignColumns, name: str) -> int:
    """Look up a confound column by suffix-matching its user-visible name.

    Pinned as a pain point: fmrimod prefixes confound columns with
    ``"nuis_runK_"`` so the original DataFrame column name is not
    addressable directly via ``cols.where(name=name)``. Until a
    proper typed lookup lands (separate `role="confound"` plus
    user-name preservation), suffix matching is the recommended
    workaround.
    """
    for c in columns.columns:
        if (c.name or "").endswith(name):
            return c.index
    raise KeyError(
        f"confound column ending in {name!r} not found in design; "
        f"available baseline names: "
        f"{[col.name for col in columns.columns if col.role == 'baseline']}"
    )


def _build_contrasts(
    task_indices: dict[str, int],
    motion_indices: tuple[int, ...],
    fd_index: int,
    n_total: int,
) -> tuple[Array, Array, Array, Array]:
    """Four headline contrasts for the realistic-confounds workflow."""
    c_t_AB = np.zeros(n_total, dtype=np.float64)
    c_t_AB[task_indices["A"]] = +1.0
    c_t_AB[task_indices["B"]] = -1.0

    c_t_fd = np.zeros(n_total, dtype=np.float64)
    c_t_fd[fd_index] = 1.0

    c_f_task = np.zeros((len(TRIAL_TYPES), n_total), dtype=np.float64)
    for row, name in enumerate(TRIAL_TYPES):
        c_f_task[row, task_indices[name]] = 1.0

    c_f_motion = np.zeros((len(motion_indices), n_total), dtype=np.float64)
    for row, idx in enumerate(motion_indices):
        c_f_motion[row, idx] = 1.0

    return c_t_AB, c_t_fd, c_f_task, c_f_motion


def load_inputs(
    max_voxels: int = MAX_VOXELS, seed: int = 20260520
) -> ConfoundsInputs:
    """Synthesize a single-run BOLD time series with task + motion structure.

    The generating model places:
    - Modest task amplitudes (so A−B is detectable above noise).
    - Non-zero coupling to the framewise-displacement column (so the
      FD nuisance contrast has non-trivial signal — simulating a
      voxel where head motion leaves an artifact).
    - Zero coupling to the motion params themselves (so the
      6-DF motion F null distribution is correct).
    """
    rng = np.random.default_rng(seed)
    events = _make_events(seed)
    confounds_df = _make_confounds_df(seed)
    design, columns = _realize_design(events, confounds_df)
    n_total = design.shape[1]

    # Typed task lookup.
    task_indices = {
        t: columns.where(term="trial_type", level=t).one().index
        for t in TRIAL_TYPES
    }
    motion_indices = tuple(
        _find_confound_index(columns, name) for name in MOTION_REGRESSORS
    )
    fd_index = _find_confound_index(columns, "framewise_displacement")

    n_voxels = min(int(max_voxels), MAX_VOXELS)
    betas = np.zeros((n_total, n_voxels), dtype=np.float64)
    voxel_ramp = np.linspace(0.5, 1.5, n_voxels, dtype=np.float64)
    betas[task_indices["A"]] = 0.95 * voxel_ramp
    betas[task_indices["B"]] = 0.30 * voxel_ramp
    # FD nuisance coupling: half the voxels are FD-sensitive.
    betas[fd_index, : n_voxels // 2] = 0.6
    # All motion params get zero true coupling (so the F-test is
    # under H0 for the workflow synthetic data).
    for idx in motion_indices:
        betas[idx] = 0.0
    # Baseline drift / intercept.
    for c in columns.columns:
        if c.role == "baseline" and "constant" in (c.name or ""):
            betas[c.index] = 100.0 + rng.normal(scale=0.3, size=n_voxels)
        elif c.role == "baseline" and "poly" in (c.name or ""):
            betas[c.index] = rng.normal(scale=0.15, size=n_voxels)

    data = design @ betas + rng.normal(scale=0.35, size=(N_SCANS, n_voxels))

    c_t_AB, c_t_fd, c_f_task, c_f_motion = _build_contrasts(
        task_indices, motion_indices, fd_index, n_total
    )
    return ConfoundsInputs(
        events=events,
        confounds_df=confounds_df,
        data=data.astype(np.float64),
        design=design,
        design_columns=columns,
        task_indices=task_indices,
        motion_indices=motion_indices,
        fd_index=fd_index,
        c_t_A_minus_B=c_t_AB,
        c_t_fd=c_t_fd,
        c_f_task_2df=c_f_task,
        c_f_motion_6df=c_f_motion,
    )


def nilearn_pipeline(inputs: ConfoundsInputs) -> PipelineOutput:
    """Reference: ``run_glm`` on the fmrimod-realised confounds-aware design."""
    labels, estimates = run_glm(inputs.data, inputs.design, noise_model="ols")
    t_AB = compute_contrast(
        labels, estimates, inputs.c_t_A_minus_B, stat_type="t"
    )
    t_fd = compute_contrast(
        labels, estimates, inputs.c_t_fd, stat_type="t"
    )
    f_task = compute_contrast(
        labels, estimates, inputs.c_f_task_2df, stat_type="F"
    )
    f_motion = compute_contrast(
        labels, estimates, inputs.c_f_motion_6df, stat_type="F"
    )
    rank_observed = int(np.linalg.matrix_rank(inputs.design))
    return PipelineOutput(
        arrays={
            "design": inputs.design,
            "effect_A_minus_B": np.asarray(t_AB.effect_size(), np.float64),
            "t_A_minus_B": np.asarray(t_AB.stat(), np.float64),
            "effect_fd": np.asarray(t_fd.effect_size(), np.float64),
            "t_fd": np.asarray(t_fd.stat(), np.float64),
            "f_task_2df": np.asarray(f_task.stat(), np.float64),
            "f_motion_6df": np.asarray(f_motion.stat(), np.float64),
            "rank": np.array([rank_observed], dtype=np.float64),
        }
    )


def fmrimod_pipeline(inputs: ConfoundsInputs) -> PipelineOutput:
    """Typed fmrimod path: ``hrf(...) + confounds(*names, source=df) + drift + intercept``.

    The typed-spec line for the confounds is one call:

        confounds("trans_x", "trans_y", ..., source=confounds_df)

    Compare to the Nilearn path: one ``make_first_level_design_matrix``
    call with ``add_regs=df.to_numpy()`` and ``add_reg_names=list(df.columns)``,
    where the user has to maintain the name ordering by hand and
    track confound indices manually for any contrast that involves
    them.
    """
    spec = (
        hrf("trial_type", basis="spm", norm="spm")
        + confounds(*ALL_CONFOUNDS, source=inputs.confounds_df)
        + drift("poly", degree=2)
        + intercept(per="run")
    )
    ds = fm.fmri_dataset(
        inputs.data, tr=TR, events=inputs.events, slice_timing_offset=0.0
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        fit = fm.fmri_lm(spec, ds, engine="concat")

    t_AB = fit.contrast(inputs.c_t_A_minus_B, name="A_minus_B")
    t_fd = fit.contrast(inputs.c_t_fd, name="fd")
    f_task = fit.contrast(inputs.c_f_task_2df, name="task_2df")
    f_motion = fit.contrast(inputs.c_f_motion_6df, name="motion_6df")

    return PipelineOutput(
        arrays={
            "design": fit.model.design_matrix_array(run=None),
            "effect_A_minus_B": np.asarray(t_AB.estimate, np.float64),
            "t_A_minus_B": np.asarray(t_AB.stat, np.float64),
            "effect_fd": np.asarray(t_fd.estimate, np.float64),
            "t_fd": np.asarray(t_fd.stat, np.float64),
            "f_task_2df": np.asarray(f_task.stat, np.float64),
            "f_motion_6df": np.asarray(f_motion.stat, np.float64),
            "rank": np.array(
                [int(fit.condition_report().runs[0].rank)],
                dtype=np.float64,
            ),
        }
    )


def make_case(max_voxels: int = MAX_VOXELS) -> ParityCase:
    """Build the realistic-confounds parity case."""
    return ParityCase(
        name="tier_a_realistic_confounds",
        fmrimod_pipeline=fmrimod_pipeline,
        reference_pipeline=nilearn_pipeline,
        inputs=load_inputs(max_voxels=max_voxels),
        tolerances={
            "design": ParityTolerance(rtol=0.0, atol=0.0),
            "effect_A_minus_B": ParityTolerance(rtol=1e-8, atol=1e-9),
            "t_A_minus_B": ParityTolerance(rtol=1e-7, atol=1e-8),
            "effect_fd": ParityTolerance(rtol=1e-8, atol=1e-9),
            "t_fd": ParityTolerance(rtol=1e-7, atol=1e-8),
            "f_task_2df": ParityTolerance(rtol=1e-7, atol=1e-8),
            "f_motion_6df": ParityTolerance(rtol=1e-7, atol=1e-8),
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
