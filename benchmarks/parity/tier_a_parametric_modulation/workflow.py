"""Parametric-modulation first-level parity against Nilearn.

This case targets an overlap slice not exercised by the other Tier A
workflows: a *categorical* main effect (trial_type ∈ {A, B}) combined with a
*continuous* parametric modulator (centered reaction time) for the same
trial type, in a single first-level fit.

Why this is a stress test
-------------------------
Nilearn's :class:`~nilearn.glm.first_level.FirstLevelModel` reads an events
DataFrame with optional ``modulation`` column. That column is *per row*, so a
design that needs both a main effect *and* a parametric modulator for the
same ``trial_type`` cannot be expressed by the events parser alone — the
user must either duplicate every event row with a renamed ``trial_type``
label, or bypass the events parser entirely and pre-build the parametric
regressors with
:func:`nilearn.glm.first_level.hemodynamic_models.compute_regressor`.

This workflow takes the latter path on the Nilearn side: it is the more
honest reference, free from the row-duplication trick and its naming
contortions.

fmrimod handles the same scenario as a single composed spec::

    hrf("trial_type", basis="spm", norm="spm")
      + hrf("trial_type", "rt_c", basis="spm", norm="spm")

The continuous-by-categorical interaction term yields one parametric
regressor per trial-type level (deterministic alphabetical order), and the
typed contrast vectors index the resulting columns without
the user tracking design-matrix layout by hand. The user-code surface is the
canonical three-line path: ``fmri_dataset → fmri_lm → fit.contrast``.

What we compare
---------------
Per-column design parity (the four task regressors) and three contrasts:

- ``main_A_minus_B``: t-contrast on the main-effect columns,
- ``param_A_minus_B``: t-contrast on the rt-modulator difference,
- ``param_omnibus``: joint F-contrast over both parametric columns.

The SPM HRF parameterization gap (``scipy.stats.gamma.pdf`` vs the SPM
``exp(-t)`` form) is the same residual present in
``tier_a_spm_auditory``; the design tolerances therefore use the same
``allow_rescale`` + Pearson/Spearman envelope as that case.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn.glm.contrasts import compute_contrast
from nilearn.glm.first_level import run_glm
from nilearn.glm.first_level.hemodynamic_models import compute_regressor
from numpy.typing import NDArray

import fmrimod as fm
from cross_testing.harness import (
    ParityCase,
    ParityTolerance,
    PipelineOutput,
    render,
    run,
)
from fmrimod.spec import hrf

Array = NDArray[np.float64]

TR = 2.0
N_SCANS = 160
MAX_VOXELS = 1024
TRIAL_TYPES: tuple[str, ...] = ("A", "B")


@dataclass(frozen=True)
class ParametricInputs:
    """Synthetic events + BOLD with a true parametric-modulation effect."""

    events: pd.DataFrame
    data: Array
    frame_times: Array


def _make_events(seed: int) -> pd.DataFrame:
    """Build deterministic alternating A/B trials with a continuous rt."""
    rng = np.random.default_rng(seed)
    n_trials = 24
    onsets = np.linspace(8.0, N_SCANS * TR - 24.0, n_trials, dtype=np.float64)
    labels = np.array(list(TRIAL_TYPES) * (n_trials // len(TRIAL_TYPES)))
    rt = rng.uniform(0.4, 1.2, n_trials).astype(np.float64)
    return pd.DataFrame(
        {
            "onset": onsets,
            "duration": np.zeros(n_trials, dtype=np.float64),
            "trial_type": labels,
            "rt": rt,
            "rt_c": rt - rt.mean(),
            "run": 1,
        }
    )


def _nilearn_design(
    events: pd.DataFrame,
    frame_times: Array,
) -> tuple[dict[str, Array], Array]:
    """Pre-build the four parametric task columns plus an intercept.

    Returns a ``(named_columns, design)`` pair. ``named_columns`` exposes the
    individual task regressors for column-by-column parity; ``design`` is the
    full ``(N_SCANS, 5)`` first-level design matrix used by Nilearn's GLM.
    """
    cols: dict[str, Array] = {}
    for label in TRIAL_TYPES:
        sub = events.loc[events.trial_type == label]
        onsets = sub.onset.to_numpy(dtype=np.float64)
        durations = sub.duration.to_numpy(dtype=np.float64)
        rt_centered = sub.rt_c.to_numpy(dtype=np.float64)

        main_amp = np.ones_like(rt_centered)
        main_exp = np.vstack([onsets, durations, main_amp])
        main_col, _ = compute_regressor(
            main_exp, "spm", frame_times, con_id=f"main_{label}"
        )
        cols[f"regressor_main_{label}"] = np.asarray(
            main_col, dtype=np.float64
        ).ravel()

        param_exp = np.vstack([onsets, durations, rt_centered])
        param_col, _ = compute_regressor(
            param_exp, "spm", frame_times, con_id=f"param_{label}"
        )
        cols[f"regressor_param_{label}"] = np.asarray(
            param_col, dtype=np.float64
        ).ravel()

    design = np.column_stack(
        [
            cols[f"regressor_main_{TRIAL_TYPES[0]}"],
            cols[f"regressor_main_{TRIAL_TYPES[1]}"],
            cols[f"regressor_param_{TRIAL_TYPES[0]}"],
            cols[f"regressor_param_{TRIAL_TYPES[1]}"],
            np.ones(frame_times.size, dtype=np.float64),
        ]
    )
    return cols, design


def load_inputs(
    max_voxels: int = MAX_VOXELS, seed: int = 20260513
) -> ParametricInputs:
    """Synthesize a BOLD time-series with known main + parametric effects."""

    rng = np.random.default_rng(seed)
    events = _make_events(seed)
    # fmri_dataset(..., tr=TR) defaults its SamplingFrame to a mid-TR grid
    # (start_time=TR/2). Match it on the Nilearn side so both pipelines
    # evaluate the same convolution grid.
    frame_times = np.arange(N_SCANS, dtype=np.float64) * TR + TR / 2.0

    _, X = _nilearn_design(events, frame_times)
    n_voxels = min(int(max_voxels), MAX_VOXELS)

    betas = np.zeros((X.shape[1], n_voxels), dtype=np.float64)
    betas[0] = np.linspace(0.6, 1.8, n_voxels)   # main A
    betas[1] = np.linspace(1.4, 0.2, n_voxels)   # main B
    betas[2] = np.linspace(0.0, 0.6, n_voxels)   # param A
    betas[3] = np.linspace(0.5, -0.1, n_voxels)  # param B
    betas[4] = 100.0 + rng.normal(scale=0.5, size=n_voxels)
    data = X @ betas + rng.normal(scale=0.3, size=(N_SCANS, n_voxels))
    return ParametricInputs(
        events=events, data=data.astype(np.float64), frame_times=frame_times
    )


def nilearn_pipeline(inputs: ParametricInputs) -> PipelineOutput:
    """Pre-build parametric regressors + Nilearn low-level GLM."""

    cols, X = _nilearn_design(inputs.events, inputs.frame_times)
    labels, estimates = run_glm(inputs.data, X, noise_model="ols")

    c_main = np.array([1.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    c_param_diff = np.array([0.0, 0.0, 1.0, -1.0, 0.0], dtype=np.float64)
    c_param_F = np.array(
        [[0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0]],
        dtype=np.float64,
    )

    t_main = compute_contrast(labels, estimates, c_main, stat_type="t")
    t_param_diff = compute_contrast(labels, estimates, c_param_diff, stat_type="t")
    f_param = compute_contrast(labels, estimates, c_param_F, stat_type="F")

    arrays: dict[str, Array] = dict(cols)
    arrays.update(
        {
            "effect_main_A_minus_B": np.asarray(t_main.effect_size(), np.float64),
            "t_main_A_minus_B": np.asarray(t_main.stat(), np.float64),
            "effect_param_A_minus_B": np.asarray(
                t_param_diff.effect_size(), np.float64
            ),
            "t_param_A_minus_B": np.asarray(t_param_diff.stat(), np.float64),
            "f_param_omnibus": np.asarray(f_param.stat(), np.float64),
        }
    )
    return PipelineOutput(arrays=arrays)


def _task_column_indices(
    event_model: object,
) -> tuple[int, int, int, int]:
    """Locate the four task columns in fmrimod's realized event design.

    The two HrfTerms compose deterministically: the main term resolves to
    one column per ``trial_type`` level in alphabetical level order, then the
    interaction term resolves to one column per level in the same order.
    This helper filters ``column_facts`` by owning term and sorts by index
    so it does not rely on the fragile mixed-interaction column-name format.
    """

    facts = list(getattr(event_model, "column_facts"))
    main = sorted(
        (f for f in facts if f["term"] == "trial_type"),
        key=lambda f: f["index"],
    )
    param = sorted(
        (f for f in facts if f["term"] == "trial_type:rt_c"),
        key=lambda f: f["index"],
    )
    if len(main) != 2 or len(param) != 2:
        raise RuntimeError(
            "Expected exactly two main and two parametric columns; got "
            f"main={[f['name'] for f in main]} "
            f"param={[f['name'] for f in param]}"
        )
    return (
        int(main[0]["index"]),
        int(main[1]["index"]),
        int(param[0]["index"]),
        int(param[1]["index"]),
    )


def fmrimod_pipeline(inputs: ParametricInputs) -> PipelineOutput:
    """Three-line fmrimod path: dataset → fit → contrast.

    The same parametric-modulation design is expressed as one composed spec
    instead of pre-computed regressors. ``hrf("trial_type", "rt_c", ...)`` is
    a continuous-by-categorical interaction term, evaluated per
    ``trial_type`` level.
    """

    spec = (
        hrf("trial_type", basis="spm", norm="spm")
        + hrf("trial_type", "rt_c", basis="spm", norm="spm")
    )
    ds = fm.fmri_dataset(inputs.data, tr=TR, events=inputs.events)
    fit = fm.fmri_lm(spec, ds)

    em = fit.model.event_model
    design = np.asarray(em.design_matrix, dtype=np.float64)
    iA, iB, pA, pB = _task_column_indices(em)
    full_n_cols = fit.model.design_matrix_array(run=0).shape[1]

    cols = {
        "regressor_main_A": design[:, iA].copy(),
        "regressor_main_B": design[:, iB].copy(),
        "regressor_param_A": design[:, pA].copy(),
        "regressor_param_B": design[:, pB].copy(),
    }

    c_main = np.zeros(full_n_cols, dtype=np.float64)
    c_main[iA] = 1.0
    c_main[iB] = -1.0
    c_param_diff = np.zeros(full_n_cols, dtype=np.float64)
    c_param_diff[pA] = 1.0
    c_param_diff[pB] = -1.0
    c_param_F = np.zeros((2, full_n_cols), dtype=np.float64)
    c_param_F[0, pA] = 1.0
    c_param_F[1, pB] = 1.0

    t_main = fit.contrast(c_main, name="main_A_minus_B")
    t_param_diff = fit.contrast(c_param_diff, name="param_A_minus_B")
    f_param = fit.contrast(c_param_F, name="param_omnibus")

    arrays: dict[str, Array] = dict(cols)
    arrays.update(
        {
            "effect_main_A_minus_B": np.asarray(t_main.estimate, np.float64),
            "t_main_A_minus_B": np.asarray(t_main.stat, np.float64),
            "effect_param_A_minus_B": np.asarray(t_param_diff.estimate, np.float64),
            "t_param_A_minus_B": np.asarray(t_param_diff.stat, np.float64),
            "f_param_omnibus": np.asarray(f_param.stat, np.float64),
        }
    )
    return PipelineOutput(arrays=arrays)


def make_case(max_voxels: int = MAX_VOXELS) -> ParityCase:
    """Build the parametric-modulation parity case.

    The four regressor-column outputs use ``allow_rescale`` + Pearson/Spearman
    bands — Nilearn evaluates the SPM HRF via ``scipy.stats.gamma.pdf`` while
    fmrimod evaluates the SPM ``exp(-t) * (a1*t^P1 - C*t^P2)`` form at the
    same mid-TR grid, leaving the same small (~4%) amplitude residual handled
    in-band by ``tier_a_spm_auditory``. Effect/t/F statistics are less
    scale-sensitive and gate on tight MAE bands.
    """

    regressor_tolerance = ParityTolerance(
        check_allclose=False,
        allow_rescale=True,
        min_pearson=0.99,
        min_spearman=0.98,
        max_abs=0.10,
    )

    return ParityCase(
        name="tier_a_parametric_modulation",
        fmrimod_pipeline=fmrimod_pipeline,
        reference_pipeline=nilearn_pipeline,
        inputs=load_inputs(max_voxels=max_voxels),
        tolerances={
            "regressor_main_A": regressor_tolerance,
            "regressor_main_B": regressor_tolerance,
            "regressor_param_A": regressor_tolerance,
            "regressor_param_B": regressor_tolerance,
            "effect_main_A_minus_B": ParityTolerance(
                check_allclose=False,
                min_pearson=0.99,
                min_spearman=0.98,
                max_mae=0.50,
            ),
            "t_main_A_minus_B": ParityTolerance(
                check_allclose=False,
                min_pearson=0.99,
                min_spearman=0.98,
                max_mae=0.30,
            ),
            "effect_param_A_minus_B": ParityTolerance(
                check_allclose=False,
                min_pearson=0.99,
                min_spearman=0.98,
                max_mae=0.50,
            ),
            "t_param_A_minus_B": ParityTolerance(
                check_allclose=False,
                min_pearson=0.99,
                min_spearman=0.98,
                max_mae=0.30,
            ),
            "f_param_omnibus": ParityTolerance(
                check_allclose=False,
                min_pearson=0.99,
                min_spearman=0.98,
                max_mae=0.50,
            ),
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
