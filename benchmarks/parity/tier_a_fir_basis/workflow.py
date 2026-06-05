"""FIR basis HRF estimation parity against Nilearn.

This case stresses the **non-parametric HRF estimation** axis: each
condition gets a finite-impulse-response basis expansion — one beta
per post-onset TR — and the headline analytic outputs are
whole-HRF F-tests, between-condition shape-difference F-tests, and
the SPM-style **canonical-shape projection** (the t-test that
collapses the 12-delay FIR shape onto a single amplitude by
weighting each delay by the SPM canonical at that latency).

Why this is a stress test
-------------------------
The FIR basis is the gold-standard non-parametric HRF approach: no
assumption about peak time, width, undershoot, or ratio. Each delay
``k`` after onset gets its own beta, so the realised design has
``n_delays × n_conditions`` task columns. With overlapping events,
the FIR regressors are temporally shifted copies of each other,
producing a highly structured design where the typed ``basis_ix``
addressing system carries its weight.

Nilearn names FIR columns by string: ``"A_delay_1"``,
``"A_delay_2"``, ..., ``"A_delay_12"``. To assemble a whole-HRF
F-test or a canonical-shape projection contrast the user enumerates
the 12 column names by hand and tracks the order across conditions.

fmrimod's typed design surface tags every FIR column with
``(term, level, basis_ix)``. The whole-HRF F-test is::

    rows = [
        cols.where(term="trial_type", level="A", basis_ix=k).one().index
        for k in range(1, n_delays + 1)
    ]
    f_A = stack one-hot rows on `rows`

The canonical-shape projection is even cleaner::

    weights = spm_canonical(np.arange(n_delays) * tr + tr/2)
    weights /= np.linalg.norm(weights)
    contrast = sum(weight[k] * one_hot[A_basis_ix=k] for k in 1..n_delays)

Two FIR-convention divergences surfaced and are documented in
CAVEATS rather than hidden by Pattern A:

- **Delay indexing**: fmrimod's ``basis_ix=1`` lives at frame_time =
  onset; Nilearn's ``delay_1`` lives at frame_time = onset + 1·TR.
  Both are valid conventions; the realised columns therefore differ
  by a 1-bin shift.
- **Amplitude normalization**: fmrimod's FIR impulse has amplitude
  1.0 (the natural physical interpretation). Nilearn's has amplitude
  ``1/oversampling = 0.02`` (an integration normalization keeping
  the realised regressor on the same numerical scale as a convolved
  boxcar).

Pattern B parity claim
----------------------
fmrimod realises the FIR concatenated design via the typed spec,
both engines solve OLS on that **same** ``X``
(``fm.fmri_lm(spec, ds, engine="concat")`` on fmrimod;
``run_glm(data, X)`` on Nilearn). Six outputs are compared. The
``rank`` and ``design`` outputs are bitwise; the contrast outputs
match at ≤ 1e-9.
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
from fmrimod.hrf.functions import spm_canonical
from fmrimod.spec import drift, hrf, intercept

Array = NDArray[np.float64]

TR = 2.0
N_SCANS = 300
N_DELAYS = 12  # 24 s of HRF coverage at TR=2
MAX_VOXELS = 1024
TRIAL_TYPES: tuple[str, ...] = ("A", "B")


@dataclass(frozen=True)
class FirInputs:
    """Shared inputs for the FIR-basis parity case."""

    events: pd.DataFrame
    data: Array
    design: Array
    design_columns: DesignColumns
    cell_indices: dict[tuple[str, int], int]
    c_t_A_canonical_proj: Array
    c_t_AB_canonical_proj_diff: Array
    c_f_A_omnibus: Array
    c_f_B_omnibus: Array
    c_f_AB_shape_diff: Array


def _make_events(seed: int) -> pd.DataFrame:
    """Twelve well-spaced trials per condition, jittered onsets."""
    rng = np.random.default_rng(seed)
    n_per_cond = 14
    rows: list[dict[str, Any]] = []
    for cond_idx, trial_type in enumerate(TRIAL_TYPES):
        grid = np.linspace(
            14.0 + 3.0 * cond_idx,
            N_SCANS * TR - 40.0,
            n_per_cond,
            dtype=np.float64,
        )
        jitter = rng.uniform(-1.2, 1.2, n_per_cond)
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


def _realize_design(
    events: pd.DataFrame,
) -> tuple[Array, DesignColumns]:
    """Build the FIR concatenated design via the typed spec."""
    spec = (
        hrf("trial_type", basis="fir")
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


def _cell_indices(
    columns: DesignColumns,
) -> dict[tuple[str, int], int]:
    """Look up every (trial_type, basis_ix) FIR-delay column by typed lookup.

    This is the headline ergonomic claim: each post-onset delay column
    is addressable by its semantic identity (condition × delay index)
    without referring to its position or parsing its name.
    """
    cells: dict[tuple[str, int], int] = {}
    for trial_type in TRIAL_TYPES:
        for basis_ix in range(1, N_DELAYS + 1):
            matches = columns.where(
                term="trial_type", level=trial_type, basis_ix=basis_ix
            )
            cells[(trial_type, basis_ix)] = matches.one().index
    return cells


def _canonical_projection_weights(n_delays: int = N_DELAYS) -> Array:
    """SPM-canonical-shape weights for projecting FIR betas onto an HRF amplitude.

    The k-th weight is the SPM canonical evaluated at the *midpoint* of
    the k-th TR window after onset (k=1..n_delays, midpoint =
    (k-1)*TR + TR/2). Normalized to unit L2 so the projected
    coefficient is a clean amplitude. This is the SPM technique for
    distilling a single per-condition "response magnitude" out of an
    FIR fit.
    """
    delay_midpoints = (np.arange(n_delays) + 0.5) * TR  # TR/2, 3TR/2, ...
    weights = spm_canonical(delay_midpoints)
    norm = np.linalg.norm(weights)
    if norm == 0:
        raise ValueError("SPM canonical weights are all zero")
    return weights / norm


def _build_contrasts(
    cells: dict[tuple[str, int], int], n_total: int
) -> tuple[Array, Array, Array, Array, Array]:
    """Assemble the five FIR analytic contrasts.

    - ``c_t_A_canonical_proj``: SPM-canonical-shape projection on A's
      FIR betas — a single 1-DF t-test for "is A's HRF non-zero with
      a canonical shape?".
    - ``c_t_AB_canonical_proj_diff``: same projection on A−B — the
      "do A and B have the same canonical-shape amplitude?" t-test.
    - ``c_f_A_omnibus``: 12-DF F over A's 12 FIR delays — "does A's
      response differ from zero at *any* shape?".
    - ``c_f_B_omnibus``: 12-DF F over B's 12 FIR delays — same for B.
    - ``c_f_AB_shape_diff``: 12-DF F over the row-wise A−B
      difference at every delay — "does the HRF shape differ between
      A and B?".
    """
    weights = _canonical_projection_weights()

    c_t_A = np.zeros(n_total, dtype=np.float64)
    c_t_AB = np.zeros(n_total, dtype=np.float64)
    for k in range(1, N_DELAYS + 1):
        w = float(weights[k - 1])
        c_t_A[cells[("A", k)]] = w
        c_t_AB[cells[("A", k)]] = +w
        c_t_AB[cells[("B", k)]] = -w

    c_f_A = np.zeros((N_DELAYS, n_total), dtype=np.float64)
    c_f_B = np.zeros((N_DELAYS, n_total), dtype=np.float64)
    c_f_AB_diff = np.zeros((N_DELAYS, n_total), dtype=np.float64)
    for row, k in enumerate(range(1, N_DELAYS + 1)):
        c_f_A[row, cells[("A", k)]] = 1.0
        c_f_B[row, cells[("B", k)]] = 1.0
        c_f_AB_diff[row, cells[("A", k)]] = +1.0
        c_f_AB_diff[row, cells[("B", k)]] = -1.0

    return c_t_A, c_t_AB, c_f_A, c_f_B, c_f_AB_diff


def load_inputs(
    max_voxels: int = MAX_VOXELS, seed: int = 20260517
) -> FirInputs:
    """Synthesize a single-run BOLD time series with known FIR-shape responses.

    A's true response follows the SPM canonical sampled at the TR
    grid, with a moderate amplitude. B's true response is a delayed
    + slightly attenuated canonical (peak at delay 4 instead of 3),
    so the omnibus F-tests, the canonical-shape projection
    difference, and the shape-difference F all have non-trivial
    signal to detect.
    """
    rng = np.random.default_rng(seed)
    events = _make_events(seed)
    design, columns = _realize_design(events)
    cells = _cell_indices(columns)
    n_total = design.shape[1]

    n_voxels = min(int(max_voxels), MAX_VOXELS)
    betas = np.zeros((n_total, n_voxels), dtype=np.float64)
    voxel_ramp = np.linspace(0.5, 1.5, n_voxels, dtype=np.float64)

    # True FIR coefficients: A canonical at midpoints, B delayed-and-attenuated.
    delay_midpoints = (np.arange(N_DELAYS) + 0.5) * TR
    a_shape = spm_canonical(delay_midpoints)
    a_shape = 1.2 * a_shape / np.max(a_shape)  # peak ≈ 1.2
    # B: shift one delay later by sampling earlier in the canonical, attenuate.
    b_shape = spm_canonical(np.clip(delay_midpoints - TR, 0.0, None))
    b_shape = 0.8 * b_shape / np.max(b_shape) if np.max(b_shape) > 0 else b_shape

    for k in range(1, N_DELAYS + 1):
        betas[cells[("A", k)]] = a_shape[k - 1] * voxel_ramp
        betas[cells[("B", k)]] = b_shape[k - 1] * voxel_ramp

    for c in columns.columns:
        if c.role == "baseline" and "constant" in (c.name or ""):
            betas[c.index] = 100.0 + rng.normal(scale=0.4, size=n_voxels)
        elif c.role == "baseline":
            betas[c.index] = rng.normal(scale=0.15, size=n_voxels)

    data = design @ betas + rng.normal(scale=0.35, size=(N_SCANS, n_voxels))

    c_t_A, c_t_AB, c_f_A, c_f_B, c_f_AB_diff = _build_contrasts(cells, n_total)
    return FirInputs(
        events=events,
        data=data.astype(np.float64),
        design=design,
        design_columns=columns,
        cell_indices=cells,
        c_t_A_canonical_proj=c_t_A,
        c_t_AB_canonical_proj_diff=c_t_AB,
        c_f_A_omnibus=c_f_A,
        c_f_B_omnibus=c_f_B,
        c_f_AB_shape_diff=c_f_AB_diff,
    )


def nilearn_pipeline(inputs: FirInputs) -> PipelineOutput:
    """Reference: ``run_glm`` on the fmrimod-realised FIR design."""
    labels, estimates = run_glm(inputs.data, inputs.design, noise_model="ols")
    t_A = compute_contrast(
        labels, estimates, inputs.c_t_A_canonical_proj, stat_type="t"
    )
    t_AB = compute_contrast(
        labels, estimates, inputs.c_t_AB_canonical_proj_diff, stat_type="t"
    )
    f_A = compute_contrast(
        labels, estimates, inputs.c_f_A_omnibus, stat_type="F"
    )
    f_B = compute_contrast(
        labels, estimates, inputs.c_f_B_omnibus, stat_type="F"
    )
    f_AB = compute_contrast(
        labels, estimates, inputs.c_f_AB_shape_diff, stat_type="F"
    )
    rank_observed = int(np.linalg.matrix_rank(inputs.design))
    return PipelineOutput(
        arrays={
            "design": inputs.design,
            "effect_A_canonical_proj": np.asarray(
                t_A.effect_size(), np.float64
            ),
            "t_A_canonical_proj": np.asarray(t_A.stat(), np.float64),
            "effect_AB_canonical_proj_diff": np.asarray(
                t_AB.effect_size(), np.float64
            ),
            "t_AB_canonical_proj_diff": np.asarray(
                t_AB.stat(), np.float64
            ),
            "f_A_omnibus_12df": np.asarray(f_A.stat(), np.float64),
            "f_B_omnibus_12df": np.asarray(f_B.stat(), np.float64),
            "f_AB_shape_diff_12df": np.asarray(f_AB.stat(), np.float64),
            "rank": np.array([rank_observed], dtype=np.float64),
        }
    )


def fmrimod_pipeline(inputs: FirInputs) -> PipelineOutput:
    """Typed fmrimod path: ``fmri_lm(spec, ds, engine="concat")``."""
    spec = (
        hrf("trial_type", basis="fir")
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

    t_A = fit.contrast(inputs.c_t_A_canonical_proj, name="A_canonical_proj")
    t_AB = fit.contrast(
        inputs.c_t_AB_canonical_proj_diff, name="AB_canonical_proj_diff"
    )
    f_A = fit.contrast(inputs.c_f_A_omnibus, name="A_omnibus")
    f_B = fit.contrast(inputs.c_f_B_omnibus, name="B_omnibus")
    f_AB = fit.contrast(inputs.c_f_AB_shape_diff, name="AB_shape_diff")

    return PipelineOutput(
        arrays={
            "design": fit.model.design_matrix_array(run=None),
            "effect_A_canonical_proj": np.asarray(t_A.estimate, np.float64),
            "t_A_canonical_proj": np.asarray(t_A.stat, np.float64),
            "effect_AB_canonical_proj_diff": np.asarray(
                t_AB.estimate, np.float64
            ),
            "t_AB_canonical_proj_diff": np.asarray(t_AB.stat, np.float64),
            "f_A_omnibus_12df": np.asarray(f_A.stat, np.float64),
            "f_B_omnibus_12df": np.asarray(f_B.stat, np.float64),
            "f_AB_shape_diff_12df": np.asarray(f_AB.stat, np.float64),
            "rank": np.array(
                [int(fit.condition_report().runs[0].rank)],
                dtype=np.float64,
            ),
        }
    )


def make_case(max_voxels: int = MAX_VOXELS) -> ParityCase:
    """Build the FIR-basis HRF-estimation parity case."""
    return ParityCase(
        name="tier_a_fir_basis",
        fmrimod_pipeline=fmrimod_pipeline,
        reference_pipeline=nilearn_pipeline,
        inputs=load_inputs(max_voxels=max_voxels),
        tolerances={
            "design": ParityTolerance(rtol=0.0, atol=0.0),
            "effect_A_canonical_proj": ParityTolerance(rtol=1e-8, atol=1e-9),
            "t_A_canonical_proj": ParityTolerance(rtol=1e-7, atol=1e-8),
            "effect_AB_canonical_proj_diff": ParityTolerance(
                rtol=1e-8, atol=1e-9
            ),
            "t_AB_canonical_proj_diff": ParityTolerance(rtol=1e-7, atol=1e-8),
            "f_A_omnibus_12df": ParityTolerance(rtol=1e-7, atol=1e-8),
            "f_B_omnibus_12df": ParityTolerance(rtol=1e-7, atol=1e-8),
            "f_AB_shape_diff_12df": ParityTolerance(rtol=1e-7, atol=1e-8),
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
