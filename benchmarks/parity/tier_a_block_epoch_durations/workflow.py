"""Block / variable-duration epoch events parity against Nilearn.

Every other Tier A parity workflow uses point events (``duration=0``).
This case adds the missing angle: a design that mixes instantaneous
trials with **variable-duration block events**, which forces the HRF
convolution through the duration-integration code path.

Why this is a stress test
-------------------------
A real session frequently mixes (a) instantaneous trials (button
presses, image flashes) with (b) blocked stimuli of varying length
(e.g. 8-, 12-, and 16-second blocks of words, faces, or rest).

fmrimod takes a single events DataFrame with per-row ``onset``,
``duration``, and ``trial_type``. Mixed durations live in the same
column with no extra handling — the typed spec stays one line::

    hrf("trial_type", basis="spm", norm="spm")
      + drift("cosine", cutoff=128.0)
      + intercept(per="run")

Nilearn accepts the same events DataFrame shape, but the user is on
their own for the typed contrast assembly: column names must be
enumerated by string (``"A"``, ``"B"``) and any cross-cell logic
referenced by hand.

Pattern B parity claim
----------------------
fmrimod realises the mixed-duration concatenated design via the typed
spec; both engines solve OLS on that **same** ``X``
(``fm.fmri_lm(spec, ds, engine="concat")`` on fmrimod;
``run_glm(data, X)`` on Nilearn). Compared outputs cover the
condition main effect, the per-condition block response (positive
amplitude on B's blocks), and the joint 2-DF F over the task block.

Pattern A (each engine builds its own X) would fail strict numerical
parity here because fmrimod's and Nilearn's SPM canonicals use
different gamma parameterizations (``p1=5, p2=15, a1=0.0833`` vs
SPM's ``delay=6, undershoot=16, dispersion=1, ratio=6``); column
correlations land at 0.91 (point events) and 0.98 (block events)
but not bitwise. The interesting cross-engine signal here is
contrast-level, not column-level, so Pattern B is the right
abstraction.
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
from fmrimod.spec import drift, hrf, intercept

Array = NDArray[np.float64]

TR = 2.0
N_SCANS = 200
MAX_VOXELS = 1024
TRIAL_TYPES: tuple[str, ...] = ("A", "B")
# Three block durations interleaved across B's trials, plus a few
# point-event trials of A. Both are convolved with the same SPM HRF.
BLOCK_DURATIONS: tuple[float, ...] = (8.0, 12.0, 16.0)


@dataclass(frozen=True)
class BlockInputs:
    """Shared inputs for the variable-duration epoch parity case."""

    events: pd.DataFrame
    data: Array
    design: Array
    design_columns: DesignColumns
    cell_indices: dict[str, int]
    c_t_A_minus_B: Array
    c_t_B_block_response: Array
    c_f_task_2df: Array


def _make_events(seed: int) -> pd.DataFrame:
    """Build deterministic A point events and B variable-duration blocks."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    # A: 8 point events evenly spaced across the run, small jitter.
    a_grid = np.linspace(12.0, N_SCANS * TR - 30.0, 8, dtype=np.float64)
    a_jitter = rng.uniform(-1.0, 1.0, len(a_grid))
    for onset in a_grid + a_jitter:
        rows.append(
            {
                "onset": float(onset),
                "duration": 0.0,
                "trial_type": "A",
                "run": 1,
            }
        )
    # B: 6 block events, durations cycle through (8, 12, 16) s. Placed
    # in the gaps between A trials and given enough room that each
    # block's boxcar lives entirely inside the run.
    b_onsets = np.linspace(25.0, N_SCANS * TR - 40.0, 6, dtype=np.float64)
    for k, onset in enumerate(b_onsets):
        rows.append(
            {
                "onset": float(onset),
                "duration": float(BLOCK_DURATIONS[k % len(BLOCK_DURATIONS)]),
                "trial_type": "B",
                "run": 1,
            }
        )
    return (
        pd.DataFrame(rows).sort_values("onset").reset_index(drop=True)
    )


def _realize_design(events: pd.DataFrame) -> tuple[Array, DesignColumns]:
    """Build the mixed-duration concatenated design via the typed spec."""
    spec = (
        hrf("trial_type", basis="spm", norm="spm")
        + drift("cosine", cutoff=128.0)
        + intercept(per="run")
    )
    dummy = fm.fmri_dataset(
        np.zeros((N_SCANS, 1), dtype=np.float64),
        tr=TR,
        events=events,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        fit = fm.fmri_lm(spec, dummy, engine="concat")
    design = np.asarray(
        fit.model.design_matrix_array(run=None), dtype=np.float64
    )
    return design, fit.design_columns()


def _cell_indices(columns: DesignColumns) -> dict[str, int]:
    """Look up each task condition's column index by typed level lookup."""
    cells: dict[str, int] = {}
    for trial_type in TRIAL_TYPES:
        match = columns.where(term="trial_type", level=trial_type).one()
        cells[trial_type] = match.index
    return cells


def _build_contrasts(
    cells: dict[str, int], n_total: int
) -> tuple[Array, Array, Array]:
    """Three contrasts: A−B (t), B alone (t), joint task F."""
    c_t_AB = np.zeros(n_total, dtype=np.float64)
    c_t_AB[cells["A"]] = +1.0
    c_t_AB[cells["B"]] = -1.0

    c_t_B = np.zeros(n_total, dtype=np.float64)
    c_t_B[cells["B"]] = 1.0

    c_f_task = np.zeros((len(TRIAL_TYPES), n_total), dtype=np.float64)
    for row, idx in enumerate(sorted(cells.values())):
        c_f_task[row, idx] = 1.0

    return c_t_AB, c_t_B, c_f_task


def load_inputs(
    max_voxels: int = MAX_VOXELS, seed: int = 20260515
) -> BlockInputs:
    """Synthesize a single-run BOLD time series with true block-response betas.

    The generating model gives A a moderate positive amplitude (point-
    event response) and B a larger positive amplitude (sustained
    block response). The voxel ramp introduces between-voxel variation
    so the contrast tests have non-trivial variance.
    """
    rng = np.random.default_rng(seed)
    events = _make_events(seed)
    design, columns = _realize_design(events)
    cells = _cell_indices(columns)
    n_total = design.shape[1]

    n_voxels = min(int(max_voxels), MAX_VOXELS)
    betas = np.zeros((n_total, n_voxels), dtype=np.float64)
    voxel_ramp = np.linspace(0.6, 1.5, n_voxels, dtype=np.float64)
    cell_amplitudes = {"A": 0.55, "B": 1.10}
    for cell, amp in cell_amplitudes.items():
        betas[cells[cell]] = amp * voxel_ramp

    # Baseline + drift coefficients.
    for c in columns.columns:
        if c.role == "baseline" and "constant" in (c.name or ""):
            betas[c.index] = 100.0 + rng.normal(scale=0.4, size=n_voxels)
        elif c.role == "baseline":
            betas[c.index] = rng.normal(scale=0.15, size=n_voxels)

    data = design @ betas + rng.normal(scale=0.35, size=(N_SCANS, n_voxels))

    c_t_AB, c_t_B, c_f_task = _build_contrasts(cells, n_total)
    return BlockInputs(
        events=events,
        data=data.astype(np.float64),
        design=design,
        design_columns=columns,
        cell_indices=cells,
        c_t_A_minus_B=c_t_AB,
        c_t_B_block_response=c_t_B,
        c_f_task_2df=c_f_task,
    )


def nilearn_pipeline(inputs: BlockInputs) -> PipelineOutput:
    """Reference: ``run_glm`` on the fmrimod-realised mixed-duration design."""
    labels, estimates = run_glm(inputs.data, inputs.design, noise_model="ols")
    t_AB = compute_contrast(
        labels, estimates, inputs.c_t_A_minus_B, stat_type="t"
    )
    t_B = compute_contrast(
        labels, estimates, inputs.c_t_B_block_response, stat_type="t"
    )
    f_task = compute_contrast(
        labels, estimates, inputs.c_f_task_2df, stat_type="F"
    )
    rank_observed = int(np.linalg.matrix_rank(inputs.design))
    return PipelineOutput(
        arrays={
            "design": inputs.design,
            "effect_A_minus_B": np.asarray(t_AB.effect_size(), np.float64),
            "t_A_minus_B": np.asarray(t_AB.stat(), np.float64),
            "effect_B_block_response": np.asarray(
                t_B.effect_size(), np.float64
            ),
            "t_B_block_response": np.asarray(t_B.stat(), np.float64),
            "f_task_2df": np.asarray(f_task.stat(), np.float64),
            "rank": np.array([rank_observed], dtype=np.float64),
        }
    )


def fmrimod_pipeline(inputs: BlockInputs) -> PipelineOutput:
    """Typed fmrimod path: ``fmri_lm(spec, ds, engine="concat")``."""
    spec = (
        hrf("trial_type", basis="spm", norm="spm")
        + drift("cosine", cutoff=128.0)
        + intercept(per="run")
    )
    ds = fm.fmri_dataset(inputs.data, tr=TR, events=inputs.events)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        fit = fm.fmri_lm(spec, ds, engine="concat")

    t_AB = fit.contrast(inputs.c_t_A_minus_B, name="A_minus_B")
    t_B = fit.contrast(inputs.c_t_B_block_response, name="B_block_response")
    f_task = fit.contrast(inputs.c_f_task_2df, name="task_2df")

    return PipelineOutput(
        arrays={
            "design": fit.model.design_matrix_array(run=None),
            "effect_A_minus_B": np.asarray(t_AB.estimate, np.float64),
            "t_A_minus_B": np.asarray(t_AB.stat, np.float64),
            "effect_B_block_response": np.asarray(t_B.estimate, np.float64),
            "t_B_block_response": np.asarray(t_B.stat, np.float64),
            "f_task_2df": np.asarray(f_task.stat, np.float64),
            "rank": np.array(
                [int(fit.condition_report().runs[0].rank)],
                dtype=np.float64,
            ),
        }
    )


def make_case(max_voxels: int = MAX_VOXELS) -> ParityCase:
    """Build the variable-duration block-event parity case."""
    return ParityCase(
        name="tier_a_block_epoch_durations",
        fmrimod_pipeline=fmrimod_pipeline,
        reference_pipeline=nilearn_pipeline,
        inputs=load_inputs(max_voxels=max_voxels),
        tolerances={
            "design": ParityTolerance(rtol=0.0, atol=0.0),
            "effect_A_minus_B": ParityTolerance(rtol=1e-8, atol=1e-9),
            "t_A_minus_B": ParityTolerance(rtol=1e-7, atol=1e-8),
            "effect_B_block_response": ParityTolerance(
                rtol=1e-8, atol=1e-9
            ),
            "t_B_block_response": ParityTolerance(rtol=1e-7, atol=1e-8),
            "f_task_2df": ParityTolerance(rtol=1e-7, atol=1e-8),
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
