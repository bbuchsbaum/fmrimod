"""Concatenated multi-run with censored / scrubbed timepoints parity.

Real fMRI has motion-corrupted frames that need to be censored
("scrubbed"). This case stresses the ergonomic gap between fmrimod's
typed ``censor=`` plumbing and the standard Nilearn approach of
appending one-hot spike regressors to the design matrix, against the
mathematically equivalent baseline of row deletion in the
concatenated system.

Why this is a stress test
-------------------------
fmrimod exposes censoring as a dataset-level boolean mask::

    ds = fm.fmri_dataset(..., censor=mask)

The mask is consumed inside the engine: ``fmri_lm(spec, ds,
engine="concat")`` row-deletes the censored frames from both the
realised design and the data before solving, and the residual DoF
falls out as ``n_kept - rank``. The rest of the typed spec is
identical to the un-censored case — nothing in the spec, the
contrast assembly, or the result API changes shape.

The Nilearn idiom requires the user to:

- Either subset the design and data themselves before
  ``run_glm`` (row deletion), or
- Append ``n_censored`` one-hot "spike" columns to the design matrix
  and track which indices they live at so they don't pollute task-
  contrast assembly (orthogonal spike-projection).

Both are mathematically equivalent (row deletion is the limit of
spike-projection as the spike amplitude → ∞), but the spec-
ergonomics gap is real.

Pattern B parity claim
----------------------
fmrimod realises the multi-run concatenated design via the typed
spec, then ``fmri_lm(..., engine="concat")`` honors the censor mask
by row-deleting the marked frames. Nilearn's reference path
explicitly performs the same row deletion on the realised X and Y,
then calls ``run_glm`` on the censored system. The compared outputs
exercise the standard t/F machinery on the censored design plus
the headline ``effective_dfres`` check: fmrimod's reported
``residual_df`` must equal Nilearn's ``n_kept - rank`` exactly.
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
from fmrimod.dataset.constructors import matrix_dataset
from fmrimod.design.columns import DesignColumns
from fmrimod.spec import drift, hrf, intercept

Array = NDArray[np.float64]

TR = 2.0
RUN_LENGTH = 100
N_RUNS = 2
N_SCANS = RUN_LENGTH * N_RUNS
MAX_VOXELS = 1024
TRIAL_TYPES: tuple[str, ...] = ("A", "B")


@dataclass(frozen=True)
class CensorInputs:
    """Shared inputs for the censored-multirun parity case."""

    events: pd.DataFrame
    data: Array
    design: Array
    design_columns: DesignColumns
    censor: NDArray[np.bool_]  # True = drop
    cell_indices: dict[tuple[str, str], int]
    c_t_A_minus_B: Array
    c_f_task_4df: Array


def _make_events(seed: int) -> pd.DataFrame:
    """Deterministic A/B trials, six per condition per run."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for run_idx in range(N_RUNS):
        for cond_idx, trial_type in enumerate(TRIAL_TYPES):
            grid = np.linspace(
                12.0 + 2.0 * cond_idx,
                RUN_LENGTH * TR - 22.0,
                6,
                dtype=np.float64,
            )
            jitter = rng.uniform(-1.0, 1.0, len(grid))
            for onset_run_relative in grid + jitter:
                rows.append(
                    {
                        "onset": float(onset_run_relative),
                        "duration": 0.0,
                        "trial_type": trial_type,
                        "run_label": f"run{run_idx + 1}",
                        "run": run_idx + 1,
                    }
                )
    return (
        pd.DataFrame(rows).sort_values(["run", "onset"]).reset_index(drop=True)
    )


def _make_censor_mask(seed: int) -> NDArray[np.bool_]:
    """A motion-realistic censor pattern: singletons + one short burst per run.

    Approximately 8% of frames dropped. The censored positions are
    *deterministic* so the parity test is reproducible.
    """
    rng = np.random.default_rng(seed)
    censor = np.zeros(N_SCANS, dtype=bool)
    for run_idx in range(N_RUNS):
        offset = run_idx * RUN_LENGTH
        # 4 random singleton spikes per run
        singleton_positions = rng.choice(
            np.arange(5, RUN_LENGTH - 5),
            size=4,
            replace=False,
        )
        censor[offset + singleton_positions] = True
        # One 4-frame burst per run (motion event), placed mid-run
        burst_start = offset + RUN_LENGTH // 2 - 2
        censor[burst_start:burst_start + 4] = True
    return censor


def _realize_design(
    events: pd.DataFrame,
) -> tuple[Array, DesignColumns]:
    """Build the multi-run concatenated design via the typed spec."""
    spec = (
        hrf("trial_type", "run_label", basis="spm", norm="spm")
        + drift("cosine", cutoff=128.0)
        + intercept(per="run")
    )
    dummy = matrix_dataset(
        np.zeros((N_SCANS, 1), dtype=np.float64),
        tr=TR,
        run_length=RUN_LENGTH,
        event_table=events,
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
) -> dict[tuple[str, str], int]:
    """Locate the four ``(trial_type, run_label)`` task columns."""
    cells: dict[tuple[str, str], int] = {}
    for trial_type in TRIAL_TYPES:
        for run_label in ("run1", "run2"):
            level = f"trial_type.{trial_type}_run_label.{run_label}"
            cells[(trial_type, run_label)] = columns.where(
                term="trial_type:run_label", level=level
            ).one().index
    return cells


def _build_contrasts(
    cells: dict[tuple[str, str], int], n_total: int
) -> tuple[Array, Array]:
    """Two contrasts: A−B across runs (t), joint F over all 4 task cells."""
    c_t_AB = np.zeros(n_total, dtype=np.float64)
    for run_label in ("run1", "run2"):
        c_t_AB[cells[("A", run_label)]] = +0.5
        c_t_AB[cells[("B", run_label)]] = -0.5

    c_f_task = np.zeros((len(cells), n_total), dtype=np.float64)
    for row, idx in enumerate(sorted(cells.values())):
        c_f_task[row, idx] = 1.0

    return c_t_AB, c_f_task


def load_inputs(
    max_voxels: int = MAX_VOXELS, seed: int = 20260516
) -> CensorInputs:
    """Synthesize a 2-run BOLD time series with a realistic censor pattern."""
    rng = np.random.default_rng(seed)
    events = _make_events(seed)
    design, columns = _realize_design(events)
    cells = _cell_indices(columns)
    n_total = design.shape[1]
    censor = _make_censor_mask(seed)

    n_voxels = min(int(max_voxels), MAX_VOXELS)
    betas = np.zeros((n_total, n_voxels), dtype=np.float64)
    voxel_ramp = np.linspace(0.4, 1.6, n_voxels, dtype=np.float64)
    cell_amplitudes = {
        ("A", "run1"): 1.20,
        ("A", "run2"): 1.05,
        ("B", "run1"): 0.35,
        ("B", "run2"): 0.40,
    }
    for cell, amp in cell_amplitudes.items():
        betas[cells[cell]] = amp * voxel_ramp
    for c in columns.columns:
        if c.role == "baseline" and "constant" in (c.name or ""):
            betas[c.index] = 100.0 + rng.normal(scale=0.4, size=n_voxels)
        elif c.role == "baseline":
            betas[c.index] = rng.normal(scale=0.15, size=n_voxels)

    data = design @ betas + rng.normal(scale=0.30, size=(N_SCANS, n_voxels))
    # Inject large spike artifacts on the censored frames — this is
    # what scrubbing exists to remove. Any pipeline that fails to
    # honor the censor will see large residuals from these frames.
    data[censor] += rng.normal(scale=8.0, size=(int(censor.sum()), n_voxels))

    c_t_AB, c_f_task = _build_contrasts(cells, n_total)
    return CensorInputs(
        events=events,
        data=data.astype(np.float64),
        design=design,
        design_columns=columns,
        censor=censor,
        cell_indices=cells,
        c_t_A_minus_B=c_t_AB,
        c_f_task_4df=c_f_task,
    )


def nilearn_pipeline(inputs: CensorInputs) -> PipelineOutput:
    """Reference: row-delete censored frames and ``run_glm`` on the rest.

    Mathematically equivalent to the spike-regressor approach (a
    one-hot column per censored frame), but row deletion gives the
    same ``dfres = n_kept - rank`` denominator fmrimod's concat
    engine uses internally, so the t/F denominators match exactly.
    """
    keep = ~inputs.censor
    X_kept = inputs.design[keep]
    Y_kept = inputs.data[keep]
    labels, estimates = run_glm(Y_kept, X_kept, noise_model="ols")
    t_AB = compute_contrast(
        labels, estimates, inputs.c_t_A_minus_B, stat_type="t"
    )
    f_task = compute_contrast(
        labels, estimates, inputs.c_f_task_4df, stat_type="F"
    )
    n_kept = int(keep.sum())
    rank_observed = int(np.linalg.matrix_rank(X_kept))
    dfres_expected = float(n_kept - rank_observed)
    return PipelineOutput(
        arrays={
            "design": inputs.design,  # full, uncensored design
            "effect_A_minus_B": np.asarray(t_AB.effect_size(), np.float64),
            "t_A_minus_B": np.asarray(t_AB.stat(), np.float64),
            "f_task_4df": np.asarray(f_task.stat(), np.float64),
            "n_kept": np.array([float(n_kept)], dtype=np.float64),
            "effective_dfres": np.array([dfres_expected], dtype=np.float64),
            "rank": np.array([float(rank_observed)], dtype=np.float64),
        }
    )


def fmrimod_pipeline(inputs: CensorInputs) -> PipelineOutput:
    """Typed fmrimod path: censor lives on the dataset, engine consumes it.

    Critically, the typed spec and contrast assembly are *identical*
    to the un-censored case. The censor mask passes in via
    ``fm.fmri_dataset(..., censor=mask)`` and the concat engine
    row-deletes inside the solve — the rest of the user-facing code
    doesn't know censoring is happening.
    """
    spec = (
        hrf("trial_type", "run_label", basis="spm", norm="spm")
        + drift("cosine", cutoff=128.0)
        + intercept(per="run")
    )
    keep = ~inputs.censor
    ds = matrix_dataset(
        inputs.data,
        tr=TR,
        run_length=RUN_LENGTH,
        event_table=inputs.events,
        censor=inputs.censor,  # auto-split across runs via run_length
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        fit = fm.fmri_lm(spec, ds, engine="concat")

    t_AB = fit.contrast(inputs.c_t_A_minus_B, name="A_minus_B")
    f_task = fit.contrast(inputs.c_f_task_4df, name="task_4df")

    return PipelineOutput(
        arrays={
            "design": fit.model.design_matrix_array(run=None),
            "effect_A_minus_B": np.asarray(t_AB.estimate, np.float64),
            "t_A_minus_B": np.asarray(t_AB.stat, np.float64),
            "f_task_4df": np.asarray(f_task.stat, np.float64),
            "n_kept": np.array([float(keep.sum())], dtype=np.float64),
            "effective_dfres": np.array(
                [float(fit.residual_df)], dtype=np.float64
            ),
            "rank": np.array(
                [float(fit.condition_report().runs[0].rank)],
                dtype=np.float64,
            ),
        }
    )


def make_case(max_voxels: int = MAX_VOXELS) -> ParityCase:
    """Build the censored-multirun parity case."""
    return ParityCase(
        name="tier_a_censored_concat",
        fmrimod_pipeline=fmrimod_pipeline,
        reference_pipeline=nilearn_pipeline,
        inputs=load_inputs(max_voxels=max_voxels),
        tolerances={
            "design": ParityTolerance(rtol=0.0, atol=0.0),
            "effect_A_minus_B": ParityTolerance(rtol=1e-8, atol=1e-9),
            "t_A_minus_B": ParityTolerance(rtol=1e-7, atol=1e-8),
            "f_task_4df": ParityTolerance(rtol=1e-7, atol=1e-8),
            "n_kept": ParityTolerance(rtol=0.0, atol=0.0),
            "effective_dfres": ParityTolerance(rtol=0.0, atol=0.0),
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
