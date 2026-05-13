"""Concatenated multi-run first-level parity against Nilearn.

This case stresses an angle absent from the other Tier A workflows:
two runs stitched into a *single* concatenated design with run-specific
intercepts, run-wise polynomial drift, and run-specific task cells,
then evaluated with **cross-run contrasts** that explicitly compare
one run's task effect against the other's.

Why this is a stress test
-------------------------
Nilearn's :class:`~nilearn.glm.first_level.FirstLevelModel` is
structurally single-run. The standard multi-run analysis path is to
fit each run independently and then pool with
``compute_fixed_effects`` — but that fixed-effects path *cannot*
express a cross-run difference contrast (e.g. "is condition A's effect
in run 1 different from condition A's effect in run 2?") because the
per-run designs are independently scaled and per-run sigma estimates
do not see each other. The genuine cross-run contrast requires a
single concatenated design with the run-specific intercept and drift
columns sitting next to each other, so the contrast vector can place
weights on both runs simultaneously.

Building that concatenated design by hand in Nilearn means the user
writes the block-diagonal drift basis themselves, threads the
run-specific intercepts into the design matrix, tracks which column
index belongs to which (run, condition) cell, and pre-computes every
contrast vector by hand. The events parser doesn't help.

fmrimod expresses the same design as one typed spec::

    hrf("trial_type", "run_label", basis="spm", norm="spm")
      + drift("poly", degree=2)
      + intercept(per="run")

The `(trial_type × run_label)` factorial term yields one task column
per (run, condition) cell (four cells); ``intercept(per="run")``
delivers per-run intercepts; ``drift("poly", degree=2)`` block-
diagonalises the polynomial drift basis per run. The cells are then
addressable by typed level lookup —
``columns.where(term="trial_type:run_label",
level="trial_type.A_run_label.run1")`` — and the cross-run contrasts
are placed onto the looked-up indices.

Pattern B parity claim
----------------------
fmrimod realises the concatenated design via the typed spec and
solves it through the typed ``fmri_lm(spec, ds, engine="concat")``
path — a single OLS on the stacked ``X`` / ``Y`` with
``dfres = n - rank``. Nilearn's ``run_glm`` is fed the same realised
``X`` for a strict cross-engine comparison. The default ``fmri_lm``
strategy on multi-run datasets is per-run + pool, which gives
identical betas (~1e-12) but estimates variance per-run rather than
over the concatenated residual; the concat engine is the right
choice when contrast vectors span runs.

The compared quantities:

- ``design``: bitwise-equal 10-column concatenated design (4 task +
  4 drift + 2 intercept).
- ``effect_main_A_minus_B``, ``t_main_A_minus_B``: condition main
  effect collapsed across runs.
- ``effect_run_diff_A``, ``t_run_diff_A``: cross-run difference for
  condition A (genuinely requires the concatenated design).
- ``effect_trial_x_run_interaction``,
  ``t_trial_x_run_interaction``: the 2x2 (trial_type × run)
  interaction.
- ``f_task_omnibus``: joint F over the four task cells.
- ``rank``: full concatenated design rank, pinned at full.
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
RUN_LABELS: tuple[str, ...] = ("run1", "run2")


@dataclass(frozen=True)
class MultirunInputs:
    """Shared inputs for the multi-run concatenated parity case."""

    events: pd.DataFrame
    data: Array
    design: Array
    design_columns: DesignColumns
    cell_indices: dict[tuple[str, str], int]
    c_main_A_minus_B: Array
    c_run_diff_A: Array
    c_trial_x_run: Array
    c_task_F: Array


def _make_events(seed: int) -> pd.DataFrame:
    """Build deterministic interleaved A/B trials in each of two runs."""
    rng = np.random.default_rng(seed)
    n_per_cell = 6
    rows: list[dict[str, Any]] = []
    for run_idx in range(N_RUNS):
        run_label = RUN_LABELS[run_idx]
        for cond_idx, trial_type in enumerate(TRIAL_TYPES):
            grid = np.linspace(
                10.0 + 2.5 * cond_idx,
                RUN_LENGTH * TR - 24.0,
                n_per_cell,
                dtype=np.float64,
            )
            jitter = rng.uniform(-1.0, 1.0, n_per_cell)
            for onset_run_relative in grid + jitter:
                rows.append(
                    {
                        # Run-relative onsets — matrix_dataset's per-run
                        # SamplingFrame handles the offset internally.
                        "onset": float(onset_run_relative),
                        "duration": 0.0,
                        "trial_type": trial_type,
                        "run_label": run_label,
                        "run": run_idx + 1,
                    }
                )
    return (
        pd.DataFrame(rows).sort_values(["run", "onset"]).reset_index(drop=True)
    )


def _realize_design(
    events: pd.DataFrame,
) -> tuple[Any, Array, DesignColumns]:
    """Build the multi-run concatenated design via the typed spec."""
    spec = (
        hrf("trial_type", "run_label", basis="spm", norm="spm")
        + drift("poly", degree=2)
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
        fit = fm.fmri_lm(spec, dummy)
    design = np.asarray(
        fit.model.design_matrix_array(run=None), dtype=np.float64
    )
    return spec, design, fit.design_columns()


def _cell_indices(
    columns: DesignColumns,
) -> dict[tuple[str, str], int]:
    """Locate the four (trial_type, run_label) task columns."""
    cells: dict[tuple[str, str], int] = {}
    for trial_type in TRIAL_TYPES:
        for run_label in RUN_LABELS:
            level = f"trial_type.{trial_type}_run_label.{run_label}"
            cells[(trial_type, run_label)] = columns.where(
                term="trial_type:run_label", level=level
            ).one().index
    return cells


def _build_contrasts(
    cells: dict[tuple[str, str], int],
    n_total: int,
) -> tuple[Array, Array, Array, Array]:
    """Construct the four typed contrasts.

    - ``c_main_A_minus_B``: 0.5 × (A_run1 + A_run2) − 0.5 × (B_run1 + B_run2)
      — condition effect collapsed across runs.
    - ``c_run_diff_A``: A_run1 − A_run2 — genuine cross-run difference,
      only meaningful in a concatenated design.
    - ``c_trial_x_run``: 2x2 interaction (A_run1 − B_run1) − (A_run2 − B_run2).
    - ``c_task_F``: joint F over the four task cells.
    """
    c_main = np.zeros(n_total, dtype=np.float64)
    for run_label in RUN_LABELS:
        c_main[cells[("A", run_label)]] = +0.5
        c_main[cells[("B", run_label)]] = -0.5

    c_run_diff = np.zeros(n_total, dtype=np.float64)
    c_run_diff[cells[("A", "run1")]] = +1.0
    c_run_diff[cells[("A", "run2")]] = -1.0

    c_inter = np.zeros(n_total, dtype=np.float64)
    c_inter[cells[("A", "run1")]] = +1.0
    c_inter[cells[("B", "run1")]] = -1.0
    c_inter[cells[("A", "run2")]] = -1.0
    c_inter[cells[("B", "run2")]] = +1.0

    c_task_F = np.zeros((len(cells), n_total), dtype=np.float64)
    for row, idx in enumerate(sorted(cells.values())):
        c_task_F[row, idx] = 1.0

    return c_main, c_run_diff, c_inter, c_task_F


def load_inputs(
    max_voxels: int = MAX_VOXELS, seed: int = 20260513
) -> MultirunInputs:
    """Synthesize a 2-run BOLD time series with known cross-run differences."""
    rng = np.random.default_rng(seed)
    events = _make_events(seed)
    spec, design, columns = _realize_design(events)
    cells = _cell_indices(columns)
    n_total = design.shape[1]

    n_voxels = min(int(max_voxels), MAX_VOXELS)
    betas = np.zeros((n_total, n_voxels), dtype=np.float64)
    # Per-cell amplitudes: A larger in run 1 than run 2 (drives the
    # cross-run-difference contrast), B roughly stable (provides a
    # baseline against which the trial×run interaction shows up).
    cell_offsets = {
        ("A", "run1"): 1.40,
        ("A", "run2"): 0.55,
        ("B", "run1"): 0.30,
        ("B", "run2"): 0.20,
    }
    voxel_ramp = np.linspace(0.4, 1.6, n_voxels, dtype=np.float64)
    for cell, idx in cells.items():
        betas[idx] = cell_offsets[cell] * voxel_ramp

    # Run-specific intercept and small drift coefficients.
    for c in columns.columns:
        if c.role == "baseline" and "constant" in (c.name or ""):
            betas[c.index] = 100.0 + rng.normal(scale=0.5, size=n_voxels)
        elif c.role == "baseline":
            betas[c.index] = rng.normal(scale=0.2, size=n_voxels)

    data = design @ betas + rng.normal(scale=0.3, size=(N_SCANS, n_voxels))

    c_main, c_run_diff, c_inter, c_F = _build_contrasts(cells, n_total)
    return MultirunInputs(
        events=events,
        data=data.astype(np.float64),
        design=design,
        design_columns=columns,
        cell_indices=cells,
        c_main_A_minus_B=c_main,
        c_run_diff_A=c_run_diff,
        c_trial_x_run=c_inter,
        c_task_F=c_F,
    )


def nilearn_pipeline(inputs: MultirunInputs) -> PipelineOutput:
    """Reference: Nilearn's run_glm on the realised concatenated design."""
    labels, estimates = run_glm(inputs.data, inputs.design, noise_model="ols")
    t_main = compute_contrast(
        labels, estimates, inputs.c_main_A_minus_B, stat_type="t"
    )
    t_run_diff = compute_contrast(
        labels, estimates, inputs.c_run_diff_A, stat_type="t"
    )
    t_inter = compute_contrast(
        labels, estimates, inputs.c_trial_x_run, stat_type="t"
    )
    f_task = compute_contrast(
        labels, estimates, inputs.c_task_F, stat_type="F"
    )
    rank_observed = int(np.linalg.matrix_rank(inputs.design))
    return PipelineOutput(
        arrays={
            "design": inputs.design,
            "effect_main_A_minus_B": np.asarray(
                t_main.effect_size(), np.float64
            ),
            "t_main_A_minus_B": np.asarray(t_main.stat(), np.float64),
            "effect_run_diff_A": np.asarray(
                t_run_diff.effect_size(), np.float64
            ),
            "t_run_diff_A": np.asarray(t_run_diff.stat(), np.float64),
            "effect_trial_x_run_interaction": np.asarray(
                t_inter.effect_size(), np.float64
            ),
            "t_trial_x_run_interaction": np.asarray(
                t_inter.stat(), np.float64
            ),
            "f_task_omnibus": np.asarray(f_task.stat(), np.float64),
            "rank": np.array([rank_observed], dtype=np.float64),
        }
    )


def fmrimod_pipeline(inputs: MultirunInputs) -> PipelineOutput:
    """Typed fmrimod path: ``fmri_lm(spec, ds, engine="concat")``.

    The concat engine fits a single OLS on the stacked concatenated
    ``X`` and ``Y`` with ``dfres = n - rank``, so the variance
    denominator matches Nilearn's single-design convention exactly.
    The default ``runwise`` strategy gives identical betas (~1e-12) but
    estimates variance per-run; for cross-run contrasts we want
    single-design semantics.
    """
    spec = (
        hrf("trial_type", "run_label", basis="spm", norm="spm")
        + drift("poly", degree=2)
        + intercept(per="run")
    )
    ds = matrix_dataset(
        inputs.data, tr=TR, run_length=RUN_LENGTH, event_table=inputs.events
    )
    fit = fm.fmri_lm(spec, ds, engine="concat")

    t_main = fit.contrast(inputs.c_main_A_minus_B, name="main_A_minus_B")
    t_run_diff = fit.contrast(inputs.c_run_diff_A, name="run_diff_A")
    t_inter = fit.contrast(inputs.c_trial_x_run, name="trial_x_run")
    f_task = fit.contrast(inputs.c_task_F, name="task_omnibus")

    return PipelineOutput(
        arrays={
            "design": fit.model.design_matrix_array(run=None),
            "effect_main_A_minus_B": np.asarray(t_main.estimate, np.float64),
            "t_main_A_minus_B": np.asarray(t_main.stat, np.float64),
            "effect_run_diff_A": np.asarray(t_run_diff.estimate, np.float64),
            "t_run_diff_A": np.asarray(t_run_diff.stat, np.float64),
            "effect_trial_x_run_interaction": np.asarray(
                t_inter.estimate, np.float64
            ),
            "t_trial_x_run_interaction": np.asarray(
                t_inter.stat, np.float64
            ),
            "f_task_omnibus": np.asarray(f_task.stat, np.float64),
            "rank": np.array(
                [int(fit.condition_report().runs[0].rank)], dtype=np.float64
            ),
        }
    )


def make_case(max_voxels: int = MAX_VOXELS) -> ParityCase:
    """Build the concatenated multi-run parity case."""
    return ParityCase(
        name="tier_a_multirun_concat",
        fmrimod_pipeline=fmrimod_pipeline,
        reference_pipeline=nilearn_pipeline,
        inputs=load_inputs(max_voxels=max_voxels),
        tolerances={
            "design": ParityTolerance(rtol=0.0, atol=0.0),
            "effect_main_A_minus_B": ParityTolerance(rtol=1e-8, atol=1e-9),
            "t_main_A_minus_B": ParityTolerance(rtol=1e-7, atol=1e-8),
            "effect_run_diff_A": ParityTolerance(rtol=1e-8, atol=1e-9),
            "t_run_diff_A": ParityTolerance(rtol=1e-7, atol=1e-8),
            "effect_trial_x_run_interaction": ParityTolerance(
                rtol=1e-8, atol=1e-9
            ),
            "t_trial_x_run_interaction": ParityTolerance(
                rtol=1e-7, atol=1e-8
            ),
            "f_task_omnibus": ParityTolerance(rtol=1e-7, atol=1e-8),
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
