"""Mixed-TR multi-run parity against Nilearn.

This case stresses an architectural axis no other Tier A workflow
touches: a single analysis with **heterogeneous repetition times
across runs** — run 1 at TR=1.5 s and run 2 at TR=2.0 s. This is
the daily reality for studies that merge sessions collected on
different scanners, change sequence between sessions, or pool
data from multi-site consortia.

Why this is a stress test
-------------------------
Nilearn's :class:`~nilearn.glm.first_level.FirstLevelModel` is
structurally single-TR — ``make_first_level_design_matrix`` takes a
single ``t_r`` and a single ``frame_times``. To fit a multi-run
analysis with mixed TRs the user has to either:

1. **Fit each run independently** with its own ``FirstLevelModel``
   and pool via ``compute_fixed_effects``. That loses cross-run
   contrast inference (the per-run designs do not see each other's
   columns; "is A's effect in run 1 different from run 2?" is not
   expressible).

2. **Hand-roll a block-diagonal concat** — build per-run designs
   with per-run frame_times, stack them block-diagonally into a
   single ``X``, attach per-run intercept and drift block-diagonal
   blocks, manually track which task column belongs to which (run,
   condition) cell, and call ``run_glm`` on the concatenated system.

fmrimod takes the mixed TR in one constructor::

    ds = matrix_dataset(
        bold, tr=[1.5, 2.0], run_length=[n1, n2], event_table=events
    )

The :class:`~fmrimod.sampling.SamplingFrame` carries per-run TRs;
the convolution path evaluates the HRF on each run's own frame
times; ``intercept(per="run")`` and ``drift(..., per="run")`` (where
relevant) block-diagonalise per-run baseline columns; and the
:class:`~fmrimod.glm.engine.ConcatEngine` stacks the result into a
single concat OLS that supports cross-run contrasts directly.

The recently-added (this branch):

- ``slice_timing_offset=`` knob (per-run grids: each run can carry
  its own offset, defaulting to the BOLD-midpoint convention
  ``TR/2``).
- TR-relative default precision (``min(TR)/16``, picking the
  *minimum* TR so the convolution grid is fine enough for the
  faster-TR run).

are exercised here in their natural multi-TR setting.

Pattern B parity claim
----------------------
fmrimod realises the mixed-TR concatenated design via the typed
spec; both engines solve OLS on that **same** ``X``
(``fm.fmri_lm(spec, ds, engine="concat")`` on fmrimod; ``run_glm``
on Nilearn). Compared outputs cover the main A-vs-B effect
collapsed across runs, the genuine cross-run difference for
condition A (``A_run1 - A_run2`` — only meaningful in the concat
design), the joint 4-DF F over all task cells, and the
``rank``/``design`` bitwise checks.
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

# Heterogeneous per-run TRs. Roughly matched in duration: run 1 is
# 100 scans * 1.5 s = 150 s; run 2 is 75 scans * 2.0 s = 150 s.
TRS: tuple[float, ...] = (1.5, 2.0)
RUN_LENGTHS: tuple[int, ...] = (100, 75)
N_SCANS = sum(RUN_LENGTHS)
N_RUNS = len(TRS)
MAX_VOXELS = 1024
TRIAL_TYPES: tuple[str, ...] = ("A", "B")
RUN_LABELS: tuple[str, ...] = ("run1", "run2")


@dataclass(frozen=True)
class MixedTRInputs:
    """Shared inputs for the mixed-TR multi-run parity case."""

    events: pd.DataFrame
    data: Array
    design: Array
    design_columns: DesignColumns
    cell_indices: dict[tuple[str, str], int]
    c_t_main_A_minus_B: Array
    c_t_A_run_diff: Array
    c_f_task_4df: Array


def _make_events(seed: int) -> pd.DataFrame:
    """Build deterministic A/B trials in each run with run-relative onsets.

    Six trials per condition per run, evenly spaced with light jitter.
    Onsets are run-relative — the per-run :class:`SamplingFrame`
    block converts to global time during convolution.
    """
    rng = np.random.default_rng(seed)
    n_per_cell = 6
    rows: list[dict[str, Any]] = []
    for run_idx, tr in enumerate(TRS):
        run_duration = RUN_LENGTHS[run_idx] * tr
        run_label = RUN_LABELS[run_idx]
        for cond_idx, trial_type in enumerate(TRIAL_TYPES):
            grid = np.linspace(
                10.0 + 3.0 * cond_idx,
                run_duration - 22.0,
                n_per_cell,
                dtype=np.float64,
            )
            jitter = rng.uniform(-1.0, 1.0, n_per_cell)
            for onset in grid + jitter:
                rows.append(
                    {
                        "onset": float(onset),
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
) -> tuple[Array, DesignColumns]:
    """Build the mixed-TR concatenated design via the typed spec."""
    spec = (
        hrf("trial_type", "run_label", basis="spm", norm="spm")
        + drift("poly", degree=2)
        + intercept(per="run")
    )
    dummy = matrix_dataset(
        np.zeros((N_SCANS, 1), dtype=np.float64),
        tr=list(TRS),
        run_length=list(RUN_LENGTHS),
        event_table=events,
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
) -> dict[tuple[str, str], int]:
    """Locate the four ``(trial_type, run_label)`` task columns."""
    cells: dict[tuple[str, str], int] = {}
    for trial_type in TRIAL_TYPES:
        for run_label in RUN_LABELS:
            level = f"trial_type.{trial_type}_run_label.{run_label}"
            cells[(trial_type, run_label)] = columns.where(
                term="trial_type:run_label", level=level
            ).one().index
    return cells


def _build_contrasts(
    cells: dict[tuple[str, str], int], n_total: int
) -> tuple[Array, Array, Array]:
    """Three headline contrasts for the mixed-TR design.

    - ``c_t_main_A_minus_B``: A − B collapsed across runs (1-DF t).
    - ``c_t_A_run_diff``: ``A_run1 − A_run2`` — genuine cross-run
      difference; meaningful *only* in a concat design.
    - ``c_f_task_4df``: joint F over the four (trial × run) cells.
    """
    c_t_AB = np.zeros(n_total, dtype=np.float64)
    for run_label in RUN_LABELS:
        c_t_AB[cells[("A", run_label)]] = +0.5
        c_t_AB[cells[("B", run_label)]] = -0.5

    c_t_run_diff = np.zeros(n_total, dtype=np.float64)
    c_t_run_diff[cells[("A", "run1")]] = +1.0
    c_t_run_diff[cells[("A", "run2")]] = -1.0

    c_f_task = np.zeros((len(cells), n_total), dtype=np.float64)
    for row, idx in enumerate(sorted(cells.values())):
        c_f_task[row, idx] = 1.0

    return c_t_AB, c_t_run_diff, c_f_task


def load_inputs(
    max_voxels: int = MAX_VOXELS, seed: int = 20260519
) -> MixedTRInputs:
    """Synthesize a 2-run BOLD time series with known per-run amplitudes."""
    rng = np.random.default_rng(seed)
    events = _make_events(seed)
    design, columns = _realize_design(events)
    cells = _cell_indices(columns)
    n_total = design.shape[1]

    n_voxels = min(int(max_voxels), MAX_VOXELS)
    betas = np.zeros((n_total, n_voxels), dtype=np.float64)
    voxel_ramp = np.linspace(0.4, 1.6, n_voxels, dtype=np.float64)
    # A is larger in run 1 than run 2 (drives the cross-run difference
    # contrast); B is roughly stable.
    cell_amplitudes = {
        ("A", "run1"): 1.30,
        ("A", "run2"): 0.55,
        ("B", "run1"): 0.30,
        ("B", "run2"): 0.25,
    }
    for cell, amp in cell_amplitudes.items():
        betas[cells[cell]] = amp * voxel_ramp

    for c in columns.columns:
        if c.role == "baseline" and "constant" in (c.name or ""):
            betas[c.index] = 100.0 + rng.normal(scale=0.4, size=n_voxels)
        elif c.role == "baseline":
            betas[c.index] = rng.normal(scale=0.15, size=n_voxels)

    data = design @ betas + rng.normal(scale=0.30, size=(N_SCANS, n_voxels))

    c_t_AB, c_t_run_diff, c_f_task = _build_contrasts(cells, n_total)
    return MixedTRInputs(
        events=events,
        data=data.astype(np.float64),
        design=design,
        design_columns=columns,
        cell_indices=cells,
        c_t_main_A_minus_B=c_t_AB,
        c_t_A_run_diff=c_t_run_diff,
        c_f_task_4df=c_f_task,
    )


def nilearn_pipeline(inputs: MixedTRInputs) -> PipelineOutput:
    """Reference: ``run_glm`` on the mixed-TR concatenated design.

    Nilearn does not natively fit a single multi-TR model — we
    short-cut that here by feeding it the same ``X`` fmrimod
    realised. The architectural ergonomic claim is at the
    *construction* level (one fmrimod constructor vs hand-rolled
    block-diagonal in Nilearn); at the *solve* level both engines
    are running the same OLS on the same X.
    """
    labels, estimates = run_glm(inputs.data, inputs.design, noise_model="ols")
    t_AB = compute_contrast(
        labels, estimates, inputs.c_t_main_A_minus_B, stat_type="t"
    )
    t_run_diff = compute_contrast(
        labels, estimates, inputs.c_t_A_run_diff, stat_type="t"
    )
    f_task = compute_contrast(
        labels, estimates, inputs.c_f_task_4df, stat_type="F"
    )
    rank_observed = int(np.linalg.matrix_rank(inputs.design))
    return PipelineOutput(
        arrays={
            "design": inputs.design,
            "effect_main_A_minus_B": np.asarray(
                t_AB.effect_size(), np.float64
            ),
            "t_main_A_minus_B": np.asarray(t_AB.stat(), np.float64),
            "effect_A_run_diff": np.asarray(
                t_run_diff.effect_size(), np.float64
            ),
            "t_A_run_diff": np.asarray(t_run_diff.stat(), np.float64),
            "f_task_4df": np.asarray(f_task.stat(), np.float64),
            "rank": np.array([rank_observed], dtype=np.float64),
        }
    )


def fmrimod_pipeline(inputs: MixedTRInputs) -> PipelineOutput:
    """Typed fmrimod path: one constructor takes the mixed TR list.

    Critically, the typed spec and contrast assembly are *identical*
    to a uniform-TR multi-run design (cf.
    ``tier_a_multirun_concat``). The mixed-TR machinery is invisible
    to the user — ``matrix_dataset(..., tr=[1.5, 2.0])`` plus the
    typed spec plus ``engine="concat"`` is the whole story.
    """
    spec = (
        hrf("trial_type", "run_label", basis="spm", norm="spm")
        + drift("poly", degree=2)
        + intercept(per="run")
    )
    ds = matrix_dataset(
        inputs.data,
        tr=list(TRS),
        run_length=list(RUN_LENGTHS),
        event_table=inputs.events,
        slice_timing_offset=0.0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        fit = fm.fmri_lm(spec, ds, engine="concat")

    t_AB = fit.contrast(inputs.c_t_main_A_minus_B, name="main_A_minus_B")
    t_run_diff = fit.contrast(inputs.c_t_A_run_diff, name="A_run_diff")
    f_task = fit.contrast(inputs.c_f_task_4df, name="task_4df")

    return PipelineOutput(
        arrays={
            "design": fit.model.design_matrix_array(run=None),
            "effect_main_A_minus_B": np.asarray(t_AB.estimate, np.float64),
            "t_main_A_minus_B": np.asarray(t_AB.stat, np.float64),
            "effect_A_run_diff": np.asarray(t_run_diff.estimate, np.float64),
            "t_A_run_diff": np.asarray(t_run_diff.stat, np.float64),
            "f_task_4df": np.asarray(f_task.stat, np.float64),
            "rank": np.array(
                [int(fit.condition_report().runs[0].rank)],
                dtype=np.float64,
            ),
        }
    )


def make_case(max_voxels: int = MAX_VOXELS) -> ParityCase:
    """Build the mixed-TR multi-run parity case."""
    return ParityCase(
        name="tier_a_mixed_tr_multirun",
        fmrimod_pipeline=fmrimod_pipeline,
        reference_pipeline=nilearn_pipeline,
        inputs=load_inputs(max_voxels=max_voxels),
        tolerances={
            "design": ParityTolerance(rtol=0.0, atol=0.0),
            "effect_main_A_minus_B": ParityTolerance(rtol=1e-8, atol=1e-9),
            "t_main_A_minus_B": ParityTolerance(rtol=1e-7, atol=1e-8),
            "effect_A_run_diff": ParityTolerance(rtol=1e-8, atol=1e-9),
            "t_A_run_diff": ParityTolerance(rtol=1e-7, atol=1e-8),
            "f_task_4df": ParityTolerance(rtol=1e-7, atol=1e-8),
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
