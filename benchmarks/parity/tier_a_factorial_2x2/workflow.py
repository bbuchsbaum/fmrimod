"""2x2 factorial first-level parity against Nilearn.

This case targets a stress angle absent from the other Tier A workflows:
a fully crossed categorical-by-categorical design with three named
hypotheses — main effect of factor A, main effect of factor B, and the
2x2 interaction — fit as a single first-level model.

Why this is a stress test
-------------------------
Nilearn's events DataFrame carries a single ``trial_type`` column, so a
2x2 factorial design must be expressed as four pre-crossed pseudo-trial
types (e.g. ``encode_neutral``, ``encode_emotional``,
``recall_neutral``, ``recall_emotional``). The contrast vectors are then
position-based: the user must remember which column index belongs to
which cell, hand-construct the canonical interaction weights
``[+1, -1, -1, +1]``, and re-derive that bookkeeping every time the
design changes (an added factor, an added level, an added confound).

fmrimod expresses the same scenario as one composed spec::

    hrf("task", "valence", basis="spm", norm="spm")

The categorical-by-categorical interaction term produces one column per
(task, valence) cell, with construction-time provenance carrying the
combined level string. Cells are addressed by authored typed references such
as ``cell("task:valence", task="recall", valence="emotional")``. Main effects
and interactions are ordinary Python algebra over those cells, so the
hypothesis is written before it is lowered to design-column weights.

Pattern B parity claim
----------------------
Both pipelines share the realised fmrimod design. Nilearn receives raw
contrast vectors because that is its public API; fmrimod receives authored
cell-level contrast objects that resolve through declared design-column
provenance. The parity claim is cross-engine: fmrimod's GLM and Nilearn's
``run_glm + compute_contrast`` should produce identical effect, t, and F
statistics on identical inputs.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from itertools import product as iter_product
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
from fmrimod.contrast import OmnibusContrast
from fmrimod.contrast import cell as contrast_cell
from fmrimod.design.columns import DesignColumns
from fmrimod.spec import hrf

Array = NDArray[np.float64]

TR = 2.0
N_SCANS = 200
MAX_VOXELS = 1024
TASK_LEVELS: tuple[str, ...] = ("encode", "recall")
VALENCE_LEVELS: tuple[str, ...] = ("emotional", "neutral")
CELLS: tuple[tuple[str, str], ...] = tuple(
    iter_product(TASK_LEVELS, VALENCE_LEVELS)
)


@dataclass(frozen=True)
class FactorialInputs:
    """Shared inputs plus raw Nilearn contrast vectors."""

    events: pd.DataFrame
    data: Array
    design: Array
    design_columns: DesignColumns
    cell_indices: dict[tuple[str, str], int]
    c_task_main: Array
    c_valence_main: Array
    c_interaction: Array
    c_cells_F: Array


def _make_events(seed: int) -> pd.DataFrame:
    """Build deterministic interleaved 2x2 trials across one run."""
    rng = np.random.default_rng(seed)
    n_per_cell = 8
    rows = []
    for cell_idx, (task, valence) in enumerate(CELLS):
        # Per-cell onset grid, offset so cells interleave in time.
        grid = np.linspace(
            8.0 + 2.0 * cell_idx,
            N_SCANS * TR - 32.0,
            n_per_cell,
            dtype=np.float64,
        )
        jitter = rng.uniform(-1.5, 1.5, n_per_cell)
        for onset in grid + jitter:
            rows.append(
                {
                    "onset": float(onset),
                    "duration": 0.0,
                    "task": task,
                    "valence": valence,
                    "run": 1,
                }
            )
    return pd.DataFrame(rows).sort_values("onset").reset_index(drop=True)


def _realize_design(events: pd.DataFrame) -> tuple[Any, Array, DesignColumns]:
    """Build the 2x2 factorial design via fmrimod's typed spec."""
    spec = hrf("task", "valence", basis="spm", norm="spm")
    dummy = fm.fmri_dataset(
        np.zeros((N_SCANS, 1), dtype=np.float64),
        tr=TR,
        events=events,
    )
    fit = fm.fmri_lm(spec, dummy)
    design = np.asarray(
        fit.model.design_matrix_array(run=0), dtype=np.float64
    )
    return spec, design, fit.design_columns()


def _cell_indices(columns: DesignColumns) -> dict[tuple[str, str], int]:
    """Find each (task, valence) cell column via typed level lookup.

    The categorical-by-categorical level format is ``task.<task>_valence.<v>``
    — the same join that ``_get_condition_tags`` emits at design-build time.
    """
    cells: dict[tuple[str, str], int] = {}
    for task, valence in CELLS:
        level = f"task.{task}_valence.{valence}"
        cells[(task, valence)] = columns.where(
            term="task:valence", level=level
        ).one().index
    return cells


def _build_contrasts(
    cells: dict[tuple[str, str], int],
    n_total: int,
) -> tuple[Array, Array, Array, Array]:
    """Construct the four canonical 2x2 contrasts.

    - ``c_task_main``: recall − encode (average over valence)
    - ``c_valence_main``: emotional − neutral (average over task)
    - ``c_interaction``: 2x2 interaction with canonical [+1, -1, -1, +1]
    - ``c_cells_F``: joint F over all four cells
    """
    c_task_main = np.zeros(n_total, dtype=np.float64)
    for valence in VALENCE_LEVELS:
        c_task_main[cells[("recall", valence)]] = 0.5
        c_task_main[cells[("encode", valence)]] = -0.5

    c_valence_main = np.zeros(n_total, dtype=np.float64)
    for task in TASK_LEVELS:
        c_valence_main[cells[(task, "emotional")]] = 0.5
        c_valence_main[cells[(task, "neutral")]] = -0.5

    c_interaction = np.zeros(n_total, dtype=np.float64)
    c_interaction[cells[("recall", "emotional")]] = 1.0
    c_interaction[cells[("recall", "neutral")]] = -1.0
    c_interaction[cells[("encode", "emotional")]] = -1.0
    c_interaction[cells[("encode", "neutral")]] = 1.0

    c_cells_F = np.zeros((4, n_total), dtype=np.float64)
    for row, factorial_cell in enumerate(CELLS):
        c_cells_F[row, cells[factorial_cell]] = 1.0

    return c_task_main, c_valence_main, c_interaction, c_cells_F


def _cell(task: str, valence: str):
    """Return an authored 2x2 cell reference for the fmrimod contrast path."""
    return contrast_cell("task:valence", task=task, valence=valence)


def _authored_task_main():
    """recall - encode, averaged over valence."""
    return 0.5 * (
        _cell("recall", "emotional")
        + _cell("recall", "neutral")
        - _cell("encode", "emotional")
        - _cell("encode", "neutral")
    )


def _authored_valence_main():
    """emotional - neutral, averaged over task."""
    return 0.5 * (
        _cell("encode", "emotional")
        + _cell("recall", "emotional")
        - _cell("encode", "neutral")
        - _cell("recall", "neutral")
    )


def _authored_interaction():
    """2x2 interaction with canonical [+1, -1, -1, +1] cell algebra."""
    return (
        _cell("recall", "emotional")
        - _cell("recall", "neutral")
        - _cell("encode", "emotional")
        + _cell("encode", "neutral")
    )


def load_inputs(
    max_voxels: int = MAX_VOXELS, seed: int = 20260513
) -> FactorialInputs:
    """Synthesize a BOLD time-series with a known 2x2 factorial effect."""
    rng = np.random.default_rng(seed)
    events = _make_events(seed)
    spec, design, columns = _realize_design(events)
    cells = _cell_indices(columns)
    n_total = design.shape[1]

    n_voxels = min(int(max_voxels), MAX_VOXELS)
    betas = np.zeros((n_total, n_voxels), dtype=np.float64)
    # Per-cell amplitude profile that produces non-zero task main effect,
    # non-zero valence main effect, and a real (non-additive) interaction.
    # Cell amplitudes per voxel: split into ramp + cell-specific intercept.
    cell_offsets = {
        ("encode", "emotional"): 0.40,
        ("encode", "neutral"): 0.10,
        ("recall", "emotional"): 1.10,
        ("recall", "neutral"): 0.30,
    }
    voxel_ramp = np.linspace(0.4, 1.6, n_voxels, dtype=np.float64)
    for factorial_cell, idx in cells.items():
        betas[idx] = cell_offsets[factorial_cell] * voxel_ramp

    baseline_idx = next(
        (c.index for c in columns.columns if c.role != "task"),
        n_total - 1,
    )
    betas[baseline_idx] = 100.0 + rng.normal(scale=0.5, size=n_voxels)

    data = design @ betas + rng.normal(scale=0.3, size=(N_SCANS, n_voxels))

    c_task, c_valence, c_inter, c_F = _build_contrasts(cells, n_total)
    return FactorialInputs(
        events=events,
        data=data.astype(np.float64),
        design=design,
        design_columns=columns,
        cell_indices=cells,
        c_task_main=c_task,
        c_valence_main=c_valence,
        c_interaction=c_inter,
        c_cells_F=c_F,
    )


def nilearn_pipeline(inputs: FactorialInputs) -> PipelineOutput:
    """Reference low-level OLS on the realised 2x2 factorial design."""
    labels, estimates = run_glm(inputs.data, inputs.design, noise_model="ols")
    t_task = compute_contrast(
        labels, estimates, inputs.c_task_main, stat_type="t"
    )
    t_valence = compute_contrast(
        labels, estimates, inputs.c_valence_main, stat_type="t"
    )
    t_inter = compute_contrast(
        labels, estimates, inputs.c_interaction, stat_type="t"
    )
    f_cells = compute_contrast(
        labels, estimates, inputs.c_cells_F, stat_type="F"
    )
    return PipelineOutput(
        arrays={
            "design": inputs.design,
            "effect_task_main": np.asarray(t_task.effect_size(), np.float64),
            "t_task_main": np.asarray(t_task.stat(), np.float64),
            "effect_valence_main": np.asarray(t_valence.effect_size(), np.float64),
            "t_valence_main": np.asarray(t_valence.stat(), np.float64),
            "effect_interaction": np.asarray(t_inter.effect_size(), np.float64),
            "t_interaction": np.asarray(t_inter.stat(), np.float64),
            "f_cells_omnibus": np.asarray(f_cells.stat(), np.float64),
        }
    )


def fmrimod_pipeline(
    inputs: FactorialInputs,
    *,
    timing_sink: dict[str, float] | None = None,
) -> PipelineOutput:
    """Typed fmrimod path: dataset → fit → authored factorial contrasts."""
    dataset_start = time.perf_counter()
    spec = hrf("task", "valence", basis="spm", norm="spm")
    ds = fm.fmri_dataset(inputs.data, tr=TR, events=inputs.events)
    if timing_sink is not None:
        timing_sink["fmrimod_dataset_seconds"] = time.perf_counter() - dataset_start

    fit_start = time.perf_counter()
    fit = fm.fmri_lm(spec, ds)
    if timing_sink is not None:
        timing_sink["fmrimod_fit_seconds"] = time.perf_counter() - fit_start

    contrast_start = time.perf_counter()
    t_task = fit.contrast(_authored_task_main(), name="task_main")
    t_valence = fit.contrast(_authored_valence_main(), name="valence_main")
    t_inter = fit.contrast(_authored_interaction(), name="interaction")
    f_cells = fit.contrast(
        OmnibusContrast(term="task:valence", name="cells_omnibus")
    )
    if timing_sink is not None:
        timing_sink["fmrimod_contrast_seconds"] = time.perf_counter() - contrast_start

    return PipelineOutput(
        arrays={
            "design": fit.model.design_matrix_array(run=0),
            "effect_task_main": np.asarray(t_task.estimate, np.float64),
            "t_task_main": np.asarray(t_task.stat, np.float64),
            "effect_valence_main": np.asarray(t_valence.estimate, np.float64),
            "t_valence_main": np.asarray(t_valence.stat, np.float64),
            "effect_interaction": np.asarray(t_inter.estimate, np.float64),
            "t_interaction": np.asarray(t_inter.stat, np.float64),
            "f_cells_omnibus": np.asarray(f_cells.stat, np.float64),
        }
    )


def make_case(max_voxels: int = MAX_VOXELS) -> ParityCase:
    """Build the 2x2 factorial parity case."""
    return ParityCase(
        name="tier_a_factorial_2x2",
        fmrimod_pipeline=fmrimod_pipeline,
        reference_pipeline=nilearn_pipeline,
        inputs=load_inputs(max_voxels=max_voxels),
        tolerances={
            "design": ParityTolerance(rtol=0.0, atol=0.0),
            "effect_task_main": ParityTolerance(rtol=1e-8, atol=1e-9),
            "t_task_main": ParityTolerance(rtol=1e-7, atol=1e-8),
            "effect_valence_main": ParityTolerance(rtol=1e-8, atol=1e-9),
            "t_valence_main": ParityTolerance(rtol=1e-7, atol=1e-8),
            "effect_interaction": ParityTolerance(rtol=1e-8, atol=1e-9),
            "t_interaction": ParityTolerance(rtol=1e-7, atol=1e-8),
            "f_cells_omnibus": ParityTolerance(rtol=1e-7, atol=1e-8),
        },
    )


def _make_timed_case(
    timings: dict[str, float], max_voxels: int = MAX_VOXELS,
) -> ParityCase:
    start = time.perf_counter()
    inputs = load_inputs(max_voxels=max_voxels)
    timings["load_inputs_seconds"] = time.perf_counter() - start

    def _timed_fmrimod(shared_inputs: FactorialInputs) -> PipelineOutput:
        return fmrimod_pipeline(shared_inputs, timing_sink=timings)

    def _timed_nilearn(shared_inputs: FactorialInputs) -> PipelineOutput:
        nilearn_start = time.perf_counter()
        output = nilearn_pipeline(shared_inputs)
        timings["nilearn_pipeline_seconds"] = time.perf_counter() - nilearn_start
        return output

    return ParityCase(
        name="tier_a_factorial_2x2",
        fmrimod_pipeline=_timed_fmrimod,
        reference_pipeline=_timed_nilearn,
        inputs=inputs,
        tolerances={
            "design": ParityTolerance(rtol=0.0, atol=0.0),
            "effect_task_main": ParityTolerance(rtol=1e-8, atol=1e-9),
            "t_task_main": ParityTolerance(rtol=1e-7, atol=1e-8),
            "effect_valence_main": ParityTolerance(rtol=1e-8, atol=1e-9),
            "t_valence_main": ParityTolerance(rtol=1e-7, atol=1e-8),
            "effect_interaction": ParityTolerance(rtol=1e-8, atol=1e-9),
            "t_interaction": ParityTolerance(rtol=1e-7, atol=1e-8),
            "f_cells_omnibus": ParityTolerance(rtol=1e-7, atol=1e-8),
        },
    )


def _write_timing_payload(json_path: Path, timings: dict[str, float]) -> None:
    payload = json.loads(json_path.read_text())
    stages = {key: float(value) for key, value in sorted(timings.items())}
    payload["timings"] = {
        "status": "recorded",
        "seconds": float(sum(stages.values())),
        "stages": stages,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    timings: dict[str, float] = {}
    result = run(_make_timed_case(timings))
    out_dir = Path(__file__).resolve().parent / "reports"
    json_path, _ = render(result, out_dir)
    _write_timing_payload(json_path, timings)
    if result.status == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
