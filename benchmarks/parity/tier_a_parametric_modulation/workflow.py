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

fmrimod handles the same scenario as a single composed spec::

    hrf("trial_type", basis="spm", norm="spm", modulators=["rt_c"])

The ``modulators=`` kwarg expands into one main-effect term plus one
interaction term per modulator, so the realised design is identical to the
hand-composed ``hrf("trial_type") + hrf("trial_type", "rt_c")`` form. The
typed level metadata survives through to ``DesignColumns``, so
``columns.where(term="trial_type:rt_c", level="A").one().index`` finds the
parametric A column directly instead of falling back to convolver-ordering
guesses.

Pattern B parity claim
----------------------
The realised fmrimod design is reused on the Nilearn side (same X, same
typed contrast vectors), so the parity claim is cross-engine: fmrimod's
GLM and Nilearn's ``run_glm + compute_contrast`` should produce identical
effect, t, and F statistics on identical inputs. The Nilearn pain point
(no main-effect-plus-modulator support in the events parser) is captured
in the docstring; the numerical test pins solver agreement.
"""

from __future__ import annotations

import json
import time
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
from fmrimod.spec import hrf

Array = NDArray[np.float64]

TR = 2.0
N_SCANS = 160
MAX_VOXELS = 1024
TRIAL_TYPES: tuple[str, ...] = ("A", "B")


@dataclass(frozen=True)
class ParametricInputs:
    """Shared inputs: events, BOLD, realised design, and typed contrasts."""

    events: pd.DataFrame
    data: Array
    design: Array
    design_columns: DesignColumns
    c_main_diff: Array
    c_param_diff: Array
    c_param_F: Array


def _make_events(seed: int) -> pd.DataFrame:
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


def _realize_design(events: pd.DataFrame) -> tuple[Any, Array, DesignColumns]:
    """Build the parametric-modulation design via fmrimod's typed spec."""

    spec = hrf("trial_type", basis="spm", norm="spm", modulators=["rt_c"])
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


def _task_column_indices(
    columns: DesignColumns,
) -> tuple[int, int, int, int]:
    """Locate the four task columns via typed level metadata.

    With the column-fact metadata correctly carrying the categorical
    level for both the main-effect term and the parametric interaction
    term, the typed ``where(term=..., level=...)`` lookup returns each
    column directly. No ordering fallback required.
    """
    iA = columns.where(term="trial_type", level="A").one().index
    iB = columns.where(term="trial_type", level="B").one().index
    pA = columns.where(term="trial_type:rt_c", level="A").one().index
    pB = columns.where(term="trial_type:rt_c", level="B").one().index
    return int(iA), int(iB), int(pA), int(pB)


def load_inputs(
    max_voxels: int = MAX_VOXELS, seed: int = 20260513
) -> ParametricInputs:
    """Synthesize a BOLD time-series with a known parametric effect."""

    rng = np.random.default_rng(seed)
    events = _make_events(seed)
    spec, design, columns = _realize_design(events)
    iA, iB, pA, pB = _task_column_indices(columns)
    n_total = design.shape[1]

    n_voxels = min(int(max_voxels), MAX_VOXELS)
    betas = np.zeros((n_total, n_voxels), dtype=np.float64)
    betas[iA] = np.linspace(0.6, 1.8, n_voxels)
    betas[iB] = np.linspace(1.4, 0.2, n_voxels)
    betas[pA] = np.linspace(0.0, 0.6, n_voxels)
    betas[pB] = np.linspace(0.5, -0.1, n_voxels)
    baseline_idx = next(
        (c.index for c in columns.columns if c.role != "task"),
        n_total - 1,
    )
    betas[baseline_idx] = 100.0 + rng.normal(scale=0.5, size=n_voxels)
    data = design @ betas + rng.normal(scale=0.3, size=(N_SCANS, n_voxels))

    c_main = np.zeros(n_total, dtype=np.float64)
    c_main[iA] = 1.0
    c_main[iB] = -1.0
    c_param_diff = np.zeros(n_total, dtype=np.float64)
    c_param_diff[pA] = 1.0
    c_param_diff[pB] = -1.0
    c_param_F = np.zeros((2, n_total), dtype=np.float64)
    c_param_F[0, pA] = 1.0
    c_param_F[1, pB] = 1.0

    return ParametricInputs(
        events=events,
        data=data.astype(np.float64),
        design=design,
        design_columns=columns,
        c_main_diff=c_main,
        c_param_diff=c_param_diff,
        c_param_F=c_param_F,
    )


def nilearn_pipeline(inputs: ParametricInputs) -> PipelineOutput:
    """Reference low-level OLS on fmrimod's realised parametric design."""

    labels, estimates = run_glm(inputs.data, inputs.design, noise_model="ols")
    t_main = compute_contrast(
        labels, estimates, inputs.c_main_diff, stat_type="t"
    )
    t_param_diff = compute_contrast(
        labels, estimates, inputs.c_param_diff, stat_type="t"
    )
    f_param = compute_contrast(
        labels, estimates, inputs.c_param_F, stat_type="F"
    )
    return PipelineOutput(
        arrays={
            "design": inputs.design,
            "effect_main_A_minus_B": np.asarray(
                t_main.effect_size(), np.float64
            ),
            "t_main_A_minus_B": np.asarray(t_main.stat(), np.float64),
            "effect_param_A_minus_B": np.asarray(
                t_param_diff.effect_size(), np.float64
            ),
            "t_param_A_minus_B": np.asarray(t_param_diff.stat(), np.float64),
            "f_param_omnibus": np.asarray(f_param.stat(), np.float64),
        }
    )


def fmrimod_pipeline(
    inputs: ParametricInputs,
    *,
    timing_sink: dict[str, float] | None = None,
) -> PipelineOutput:
    """Four-line fmrimod path: dataset → fmri_lm with modulators → contrast.

    The user-code surface is the canonical typed pipeline. The
    ``modulators=["rt_c"]`` kwarg expands into the same main +
    parametric structure that the Nilearn side gets via pre-built
    regressors, so the numerical comparison stays cross-engine.
    """
    dataset_start = time.perf_counter()
    spec = hrf("trial_type", basis="spm", norm="spm", modulators=["rt_c"])
    ds = fm.fmri_dataset(inputs.data, tr=TR, events=inputs.events)
    if timing_sink is not None:
        timing_sink["fmrimod_dataset_seconds"] = time.perf_counter() - dataset_start

    fit_start = time.perf_counter()
    fit = fm.fmri_lm(spec, ds)
    if timing_sink is not None:
        timing_sink["fmrimod_fit_seconds"] = time.perf_counter() - fit_start

    contrast_start = time.perf_counter()
    t_main = fit.contrast(inputs.c_main_diff, name="main_A_minus_B")
    t_param_diff = fit.contrast(inputs.c_param_diff, name="param_A_minus_B")
    f_param = fit.contrast(inputs.c_param_F, name="param_omnibus")
    if timing_sink is not None:
        timing_sink["fmrimod_contrast_seconds"] = time.perf_counter() - contrast_start

    return PipelineOutput(
        arrays={
            "design": fit.model.design_matrix_array(run=0),
            "effect_main_A_minus_B": np.asarray(t_main.estimate, np.float64),
            "t_main_A_minus_B": np.asarray(t_main.stat, np.float64),
            "effect_param_A_minus_B": np.asarray(
                t_param_diff.estimate, np.float64
            ),
            "t_param_A_minus_B": np.asarray(t_param_diff.stat, np.float64),
            "f_param_omnibus": np.asarray(f_param.stat, np.float64),
        }
    )


def make_case(max_voxels: int = MAX_VOXELS) -> ParityCase:
    """Build the parametric-modulation parity case."""
    return ParityCase(
        name="tier_a_parametric_modulation",
        fmrimod_pipeline=fmrimod_pipeline,
        reference_pipeline=nilearn_pipeline,
        inputs=load_inputs(max_voxels=max_voxels),
        tolerances={
            "design": ParityTolerance(rtol=0.0, atol=0.0),
            "effect_main_A_minus_B": ParityTolerance(rtol=1e-8, atol=1e-9),
            "t_main_A_minus_B": ParityTolerance(rtol=1e-7, atol=1e-8),
            "effect_param_A_minus_B": ParityTolerance(rtol=1e-8, atol=1e-9),
            "t_param_A_minus_B": ParityTolerance(rtol=1e-7, atol=1e-8),
            "f_param_omnibus": ParityTolerance(rtol=1e-7, atol=1e-8),
        },
    )


def _make_timed_case(
    timings: dict[str, float],
    max_voxels: int = MAX_VOXELS,
) -> ParityCase:
    start = time.perf_counter()
    inputs = load_inputs(max_voxels=max_voxels)
    timings["load_inputs_seconds"] = time.perf_counter() - start

    def _timed_fmrimod(shared_inputs: ParametricInputs) -> PipelineOutput:
        return fmrimod_pipeline(shared_inputs, timing_sink=timings)

    def _timed_nilearn(shared_inputs: ParametricInputs) -> PipelineOutput:
        nilearn_start = time.perf_counter()
        output = nilearn_pipeline(shared_inputs)
        timings["nilearn_pipeline_seconds"] = time.perf_counter() - nilearn_start
        return output

    return ParityCase(
        name="tier_a_parametric_modulation",
        fmrimod_pipeline=_timed_fmrimod,
        reference_pipeline=_timed_nilearn,
        inputs=inputs,
        tolerances={
            "design": ParityTolerance(rtol=0.0, atol=0.0),
            "effect_main_A_minus_B": ParityTolerance(rtol=1e-8, atol=1e-9),
            "t_main_A_minus_B": ParityTolerance(rtol=1e-7, atol=1e-8),
            "effect_param_A_minus_B": ParityTolerance(rtol=1e-8, atol=1e-9),
            "t_param_A_minus_B": ParityTolerance(rtol=1e-7, atol=1e-8),
            "f_param_omnibus": ParityTolerance(rtol=1e-7, atol=1e-8),
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
