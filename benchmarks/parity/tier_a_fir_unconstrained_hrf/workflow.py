"""Unconstrained FIR-basis first-level parity against Nilearn.

This case targets a stress angle absent from the other Tier A workflows:
the user does *not* assume a canonical HRF shape. Instead each trial type
is modeled with a 12-bin finite-impulse-response (FIR) basis that spans
0–24 s post-stimulus, producing 24 task regressors total. The hypothesis of
interest is omnibus: *is there any non-zero response at any FIR lag for any
trial type?*

Why this is a stress test
-------------------------
Nilearn does carry an ``hrf_model="fir"`` option, but the per-lag lag-column
naming + the joint F-contrast over "all lags for all conditions" is manual
bookkeeping at the user-code level: the user must enumerate which column
indices belong to the FIR expansion and assemble a ``(24, n_total)`` F-matrix
by hand. The omnibus is fragile to design edits — adding a confound, a
trial type, or another FIR bin silently moves the indices.

fmrimod expresses the same scenario through typed values:

* ``hrf("trial_type", basis="fir")`` lowers to the same 24-column design.
* ``OmnibusContrast("trial_type")`` resolves to the full-rank F-matrix over
  every column carrying that term, with one row per ``(level, basis_ix)``
  pair — the construction-time provenance carries the lag identity so the
  resolver does not have to grep column names.
* ``fit.contrast(OmnibusContrast(...))`` returns the resolved F-statistic
  alongside its typed intent record.

What we compare
---------------
Both pipelines share the *same* realized design matrix (fmrimod builds it
once at input-load time, the Nilearn side reuses the same ``X``). The
parity claim is therefore cross-engine: identical OLS solutions on
identical inputs.

The harness compares:

- ``design``: bitwise-identical realized design (rtol=0, atol=0).
- ``effect_lag_a``, ``t_lag_a``: a per-lag t-contrast on the first FIR bin
  carrying ``trial_type=A``, located via construction-time provenance.
- ``f_trial_type_omnibus``: the joint F-statistic over all 24 FIR columns.

The pre-resolved contrast vectors are computed once from typed values and
passed to *both* pipelines, so a divergence in the F-statistic indicates a
real solver disagreement rather than contrast bookkeeping drift.
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
from fmrimod.contrast import OmnibusContrast
from fmrimod.design.columns import DesignColumns
from fmrimod.spec import hrf

Array = NDArray[np.float64]

TR = 2.0
N_SCANS = 180
MAX_VOXELS = 1024
FIR_NBASIS = 12
FIR_SPAN = 24.0
TRIAL_TYPES: tuple[str, ...] = ("A", "B")


@dataclass(frozen=True)
class FirInputs:
    """Shared inputs: events, BOLD, realized design, and typed contrasts."""

    events: pd.DataFrame
    data: Array
    design: Array
    design_columns: DesignColumns
    omnibus: OmnibusContrast
    f_omnibus_weights: Array
    t_lag_a_weights: Array


def _make_events(seed: int) -> pd.DataFrame:
    """Build deterministic jittered A/B trials with no two events within 6 s."""

    rng = np.random.default_rng(seed)
    n_per_condition = 16
    onsets = []
    labels = []
    # Place A and B events on interleaved jittered grids to keep them
    # decorrelated from the run-mean and from each other.
    a_grid = np.linspace(8.0, N_SCANS * TR - 28.0, n_per_condition, dtype=np.float64)
    b_grid = a_grid + rng.uniform(2.0, 6.0, n_per_condition)
    for onset in a_grid:
        onsets.append(float(onset))
        labels.append("A")
    for onset in b_grid:
        onsets.append(float(onset))
        labels.append("B")
    df = pd.DataFrame(
        {
            "onset": np.asarray(onsets, dtype=np.float64),
            "duration": np.zeros(len(onsets), dtype=np.float64),
            "trial_type": labels,
            "run": 1,
        }
    ).sort_values("onset").reset_index(drop=True)
    return df


def _realize_design(events: pd.DataFrame) -> tuple[Any, Array, DesignColumns]:
    """Build the FIR design from a typed spec via a zero-data dry-run fit."""

    spec = hrf("trial_type", basis="fir")
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


def _t_vector(columns: DesignColumns, term: str, level: str) -> Array:
    """One-hot t-contrast on the first FIR basis carrying (term, level).

    Uses the typed ``basis_ix=`` filter to address a specific FIR lag
    without iterating the realised design by hand.
    """

    selected = columns.where(term=term, level=level, basis_ix=1)
    if len(selected) != 1:
        raise RuntimeError(
            f"expected one column with term={term!r} level={level!r} "
            f"basis_ix=1; got {len(selected)}"
        )
    vector = np.zeros(len(columns), dtype=np.float64)
    vector[selected.one().index] = 1.0
    return vector


def load_inputs(
    max_voxels: int = MAX_VOXELS, seed: int = 20260513
) -> FirInputs:
    """Synthesize a BOLD time-series with a known FIR-shaped response."""

    rng = np.random.default_rng(seed)
    events = _make_events(seed)
    spec, design, design_columns = _realize_design(events)
    n_voxels = min(int(max_voxels), MAX_VOXELS)

    n_task = sum(1 for c in design_columns.columns if c.term == "trial_type")
    n_total = design.shape[1]
    if n_task != FIR_NBASIS * len(TRIAL_TYPES):
        raise RuntimeError(
            f"Expected {FIR_NBASIS * len(TRIAL_TYPES)} FIR task columns, "
            f"got {n_task}"
        )

    # Synthesize per-column betas. Task betas vary across voxels so the
    # omnibus F is meaningful; the baseline column gets a large constant.
    betas = np.zeros((n_total, n_voxels), dtype=np.float64)
    task_cols = sorted(
        (c for c in design_columns.where(term="trial_type").columns),
        key=lambda c: c.index,
    )
    for k, column in enumerate(task_cols):
        # Decaying amplitude profile across lags, sign flipped for B so the
        # main-effect difference is non-trivial.
        sign = 1.0 if column.level == "A" else -0.6
        amp = sign * np.exp(-((k % FIR_NBASIS) - 3.0) ** 2 / 8.0)
        betas[column.index, :] = amp * np.linspace(0.4, 1.6, n_voxels)
    baseline_idx = next(
        (c.index for c in design_columns.columns if c.role != "task"),
        n_total - 1,
    )
    betas[baseline_idx, :] = 100.0 + rng.normal(scale=0.5, size=n_voxels)

    data = design @ betas + rng.normal(scale=0.3, size=(N_SCANS, n_voxels))

    omnibus = OmnibusContrast("trial_type")
    f_weights = omnibus.resolve(design_columns)
    t_weights = _t_vector(design_columns, term="trial_type", level="A")

    return FirInputs(
        events=events,
        data=data.astype(np.float64),
        design=design,
        design_columns=design_columns,
        omnibus=omnibus,
        f_omnibus_weights=f_weights,
        t_lag_a_weights=t_weights,
    )


def nilearn_pipeline(inputs: FirInputs) -> PipelineOutput:
    """Reference low-level OLS on fmrimod's realized FIR design."""

    labels, estimates = run_glm(inputs.data, inputs.design, noise_model="ols")
    t_lag_a = compute_contrast(
        labels, estimates, inputs.t_lag_a_weights, stat_type="t"
    )
    f_omnibus = compute_contrast(
        labels, estimates, inputs.f_omnibus_weights, stat_type="F"
    )
    return PipelineOutput(
        arrays={
            "design": inputs.design,
            "effect_lag_a": np.asarray(t_lag_a.effect_size(), np.float64),
            "t_lag_a": np.asarray(t_lag_a.stat(), np.float64),
            "f_trial_type_omnibus": np.asarray(f_omnibus.stat(), np.float64),
        }
    )


def fmrimod_pipeline(
    inputs: FirInputs,
    *,
    timing_sink: dict[str, float] | None = None,
) -> PipelineOutput:
    """Three-line fmrimod path: dataset → fit → typed FIR omnibus contrast."""

    dataset_start = time.perf_counter()
    spec = hrf("trial_type", basis="fir")
    ds = fm.fmri_dataset(inputs.data, tr=TR, events=inputs.events)
    if timing_sink is not None:
        timing_sink["fmrimod_dataset_seconds"] = time.perf_counter() - dataset_start

    fit_start = time.perf_counter()
    fit = fm.fmri_lm(spec, ds)
    if timing_sink is not None:
        timing_sink["fmrimod_fit_seconds"] = time.perf_counter() - fit_start

    contrast_start = time.perf_counter()
    t_lag_a = fit.contrast(inputs.t_lag_a_weights, name="lag_a")
    f_omnibus = fit.contrast(inputs.omnibus)
    if timing_sink is not None:
        timing_sink["fmrimod_contrast_seconds"] = time.perf_counter() - contrast_start

    return PipelineOutput(
        arrays={
            "design": fit.model.design_matrix_array(run=0),
            "effect_lag_a": np.asarray(t_lag_a.estimate, np.float64),
            "t_lag_a": np.asarray(t_lag_a.stat, np.float64),
            "f_trial_type_omnibus": np.asarray(f_omnibus.stat, np.float64),
        }
    )


def make_case(max_voxels: int = MAX_VOXELS) -> ParityCase:
    """Build the FIR unconstrained-HRF parity case."""

    return ParityCase(
        name="tier_a_fir_unconstrained_hrf",
        fmrimod_pipeline=fmrimod_pipeline,
        reference_pipeline=nilearn_pipeline,
        inputs=load_inputs(max_voxels=max_voxels),
        tolerances={
            "design": ParityTolerance(rtol=0.0, atol=0.0),
            "effect_lag_a": ParityTolerance(rtol=1e-8, atol=1e-9),
            "t_lag_a": ParityTolerance(rtol=1e-7, atol=1e-8),
            "f_trial_type_omnibus": ParityTolerance(rtol=1e-7, atol=1e-8),
        },
    )


def _make_timed_case(
    timings: dict[str, float],
    max_voxels: int = MAX_VOXELS,
) -> ParityCase:
    start = time.perf_counter()
    inputs = load_inputs(max_voxels=max_voxels)
    timings["load_inputs_seconds"] = time.perf_counter() - start

    def _timed_fmrimod(shared_inputs: FirInputs) -> PipelineOutput:
        return fmrimod_pipeline(shared_inputs, timing_sink=timings)

    def _timed_nilearn(shared_inputs: FirInputs) -> PipelineOutput:
        nilearn_start = time.perf_counter()
        output = nilearn_pipeline(shared_inputs)
        timings["nilearn_pipeline_seconds"] = time.perf_counter() - nilearn_start
        return output

    return ParityCase(
        name="tier_a_fir_unconstrained_hrf",
        fmrimod_pipeline=_timed_fmrimod,
        reference_pipeline=_timed_nilearn,
        inputs=inputs,
        tolerances={
            "design": ParityTolerance(rtol=0.0, atol=0.0),
            "effect_lag_a": ParityTolerance(rtol=1e-8, atol=1e-9),
            "t_lag_a": ParityTolerance(rtol=1e-7, atol=1e-8),
            "f_trial_type_omnibus": ParityTolerance(rtol=1e-7, atol=1e-8),
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
