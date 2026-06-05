"""Mumford LSS single-trial estimation parity against a hand-rolled Nilearn loop.

LSS (Least Squares Single) is Mumford's per-trial beta-series approach
for MVPA: for each trial of interest, fit a separate GLM with two task
regressors — (1) the trial of interest and (2) the sum of all other
trials — plus the usual nuisance regressors. The per-trial beta from
column 1 is the single-trial estimate. This avoids the high
collinearity between same-condition trials that plagues the LSA
(Least Squares All) approach, at the cost of fitting N models
instead of one.

fmrimod ships two surfaces for single-trial estimation:

- ``trialwise(basis="spm")`` — a typed-spec term that produces N
  trial regressors in *one* big design (LSA). The realised columns
  are addressable as ``cols.where(term="trial")``.
- ``lss_single_trial(Y, X, ...)`` — a matrix-first vectorized LSS
  estimator. Take a per-trial design ``X`` (shape ``(n, n_trials)``,
  one HRF-convolved column per trial) plus the data ``Y``; the
  function does the inner LSS loop in vectorized form and returns
  per-trial betas.

Nilearn does not ship an LSS estimator. The Nilearn-side reference
in this workflow is a hand-rolled loop following the Mumford
algorithm exactly: for each trial k, build a design with the trial-k
column, a "sum of other trials" column, the nuisance regressors;
solve OLS; extract the trial-k beta.

Pattern B parity claim
----------------------
fmrimod's ``trialwise()`` typed spec builds the per-trial design X
(``n_scans x n_trials``); both engines then take that X and run:

- fmrimod: ``lss_single_trial(Y, X, baseline_regressors=baseline)``
  (vectorized).
- Nilearn-equivalent: hand-rolled per-trial loop with
  ``np.linalg.lstsq`` on the (trial_k, others_sum, baseline)
  design.

All four headline outputs match at <= 1e-9:

- ``lss_betas``: full per-trial beta matrix
  (``n_trials x n_voxels``).
- ``effect_first_A_trial``: LSS beta for the first condition-A trial.
- ``mean_lss_beta_A``: per-voxel mean LSS beta across the
  condition-A trials.
- ``mean_lss_beta_B``: per-voxel mean LSS beta across the
  condition-B trials.

Pain points logged for follow-up
--------------------------------

Two ergonomic gaps surfaced while wiring this and are pinned in
``tests/test_lss/test_lss_typed_spec_pain_points.py``:

1. **The typed ``trialwise()`` spec was broken** at the start of
   this work — the lowering didn't propagate the ``_is_trialwise``
   marker, so ``fmri_lm(trialwise(...), ds)`` failed with
   ``ValueError: Event '__trial__' not found in model``. Fixed in
   the same commit that landed this workflow.

2. **No typed ``lss_single_trial`` wrapper on the high-level
   ``fmri_lm`` surface.** The user has to extract the per-trial
   design themselves (via the ``trialwise()`` typed spec) and the
   baseline regressors (via the baseline columns of the realised
   design), then call the matrix-first ``lss_single_trial``
   directly. A typed ``single_trial(method="lss")`` or
   ``fmri_lm(..., engine="lss")`` would be a clean win — the
   matrix-first API already does the heavy lifting.

3. **Trial labels are not surfaced on the per-trial column
   provenance.** The realised ``trialwise()`` columns get
   ``term="trial"`` and ``name="trial_NN"`` but no link back to
   the original event-table row (trial_type, original onset).
   Users doing MVPA need to join their per-trial betas back to a
   stimulus / condition label by parsing the trial number out of
   the column name. Adding ``condition`` / ``level`` provenance
   on the trialwise columns would let downstream MVPA tooling
   pull labels via typed lookup.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
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
from fmrimod.single import lss_single_trial
from fmrimod.spec import drift, hrf, intercept, trialwise

Array = NDArray[np.float64]

TR = 2.0
N_SCANS = 220
MAX_VOXELS = 1024
TRIAL_TYPES: tuple[str, ...] = ("A", "B")
N_TRIALS_PER_CONDITION = 8


@dataclass(frozen=True)
class LssInputs:
    """Shared inputs for the LSS parity case."""

    events: pd.DataFrame
    data: Array
    trialwise_design: Array  # (n_scans, n_trials)
    baseline_design: Array   # (n_scans, n_baseline_cols)
    full_design: Array       # (n_scans, n_trials + n_baseline_cols)
    design_columns: DesignColumns
    trial_indices: tuple[int, ...]
    condition_of_trial: tuple[str, ...]
    a_trial_positions: tuple[int, ...]
    b_trial_positions: tuple[int, ...]


def _make_events(seed: int) -> pd.DataFrame:
    """Interleaved A/B trials, eight each, jittered onsets."""
    rng = np.random.default_rng(seed)
    n_each = N_TRIALS_PER_CONDITION
    onsets = np.linspace(
        14.0, N_SCANS * TR - 30.0, 2 * n_each, dtype=np.float64
    )
    jitter = rng.uniform(-0.8, 0.8, len(onsets))
    onsets = onsets + jitter
    rows = []
    for k, onset in enumerate(onsets):
        rows.append(
            {
                "onset": float(onset),
                "duration": 0.0,
                "trial_type": TRIAL_TYPES[k % 2],
                "run": 1,
            }
        )
    return (
        pd.DataFrame(rows).sort_values("onset").reset_index(drop=True)
    )


def _realize_designs(
    events: pd.DataFrame,
) -> tuple[Array, Array, Array, DesignColumns, tuple[int, ...]]:
    """Build the trialwise design X and baseline block Z via the typed spec."""
    spec = (
        trialwise(basis="spm")
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
    full_design = np.asarray(
        fit.model.design_matrix_array(run=None), dtype=np.float64
    )
    cols = fit.design_columns()
    # Trial regressors (one column per event).
    trial_indices = tuple(
        c.index for c in cols.where(term="trial").columns
    )
    # Baseline = drift + intercept (typed roles distinguish them).
    baseline_indices = tuple(
        c.index for c in cols.columns
        if c.role in ("drift", "intercept", "confound", "baseline")
    )
    trialwise_design = full_design[:, list(trial_indices)]
    baseline_design = full_design[:, list(baseline_indices)]
    return (
        trialwise_design,
        baseline_design,
        full_design,
        cols,
        trial_indices,
    )


def load_inputs(
    max_voxels: int = MAX_VOXELS, seed: int = 20260521
) -> LssInputs:
    """Synthesize a BOLD time series with per-trial activity."""
    rng = np.random.default_rng(seed)
    events = _make_events(seed)
    trialwise_design, baseline_design, full_design, cols, trial_indices = (
        _realize_designs(events)
    )
    n_total = full_design.shape[1]
    n_trials = trialwise_design.shape[1]
    assert n_trials == len(events)

    condition_of_trial = tuple(events["trial_type"].tolist())
    a_positions = tuple(
        i for i, c in enumerate(condition_of_trial) if c == "A"
    )
    b_positions = tuple(
        i for i, c in enumerate(condition_of_trial) if c == "B"
    )

    n_voxels = min(int(max_voxels), MAX_VOXELS)

    # True per-trial amplitudes — A trials larger than B, small jitter.
    true_betas = np.zeros((n_total, n_voxels), dtype=np.float64)
    voxel_ramp = np.linspace(0.5, 1.5, n_voxels, dtype=np.float64)
    for trial_pos, trial_col in enumerate(trial_indices):
        cond = condition_of_trial[trial_pos]
        base_amp = 1.20 if cond == "A" else 0.45
        trial_amp = base_amp + rng.normal(scale=0.08)
        true_betas[trial_col] = trial_amp * voxel_ramp
    for c in cols.columns:
        if c.role == "intercept":
            true_betas[c.index] = 100.0 + rng.normal(scale=0.4, size=n_voxels)
        elif c.role == "drift":
            true_betas[c.index] = rng.normal(scale=0.15, size=n_voxels)

    data = full_design @ true_betas + rng.normal(
        scale=0.30, size=(N_SCANS, n_voxels)
    )

    return LssInputs(
        events=events,
        data=data.astype(np.float64),
        trialwise_design=trialwise_design,
        baseline_design=baseline_design,
        full_design=full_design,
        design_columns=cols,
        trial_indices=trial_indices,
        condition_of_trial=condition_of_trial,
        a_trial_positions=a_positions,
        b_trial_positions=b_positions,
    )


def _hand_rolled_nilearn_lss(
    Y: Array, X: Array, baseline: Array
) -> Array:
    """Hand-rolled Mumford LSS loop using NumPy lstsq.

    For each trial k, build the (trial_k, others_sum) 2-column task
    design plus the baseline regressors, solve OLS, extract the
    trial-k beta. Nilearn doesn't ship an LSS estimator, so the
    reference is the explicit loop following the Mumford algorithm.
    """
    n, n_trials = X.shape
    n_voxels = Y.shape[1]
    betas = np.zeros((n_trials, n_voxels), dtype=np.float64)
    for k in range(n_trials):
        others_sum = X.sum(axis=1) - X[:, k]
        design_k = np.column_stack([X[:, k], others_sum, baseline])
        beta_k, *_ = np.linalg.lstsq(design_k, Y, rcond=None)
        betas[k] = beta_k[0]  # trial-k beta is column 0
    return betas


def fmrimod_pipeline(inputs: LssInputs) -> PipelineOutput:
    """Typed-spec build of X + matrix-first vectorized LSS solver."""
    result = lss_single_trial(
        Y=inputs.data,
        X=inputs.trialwise_design,
        baseline_regressors=inputs.baseline_design,
        include_intercept=False,
    )
    lss_betas = np.asarray(result.betas, dtype=np.float64)
    return PipelineOutput(
        arrays={
            "lss_betas": lss_betas,
            "effect_first_A_trial": lss_betas[inputs.a_trial_positions[0]],
            "mean_lss_beta_A": np.mean(
                lss_betas[list(inputs.a_trial_positions)], axis=0
            ),
            "mean_lss_beta_B": np.mean(
                lss_betas[list(inputs.b_trial_positions)], axis=0
            ),
        }
    )


def nilearn_pipeline(inputs: LssInputs) -> PipelineOutput:
    """Hand-rolled per-trial loop following the Mumford LSS algorithm."""
    lss_betas = _hand_rolled_nilearn_lss(
        Y=inputs.data,
        X=inputs.trialwise_design,
        baseline=inputs.baseline_design,
    )
    return PipelineOutput(
        arrays={
            "lss_betas": lss_betas,
            "effect_first_A_trial": lss_betas[inputs.a_trial_positions[0]],
            "mean_lss_beta_A": np.mean(
                lss_betas[list(inputs.a_trial_positions)], axis=0
            ),
            "mean_lss_beta_B": np.mean(
                lss_betas[list(inputs.b_trial_positions)], axis=0
            ),
        }
    )


def make_case(max_voxels: int = MAX_VOXELS) -> ParityCase:
    """Build the LSS single-trial parity case."""
    return ParityCase(
        name="tier_a_single_trial_lss",
        fmrimod_pipeline=fmrimod_pipeline,
        reference_pipeline=nilearn_pipeline,
        inputs=load_inputs(max_voxels=max_voxels),
        tolerances={
            "lss_betas": ParityTolerance(rtol=1e-8, atol=1e-9),
            "effect_first_A_trial": ParityTolerance(rtol=1e-8, atol=1e-9),
            "mean_lss_beta_A": ParityTolerance(rtol=1e-8, atol=1e-9),
            "mean_lss_beta_B": ParityTolerance(rtol=1e-8, atol=1e-9),
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
