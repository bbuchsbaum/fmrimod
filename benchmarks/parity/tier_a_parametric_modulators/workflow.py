"""Multiple parametric modulators per condition — modern-correct workflow.

This case stresses fmrimod's handling of multiple parametric modulators
on the same condition (e.g. reaction time *and* accuracy modulating
the BOLD response to a single trial type), and demonstrates the
**modern-correct** analytic workflow as advocated by Mumford et al.
(2015, *PLoS ONE*): pre-center each modulator, do **not** orthogonalize
between modulators, and answer "do these modulators add variance
beyond the unmodulated regressor?" with an explicit joint F-test.

Why "no orthogonalization by default" is the right call
-------------------------------------------------------
SPM12 still defaults to *sequential Gram-Schmidt orthogonalization*
of parametric modulators against (a) the unmodulated regressor and
(b) each prior modulator. The Mumford et al. paper showed this is
statistically problematic — beta estimates become order-dependent,
the second modulator's beta no longer means "the effect of this
modulator" but "residual variance after partialing out everything
listed before me," and the implicit asymmetric variance attribution
is rarely defensible. The modern consensus (FSL, AFNI, recent
Nilearn): don't orthogonalize, mean-center instead, and test for
"does this modulator add unique variance?" with a proper
nested-model F.

fmrimod's behavior matches the modern consensus by default: the
typed ``hrf("trial_type", modulators=("rt", "accuracy"))`` builds
the natural three-column structure (unmodulated boxcar + one
modulated regressor per modulator) with no orthogonalization. The
order ``"rt", "accuracy"`` is interchangeable with ``"accuracy",
"rt"`` and produces the same betas (up to column reordering).

What this case demonstrates
---------------------------

- The typed ``modulators=`` keyword expands cleanly to one
  unmodulated boxcar plus one column per modulator.
- Typed level lookup names each piece: ``cols.where(
  term="trial_type", level="A")`` for the unmodulated boxcar,
  ``cols.where(term="trial_type:rt", level="A")`` for the
  RT-modulated regressor, etc.
- The headline analytic contrasts:
    - Three 1-DF t-tests (unmodulated effect, RT modulation
      amplitude, accuracy modulation amplitude).
    - One 2-DF joint F-test over both modulators ("do RT and
      accuracy together add variance beyond the unmodulated
      regressor?") — the *correct* inferential answer to the
      "are these modulators worth including?" question.

The workflow pre-centers each modulator in the events DataFrame
before building the spec. That gives the unmodulated boxcar and
modulator columns the near-orthogonal structure that makes the
betas interpretable, without the asymmetric-attribution problems
of sequential orthogonalization.

Pain point logged for follow-up: fmrimod has no
``hrf(..., center_modulators=True)`` keyword, so the user must
remember to pre-center modulators themselves (same as Nilearn).
A typed opt-in would be a clean ergonomic win since the modern-
correct workflow always wants this.

Pattern B parity claim
----------------------
fmrimod realises the parametric-modulator design via the typed
spec; both engines solve OLS on that **same** ``X``
(``fm.fmri_lm(spec, ds, engine="concat")`` on fmrimod;
``run_glm(data, X)`` on Nilearn). Eight outputs are compared. The
``rank`` and ``design`` outputs are bitwise; the contrast outputs
match at <= 1e-9.
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
N_SCANS = 220
MAX_VOXELS = 1024


@dataclass(frozen=True)
class ModulatorInputs:
    """Shared inputs for the parametric-modulator parity case."""

    events: pd.DataFrame
    data: Array
    design: Array
    design_columns: DesignColumns
    cell_indices: dict[str, int]
    c_t_unmod: Array
    c_t_rt: Array
    c_t_accuracy: Array
    c_f_modulators_joint: Array


def _make_events(seed: int) -> pd.DataFrame:
    """Build deterministic single-condition trials with two pre-centered modulators.

    The modulators (``rt``, ``accuracy``) are mean-centered in the
    events DataFrame *before* convolution so the unmodulated boxcar
    and the modulated regressors are near-orthogonal. The true mean
    of ``rt`` is around 1.0 s and ``accuracy`` around 0.5; the
    modulators in the resulting events table sum to zero across
    trials.
    """
    rng = np.random.default_rng(seed)
    n_trials = 18
    onsets = np.linspace(14.0, N_SCANS * TR - 30.0, n_trials, dtype=np.float64)
    jitter = rng.uniform(-1.0, 1.0, n_trials)
    onsets = onsets + jitter

    raw_rt = rng.uniform(0.5, 1.5, n_trials)
    raw_acc = rng.uniform(0.2, 1.0, n_trials)
    rt_centered = raw_rt - raw_rt.mean()
    acc_centered = raw_acc - raw_acc.mean()

    return pd.DataFrame({
        "onset": onsets,
        "duration": 0.0,
        "trial_type": "A",
        "rt": rt_centered,
        "accuracy": acc_centered,
        "run": 1,
    }).sort_values("onset").reset_index(drop=True)


def _realize_design(
    events: pd.DataFrame,
) -> tuple[Array, DesignColumns]:
    """Build the unmodulated + parametric-modulator design via the typed spec.

    The headline spec line is::

        hrf("trial_type", modulators=("rt", "accuracy"))

    which expands to three task columns:
      1. ``trial_type:A`` — unmodulated boxcar.
      2. ``trial_type:rt:A`` — RT-modulated regressor.
      3. ``trial_type:accuracy:A`` — accuracy-modulated regressor.
    """
    spec = (
        hrf("trial_type", modulators=("rt", "accuracy"))
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
) -> dict[str, int]:
    """Locate the three task columns by typed term lookup.

    The unmodulated boxcar lives at ``term="trial_type"``; each
    modulator's column lives at ``term=f"trial_type:{modname}"``.
    All three have ``level="A"`` (the single trial type).
    """
    cells: dict[str, int] = {}
    cells["unmod"] = columns.where(
        term="trial_type", level="A"
    ).one().index
    cells["rt"] = columns.where(
        term="trial_type:rt", level="A"
    ).one().index
    cells["accuracy"] = columns.where(
        term="trial_type:accuracy", level="A"
    ).one().index
    return cells


def _build_contrasts(
    cells: dict[str, int], n_total: int
) -> tuple[Array, Array, Array, Array]:
    """Three 1-DF t-tests + one 2-DF joint F-test."""
    c_t_unmod = np.zeros(n_total, dtype=np.float64)
    c_t_unmod[cells["unmod"]] = 1.0

    c_t_rt = np.zeros(n_total, dtype=np.float64)
    c_t_rt[cells["rt"]] = 1.0

    c_t_acc = np.zeros(n_total, dtype=np.float64)
    c_t_acc[cells["accuracy"]] = 1.0

    # Joint F over both modulators: tests "do RT and accuracy together
    # add variance beyond the unmodulated boxcar?". This is the
    # nested-model F-test that Mumford et al. recommend as the
    # *correct* answer to "are these modulators worth including?"
    # — rather than relying on SPM-style sequential orthogonalization
    # to make per-modulator betas look identifiable.
    c_f_joint = np.zeros((2, n_total), dtype=np.float64)
    c_f_joint[0, cells["rt"]] = 1.0
    c_f_joint[1, cells["accuracy"]] = 1.0

    return c_t_unmod, c_t_rt, c_t_acc, c_f_joint


def load_inputs(
    max_voxels: int = MAX_VOXELS, seed: int = 20260518
) -> ModulatorInputs:
    """Synthesize a BOLD time series with known true effects per modulator.

    The generating model places:
    - A moderate unmodulated boxcar amplitude (so the unmodulated
      effect is detectable).
    - A clear positive RT modulation (so the RT contrast is detectable).
    - A near-zero accuracy modulation (so a *correctly-specified*
      analysis will fail to reject H0 for accuracy alone, but the
      joint F over both modulators may still be marginal).
    """
    rng = np.random.default_rng(seed)
    events = _make_events(seed)
    design, columns = _realize_design(events)
    cells = _cell_indices(columns)
    n_total = design.shape[1]

    n_voxels = min(int(max_voxels), MAX_VOXELS)
    betas = np.zeros((n_total, n_voxels), dtype=np.float64)
    voxel_ramp = np.linspace(0.5, 1.5, n_voxels, dtype=np.float64)

    cell_amplitudes = {
        "unmod": 1.10,    # clean unmodulated effect
        "rt": 0.45,       # clear positive RT modulation
        "accuracy": 0.05, # near-zero — exercises the "marginal" branch
    }
    for cell, amp in cell_amplitudes.items():
        betas[cells[cell]] = amp * voxel_ramp

    for c in columns.columns:
        if c.role == "baseline" and "constant" in (c.name or ""):
            betas[c.index] = 100.0 + rng.normal(scale=0.4, size=n_voxels)
        elif c.role == "baseline":
            betas[c.index] = rng.normal(scale=0.15, size=n_voxels)

    data = design @ betas + rng.normal(scale=0.35, size=(N_SCANS, n_voxels))

    c_t_unmod, c_t_rt, c_t_acc, c_f_joint = _build_contrasts(cells, n_total)
    return ModulatorInputs(
        events=events,
        data=data.astype(np.float64),
        design=design,
        design_columns=columns,
        cell_indices=cells,
        c_t_unmod=c_t_unmod,
        c_t_rt=c_t_rt,
        c_t_accuracy=c_t_acc,
        c_f_modulators_joint=c_f_joint,
    )


def nilearn_pipeline(inputs: ModulatorInputs) -> PipelineOutput:
    """Reference: ``run_glm`` on the fmrimod-realised parametric-modulator design."""
    labels, estimates = run_glm(inputs.data, inputs.design, noise_model="ols")
    t_unmod = compute_contrast(labels, estimates, inputs.c_t_unmod, stat_type="t")
    t_rt = compute_contrast(labels, estimates, inputs.c_t_rt, stat_type="t")
    t_acc = compute_contrast(labels, estimates, inputs.c_t_accuracy, stat_type="t")
    f_joint = compute_contrast(
        labels, estimates, inputs.c_f_modulators_joint, stat_type="F"
    )
    rank_observed = int(np.linalg.matrix_rank(inputs.design))
    return PipelineOutput(
        arrays={
            "design": inputs.design,
            "effect_unmod": np.asarray(t_unmod.effect_size(), np.float64),
            "t_unmod": np.asarray(t_unmod.stat(), np.float64),
            "effect_rt": np.asarray(t_rt.effect_size(), np.float64),
            "t_rt": np.asarray(t_rt.stat(), np.float64),
            "effect_accuracy": np.asarray(t_acc.effect_size(), np.float64),
            "t_accuracy": np.asarray(t_acc.stat(), np.float64),
            "f_modulators_joint_2df": np.asarray(f_joint.stat(), np.float64),
            "rank": np.array([rank_observed], dtype=np.float64),
        }
    )


def fmrimod_pipeline(inputs: ModulatorInputs) -> PipelineOutput:
    """Typed fmrimod path: ``fmri_lm(spec, ds, engine="concat")``."""
    spec = (
        hrf("trial_type", modulators=("rt", "accuracy"))
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

    t_unmod = fit.contrast(inputs.c_t_unmod, name="unmod")
    t_rt = fit.contrast(inputs.c_t_rt, name="rt")
    t_acc = fit.contrast(inputs.c_t_accuracy, name="accuracy")
    f_joint = fit.contrast(inputs.c_f_modulators_joint, name="modulators_joint")

    return PipelineOutput(
        arrays={
            "design": fit.model.design_matrix_array(run=None),
            "effect_unmod": np.asarray(t_unmod.estimate, np.float64),
            "t_unmod": np.asarray(t_unmod.stat, np.float64),
            "effect_rt": np.asarray(t_rt.estimate, np.float64),
            "t_rt": np.asarray(t_rt.stat, np.float64),
            "effect_accuracy": np.asarray(t_acc.estimate, np.float64),
            "t_accuracy": np.asarray(t_acc.stat, np.float64),
            "f_modulators_joint_2df": np.asarray(f_joint.stat, np.float64),
            "rank": np.array(
                [int(fit.condition_report().runs[0].rank)],
                dtype=np.float64,
            ),
        }
    )


def make_case(max_voxels: int = MAX_VOXELS) -> ParityCase:
    """Build the parametric-modulator parity case."""
    return ParityCase(
        name="tier_a_parametric_modulators",
        fmrimod_pipeline=fmrimod_pipeline,
        reference_pipeline=nilearn_pipeline,
        inputs=load_inputs(max_voxels=max_voxels),
        tolerances={
            "design": ParityTolerance(rtol=0.0, atol=0.0),
            "effect_unmod": ParityTolerance(rtol=1e-8, atol=1e-9),
            "t_unmod": ParityTolerance(rtol=1e-7, atol=1e-8),
            "effect_rt": ParityTolerance(rtol=1e-8, atol=1e-9),
            "t_rt": ParityTolerance(rtol=1e-7, atol=1e-8),
            "effect_accuracy": ParityTolerance(rtol=1e-8, atol=1e-9),
            "t_accuracy": ParityTolerance(rtol=1e-7, atol=1e-8),
            "f_modulators_joint_2df": ParityTolerance(rtol=1e-7, atol=1e-8),
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
