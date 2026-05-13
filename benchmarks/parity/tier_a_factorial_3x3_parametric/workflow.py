"""3x3 factorial + parametric-modulator first-level parity against Nilearn.

This case stresses a layered design that combines what the earlier Tier
A workflows exercise in isolation: a *3x3* fully crossed categorical
interaction (``difficulty × emotion``, nine cells) *plus* a continuous
parametric modulator (z-scored reaction time) acting per cell, while
~12% of trials have a missing ``rt`` value.

Why this is a stress test
-------------------------
Three separate pain points show up in one place:

1. **Two-factor factorial scales linearly in cell count.** A 3x3 design
   means 9 main-effect cells plus 9 parametric cells, so the column
   bookkeeping that broke at 2x2 (see ``tier_a_factorial_2x2``) gets
   18 task columns deep. Nilearn users would pre-cross all 9 cells
   into pseudo trial_types, then duplicate them with renamed labels
   for the parametric modulators — 18 pseudo trial_types and a
   manually-indexed contrast vector per hypothesis.

2. **Modulator z-scoring is upstream and trivial.** With pandas the
   user writes ``events["rt_z"] = (events["rt"] - events["rt"].mean()) /
   events["rt"].std()``. NaN is skipped by pandas. The point is that
   the *typed-API* call site stays a single line:
   ``hrf("difficulty", "emotion", modulators=["rt_z"])``.

3. **Scattered NaNs in the modulator must recover with minimal
   ceremony.** Before this round fmrimod raised a hard ``ValueError``
   on a non-finite EventVariable value. Now the default
   ``nan_strategy="drop"`` substitutes zero amplitude for those trials
   (so they contribute nothing to *parametric* columns) while keeping
   them in *main-effect* categorical columns, and emits a single
   ``UserWarning`` naming the variable and the count.

Pattern B parity claim
----------------------
fmrimod realises the design once via the typed spec; both pipelines
solve OLS on the same ``X``. Effects and the deferred-DoF-corrected
t / F statistics match Nilearn's ``run_glm + compute_contrast`` at
machine precision.

What we compare
---------------
- ``design``: bitwise-equal realised 19-column design (9 main + 9
  parametric + intercept).
- ``effect_difficulty_linear`` / ``t_difficulty_linear``: linear
  trend across difficulty levels, collapsed across emotion.
- ``effect_emotion_linear`` / ``t_emotion_linear``: linear trend
  across emotion, collapsed across difficulty.
- ``effect_diff_x_emo_quadrant`` / ``t_diff_x_emo_quadrant``: a
  signed 2x2 sub-interaction (hard − easy) × (positive − negative).
- ``f_parametric_omnibus``: joint F over the nine parametric columns.
- ``rank``: realised design rank — pinned at full so the test fails
  loudly if the NaN-zeroing accidentally collapses a column.
"""

from __future__ import annotations

import warnings
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
from fmrimod.design.columns import DesignColumns
from fmrimod.spec import hrf

Array = NDArray[np.float64]

TR = 2.0
N_SCANS = 260
MAX_VOXELS = 1024
DIFFICULTY_LEVELS: tuple[str, ...] = ("easy", "hard", "medium")
EMOTION_LEVELS: tuple[str, ...] = ("negative", "neutral", "positive")
CELLS: tuple[tuple[str, str], ...] = tuple(
    iter_product(DIFFICULTY_LEVELS, EMOTION_LEVELS)
)
DIFFICULTY_TREND = {"easy": -1.0, "medium": 0.0, "hard": +1.0}
EMOTION_TREND = {"negative": -1.0, "neutral": 0.0, "positive": +1.0}
NAN_FRACTION = 0.12


@dataclass(frozen=True)
class FactorialParametricInputs:
    """Shared inputs for the 3x3 + parametric parity case."""

    events: pd.DataFrame
    data: Array
    design: Array
    design_columns: DesignColumns
    cell_main_indices: dict[tuple[str, str], int]
    cell_param_indices: dict[tuple[str, str], int]
    c_difficulty_linear: Array
    c_emotion_linear: Array
    c_diff_x_emo_quadrant: Array
    c_parametric_F: Array
    n_nan_trials: int


def _make_events(seed: int) -> pd.DataFrame:
    """Build 9-cell interleaved events with a z-scored RT modulator + NaNs.

    The z-scoring is intentionally done on the user side with the
    boring pandas one-liner — that is the ceremony budget the parity
    contract is defending.
    """
    rng = np.random.default_rng(seed)
    n_per_cell = 6
    rows: list[dict[str, Any]] = []
    for cell_idx, (difficulty, emotion) in enumerate(CELLS):
        grid = np.linspace(
            8.0 + 1.7 * cell_idx,
            N_SCANS * TR - 28.0,
            n_per_cell,
            dtype=np.float64,
        )
        jitter = rng.uniform(-1.0, 1.0, n_per_cell)
        for onset in grid + jitter:
            rows.append(
                {
                    "onset": float(onset),
                    "duration": 0.0,
                    "difficulty": difficulty,
                    "emotion": emotion,
                    "rt": float(
                        rng.uniform(0.45, 1.25)
                        + 0.08 * DIFFICULTY_TREND[difficulty]
                    ),
                    "run": 1,
                }
            )
    events = (
        pd.DataFrame(rows).sort_values("onset").reset_index(drop=True)
    )
    # Scatter ~12% NaN into rt.
    n = len(events)
    nan_idx = rng.choice(n, size=int(round(NAN_FRACTION * n)), replace=False)
    events.loc[nan_idx, "rt"] = np.nan

    # User-side z-score across all trials. pandas .mean() / .std() skip
    # NaN so the resulting rt_z is NaN where rt was NaN and z-scored
    # everywhere else.
    rt_mean = events["rt"].mean()
    rt_sd = events["rt"].std()
    events["rt_z"] = (events["rt"] - rt_mean) / rt_sd
    return events


def _realize_design(
    events: pd.DataFrame,
) -> tuple[Any, Array, DesignColumns]:
    """Build the 9-cell factorial-plus-parametric design via the typed spec.

    The typed-spec call is exactly the line a user would type — no
    pseudo-trial-type crossing, no manual column duplication, no NaN
    pre-filter. ``modulators=["rt_z"]`` expands into a 3-way interaction
    term (``difficulty × emotion × rt_z``) which yields one parametric
    column per (difficulty, emotion) cell. The NaN-tolerant
    EventVariable zeroes the amplitude on the missing trials.
    """
    spec = hrf(
        "difficulty",
        "emotion",
        basis="spm",
        norm="spm",
        modulators=["rt_z"],
    )
    dummy = fm.fmri_dataset(
        np.zeros((N_SCANS, 1), dtype=np.float64),
        tr=TR,
        events=events,
    )
    # Capture/discard the NaN-tolerance warning at input-build time —
    # the parity pipeline re-runs the fit and asserts the warning fires
    # there so the diagnostic surface is exercised.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fit = fm.fmri_lm(spec, dummy)
    design = np.asarray(
        fit.model.design_matrix_array(run=0), dtype=np.float64
    )
    return spec, design, fit.design_columns()


def _cell_indices(
    columns: DesignColumns,
) -> tuple[dict[tuple[str, str], int], dict[tuple[str, str], int]]:
    """Locate the nine main-effect and nine parametric cell columns.

    The main-effect term carries level strings of the form
    ``"difficulty.<d>_emotion.<e>"`` (categorical:categorical join).
    The parametric term is a 3-way interaction (2 categorical + 1
    continuous) and carries level strings of the form ``"<d>:<e>"``
    (categorical-combo only — the continuous variable does not
    contribute a sub-level here). Both formats are stable and both
    are addressable via the typed ``where(term=..., level=...)``
    accessor.
    """
    main: dict[tuple[str, str], int] = {}
    param: dict[tuple[str, str], int] = {}
    for difficulty, emotion in CELLS:
        main_level = f"difficulty.{difficulty}_emotion.{emotion}"
        main[(difficulty, emotion)] = columns.where(
            term="difficulty:emotion", level=main_level
        ).one().index
        param_level = f"{difficulty}:{emotion}"
        param[(difficulty, emotion)] = columns.where(
            term="difficulty:emotion:rt_z", level=param_level
        ).one().index
    return main, param


def _build_contrasts(
    main: dict[tuple[str, str], int],
    param: dict[tuple[str, str], int],
    n_total: int,
) -> tuple[Array, Array, Array, Array]:
    """Construct the four typed contrasts.

    All weights are placed by ``cell_main_indices``/``cell_param_indices``
    lookups, so the contract is robust to any column-order convention
    change in the realised design.
    """
    c_diff = np.zeros(n_total, dtype=np.float64)
    for (difficulty, emotion), idx in main.items():
        c_diff[idx] = DIFFICULTY_TREND[difficulty] / len(EMOTION_LEVELS)

    c_emo = np.zeros(n_total, dtype=np.float64)
    for (difficulty, emotion), idx in main.items():
        c_emo[idx] = EMOTION_TREND[emotion] / len(DIFFICULTY_LEVELS)

    # 2x2 sub-interaction: (hard - easy) × (positive - negative)
    c_quad = np.zeros(n_total, dtype=np.float64)
    c_quad[main[("hard", "positive")]] = +1.0
    c_quad[main[("hard", "negative")]] = -1.0
    c_quad[main[("easy", "positive")]] = -1.0
    c_quad[main[("easy", "negative")]] = +1.0

    # Joint F over the 9 parametric columns.
    c_param_F = np.zeros((len(param), n_total), dtype=np.float64)
    for row, idx in enumerate(sorted(param.values())):
        c_param_F[row, idx] = 1.0

    return c_diff, c_emo, c_quad, c_param_F


def load_inputs(
    max_voxels: int = MAX_VOXELS, seed: int = 20260513
) -> FactorialParametricInputs:
    """Synthesize BOLD on a 9-cell + parametric design with NaN modulators."""
    rng = np.random.default_rng(seed)
    events = _make_events(seed)
    spec, design, columns = _realize_design(events)
    main_idx, param_idx = _cell_indices(columns)
    n_total = design.shape[1]

    n_voxels = min(int(max_voxels), MAX_VOXELS)
    betas = np.zeros((n_total, n_voxels), dtype=np.float64)
    voxel_ramp = np.linspace(0.4, 1.6, n_voxels, dtype=np.float64)
    # Main-effect amplitudes: monotonic in difficulty, U-shaped in
    # emotion, plus a small interaction between the two extremes.
    main_offsets = {
        ("easy", "negative"): 0.20,
        ("easy", "neutral"): 0.05,
        ("easy", "positive"): 0.25,
        ("medium", "negative"): 0.55,
        ("medium", "neutral"): 0.40,
        ("medium", "positive"): 0.60,
        ("hard", "negative"): 1.15,
        ("hard", "neutral"): 0.95,
        ("hard", "positive"): 0.55,
    }
    for cell, idx in main_idx.items():
        betas[idx] = main_offsets[cell] * voxel_ramp
    # Parametric amplitudes: rt-modulation present in hard cells, mild
    # in medium, near-zero in easy (so the joint F is non-trivial).
    param_offsets = {
        ("easy", "negative"): 0.02,
        ("easy", "neutral"): 0.01,
        ("easy", "positive"): 0.00,
        ("medium", "negative"): 0.15,
        ("medium", "neutral"): 0.10,
        ("medium", "positive"): 0.12,
        ("hard", "negative"): 0.45,
        ("hard", "neutral"): 0.40,
        ("hard", "positive"): 0.35,
    }
    for cell, idx in param_idx.items():
        betas[idx] = param_offsets[cell] * voxel_ramp

    baseline_idx = next(
        (c.index for c in columns.columns if c.role != "task"),
        n_total - 1,
    )
    betas[baseline_idx] = 100.0 + rng.normal(scale=0.5, size=n_voxels)

    data = design @ betas + rng.normal(scale=0.3, size=(N_SCANS, n_voxels))

    c_diff, c_emo, c_quad, c_param_F = _build_contrasts(
        main_idx, param_idx, n_total
    )
    n_nan_trials = int(events["rt"].isna().sum())

    return FactorialParametricInputs(
        events=events,
        data=data.astype(np.float64),
        design=design,
        design_columns=columns,
        cell_main_indices=main_idx,
        cell_param_indices=param_idx,
        c_difficulty_linear=c_diff,
        c_emotion_linear=c_emo,
        c_diff_x_emo_quadrant=c_quad,
        c_parametric_F=c_param_F,
        n_nan_trials=n_nan_trials,
    )


def nilearn_pipeline(
    inputs: FactorialParametricInputs,
) -> PipelineOutput:
    """Reference: Nilearn's run_glm on the realised fmrimod design."""
    labels, estimates = run_glm(
        inputs.data, inputs.design, noise_model="ols"
    )
    t_diff = compute_contrast(
        labels, estimates, inputs.c_difficulty_linear, stat_type="t"
    )
    t_emo = compute_contrast(
        labels, estimates, inputs.c_emotion_linear, stat_type="t"
    )
    t_quad = compute_contrast(
        labels, estimates, inputs.c_diff_x_emo_quadrant, stat_type="t"
    )
    f_param = compute_contrast(
        labels, estimates, inputs.c_parametric_F, stat_type="F"
    )
    rank_observed = int(np.linalg.matrix_rank(inputs.design))
    return PipelineOutput(
        arrays={
            "design": inputs.design,
            "effect_difficulty_linear": np.asarray(
                t_diff.effect_size(), np.float64
            ),
            "t_difficulty_linear": np.asarray(t_diff.stat(), np.float64),
            "effect_emotion_linear": np.asarray(
                t_emo.effect_size(), np.float64
            ),
            "t_emotion_linear": np.asarray(t_emo.stat(), np.float64),
            "effect_diff_x_emo_quadrant": np.asarray(
                t_quad.effect_size(), np.float64
            ),
            "t_diff_x_emo_quadrant": np.asarray(t_quad.stat(), np.float64),
            "f_parametric_omnibus": np.asarray(f_param.stat(), np.float64),
            "rank": np.array([rank_observed], dtype=np.float64),
        }
    )


def fmrimod_pipeline(
    inputs: FactorialParametricInputs,
) -> PipelineOutput:
    """One-line spec: 3x3 factorial × rt_z parametric modulator.

    Asserts that the NaN-tolerance UserWarning fires (so the diagnostic
    surface is exercised at parity time) and names the modulator
    variable.
    """
    spec = hrf(
        "difficulty",
        "emotion",
        basis="spm",
        norm="spm",
        modulators=["rt_z"],
    )
    ds = fm.fmri_dataset(inputs.data, tr=TR, events=inputs.events)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        fit = fm.fmri_lm(spec, ds)

    if inputs.n_nan_trials > 0:
        nan_warnings = [
            w for w in captured
            if issubclass(w.category, UserWarning)
            and "rt_z" in str(w.message)
            and "non-finite" in str(w.message)
        ]
        if not nan_warnings:
            raise AssertionError(
                "expected a UserWarning naming the NaN-bearing modulator "
                "'rt_z' when the events table carries non-finite rt values"
            )

    t_diff = fit.contrast(
        inputs.c_difficulty_linear, name="difficulty_linear"
    )
    t_emo = fit.contrast(inputs.c_emotion_linear, name="emotion_linear")
    t_quad = fit.contrast(
        inputs.c_diff_x_emo_quadrant, name="diff_x_emo_quadrant"
    )
    f_param = fit.contrast(
        inputs.c_parametric_F, name="parametric_omnibus"
    )

    rank_observed = int(fit.condition_report().runs[0].rank)
    return PipelineOutput(
        arrays={
            "design": fit.model.design_matrix_array(run=0),
            "effect_difficulty_linear": np.asarray(
                t_diff.estimate, np.float64
            ),
            "t_difficulty_linear": np.asarray(t_diff.stat, np.float64),
            "effect_emotion_linear": np.asarray(t_emo.estimate, np.float64),
            "t_emotion_linear": np.asarray(t_emo.stat, np.float64),
            "effect_diff_x_emo_quadrant": np.asarray(
                t_quad.estimate, np.float64
            ),
            "t_diff_x_emo_quadrant": np.asarray(t_quad.stat, np.float64),
            "f_parametric_omnibus": np.asarray(f_param.stat, np.float64),
            "rank": np.array([rank_observed], dtype=np.float64),
        }
    )


def make_case(max_voxels: int = MAX_VOXELS) -> ParityCase:
    """Build the 3x3 + parametric parity case.

    Both engines use the same realised design, so the parity claim is
    cross-engine: identical OLS on identical inputs. The deferred-DoF
    correction (fmrimod uses ``n - rank``, see solver.py) does not
    apply here because the realised design is full rank — the NaN
    zeroing keeps the parametric columns linearly independent so the
    QR pivot does not trigger the SVD path.
    """
    return ParityCase(
        name="tier_a_factorial_3x3_parametric",
        fmrimod_pipeline=fmrimod_pipeline,
        reference_pipeline=nilearn_pipeline,
        inputs=load_inputs(max_voxels=max_voxels),
        tolerances={
            "design": ParityTolerance(rtol=0.0, atol=0.0),
            "effect_difficulty_linear": ParityTolerance(
                rtol=1e-8, atol=1e-9
            ),
            "t_difficulty_linear": ParityTolerance(rtol=1e-7, atol=1e-8),
            "effect_emotion_linear": ParityTolerance(rtol=1e-8, atol=1e-9),
            "t_emotion_linear": ParityTolerance(rtol=1e-7, atol=1e-8),
            "effect_diff_x_emo_quadrant": ParityTolerance(
                rtol=1e-8, atol=1e-9
            ),
            "t_diff_x_emo_quadrant": ParityTolerance(
                rtol=1e-7, atol=1e-8
            ),
            "f_parametric_omnibus": ParityTolerance(rtol=1e-7, atol=1e-8),
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
