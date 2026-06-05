"""Full 3-way factorial with main effects, 2-way and 3-way interactions.

The deepest typed-spec stress test: ``hrf("trial_type", "difficulty",
"context")`` produces 2x2x2 = 8 cells, and the headline contrasts
cover the *full ANOVA decomposition*:

- 3 main effects (one per factor)
- 3 two-way interactions (each pair)
- 1 three-way interaction
- 1 omnibus 8-DF F over all cells

The typed ``cols.where(term="trial_type:difficulty:context",
level=...)`` lookup populates a dictionary keyed by the 8
``(trial_type, difficulty, context)`` triples, after which the
contrast vectors are simple +/- weights placed onto the lookup
indices via standard ANOVA factor coding.

Why this is a stress test
-------------------------
Nilearn's first-level surface doesn't have a native 3-way factorial
expansion from an events DataFrame; users typically pre-build the
8-cell event types ("A_easy_off", "A_easy_on", ...) by string
concatenation and then track column positions by hand. The 7 ANOVA
contrasts on top need 7 hand-assembled vectors of length 8 each.

fmrimod's typed factorial term produces the 8 cells with proper
level strings, and the contrast assembly uses a typed lookup. The
3-way interaction t-test (a.k.a. ``A x B x C``) is the hairiest
analytic move in the standard ANOVA toolkit; this case proves the
typed API handles it cleanly.

Pattern B parity claim
----------------------
fmrimod realises the design (8 task cells + drift + intercept) via
the typed spec; both engines solve OLS on that **same** X. Eleven
outputs match at <= 1e-9:

- ``t_main_trial_type``, ``t_main_difficulty``, ``t_main_context``
  (3 main-effect t-tests).
- ``t_inter_TD``, ``t_inter_TC``, ``t_inter_DC`` (3 two-way
  interaction t-tests).
- ``t_inter_TDC`` (the 3-way interaction).
- ``f_task_omnibus_8df`` (joint F over all 8 cells).
- ``rank``, ``design`` (bitwise).

Pain points observed (logged in workflow notes)
-----------------------------------------------

The headline ergonomic ask: a built-in factorial contrast
generator (``factorial_contrasts(term="trial_type:difficulty:context",
factors=("trial_type", "difficulty", "context"))``) would return
the seven standard ANOVA contrasts as a typed object that knows
which factor each weight came from. Currently users assemble the
contrast vectors by enumerating the 8 cells in a fixed order and
encoding +/- weights by hand; this is fine for 3 factors but
becomes unwieldy at 4+.

The level-string convention
(``"trial_type.A_difficulty.easy_context.off"``) is functional but
verbose: users typing contrasts by hand have to remember the exact
factor-order and value-string convention. A helper
``cell_index(trial_type="A", difficulty="easy", context="off")``
on ``DesignColumns`` would be a clean addition that doesn't
require introspecting the level string.
"""

from __future__ import annotations

import itertools
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
N_SCANS = 360
MAX_VOXELS = 1024
TRIAL_TYPES: tuple[str, ...] = ("A", "B")
DIFFICULTIES: tuple[str, ...] = ("easy", "hard")
CONTEXTS: tuple[str, ...] = ("off", "on")
N_PER_CELL = 4  # 8 cells * 4 reps = 32 trials


@dataclass(frozen=True)
class FactorialInputs:
    """Shared inputs for the 3-way factorial parity case."""

    events: pd.DataFrame
    data: Array
    design: Array
    design_columns: DesignColumns
    cell_indices: dict[tuple[str, str, str], int]
    # 7 ANOVA contrasts + 1 omnibus F.
    c_t_main_trial_type: Array
    c_t_main_difficulty: Array
    c_t_main_context: Array
    c_t_inter_TD: Array
    c_t_inter_TC: Array
    c_t_inter_DC: Array
    c_t_inter_TDC: Array
    c_f_task_omnibus: Array


def _make_events(seed: int) -> pd.DataFrame:
    """Build deterministic factorial trials, balanced across the 8 cells."""
    rng = np.random.default_rng(seed)
    cells = list(itertools.product(TRIAL_TYPES, DIFFICULTIES, CONTEXTS))
    rows: list[dict[str, Any]] = []
    for _rep in range(N_PER_CELL):
        rng.shuffle(cells)
        for trial_type, difficulty, context in cells:
            rows.append({
                "trial_type": trial_type,
                "difficulty": difficulty,
                "context": context,
            })
    onsets = np.linspace(
        12.0, N_SCANS * TR - 30.0, len(rows), dtype=np.float64
    )
    jitter = rng.uniform(-0.8, 0.8, len(rows))
    df = pd.DataFrame(rows)
    df["onset"] = onsets + jitter
    df["duration"] = 0.0
    df["run"] = 1
    return df.sort_values("onset").reset_index(drop=True)


def _realize_design(events: pd.DataFrame) -> tuple[Array, DesignColumns]:
    """Build the 3-way factorial design via the typed spec."""
    spec = (
        hrf("trial_type", "difficulty", "context", basis="spm", norm="spm")
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
) -> dict[tuple[str, str, str], int]:
    """Locate the eight (trial_type, difficulty, context) task columns.

    Pain point: the level-string convention is
    ``"trial_type.{V}_difficulty.{V}_context.{V}"``; users typing
    contrasts by hand have to remember the factor order. A typed
    ``cell_index(trial_type=..., difficulty=..., context=...)``
    helper would close the gap.
    """
    term_name = "trial_type:difficulty:context"
    cells: dict[tuple[str, str, str], int] = {}
    for tt, diff, ctx in itertools.product(
        TRIAL_TYPES, DIFFICULTIES, CONTEXTS
    ):
        level = f"trial_type.{tt}_difficulty.{diff}_context.{ctx}"
        cells[(tt, diff, ctx)] = columns.where(
            term=term_name, level=level
        ).one().index
    return cells


def _build_contrasts(
    cells: dict[tuple[str, str, str], int],
    n_total: int,
) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
    """Assemble the seven ANOVA contrasts + the joint 8-DF F.

    Factor coding (-1 / +1):
      trial_type: A=-1, B=+1
      difficulty: easy=-1, hard=+1
      context:    off=-1, on=+1

    Main effect of factor X = mean of (X_code * cell_value) over cells.
    Two-way interaction = mean of (X_code * Y_code * cell_value).
    Three-way interaction = mean of (X_code * Y_code * Z_code * cell_value).
    """
    tt_code = {"A": -1.0, "B": +1.0}
    diff_code = {"easy": -1.0, "hard": +1.0}
    ctx_code = {"off": -1.0, "on": +1.0}

    def _build(weight_fn) -> Array:
        c = np.zeros(n_total, dtype=np.float64)
        for cell, idx in cells.items():
            tt, diff, ctx = cell
            c[idx] = weight_fn(tt, diff, ctx)
        return c

    # Main effects (normalised so the contrast estimates a difference
    # of cell means).
    c_main_tt = _build(lambda tt, d, c: tt_code[tt] / 4.0)
    c_main_diff = _build(lambda tt, d, c: diff_code[d] / 4.0)
    c_main_ctx = _build(lambda tt, d, c: ctx_code[c] / 4.0)
    # Two-way interactions.
    c_inter_TD = _build(
        lambda tt, d, c: tt_code[tt] * diff_code[d] / 4.0
    )
    c_inter_TC = _build(
        lambda tt, d, c: tt_code[tt] * ctx_code[c] / 4.0
    )
    c_inter_DC = _build(
        lambda tt, d, c: diff_code[d] * ctx_code[c] / 4.0
    )
    # Three-way interaction.
    c_inter_TDC = _build(
        lambda tt, d, c: tt_code[tt] * diff_code[d] * ctx_code[c] / 4.0
    )
    # Omnibus 8-DF F over the eight cells.
    c_f_omni = np.zeros((len(cells), n_total), dtype=np.float64)
    for row, idx in enumerate(sorted(cells.values())):
        c_f_omni[row, idx] = 1.0

    return (
        c_main_tt, c_main_diff, c_main_ctx,
        c_inter_TD, c_inter_TC, c_inter_DC,
        c_inter_TDC, c_f_omni,
    )


def load_inputs(
    max_voxels: int = MAX_VOXELS, seed: int = 20260523
) -> FactorialInputs:
    """Synthesize BOLD with known main effects + a specific 3-way interaction.

    The generating model places:
    - A clear A vs B main effect.
    - A modest hard vs easy main effect.
    - A small on vs off main effect.
    - A meaningful 3-way (trial_type x difficulty x context)
      interaction so the headline T_inter_TDC contrast has
      detectable signal.
    """
    rng = np.random.default_rng(seed)
    events = _make_events(seed)
    design, columns = _realize_design(events)
    cells = _cell_indices(columns)
    n_total = design.shape[1]

    n_voxels = min(int(max_voxels), MAX_VOXELS)
    betas = np.zeros((n_total, n_voxels), dtype=np.float64)
    voxel_ramp = np.linspace(0.4, 1.6, n_voxels, dtype=np.float64)

    tt_code = {"A": -1.0, "B": +1.0}
    diff_code = {"easy": -1.0, "hard": +1.0}
    ctx_code = {"off": -1.0, "on": +1.0}
    for cell, idx in cells.items():
        tt, d, c = cell
        # Sum of main effects + 3-way interaction:
        amp = (
            0.50  # grand mean
            + 0.40 * tt_code[tt]                # A < B
            + 0.20 * diff_code[d]               # easy < hard
            + 0.10 * ctx_code[c]                # off < on
            + 0.25 * tt_code[tt] * diff_code[d] * ctx_code[c]  # 3-way
        )
        betas[idx] = amp * voxel_ramp

    for c in columns.columns:
        if c.role == "intercept":
            betas[c.index] = 100.0 + rng.normal(scale=0.4, size=n_voxels)
        elif c.role == "drift":
            betas[c.index] = rng.normal(scale=0.15, size=n_voxels)

    data = design @ betas + rng.normal(
        scale=0.35, size=(N_SCANS, n_voxels)
    )

    (c_t_main_tt, c_t_main_diff, c_t_main_ctx,
     c_t_inter_TD, c_t_inter_TC, c_t_inter_DC,
     c_t_inter_TDC, c_f_omni) = _build_contrasts(cells, n_total)

    return FactorialInputs(
        events=events,
        data=data.astype(np.float64),
        design=design,
        design_columns=columns,
        cell_indices=cells,
        c_t_main_trial_type=c_t_main_tt,
        c_t_main_difficulty=c_t_main_diff,
        c_t_main_context=c_t_main_ctx,
        c_t_inter_TD=c_t_inter_TD,
        c_t_inter_TC=c_t_inter_TC,
        c_t_inter_DC=c_t_inter_DC,
        c_t_inter_TDC=c_t_inter_TDC,
        c_f_task_omnibus=c_f_omni,
    )


def nilearn_pipeline(inputs: FactorialInputs) -> PipelineOutput:
    """Reference: ``run_glm`` on the fmrimod-realised 3-way design."""
    labels, estimates = run_glm(inputs.data, inputs.design, noise_model="ols")
    t_main_tt = compute_contrast(
        labels, estimates, inputs.c_t_main_trial_type, stat_type="t"
    )
    t_main_diff = compute_contrast(
        labels, estimates, inputs.c_t_main_difficulty, stat_type="t"
    )
    t_main_ctx = compute_contrast(
        labels, estimates, inputs.c_t_main_context, stat_type="t"
    )
    t_inter_TD = compute_contrast(
        labels, estimates, inputs.c_t_inter_TD, stat_type="t"
    )
    t_inter_TC = compute_contrast(
        labels, estimates, inputs.c_t_inter_TC, stat_type="t"
    )
    t_inter_DC = compute_contrast(
        labels, estimates, inputs.c_t_inter_DC, stat_type="t"
    )
    t_inter_TDC = compute_contrast(
        labels, estimates, inputs.c_t_inter_TDC, stat_type="t"
    )
    f_omni = compute_contrast(
        labels, estimates, inputs.c_f_task_omnibus, stat_type="F"
    )
    rank_observed = int(np.linalg.matrix_rank(inputs.design))
    return PipelineOutput(
        arrays={
            "design": inputs.design,
            "t_main_trial_type": np.asarray(t_main_tt.stat(), np.float64),
            "t_main_difficulty": np.asarray(t_main_diff.stat(), np.float64),
            "t_main_context": np.asarray(t_main_ctx.stat(), np.float64),
            "t_inter_TD": np.asarray(t_inter_TD.stat(), np.float64),
            "t_inter_TC": np.asarray(t_inter_TC.stat(), np.float64),
            "t_inter_DC": np.asarray(t_inter_DC.stat(), np.float64),
            "t_inter_TDC": np.asarray(t_inter_TDC.stat(), np.float64),
            "f_task_omnibus_8df": np.asarray(f_omni.stat(), np.float64),
            "rank": np.array([rank_observed], dtype=np.float64),
        }
    )


def fmrimod_pipeline(inputs: FactorialInputs) -> PipelineOutput:
    """Typed fmrimod path: ``fm.fmri_lm(spec, ds, engine="concat")``."""
    spec = (
        hrf("trial_type", "difficulty", "context", basis="spm", norm="spm")
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
    t_main_tt = fit.contrast(inputs.c_t_main_trial_type, name="main_TT")
    t_main_diff = fit.contrast(inputs.c_t_main_difficulty, name="main_Diff")
    t_main_ctx = fit.contrast(inputs.c_t_main_context, name="main_Ctx")
    t_inter_TD = fit.contrast(inputs.c_t_inter_TD, name="inter_TD")
    t_inter_TC = fit.contrast(inputs.c_t_inter_TC, name="inter_TC")
    t_inter_DC = fit.contrast(inputs.c_t_inter_DC, name="inter_DC")
    t_inter_TDC = fit.contrast(inputs.c_t_inter_TDC, name="inter_TDC")
    f_omni = fit.contrast(inputs.c_f_task_omnibus, name="omnibus")
    return PipelineOutput(
        arrays={
            "design": fit.model.design_matrix_array(run=None),
            "t_main_trial_type": np.asarray(t_main_tt.stat, np.float64),
            "t_main_difficulty": np.asarray(t_main_diff.stat, np.float64),
            "t_main_context": np.asarray(t_main_ctx.stat, np.float64),
            "t_inter_TD": np.asarray(t_inter_TD.stat, np.float64),
            "t_inter_TC": np.asarray(t_inter_TC.stat, np.float64),
            "t_inter_DC": np.asarray(t_inter_DC.stat, np.float64),
            "t_inter_TDC": np.asarray(t_inter_TDC.stat, np.float64),
            "f_task_omnibus_8df": np.asarray(f_omni.stat, np.float64),
            "rank": np.array(
                [int(fit.condition_report().runs[0].rank)],
                dtype=np.float64,
            ),
        }
    )


def make_case(max_voxels: int = MAX_VOXELS) -> ParityCase:
    """Build the 3-way factorial parity case."""
    return ParityCase(
        name="tier_a_factorial_3way",
        fmrimod_pipeline=fmrimod_pipeline,
        reference_pipeline=nilearn_pipeline,
        inputs=load_inputs(max_voxels=max_voxels),
        tolerances={
            "design": ParityTolerance(rtol=0.0, atol=0.0),
            "t_main_trial_type": ParityTolerance(rtol=1e-7, atol=1e-8),
            "t_main_difficulty": ParityTolerance(rtol=1e-7, atol=1e-8),
            "t_main_context": ParityTolerance(rtol=1e-7, atol=1e-8),
            "t_inter_TD": ParityTolerance(rtol=1e-7, atol=1e-8),
            "t_inter_TC": ParityTolerance(rtol=1e-7, atol=1e-8),
            "t_inter_DC": ParityTolerance(rtol=1e-7, atol=1e-8),
            "t_inter_TDC": ParityTolerance(rtol=1e-7, atol=1e-8),
            "f_task_omnibus_8df": ParityTolerance(rtol=1e-7, atol=1e-8),
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
