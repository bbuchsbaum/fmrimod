"""Regression tests for 3-way factorial typed-spec ergonomics.

The ``tier_a_factorial_3way`` parity workflow exercises a full
2x2x2 factorial expansion. The realised cells are correctly built
and all 7 ANOVA contrasts + the joint 8-DF F match Nilearn at
<= 1e-10 (pinned by the parametrised parity suite). This file pins
the *ergonomic surface* — what the typed spec produces and how a
user looks up cells — so a future convenience addition (factorial
contrast generator, per-factor cell lookup) has a clear target.

Pain points worth fixing in a follow-up:

1. **Level-string convention is verbose** — cells are addressed as
   ``"trial_type.A_difficulty.easy_context.off"``. A typed
   ``cols.cell(trial_type="A", difficulty="easy", context="off")``
   helper would let users address cells by factor values without
   reconstructing the level string.

2. **No factorial contrast generator** — users hand-assemble
   the 7 ANOVA contrasts by enumerating cells. A
   ``factorial_contrasts(term="trial_type:difficulty:context",
   factors=("trial_type", "difficulty", "context"))`` helper
   returning all standard ANOVA contrasts would close the gap.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
import pytest

import fmrimod as fm
from fmrimod.spec import drift, hrf, intercept


def _three_way_fit():
    """Build a balanced 2x2x2 factorial fit."""
    rng = np.random.default_rng(0)
    trial_types = ("A", "B")
    difficulties = ("easy", "hard")
    contexts = ("off", "on")
    rows = []
    cells = list(itertools.product(trial_types, difficulties, contexts))
    for _rep in range(3):
        rng.shuffle(cells)
        for tt, d, c in cells:
            rows.append({
                "trial_type": tt, "difficulty": d, "context": c,
            })
    n_rows = len(rows)
    onsets = np.linspace(12.0, 230.0, n_rows, dtype=np.float64)
    df = pd.DataFrame(rows)
    df["onset"] = onsets
    df["duration"] = 0.0
    df["run"] = 1
    df = df.sort_values("onset").reset_index(drop=True)
    ds = fm.fmri_dataset(
        np.zeros((140, 4)), tr=2.0, events=df, slice_timing_offset=0.0
    )
    spec = (
        hrf("trial_type", "difficulty", "context", basis="spm", norm="spm")
        + drift("poly", degree=2)
        + intercept(per="run")
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return fm.fmri_lm(spec, ds), trial_types, difficulties, contexts


def test_three_way_factorial_produces_eight_cells() -> None:
    """``hrf("a","b","c")`` on a balanced design yields 8 task cells."""
    fit, tts, ds, cs = _three_way_fit()
    cols = fit.design_columns()
    task_cells = list(
        cols.where(term="trial_type:difficulty:context").columns
    )
    assert len(task_cells) == 8


def test_level_string_follows_factor_dot_value_underscore_convention() -> None:
    """Pinning the current level-string convention.

    Currently each task cell's level looks like
    ``"trial_type.A_difficulty.easy_context.off"``. The convention is
    ``"{factor}.{value}_{factor}.{value}_{factor}.{value}"``. Pinning
    so a future change can flip this assertion to the desired shape.
    """
    fit, tts, ds, cs = _three_way_fit()
    cols = fit.design_columns()
    expected_levels = {
        f"trial_type.{tt}_difficulty.{d}_context.{c}"
        for tt in tts for d in ds for c in cs
    }
    actual_levels = {
        col.level
        for col in cols.where(term="trial_type:difficulty:context").columns
    }
    assert actual_levels == expected_levels


def test_each_cell_resolvable_via_level_string_lookup() -> None:
    """The verbose-but-functional cell lookup path.

    The typed ``cols.where(term="...", level="...")`` works as long as
    the user reconstructs the level string. This test pins that path
    explicitly so a future ``cols.cell(trial_type=..., ...)`` helper
    composes with it cleanly.
    """
    fit, tts, ds, cs = _three_way_fit()
    cols = fit.design_columns()
    for tt, d, c in itertools.product(tts, ds, cs):
        level = f"trial_type.{tt}_difficulty.{d}_context.{c}"
        match = cols.where(
            term="trial_type:difficulty:context", level=level,
        )
        assert len(list(match.columns)) == 1, (
            f"expected exactly one column for cell {(tt, d, c)}; "
            f"got {len(list(match.columns))}"
        )


def test_three_way_factorial_design_is_full_rank() -> None:
    """The realised 8-cell design + drift + intercept has no aliasing."""
    fit, tts, ds, cs = _three_way_fit()
    X = fit.model.design_matrix_array(run=None)
    assert np.linalg.matrix_rank(X) == X.shape[1], (
        "3-way factorial design unexpectedly rank-deficient"
    )
