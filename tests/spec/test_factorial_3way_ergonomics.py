"""Regression tests for 3-way factorial typed-spec ergonomics.

The ``tier_a_factorial_3way`` parity workflow exercises a full
2x2x2 factorial expansion; all 7 ANOVA contrasts + the joint 8-DF F
match Nilearn at <= 1e-10 (pinned by the parametrised parity
suite). This file pins the typed ergonomic surface added in the
same series of commits:

1. ``DesignColumns.cell(trial_type="A", difficulty="easy",
   context="off")`` — typed cell lookup that parses the level
   string against the known factor list, so users address cells
   by factor values rather than reconstructing the
   ``"<f1>.<v1>_<f2>.<v2>_..."`` convention.

2. ``fmrimod.contrast.anova_contrasts(columns, term=..., factors=
   (...))`` — high-level ANOVA contrast generator returning the
   full main-effect + interaction + omnibus decomposition as
   ready-to-use contrast vectors. Restricted to binary factors;
   higher arity uses the lower-level
   ``generate_interaction_contrast`` building block.
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


# -- Fixed: typed cell lookup -----------------------------------------------


def test_cell_lookup_by_factor_kwargs() -> None:
    """``cols.cell(trial_type="A", difficulty="easy", context="off")``."""
    fit, tts, ds, cs = _three_way_fit()
    cols = fit.design_columns()
    # Every cell resolves to a unique column.
    seen = set()
    for tt in tts:
        for d in ds:
            for c in cs:
                col = cols.cell(
                    trial_type=tt, difficulty=d, context=c,
                )
                assert col.role == "task"
                assert col.index not in seen
                seen.add(col.index)
    assert len(seen) == 8


def test_cell_lookup_infers_term_when_unambiguous() -> None:
    """``term=`` can be omitted when one factorial term covers the kwargs."""
    fit, *_ = _three_way_fit()
    cols = fit.design_columns()
    col_explicit = cols.cell(
        term="trial_type:difficulty:context",
        trial_type="A", difficulty="easy", context="off",
    )
    col_inferred = cols.cell(
        trial_type="A", difficulty="easy", context="off",
    )
    assert col_explicit.index == col_inferred.index


def test_cell_lookup_raises_on_unknown_value() -> None:
    """``cols.cell(...)`` with an unknown value raises ``KeyError``."""
    fit, *_ = _three_way_fit()
    cols = fit.design_columns()
    with pytest.raises(KeyError, match="no column matched"):
        cols.cell(
            trial_type="A", difficulty="easy", context="nonexistent",
        )


def test_cell_lookup_raises_on_unknown_factor() -> None:
    """``cols.cell(...)`` with an unknown factor raises ``ValueError``."""
    fit, *_ = _three_way_fit()
    cols = fit.design_columns()
    with pytest.raises(ValueError, match="no factorial term covers"):
        cols.cell(unknown_factor="x")


# -- Fixed: typed ANOVA contrast generator ---------------------------------


def test_anova_contrasts_returns_full_decomposition() -> None:
    """``anova_contrasts`` returns all main effects + all interactions + omnibus."""
    from fmrimod.contrast import anova_contrasts

    fit, *_ = _three_way_fit()
    ac = anova_contrasts(
        fit.design_columns(),
        term="trial_type:difficulty:context",
        factors=("trial_type", "difficulty", "context"),
    )
    # 3 main effects, 3 two-way, 1 three-way = 7 t-contrasts + omnibus.
    assert set(ac.main.keys()) == {"trial_type", "difficulty", "context"}
    assert set(ac.interaction.keys()) == {
        ("trial_type", "difficulty"),
        ("trial_type", "context"),
        ("difficulty", "context"),
        ("trial_type", "difficulty", "context"),
    }
    # Omnibus is 8-DF F over the 8 cells.
    assert ac.omnibus.shape[0] == 8


def test_anova_contrasts_main_effect_sums_to_zero() -> None:
    """Main-effect contrast weights sum to zero across the cells."""
    from fmrimod.contrast import anova_contrasts

    fit, *_ = _three_way_fit()
    ac = anova_contrasts(
        fit.design_columns(),
        term="trial_type:difficulty:context",
        factors=("trial_type", "difficulty", "context"),
    )
    for name, c in ac.main.items():
        assert abs(float(c.sum())) < 1e-12, (
            f"main effect {name!r} weights should sum to zero; "
            f"got {c.sum()}"
        )


def test_anova_contrasts_interaction_sums_to_zero() -> None:
    """Every interaction contrast weight sums to zero across cells."""
    from fmrimod.contrast import anova_contrasts

    fit, *_ = _three_way_fit()
    ac = anova_contrasts(
        fit.design_columns(),
        term="trial_type:difficulty:context",
        factors=("trial_type", "difficulty", "context"),
    )
    for name, c in ac.interaction.items():
        assert abs(float(c.sum())) < 1e-12, (
            f"interaction {':'.join(name)!r} weights should sum to "
            f"zero; got {c.sum()}"
        )


def test_anova_contrasts_rejects_non_binary_factor() -> None:
    """3-level factors raise ``NotImplementedError`` from the binary helper."""
    from fmrimod.contrast import anova_contrasts

    rng = np.random.default_rng(0)
    rows = []
    for tt in ("A", "B"):
        for d in ("easy", "med", "hard"):  # 3 levels — not supported
            for _rep in range(2):
                rows.append({"trial_type": tt, "difficulty": d})
    df = pd.DataFrame(rows)
    df["onset"] = np.linspace(10, 180, len(df))
    df["duration"] = 0.0
    df["run"] = 1
    ds = fm.fmri_dataset(
        np.zeros((120, 4)), tr=2.0, events=df, slice_timing_offset=0.0
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fit = fm.fmri_lm(
            hrf("trial_type", "difficulty", basis="spm", norm="spm")
            + intercept(per="run"),
            ds,
        )
    with pytest.raises(NotImplementedError, match="binary"):
        anova_contrasts(
            fit.design_columns(),
            term="trial_type:difficulty",
            factors=("trial_type", "difficulty"),
        )


def test_anova_contrasts_matches_hand_built_contrasts() -> None:
    """Bitwise agreement between ``anova_contrasts`` and the manual assembly.

    Pinning that the typed generator returns the same contrast
    vectors the parity workflow's ``_build_contrasts`` would
    produce, so refactors of either path can be diffed cleanly.
    """
    from fmrimod.contrast import anova_contrasts

    fit, *_ = _three_way_fit()
    cols = fit.design_columns()
    n_total = len([1 for _ in cols.columns])
    # Manual main effect for trial_type (binary): -0.25 on A cells,
    # +0.25 on B cells.
    expected_main_tt = np.zeros(n_total)
    for tt in ("A", "B"):
        for d in ("easy", "hard"):
            for c in ("off", "on"):
                idx = cols.cell(
                    trial_type=tt, difficulty=d, context=c,
                ).index
                expected_main_tt[idx] = -0.25 if tt == "A" else +0.25
    ac = anova_contrasts(
        cols,
        term="trial_type:difficulty:context",
        factors=("trial_type", "difficulty", "context"),
    )
    np.testing.assert_array_equal(ac.main["trial_type"], expected_main_tt)
