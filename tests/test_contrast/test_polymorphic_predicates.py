"""Tests for the polymorphic predicate inputs on contrast constructors.

bd-01KRFMD3F66TENJMP6BQYE32HC — accept dict/string/callable in addition to the
legacy ``Formula(...)`` wrapper so user code stops writing
``[-1, 1, 0]`` weight vectors by hand.
"""

from __future__ import annotations

import pytest

import fmrimod as fm
from fmrimod.contrast.contrast_spec import (
    Formula,
    PairContrastSpec,
    OnewayContrastSpec,
    UnitContrastSpec,
    PolyContrastSpec,
    InteractionContrastSpec,
    ContrastFormulaSpec,
    _dict_to_predicate_string,
    _to_formula,
)


# -- _to_formula helper ----------------------------------------------------


def test_to_formula_passes_formula_through_unchanged():
    f = Formula("a == 1")
    assert _to_formula(f) is f


def test_to_formula_wraps_strings():
    f = _to_formula("a == 'face'")
    assert isinstance(f, Formula)
    assert f.expr == "a == 'face'"


def test_to_formula_renders_dict_predicate():
    f = _to_formula({"condition": "face"})
    assert f.expr == "condition == 'face'"


def test_to_formula_multi_key_dict_uses_and():
    f = _to_formula({"block": 1, "condition": "face"})
    # Dict iteration order is insertion order in Py3.7+
    assert "block == 1" in f.expr
    assert "condition == 'face'" in f.expr
    assert "&" in f.expr


def test_to_formula_dict_membership_list_predicate():
    f = _to_formula({"condition": ["face", "house"]})
    assert "condition in ('face', 'house')" == f.expr


def test_to_formula_attaches_callable_with_fn():
    pred = lambda cells: cells["condition"] == "face"
    f = _to_formula(pred)
    assert isinstance(f, Formula)
    assert getattr(f, "_fn", None) is pred


def test_to_formula_rejects_unknown_type():
    with pytest.raises(TypeError, match="Formula, dict, string, or callable"):
        _to_formula(42)


# -- pair_contrast ---------------------------------------------------------


def test_pair_contrast_accepts_dict_predicates():
    c = fm.pair_contrast(
        {"condition": "face"},
        {"condition": "house"},
        name="face_vs_house",
    )
    assert isinstance(c, PairContrastSpec)
    assert c.name == "face_vs_house"
    assert c.A.expr == "condition == 'face'"
    assert c.B.expr == "condition == 'house'"


def test_pair_contrast_accepts_string_predicates():
    c = fm.pair_contrast(
        "condition == 'face'",
        "condition == 'house'",
        name="face_vs_house_str",
    )
    assert c.A.expr == "condition == 'face'"


def test_pair_contrast_accepts_legacy_formula():
    c = fm.pair_contrast(Formula("a == 1"), Formula("a == 2"), name="leg")
    assert isinstance(c, PairContrastSpec)


def test_pair_contrast_accepts_callable_predicate():
    def is_face(cells):
        return cells["condition"] == "face"

    c = fm.pair_contrast(is_face, {"condition": "house"}, name="cb")
    assert getattr(c.A, "_fn", None) is is_face


def test_pair_contrast_with_where_clause_dict():
    c = fm.pair_contrast(
        {"condition": "face"},
        {"condition": "house"},
        name="early",
        where={"block": 1},
    )
    assert c.where.expr == "block == 1"


# -- unit_contrast ---------------------------------------------------------


def test_unit_contrast_accepts_dict_and_string():
    c1 = fm.unit_contrast({"task": "memory"}, name="memory_main")
    c2 = fm.unit_contrast("task == 'memory'", name="memory_main_str")
    assert isinstance(c1, UnitContrastSpec)
    assert c1.A.expr == "task == 'memory'"
    assert c2.A.expr == "task == 'memory'"


# -- oneway_contrast -------------------------------------------------------


def test_oneway_contrast_accepts_dict_membership():
    c = fm.oneway_contrast(
        {"condition": ["face", "house", "object"]}, name="category_main"
    )
    assert isinstance(c, OnewayContrastSpec)
    assert "in ('face', 'house', 'object')" in c.A.expr


# -- interaction_contrast --------------------------------------------------


def test_interaction_contrast_accepts_string():
    c = fm.interaction_contrast("condition * time", name="cond_by_time")
    assert isinstance(c, InteractionContrastSpec)
    assert c.A.expr == "condition * time"


# -- poly_contrast ---------------------------------------------------------


def test_poly_contrast_accepts_dict_and_degree():
    c = fm.poly_contrast(
        {"dose": ["low", "med", "high"]},
        name="dose_linear",
        degree=1,
        value_map={"low": 0, "med": 2, "high": 5},
    )
    assert isinstance(c, PolyContrastSpec)
    assert c.degree == 1
    assert c.value_map == {"low": 0, "med": 2, "high": 5}


# -- column_contrast (unchanged - still strict on regex string) ------------


def test_column_contrast_still_takes_regex_strings():
    c = fm.column_contrast(pattern_A="^face_", name="face_main")
    assert c.pattern_A == "^face_"


# -- contrast (top-level form) ---------------------------------------------


def test_contrast_accepts_dict_form():
    c = fm.contrast({"condition": "face"}, name="face_main")
    assert isinstance(c, ContrastFormulaSpec)
    assert c.A.expr == "condition == 'face'"


# -- top-level exposure ----------------------------------------------------


def test_constructors_visible_at_top_level():
    for name in (
        "pair_contrast",
        "unit_contrast",
        "oneway_contrast",
        "interaction_contrast",
        "poly_contrast",
        "column_contrast",
        "contrast_set",
        "pairwise_contrasts",
        "one_against_all_contrast",
        "sliding_window_contrasts",
    ):
        assert callable(getattr(fm, name)), f"fm.{name} should be callable"


def test_contrast_submodule_still_importable_via_sys_modules():
    """The submodule is shadowed at attribute level by the legacy ``contrast``
    wrapper function (predates this bead); the underlying submodule remains
    reachable via the import system / sys.modules."""
    import sys

    # The function wrapper sits at the attribute level.
    assert callable(fm.contrast)
    # The submodule is still loaded — fetch it from sys.modules.
    contrast_mod = sys.modules.get("fmrimod.contrast")
    assert contrast_mod is not None
    assert hasattr(contrast_mod, "pair_contrast")
    assert hasattr(contrast_mod, "ContrastSpec") or hasattr(
        contrast_mod, "contrast_spec"
    )


# -- contrast-of-contrasts via __sub__ -------------------------------------


def test_contrast_minus_contrast_produces_pair_diff():
    c1 = fm.unit_contrast({"condition": "face"}, name="A")
    c2 = fm.unit_contrast({"condition": "house"}, name="B")
    diff = c1 - c2
    assert isinstance(diff, PairContrastSpec)
    assert diff.name == "A-B"
