"""Nested contrast sets flatten in argument order (inventory v2 DES-19).

Port of R fmridesign ``test-contrast-set-nesting.R`` (2a98707, #33).
"""

import numpy as np
import pandas as pd
import pytest

from fmrimod import event_model
from fmrimod.contrast import (
    contrast_set,
    contrast_weights,
    one_against_all_contrast,
    pair_contrast,
    pairwise_contrasts,
    unit_contrast,
)
from fmrimod.contrast.contrast_spec import ContrastSet, Formula
from fmrimod.contrast.errors import ContrastError
from fmrimod.formula.base import Term

LEVELS = ["Degraded", "Foreign", "Intact"]


def _pair():
    return pair_contrast(
        Formula("condition == 'Intact'"), Formula("condition == 'Degraded'"),
        name="intelligibility",
    )


def test_helper_sets_combine_with_individual_specs():
    p = _pair()
    o = one_against_all_contrast(LEVELS, "condition")
    combined = contrast_set(p, o)
    assert isinstance(combined, ContrastSet)
    assert len(combined) == 4
    assert combined[0] is p
    assert [c.name for c in combined] == ["intelligibility"] + [c.name for c in o]
    assert list(combined) == list(contrast_set(p, *o))


def test_nested_set_weights_through_the_model():
    combined = contrast_set(_pair(), one_against_all_contrast(LEVELS, "condition"))
    ev = pd.DataFrame({"onset": np.arange(0.0, 60.0, 10.0), "condition": LEVELS * 2})
    term = Term("condition")
    term.contrast_specs = combined
    em = event_model([term], data=ev, tr=1.0, n_scans=60)
    weights = contrast_weights(em)["condition"]
    assert list(weights) == [c.name for c in combined]
    expected = [[-1, 0, 1], [1, -0.5, -0.5], [-0.5, 1, -0.5], [-0.5, -0.5, 1]]
    for con, exp in zip(combined, expected, strict=True):
        np.testing.assert_allclose(weights[con.name].offset_weights[:, 0], exp)


def test_recursive_flattening_keeps_order_and_drops_empty_sets():
    a, b = _pair(), one_against_all_contrast(LEVELS, "condition")[0]
    assert list(contrast_set(contrast_set(a, contrast_set(b)), contrast_set())) == [a, b]
    assert list(contrast_set(contrast_set(), contrast_set())) == []
    pairs = pairwise_contrasts(["A", "B", "C"], "condition")
    assert list(contrast_set(contrast_set(pairs))) == list(pairs)


def test_duplicate_names_raise_after_flattening():
    # R keeps duplicates (contrast_set(a, contrast_set(a)) == contrast_set(a, a));
    # fmrimod raises because name-keyed results would overwrite (2026-10-25).
    a, b = _pair(), one_against_all_contrast(LEVELS, "condition")[0]
    for args in [(a, a), (a, contrast_set(a)), (a, b, contrast_set(b, contrast_set(a)))]:
        with pytest.raises(ContrastError) as err:
            contrast_set(*args)
        assert isinstance(err.value, ValueError)
    with pytest.raises(ContrastError, match=r"'intelligibility' at positions 0, 1"):
        contrast_set(a, contrast_set(a))
    renamed = pair_contrast(
        Formula("condition == 'Foreign'"), Formula("condition == 'Degraded'"),
        name="intelligibility",
    )
    with pytest.raises(
        ContrastError,
        match=r"'intelligibility' at positions 0, 2; 'con_Degraded_vs_other' at positions 1, 3",
    ):
        contrast_set(a, b, contrast_set(renamed, b))


def test_duplicate_names_on_one_term_raise():
    a = _pair()
    ev = pd.DataFrame({"onset": np.arange(0.0, 60.0, 10.0), "condition": LEVELS * 2})
    term = Term("condition")
    term.contrast_specs = [a, a]
    em = event_model([term], data=ev, tr=1.0, n_scans=60)
    with pytest.raises(ContrastError, match=r"Term 'condition' contrasts: duplicate"):
        contrast_weights(em)
    grown = contrast_set(a)
    grown.append(a)
    with pytest.raises(ContrastError, match="duplicate contrast names"):
        contrast_weights(grown, em._create_event_terms()[0])


def test_same_name_on_different_terms_is_allowed():
    ev = pd.DataFrame(
        {
            "onset": np.arange(0.0, 60.0, 10.0),
            "condition": LEVELS * 2,
            "task": ["x", "y"] * 3,
        }
    )
    t1, t2 = Term("condition"), Term("task")
    t1.contrast_specs = [unit_contrast(Formula("condition == 'Intact'"), name="c")]
    t2.contrast_specs = [unit_contrast(Formula("task == 'x'"), name="c")]
    em = event_model([t1, t2], data=ev, tr=1.0, n_scans=60)
    weights = contrast_weights(em)
    assert set(weights) == {"condition", "task"}


def test_invalid_leaves_report_their_path():
    a = _pair()
    with pytest.raises(TypeError, match=r"argument 2 must be a ContrastSpec or ContrastSet"):
        contrast_set(a, 42)
    with pytest.raises(TypeError, match=r"contrast_set\(\*specs\)"):
        contrast_set(a, [a])
    malformed = list.__new__(ContrastSet)
    list.extend(malformed, [a, "oops"])
    with pytest.raises(TypeError, match=r"argument 2\[1\] must be"):
        contrast_set(a, malformed)
