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
)
from fmrimod.contrast.contrast_spec import ContrastSet, Formula
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
    np.testing.assert_allclose(weights["intelligibility"].offset_weights[:, 0], [-1, 0, 1])


def test_recursive_flattening_keeps_order_duplicates_and_drops_empty_sets():
    a, b = _pair(), one_against_all_contrast(LEVELS, "condition")[0]
    assert list(contrast_set(contrast_set(a, contrast_set(b)), contrast_set())) == [a, b]
    assert list(contrast_set(contrast_set(), contrast_set())) == []
    assert list(contrast_set(a, contrast_set(a))) == [a, a]
    pairs = pairwise_contrasts(["A", "B", "C"], "condition")
    assert list(contrast_set(contrast_set(pairs))) == list(pairs)


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
