"""REG-A-18: a requested contrast that cannot be computed is an error.

fmrireg 2fe1498 turned "warn and drop the failed contrast" into a stop, and
06d2664 made unknown names and unparseable contrast formulas errors instead
of all-zero weights. The same holds for contrasts attached to a model term:
they are never silently dropped and never replaced by zero weights.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from fmrimod import event_model
from fmrimod.contrast import (
    ContrastError,
    contrast,
    contrast_set,
    contrast_weights,
    pair_contrast,
    unit_contrast,
)
from fmrimod.contrast.contrast_spec import Formula
from fmrimod.formula.base import Term


def _model(*specs):
    data = pd.DataFrame(
        {
            "onset": np.arange(9) * 2.0 + 1.0,
            "condition": ["A", "B", "C"] * 3,
        }
    )
    term = Term("condition")
    term.contrast_specs = contrast_set(*specs)
    return event_model([term], data=data, tr=2.0, n_scans=20)


def test_valid_attached_contrasts_still_resolve():
    em = _model(
        pair_contrast(Formula("A"), Formula("B"), name="A_vs_B"),
        contrast(Formula("A - C"), name="A_minus_C"),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = contrast_weights(em)
    assert set(result["condition"]) == {"A_vs_B", "A_minus_C"}
    w = np.ravel(result["condition"]["A_minus_C"].offset_weights)
    assert w.tolist() == [1.0, 0.0, -1.0]


@pytest.mark.parametrize(
    "bad, match",
    [
        (contrast(Formula("A - ???"), name="bad"), "unparseable formula"),
        (contrast(Formula("A - Z"), name="bad"), "unknown name 'Z'"),
        (contrast(Formula("condition == 'Z'"), name="bad"), "matches no condition"),
        (unit_contrast(Formula("condition == 'Z'"), name="bad"), "No conditions"),
        (
            pair_contrast(Formula("condition == 'Y'"), Formula("condition == 'Z'"), name="bad"),
            "No conditions",
        ),
    ],
)
def test_uncomputable_attached_contrast_raises_at_model_build(bad, match):
    em = _model(pair_contrast(Formula("A"), Formula("B"), name="good"), bad)
    with pytest.raises(ContrastError, match=match) as excinfo:
        contrast_weights(em)
    err = excinfo.value
    assert err.contrast == "bad"
    assert err.term == "condition"
    assert "'bad'" in str(err) and "'condition'" in str(err)


def test_uncomputable_formula_never_falls_back_to_zero_weights():
    em = _model(contrast(Formula("A - ???"), name="bad"))
    term = em._create_event_terms()[0]
    with pytest.raises(ContrastError, match="Formula contrast 'bad'"):
        contrast_weights(contrast(Formula("A - ???"), name="bad"), term)
