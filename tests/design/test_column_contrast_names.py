"""column_contrast patterns against design-matrix column names.

Port of R fmridesign ``test-column-contrast-names.R`` (53772f6, issue #24;
inventory v2 DES-12). A term-attached ``column_contrast`` pattern is matched
against the term's realized design-matrix column names
(``term_tag_condition_tag[_b##]``) first and, only if it matches none of
them, against the term-level condition names (``condition_tag[_b##]``).
A pattern that selects different columns in the two namespaces is an error.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from fmrimod import event_model
from fmrimod.contrast import column_contrast, contrast_set, contrast_weights
from fmrimod.contrast.errors import ContrastError
from fmrimod.formula.base import Term


def _events():
    return pd.DataFrame({
        "onset": np.arange(10.0, 160.0, 20.0),
        "cond": ["A", "B"] * 4,
        "task": ["face", "face", "house", "house"] * 2,
    })


def _model(*terms):
    return event_model(list(terms), data=_events(), tr=2.0, n_scans=100)


def _term(variables, contrasts, **kwargs):
    term = Term(variables, **kwargs)
    term.contrast_specs = contrast_set(*contrasts)
    return term


def _weights(em, term_key, name):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cw = contrast_weights(em)
    return dict(zip(em.column_names, cw[term_key][name].offset_weights[:, 0], strict=True))


def _expected(em, pos, neg=()):
    w = dict.fromkeys(em.column_names, 0.0)
    for name in pos:
        w[name] = 1.0 / len(pos)
    for name in neg:
        w[name] = -1.0 / len(neg)
    return w


def test_single_basis_design_names_select_columns():
    em = _model(_term("cond", [column_contrast(
        r"^cond_cond\.A$", "AvB", pattern_B=r"^cond_cond\.B$")]))
    assert em.column_names == ["cond_cond.A", "cond_cond.B"]
    assert _weights(em, "cond", "AvB") == _expected(em, ["cond_cond.A"], ["cond_cond.B"])


def test_multibasis_design_names_select_one_basis_column():
    em = _model(_term("cond", [
        column_contrast(r"^cond_cond\.A_b01$", "AvB_b1", pattern_B=r"^cond_cond\.B_b01$"),
        column_contrast(r"_b02$", "all_b2"),
        column_contrast(r"^cond_cond\.A_b02$", "AvB_b2", pattern_B=r"^cond_cond\.B_b02$"),
        column_contrast(r"^cond\.A_b01$", "legacy_b1", pattern_B=r"^cond\.B_b01$"),
    ], hrf="spmg2"))
    assert em.column_names == [
        "cond_cond.A_b01", "cond_cond.A_b02", "cond_cond.B_b01", "cond_cond.B_b02",
    ]
    b1 = _expected(em, ["cond_cond.A_b01"], ["cond_cond.B_b01"])
    assert _weights(em, "cond", "AvB_b1") == b1
    assert _weights(em, "cond", "legacy_b1") == b1
    assert _weights(em, "cond", "AvB_b2") == _expected(
        em, ["cond_cond.A_b02"], ["cond_cond.B_b02"])
    assert _weights(em, "cond", "all_b2") == _expected(
        em, ["cond_cond.A_b02", "cond_cond.B_b02"])


def test_term_level_fallback_selects_all_basis_columns():
    em = _model(_term("cond", [column_contrast(r"^cond\.A", "A")], hrf="spmg3"))
    assert _weights(em, "cond", "A") == _expected(
        em, ["cond_cond.A_b01", "cond_cond.A_b02", "cond_cond.A_b03"])


def test_factorial_design_names_select_cells():
    em = _model(_term(["task", "cond"], [
        column_contrast(r"^task_cond_task\.face_cond\.A$", "FvH_A",
                        pattern_B=r"^task_cond_task\.house_cond\.A$"),
        column_contrast(r"^task\.face_cond\.A$", "legacy_FvH_A",
                        pattern_B=r"^task\.house_cond\.A$"),
    ]))
    expected = _expected(em, ["task_cond_task.face_cond.A"], ["task_cond_task.house_cond.A"])
    assert _weights(em, "task:cond", "FvH_A") == expected
    assert _weights(em, "task:cond", "legacy_FvH_A") == expected


def test_pattern_selecting_different_columns_in_two_namespaces_is_ambiguous():
    em = _model(_term("task", [column_contrast("task_|house", "amb")]))
    # The pattern matches both design names (task_task.face, task_task.house)
    # but only the term-level name task.house.
    # A requested contrast that cannot be computed raises (REG-A-18).
    with pytest.raises(ContrastError, match="ambiguous"):
        contrast_weights(em)


def test_direct_term_level_call_still_matches_condition_names():
    em = _model(Term("cond"))
    term = em._create_event_terms()[0]
    con = contrast_weights(column_contrast(r"^cond\.A$", "x", pattern_B=r"^cond\.B$"), term)
    assert list(con.weights[:, 0]) == [1.0, -1.0]
    assert list(con.condnames) == ["cond.A", "cond.B"]
