"""hrf() basis-name edge cases (inventory v2 DES-21; R f0da93f, c69a829)."""

import warnings

import numpy as np
import pandas as pd
import pytest

from fmrimod import event_model
from fmrimod.formula.base import Term
from fmrimod.hrf.library import SPM_CANONICAL


def _events():
    return pd.DataFrame({"onset": [2.0, 20.0, 40.0, 60.0], "cond": ["A", "B", "A", "B"]})


def test_bs_is_an_alias_of_bspline():
    em_bs = event_model("onset ~ hrf(cond, basis='bs')", data=_events(), tr=2.0, n_scans=50)
    em_bspline = event_model(
        "onset ~ hrf(cond, basis='bspline')", data=_events(), tr=2.0, n_scans=50
    )
    np.testing.assert_array_equal(em_bs.design_matrix, em_bspline.design_matrix)
    assert em_bs.design_matrix.shape[1] == em_bspline.design_matrix.shape[1] > 2


def _boxcar(t):
    t = np.asarray(t, dtype=float)
    return ((t >= 0) & (t < 4)).astype(float)


def _model_with(hrf_spec):
    term = Term("cond", hrf=hrf_spec)
    term._kwargs["hrf_fun"] = _boxcar
    return event_model([term], data=_events(), tr=2.0, n_scans=50)


@pytest.mark.parametrize("basis", ["spmg3", SPM_CANONICAL])
def test_hrf_fun_overriding_a_basis_warns(basis):
    em = _model_with(basis)
    with pytest.warns(UserWarning, match="hrf_fun.*takes precedence"):
        _ = em.design_matrix


@pytest.mark.parametrize("basis", ["spmg1", "spm", None])
def test_hrf_fun_with_the_default_basis_is_silent(basis):
    em = _model_with(basis)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*takes precedence.*")
        _ = em.design_matrix
