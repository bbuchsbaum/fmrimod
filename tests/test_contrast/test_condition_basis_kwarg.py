"""Typed ``basis=`` role shortcut on :func:`fmrimod.contrast.condition`.

``basis="canonical"`` / ``"derivative"`` / ``"dispersion"`` lets callers
name an SPM informed-set role instead of leaking the internal 1-based
``basis_ix`` column convention into user code (tier_a_spm_derivative_basis
pain point #2, bd-01KRMQ4Z8MP6R2G58YZ3J2Z512). The kwarg is a closed
``typing.Literal`` mutually exclusive with ``basis_ix=``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import fmrimod as fm
from fmrimod.contrast import SemanticContrast, condition
from fmrimod.spec import hrf as hrf_term


def test_basis_role_names_resolve_to_one_based_indices() -> None:
    assert condition("x", basis="canonical").basis_ix == (1,)
    assert condition("x", basis="derivative").basis_ix == (2,)
    assert condition("x", basis="dispersion").basis_ix == (3,)


def test_basis_default_is_unchanged_empty_selection() -> None:
    """Neither selector set keeps the historical 'all bases' default."""
    assert condition("x").basis_ix == ()
    assert condition("x", basis_ix=2).basis_ix == (2,)


def test_basis_and_basis_ix_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError) as excinfo:
        condition("listening", basis="canonical", basis_ix=2)
    message = str(excinfo.value)
    assert "basis=" in message
    assert "basis_ix=" in message
    assert "mutually exclusive" in message


def _spmg2_fit() -> object:
    events = pd.DataFrame(
        {
            "onset": [0.0, 30.0, 60.0, 90.0],
            "duration": [10.0] * 4,
            "trial_type": ["listening"] * 4,
            "run": [1] * 4,
        }
    )
    y = np.zeros((50, 4), dtype=np.float64)
    ds = fm.fmri_dataset(y, tr=7.0, mask=np.ones(4, dtype=bool), events=events)
    return fm.fmri_lm(
        hrf_term("trial_type", basis="spmg2", norm="spm"), ds, precision=0.02
    )


def test_basis_canonical_subsumes_basis_ix_one_on_spmg2() -> None:
    """basis='canonical' selects the same SPMG2 column as basis_ix=1, and
    basis='derivative' selects the basis_ix=2 column — proving the role
    spelling subsumes the index spelling without changing resolution."""
    fit = _spmg2_fit()
    columns = fit.design_columns()

    by_name = SemanticContrast(
        positive=condition("listening", term="trial_type", basis="canonical"),
        name="listening",
    ).resolve(columns)
    by_index = SemanticContrast(
        positive=condition("listening", term="trial_type", basis_ix=1),
        name="listening",
    ).resolve(columns)
    np.testing.assert_array_equal(by_name, by_index)

    canonical_col = columns[int(np.flatnonzero(by_name)[0])]
    assert canonical_col.basis_ix == 1

    derivative = SemanticContrast(
        positive=condition("listening", term="trial_type", basis="derivative"),
        name="listening",
    ).resolve(columns)
    derivative_col = columns[int(np.flatnonzero(derivative)[0])]
    assert derivative_col.basis_ix == 2
    assert not np.array_equal(by_name, derivative)
