"""R-parity surface tests for the HRF dataclass.

Covers the additions made for bd-01KRFMD3E3D9N0XE0HKDZ8K42G:

- ``HRF_SPMG1`` / ``HRF_SPMG2`` / ``HRF_SPMG3`` / ``HRF_GAMMA`` / etc. aliases.
- ``HRF.__add__`` for column-binding via ``HRF_SPMG1 + HRF_GAMMA``.
- Fluent ``.lag()`` / ``.block()`` / ``.normalize()`` instance methods.
"""

from __future__ import annotations

import numpy as np
import pytest

import fmrimod as fm
from fmrimod.hrf import (
    BSPLINE_HRF,
    FIR_HRF,
    GAMMA_HRF,
    GAUSSIAN_HRF,
    HRF,
    HRF_BSPLINE,
    HRF_FIR,
    HRF_GAMMA,
    HRF_GAUSSIAN,
    HRF_SPMG1,
    HRF_SPMG2,
    HRF_SPMG3,
    SPM_CANONICAL,
    SPM_WITH_DERIVATIVE,
    SPM_WITH_DISPERSION,
    bind_basis,
    lag_hrf,
)


T = np.linspace(0, 30, 121)


def test_r_parity_aliases_are_same_objects_as_legacy_names():
    assert HRF_SPMG1 is SPM_CANONICAL
    assert HRF_SPMG2 is SPM_WITH_DERIVATIVE
    assert HRF_SPMG3 is SPM_WITH_DISPERSION
    assert HRF_GAMMA is GAMMA_HRF
    assert HRF_GAUSSIAN is GAUSSIAN_HRF
    assert HRF_BSPLINE is BSPLINE_HRF
    assert HRF_FIR is FIR_HRF


def test_r_parity_aliases_evaluate_identically():
    np.testing.assert_array_equal(HRF_SPMG1(T), SPM_CANONICAL(T))
    np.testing.assert_array_equal(HRF_SPMG3(T), SPM_WITH_DISPERSION(T))


def test_hrf_add_produces_bind_basis_result():
    combined = HRF_SPMG1 + HRF_GAMMA
    expected = bind_basis(HRF_SPMG1, HRF_GAMMA)
    np.testing.assert_array_equal(combined(T), expected(T))
    assert combined.nbasis == HRF_SPMG1.nbasis + HRF_GAMMA.nbasis


def test_hrf_add_with_non_hrf_returns_not_implemented():
    # Behavior: Python falls back to the other operand's __radd__, which is
    # also not implemented for ndarray, so we get a TypeError.
    with pytest.raises(TypeError):
        _ = HRF_SPMG1 + np.array([1.0, 2.0])  # type: ignore[operator]


def test_hrf_add_chains_three_bases():
    triple = HRF_SPMG1 + HRF_GAMMA + HRF_GAUSSIAN
    assert triple.nbasis == 3
    values = triple(T)
    assert values.shape == (T.size, 3)


def test_hrf_lag_method_matches_free_lag_hrf():
    via_method = HRF_SPMG1.lag(3.0)
    via_function = lag_hrf(HRF_SPMG1, 3.0)
    np.testing.assert_array_equal(via_method(T), via_function(T))


def test_hrf_block_method_returns_new_hrf():
    blocked = HRF_SPMG1.block(width=4.0, precision=0.1)
    assert isinstance(blocked, HRF)
    # Block convolution increases the effective area / span.
    base_vals = HRF_SPMG1(T)
    blocked_vals = blocked(T)
    assert blocked_vals.shape == base_vals.shape
    assert np.max(blocked_vals) >= np.max(base_vals) * 0.9


def test_hrf_normalize_method_yields_unit_peak():
    raw = HRF_SPMG1
    norm = raw.normalize()
    peak = np.max(np.abs(norm(T)))
    assert np.isclose(peak, 1.0, atol=1e-6)


def test_hrf_fluent_chain_lag_then_normalize():
    chained = HRF_SPMG1.lag(2.0).normalize()
    assert isinstance(chained, HRF)
    np.testing.assert_allclose(np.max(np.abs(chained(T))), 1.0, atol=1e-6)


def test_top_level_fmrimod_exposes_aliases():
    # Aliases should be reachable via the top-level package.
    assert fm.HRF_SPMG1 is HRF_SPMG1
    assert fm.HRF_GAMMA is HRF_GAMMA
