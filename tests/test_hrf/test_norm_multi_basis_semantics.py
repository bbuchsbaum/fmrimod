"""Verify ``norm="spm"`` / ``"unit_peak"`` / ``"unit_integral"`` apply a
single canonical-derived scalar uniformly across multi-basis HRFs, while
``"unit_peak_per_basis"`` rescales each column independently.

Regression target: bd-01KRMQ4PRCKWTM59XG1XW5QAKT. Pain point #4 from
``tier_a_spm_derivative_basis``: the multi-basis broadcasting rule was
undocumented in user-facing ``hrf()``. This test pins the documented
behaviour numerically so a future refactor cannot silently flip the
broadcasting rule (e.g. by lifting ``unit_peak_per_basis`` semantics
into the default ``unit_peak`` path).
"""

from __future__ import annotations

import numpy as np

from fmrimod.hrf.normalization import normalize
from fmrimod.hrf.spm_hrf import SPMG1_HRF, SPMG2_HRF, SPMG3_HRF
from fmrimod.spec import hrf


def _eval(h, t):
    return np.asarray(h(t), dtype=np.float64)


def test_norm_spm_on_spmg2_uses_canonical_scalar_uniformly() -> None:
    """``norm='spm'`` on SPMG2: the stored norm_factor is a scalar, and
    both columns of the normalized output equal raw / norm_factor exactly."""
    base = SPMG2_HRF()
    norm = normalize(base, "spm")
    t = np.linspace(0.0, 32.0, 200)
    raw = _eval(base, t)
    out = _eval(norm, t)
    assert raw.shape == out.shape == (200, 2)

    # Single scalar factor (not a per-column vector) — that is the
    # broadcasting rule the docstring promises.
    assert np.ndim(norm.norm_factor) == 0
    scalar = float(norm.norm_factor)
    np.testing.assert_allclose(out, raw / scalar, rtol=1e-12, atol=0.0)


def test_norm_unit_peak_on_spmg3_uses_canonical_scalar_uniformly() -> None:
    """``norm='unit_peak'`` on SPMG3: stored norm_factor is one scalar
    applied uniformly to all three basis columns."""
    base = SPMG3_HRF()
    norm = normalize(base, "unit_peak")
    t = np.linspace(0.0, 24.0, 240)
    raw = _eval(base, t)
    out = _eval(norm, t)
    assert raw.shape == out.shape == (240, 3)

    assert np.ndim(norm.norm_factor) == 0
    scalar = float(norm.norm_factor)
    np.testing.assert_allclose(out, raw / scalar, rtol=1e-12, atol=0.0)
    # Canonical column normalized to unit peak (on a dense grid that
    # actually includes the peak — the divisor is computed on the
    # reference grid, so on a coarse grid the observed max is slightly
    # below 1).
    dense = np.linspace(0.0, 24.0, 24001)
    dense_out = _eval(norm, dense)
    assert abs(float(np.max(np.abs(dense_out[:, 0]))) - 1.0) < 1e-6


def test_norm_unit_peak_per_basis_rescales_each_column_independently() -> None:
    """``norm='unit_peak_per_basis'`` on SPMG2: stored norm_factor is a
    length-nbasis *vector* — that is the structural difference from the
    other modes."""
    base = SPMG2_HRF()
    norm = normalize(base, "unit_peak_per_basis")

    factor = np.asarray(norm.norm_factor, dtype=np.float64)
    assert factor.ndim == 1 and factor.shape == (2,), (
        "unit_peak_per_basis must store a length-nbasis vector divisor; "
        "any scalar storage would mean it has collapsed to unit_peak"
    )
    # The two columns of the underlying SPMG2 have distinct peaks, so
    # the per-basis vector must contain two distinct values.
    assert factor[0] != factor[1]

    # Each column ends with unit peak on a dense grid.
    dense = np.linspace(0.0, 24.0, 24001)
    out = _eval(norm, dense)
    assert abs(float(np.max(np.abs(out[:, 0]))) - 1.0) < 1e-4
    assert abs(float(np.max(np.abs(out[:, 1]))) - 1.0) < 1e-4


def test_norm_spm_on_spmg1_matches_scalar_application() -> None:
    """Sanity: on a single-column HRF, the same code path stores a
    scalar divisor. Catches a regression where multi-basis special-
    casing leaks into the single-basis path."""
    base = SPMG1_HRF()
    norm = normalize(base, "spm")
    assert np.ndim(norm.norm_factor) == 0

    t = np.linspace(0.0, 32.0, 200)
    raw = _eval(base, t)
    out = _eval(norm, t)
    np.testing.assert_allclose(out, raw / float(norm.norm_factor), rtol=1e-12)


def test_hrf_builder_docstring_documents_multi_basis_broadcasting() -> None:
    """The user-facing ``hrf()`` docstring (not just the
    ``normalization.py`` module docstring) must surface the multi-basis
    broadcasting rule. Pain point #4 was that ``hrf()`` was silent on it."""
    doc = (hrf.__doc__ or "").lower()
    assert "multi-basis" in doc, (
        "hrf() docstring must mention multi-basis behaviour"
    )
    assert "canonical" in doc, (
        "hrf() docstring must name the canonical column as the scalar source"
    )
    assert "unit_peak_per_basis" in doc, (
        "hrf() docstring must list unit_peak_per_basis as the alternative"
    )
