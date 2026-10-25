"""Pin the SPM canonical HRF to R fmrihrf's ``HRF_SPMG1``.

Steward decision D2 (2026-10-25, decision note 0005): fmrimod follows R
fmrihrf. The canonical is ``exp(-t) * (t^5 / 120 - t^15 / (6 * 15!))``,
i.e. ``gamma(6) - gamma(16) / 6`` with ratio exactly 1/6, over a 24 s span.
Before D2 fmrimod used Nilearn's 0.167 ratio and a 32 s span, and kept the
pre-``daca7ba`` R form (``a1=0.0833``, ten times R's ``1/120``) as
``*_Legacy`` classes; those are removed.
"""

from __future__ import annotations

from math import factorial, gamma

import numpy as np
import pytest
from nilearn.glm.first_level.hemodynamic_models import spm_hrf as nl_spm_hrf

from fmrimod.hrf import get_hrf
from fmrimod.hrf.derivatives import spmg1_derivative, spmg1_second_derivative
from fmrimod.hrf.functions import spm_canonical
from fmrimod.hrf.spm_hrf import SPMG1_HRF, SPMG2_HRF, SPMG3_HRF


def test_default_spm_uses_r_parameterization() -> None:
    for cls in (SPMG1_HRF, SPMG2_HRF, SPMG3_HRF):
        h = cls()
        assert (h.delay, h.undershoot, h.dispersion, h.u_dispersion) == (6, 16, 1, 1)
        assert h.ratio == 1.0 / 6.0
        assert h.span == 24.0


def test_canonical_is_r_closed_form() -> None:
    """R ``hrf_spmg1``: ``exp(-t) * (A1 t^P1 - t^P2 / (6 P2!))``, A1 = 1/120."""
    t = np.arange(0.0, 30.0001, 0.05)
    expected = np.exp(-t) * (t**5 / 120.0 - t**15 / (6.0 * factorial(15)))
    np.testing.assert_allclose(SPMG1_HRF()(t), expected, rtol=1e-12, atol=1e-16)


def test_default_spm_peak_time_and_undershoot_ratio() -> None:
    t = np.linspace(0, 32, 3201)
    values = SPMG1_HRF()(t)
    assert 4.5 <= float(t[int(np.argmax(values))]) <= 5.5
    assert 6.0 < float(np.max(values)) / float(-np.min(values)) < 20.0


def test_canonical_shape_still_correlates_with_nilearn_kernel() -> None:
    """R's and Nilearn's kernels differ only in ratio (1/6 vs 0.167) and length."""
    t = np.arange(0, 32.0, 0.1)
    fm_kernel = spm_canonical(t)
    nl_full = nl_spm_hrf(t_r=2.0, oversampling=20, time_length=32.0)
    nl_t = np.linspace(0, 32.0, len(nl_full))
    fm_on_nl = np.interp(nl_t, t, fm_kernel)
    corr = float(np.corrcoef(fm_on_nl / fm_on_nl.sum(), nl_full / nl_full.sum())[0, 1])
    assert corr > 0.999, corr


def test_spm_canonical_function_rejects_invalid_dispersion() -> None:
    with pytest.raises(ValueError, match="dispersion"):
        spm_canonical(np.arange(0, 24), dispersion=0.0)
    with pytest.raises(ValueError, match="dispersion"):
        spm_canonical(np.arange(0, 24), u_dispersion=-1.0)


@pytest.mark.parametrize("name", ["spm_legacy", "spmg1_legacy", "spmg2_legacy", "spmg3_legacy"])
def test_legacy_basis_names_are_gone(name: str) -> None:
    with pytest.raises((KeyError, ValueError)):
        get_hrf(name)


@pytest.mark.parametrize("p2", [12.0, 16.5])
def test_undershoot_normaliser_scales_with_p2(p2: float) -> None:
    """R ``hrf_spmg1(P2=p2)`` divides by ``6 * Gamma(P2 + 1)``.

    fmrihrf ``d19ba00`` (test_spmg_correctness.R "scales the undershoot by
    P2, not a fixed 15!"). Here ``P2 = undershoot - 1``; the analytic
    derivatives must stay the time derivatives of that kernel.
    """
    t = np.arange(0.0, 40.0001, 0.25)
    expected = np.exp(-t) * (t**5 / 120.0 - t**p2 / (6.0 * gamma(p2 + 1.0)))
    np.testing.assert_allclose(
        spm_canonical(t, undershoot=p2 + 1.0), expected, rtol=1e-12, atol=1e-15
    )

    tp, h = t[1:], 1e-5

    def central(f):
        return (f(tp + h) - f(tp - h)) / (2 * h)

    np.testing.assert_allclose(
        spmg1_derivative(tp, undershoot=p2 + 1.0),
        central(lambda x: spm_canonical(x, undershoot=p2 + 1.0)),
        rtol=0, atol=1e-7,
    )
    np.testing.assert_allclose(
        spmg1_second_derivative(tp, undershoot=p2 + 1.0),
        central(lambda x: spmg1_derivative(x, undershoot=p2 + 1.0)),
        rtol=0, atol=1e-7,
    )


def test_spmg2_second_column_is_the_analytic_time_derivative() -> None:
    t = np.linspace(0, 24, 97)
    basis = SPMG2_HRF()(t)
    np.testing.assert_allclose(basis[:, 0], SPMG1_HRF()(t), atol=1e-16)
    np.testing.assert_allclose(basis[:, 1], spmg1_derivative(t), atol=1e-16)
