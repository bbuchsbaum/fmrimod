"""Assert SPMG2/SPMG3 docstrings document the closed-form derivative choice.

Regression target: bd-01KRMQ4G6GX8YA0HNYT5JFN7K1. The closed-form
temporal derivative used by ``SPMG2_HRF`` / ``SPMG3_HRF`` is a deliberate
divergence from Nilearn / SPM12 (finite-difference). The divergence
must be discoverable from the docstrings of the classes themselves and
from the :func:`fmrimod.spec.hrf` builder users actually call.
"""

from __future__ import annotations

from fmrimod.hrf.spm_hrf import SPMG2_HRF, SPMG3_HRF
from fmrimod.spec import hrf


def _normalised(doc: str | None) -> str:
    return (doc or "").lower()


def test_spmg2_docstring_names_the_closed_form_divergence() -> None:
    doc = _normalised(SPMG2_HRF.__doc__)
    assert "closed-form" in doc, "SPMG2_HRF docstring must name the closed-form choice"
    assert "finite difference" in doc or "finite-difference" in doc, (
        "SPMG2_HRF docstring must contrast against finite-difference"
    )
    assert "nilearn" in doc, (
        "SPMG2_HRF docstring must name Nilearn so the divergence is searchable"
    )
    assert "spm12" in doc or "spm 12" in doc, (
        "SPMG2_HRF docstring must name SPM12 (the published-paper reference)"
    )
    assert "latency" in doc, (
        "SPMG2_HRF docstring must surface the latency-calibration caveat"
    )


def test_spmg3_docstring_names_the_closed_form_divergence() -> None:
    doc = _normalised(SPMG3_HRF.__doc__)
    assert "closed-form" in doc
    assert "nilearn" in doc
    assert "spm12" in doc or "spm 12" in doc
    assert "dispersion" in doc


def test_hrf_builder_docstring_points_to_spmg2_for_informed_basis_set() -> None:
    doc = _normalised(hrf.__doc__)
    assert "closed-form" in doc, (
        "hrf() docstring must surface the closed-form derivative choice for "
        "informed basis sets so casual users do not have to read SPMG2_HRF"
    )
    assert "spmg2" in doc, "hrf() docstring must reference basis='spmg2'"
    assert "latency" in doc, (
        "hrf() docstring must surface the downstream calibration caveat"
    )
