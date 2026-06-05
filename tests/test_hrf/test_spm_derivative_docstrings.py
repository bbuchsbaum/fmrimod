"""Assert SPMG2/SPMG3 docstrings document the SPM derivative choice.

The default ``SPMG2_HRF`` / ``SPMG3_HRF`` classes now align with Nilearn
and SPM by using finite differences in the typed delay/dispersion
parameters. That choice must be discoverable from the class docstrings
and from the :func:`fmrimod.spec.hrf` builder users actually call.
"""

from __future__ import annotations

from fmrimod.hrf.spm_hrf import SPMG2_HRF, SPMG3_HRF
from fmrimod.spec import hrf


def _normalised(doc: str | None) -> str:
    return (doc or "").lower()


def test_spmg2_docstring_names_the_spm_finite_difference_choice() -> None:
    doc = _normalised(SPMG2_HRF.__doc__)
    assert "finite difference" in doc or "finite-difference" in doc, (
        "SPMG2_HRF docstring must name the finite-difference choice"
    )
    assert "delay" in doc
    assert "nilearn" in doc, (
        "SPMG2_HRF docstring must name Nilearn so the alignment is searchable"
    )
    assert "spm" in doc, (
        "SPMG2_HRF docstring must name SPM (the published-paper reference)"
    )


def test_spmg3_docstring_names_the_spm_finite_difference_choice() -> None:
    doc = _normalised(SPMG3_HRF.__doc__)
    assert "finite difference" in doc or "finite-difference" in doc
    assert "nilearn" in doc
    assert "spm" in doc
    assert "dispersion" in doc


def test_hrf_builder_docstring_points_to_spmg2_for_informed_basis_set() -> None:
    doc = _normalised(hrf.__doc__)
    assert "finite-difference" in doc or "finite difference" in doc, (
        "hrf() docstring must surface the finite-difference derivative choice for "
        "informed basis sets so casual users do not have to read SPMG2_HRF"
    )
    assert "spmg2" in doc, "hrf() docstring must reference basis='spmg2'"
    assert "delay" in doc
