"""Assert SPMG2/SPMG3 docstrings document the derivative choice.

Steward decision D2 (decision note 0005) follows R fmrihrf: the second
column is the analytic time derivative ``dh/dt`` and SPMG3's third is SPM's
dispersion difference. That choice must be discoverable from the class
docstrings and from the :func:`fmrimod.spec.hrf` builder users actually call.
"""

from __future__ import annotations

from fmrimod.hrf.spm_hrf import SPMG2_HRF, SPMG3_HRF
from fmrimod.spec import hrf


def _normalised(doc: str | None) -> str:
    return (doc or "").lower()


def test_spmg2_docstring_names_the_analytic_time_derivative() -> None:
    doc = _normalised(SPMG2_HRF.__doc__)
    assert "analytic" in doc and "time derivative" in doc
    assert "hrf_spmg2" in doc, "SPMG2_HRF docstring must name its R reference"


def test_spmg3_docstring_names_the_dispersion_column() -> None:
    doc = _normalised(SPMG3_HRF.__doc__)
    assert "time derivative" in doc
    assert "dispersion" in doc
    assert "hrf_spmg3" in doc


def test_hrf_builder_docstring_points_to_spmg2_for_informed_basis_set() -> None:
    doc = _normalised(hrf.__doc__)
    assert "spmg2" in doc, "hrf() docstring must reference basis='spmg2'"
    assert "analytic time derivative" in doc, (
        "hrf() docstring must surface the derivative choice for informed basis "
        "sets so casual users do not have to read SPMG2_HRF"
    )
    assert "dispersion" in doc
