"""Verify DesignProvenanceError surfaces the repair_path to users.

The error class accepts a ``repair_path`` argument but historically only
attached it as an instance attribute — ``str(err)`` rendered the bare
message. The multi-basis ambiguity error in :mod:`fmrimod.contrast.semantic`
relies on ``repair_path`` to teach the user that ``basis_ix=`` is the
spelling that disambiguates the contrast.

Regression target: tier_a_spm_derivative_basis pain point #1
(bd-01KRMQ47HB20TF523VYV1HVE19).
"""

from __future__ import annotations

import pandas as pd
import pytest

import fmrimod as fm
from fmrimod.contrast import SemanticContrast, condition
from fmrimod.contrast.errors import DesignProvenanceError
from fmrimod.spec import hrf as hrf_term


def test_str_renders_repair_path_when_present() -> None:
    err = DesignProvenanceError(
        "ambiguous condition",
        weak_fields=("basis_ix",),
        repair_path="pass basis_ix=1 to condition(...)",
    )
    rendered = str(err)
    assert "ambiguous condition" in rendered
    assert "Repair:" in rendered
    assert "basis_ix=1" in rendered


def test_str_falls_back_to_message_without_repair_path() -> None:
    err = DesignProvenanceError("plain failure")
    assert str(err) == "plain failure"


def test_multi_basis_ambiguity_names_basis_ix_with_concrete_example() -> None:
    """When a SemanticContrast hits an SPMG2 design, the error must teach
    the ``basis_ix=`` repair, including a concrete example value."""

    import numpy as np

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
    fit = fm.fmri_lm(
        hrf_term("trial_type", basis="spmg2", norm="spm"), ds, precision=0.02
    )

    with pytest.raises(DesignProvenanceError) as excinfo:
        fit.contrast(
            SemanticContrast(
                positive=condition("listening", term="trial_type"),
                name="listening",
            )
        )

    rendered = str(excinfo.value)
    assert "ambiguous across basis columns" in rendered
    assert "Repair:" in rendered
    assert "basis_ix=" in rendered
    assert "canonical" in rendered.lower()
