"""Regression tests for the SPM dispersion derivative on SPMG3.

History: fmrimod's :class:`SPMG3_HRF` originally placed the second
time derivative ``∂²h/∂t²`` in its third basis column, inheriting
that convention from R ``fmrireg::HRF_SPMG3``. The SPM informed-
basis definition (matching Nilearn) is the *dispersion derivative*
``∂h/∂σ`` taken at ``σ=1`` with the negative-forward sign
``(h(σ=1) - h(σ=1+dx)) / dx`` — a finite difference in the
dispersion (width) parameter, not the second derivative in time.

These two basis functions are mathematically distinct (correlation
near zero on the canonical's support), and the interpretive claim
that "the third coefficient captures HRF width changes" only holds
for the dispersion derivative.

This file pins the corrected behavior so SPMG3 cannot silently
regress to the second-time-derivative convention.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import fmrimod as fm
from fmrimod.hrf.derivatives import (
    spmg1_dispersion_derivative,
    spmg1_dispersion_derivative_spm,
    spmg1_second_derivative,
)
from fmrimod.hrf.functions import spm_canonical
from fmrimod.hrf.spm_hrf import SPMG3_HRF
from fmrimod.spec import hrf


def test_spm_canonical_dispersion_param_narrows_response() -> None:
    """``dispersion > 1`` narrows / shifts the peak earlier.

    SPM's convention: ``shape = p/σ`` and ``scale = 1/σ`` means
    larger ``σ`` decreases both the gamma's mean and its variance,
    so the response peaks earlier and is narrower. fmrimod mirrors
    this via ``t -> t * σ`` in :func:`spm_canonical`.
    """
    t = np.linspace(0, 20, 200)
    h_default = spm_canonical(t, dispersion=1.0)
    h_higher = spm_canonical(t, dispersion=1.10)

    peak_default = float(t[int(np.argmax(h_default))])
    peak_higher = float(t[int(np.argmax(h_higher))])
    assert peak_higher < peak_default, (
        f"larger dispersion should shift the peak earlier; "
        f"got peak(σ=1.10)={peak_higher} >= peak(σ=1.00)={peak_default}"
    )


def test_spm_dispersion_derivative_sign_matches_nilearn() -> None:
    """SPMG3's third column has the SPM/Nilearn dispersion-derivative sign.

    SPM and Nilearn use the negative forward difference
    ``(h(σ=1) - h(σ=1+dx)) / dx``. The resulting derivative is
    negative on the rising flank of the canonical (because increasing
    σ shifts the rising edge earlier, making ``h(σ=1+dx) > h(σ=1)``
    on the rising flank, so ``(h - h_pert)`` is negative there).

    The previous second-time-derivative implementation was POSITIVE
    on the rising flank (the canonical's curvature is concave-up
    there before the peak). This sign sanity check therefore
    distinguishes the corrected dispersion derivative from the
    legacy second-time-derivative basis.
    """
    t = np.linspace(0, 16, 161)
    hrf = SPMG3_HRF()
    basis = hrf(t)
    disp_col = basis[:, 2]

    # Rising flank: 1.5 s to 3.0 s (well before the canonical's peak
    # near t≈5 s). On this segment the SPM dispersion derivative is
    # negative.
    rising = (t >= 1.5) & (t <= 3.0)
    assert np.all(disp_col[rising] <= 0.0), (
        "SPM dispersion derivative should be non-positive on the "
        f"canonical's rising flank; got max={disp_col[rising].max():.4g}"
    )

    # Past the peak (8-10 s), the dispersion derivative is positive
    # — the response there is INCREASED by tighter dispersion because
    # the undershoot starts later under SPM's σ convention.
    post_peak = (t >= 8.0) & (t <= 10.0)
    assert np.any(disp_col[post_peak] > 0.0), (
        "SPM dispersion derivative should be positive somewhere past "
        f"the canonical peak; got max={disp_col[post_peak].max():.4g}"
    )


def test_spm_dispersion_derivative_is_not_second_time_derivative() -> None:
    """The third basis column is distinct from the second time derivative.

    On the canonical's support, the corrected dispersion derivative
    should differ from ``spmg1_second_derivative`` by a clearly
    different shape (negative-vs-positive sign on the rising flank
    is enough to make this a robust check that resists numerical
    drift).
    """
    t = np.linspace(0, 20, 201)
    hrf = SPMG3_HRF()
    disp_col = hrf(t)[:, 2]
    # SPM canonical's second time derivative computed on the legacy form
    # — we only want a distinguishing-shape check, not numerical
    # equivalence between the two parameterizations.
    second_time = spmg1_second_derivative(t)

    # The two have opposite signs on the rising flank — they are not
    # the same function.
    rising = (t >= 1.5) & (t <= 3.0)
    sign_disp = np.sign(np.mean(disp_col[rising]))
    sign_d2t = np.sign(np.mean(second_time[rising]))
    assert sign_disp != sign_d2t, (
        "dispersion derivative and second time derivative should have "
        f"different signs on the rising flank; both are {sign_disp}"
    )


def test_spm_dispersion_derivative_function_matches_class_column() -> None:
    """``spmg1_dispersion_derivative_spm`` and SPMG3()(t)[:,2] match exactly.

    The new SPMG3 (post canonical alignment) uses the SPM dispersion
    derivative (finite difference w.r.t. dispersion parameter), so the
    module-level helper that matches the class column is
    :func:`spmg1_dispersion_derivative_spm`, not the legacy
    :func:`spmg1_dispersion_derivative`.
    """
    t = np.linspace(0, 24, 49)
    h = SPMG3_HRF()
    from_class = h(t)[:, 2]
    from_function = spmg1_dispersion_derivative_spm(
        t, delay=h.delay, undershoot=h.undershoot,
        dispersion=h.dispersion, u_dispersion=h.u_dispersion, ratio=h.ratio,
    )
    np.testing.assert_allclose(from_class, from_function, atol=1e-14)


def test_spmg3_design_column_uses_dispersion_derivative() -> None:
    """The realised SPMG3 design's third basis column is the dispersion derivative.

    This is the typed-spec-side regression: build a real design via
    ``hrf("trial_type", basis="spmg3")`` and confirm the ``basis_ix=3``
    column is the SPM dispersion derivative (not the second time
    derivative), by checking the sign on the canonical's rising flank
    on a clean dirac-onset row.
    """
    # Single onset at t=0 makes the convolution equal to the basis
    # itself (sampled at the frame times). A 0.25 s TR keeps enough
    # resolution to see the rising-flank sign on the dispersion col.
    events = pd.DataFrame(
        {"onset": [0.0], "duration": [0.0], "trial_type": ["A"], "run": [1]}
    )
    n_frames = 128
    tr = 0.25
    ds = fm.fmri_dataset(np.zeros((n_frames, 1)), tr=tr, events=events)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fit = fm.fmri_lm(
            hrf("trial_type", basis="spmg3", norm="spm"), ds
        )
    cols = fit.design_columns()
    disp_idx = (
        cols.where(term="trial_type", level="A", basis_ix=3).one().index
    )
    X = fit.model.design_matrix_array(run=0)
    disp_col = X[:, disp_idx]
    frame_times = np.arange(n_frames) * tr
    rising = (frame_times >= 1.5) & (frame_times <= 3.0)
    assert np.all(disp_col[rising] <= 0.0), (
        "Realised SPMG3 design's basis_ix=3 column should be the "
        "SPM dispersion derivative (non-positive on the rising flank); "
        f"got max={disp_col[rising].max():.4g}"
    )
