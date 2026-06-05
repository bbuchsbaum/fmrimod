"""Regression tests for per-run TR plumbing.

Multi-run datasets can carry heterogeneous repetition times across
runs (common when merging data across sessions, scanners, or sites).
The :class:`~fmrimod.sampling.SamplingFrame` carries per-run TR in
its ``tr`` array; the convolution path evaluates the HRF on each
run's own frame times; the concat engine stacks the result into a
single OLS. The TR-relative default precision uses ``min(TR)``
across runs so the convolution grid is fine enough for the faster
run.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import fmrimod as fm
from fmrimod.dataset.constructors import matrix_dataset
from fmrimod.design.event_model import DEFAULT_PRECISION_OVERSAMPLING
from fmrimod.spec import drift, hrf, intercept


def test_matrix_dataset_accepts_per_run_tr_list() -> None:
    """``tr=[1.5, 2.0]`` plumbs through to per-run SamplingFrame.tr."""
    ds = matrix_dataset(
        np.zeros((150, 1)),
        tr=[1.5, 2.0],
        run_length=[100, 50],
        slice_timing_offset=0.0,
    )
    sf = ds.get_sampling_frame()
    np.testing.assert_array_equal(sf.tr, np.array([1.5, 2.0]))
    np.testing.assert_array_equal(sf.blocklens, np.array([100, 50]))


def test_per_run_sampling_grids_use_correct_local_tr() -> None:
    """Each run's frame times step at its own TR."""
    ds = matrix_dataset(
        np.zeros((150, 1)),
        tr=[1.5, 2.0],
        run_length=[100, 50],
        slice_timing_offset=0.0,
    )
    sf = ds.get_sampling_frame()
    # Run 0 (TR=1.5): frames at 0, 1.5, 3.0, ...
    np.testing.assert_allclose(
        sf.block_samples(0)[:4], np.array([0.0, 1.5, 3.0, 4.5])
    )
    # Run 1 (TR=2.0): frames at 0, 2.0, 4.0, ...
    np.testing.assert_allclose(
        sf.block_samples(1)[:4], np.array([0.0, 2.0, 4.0, 6.0])
    )


def test_default_precision_uses_min_tr_across_runs() -> None:
    """TR-relative default uses ``min(TR)`` so the grid is fine enough for the fastest run."""
    events = pd.DataFrame({
        "onset": [10.0, 10.0], "duration": [0.0, 0.0],
        "trial_type": ["A", "A"], "run": [1, 2],
    })
    ds = matrix_dataset(
        np.zeros((150, 1)),
        tr=[1.5, 2.0],
        run_length=[100, 50],
        event_table=events,
        slice_timing_offset=0.0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fit = fm.fmri_lm(hrf("trial_type", basis="spm"), ds)
    expected = min(1.5, 2.0) / DEFAULT_PRECISION_OVERSAMPLING
    assert fit.model._event_model.precision == pytest.approx(expected), (
        f"min(TR)/{DEFAULT_PRECISION_OVERSAMPLING}={expected}, "
        f"got {fit.model._event_model.precision}"
    )


def test_hrf_peak_lands_at_correct_global_time_per_run() -> None:
    """A single event at run-relative t=10s produces a peak at global t=15s in each run."""
    n1, n2 = 60, 60
    events = pd.DataFrame({
        "onset": [10.0, 10.0],
        "duration": 0.0,
        "trial_type": ["A", "A"],
        "run": [1, 2],
    })
    ds = matrix_dataset(
        np.zeros((n1 + n2, 1)),
        tr=[1.5, 2.0],
        run_length=[n1, n2],
        event_table=events,
        slice_timing_offset=0.0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fit = fm.fmri_lm(hrf("trial_type", basis="spm", norm="spm"), ds, engine="concat")
    X = fit.model.design_matrix_array(run=None)
    A_col = X[:, 0]

    # Run 0 frames at 0, 1.5, 3, ... — peak at index 10 (t=15.0, exact peak)
    run0_peak = int(np.argmax(A_col[:n1]))
    assert run0_peak == 10, (
        f"Run 0 (TR=1.5) peak should land at frame 10 (t=15.0s); got {run0_peak}"
    )
    # Run 1 frames at 0, 2, 4, ... — peak at frame 7 or 8 (t=14 or 16s, both close to 15)
    run1_peak = int(np.argmax(A_col[n1:]))
    assert run1_peak in (7, 8), (
        f"Run 1 (TR=2.0) peak should be frame 7 or 8 (t=14 or 16s); got {run1_peak}"
    )


def test_concat_engine_handles_mixed_tr_cross_run_contrast() -> None:
    """Cross-run difference contrast is computable through the concat engine."""
    rng = np.random.default_rng(0)
    n1, n2 = 100, 75
    rows = []
    for run_idx, tr in enumerate([1.5, 2.0]):
        for k, onset in enumerate(np.linspace(10.0, 130.0, 6)):
            rows.append({
                "onset": float(onset),
                "duration": 0.0,
                "trial_type": "A" if k % 2 == 0 else "B",
                "run_label": f"run{run_idx + 1}",
                "run": run_idx + 1,
            })
    events = pd.DataFrame(rows)
    Y = rng.normal(size=(n1 + n2, 8))
    ds = matrix_dataset(
        Y,
        tr=[1.5, 2.0],
        run_length=[n1, n2],
        event_table=events,
        slice_timing_offset=0.0,
    )
    spec = (
        hrf("trial_type", "run_label", basis="spm", norm="spm")
        + intercept(per="run")
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fit = fm.fmri_lm(spec, ds, engine="concat")

    cols = fit.design_columns()
    a_run1_idx = cols.where(
        term="trial_type:run_label",
        level="trial_type.A_run_label.run1",
    ).one().index
    a_run2_idx = cols.where(
        term="trial_type:run_label",
        level="trial_type.A_run_label.run2",
    ).one().index
    n_cols = sum(1 for _ in cols.columns)
    c = np.zeros(n_cols, dtype=np.float64)
    c[a_run1_idx] = +1.0
    c[a_run2_idx] = -1.0
    result = fit.contrast(c, name="A_run_diff")
    assert result.stat.shape == (8,)
    assert np.all(np.isfinite(result.stat))


def test_mixed_tr_with_explicit_slice_timing_offset_per_run() -> None:
    """``slice_timing_offset=[0.5, 1.0]`` plumbs to per-run SamplingFrame offsets."""
    ds = matrix_dataset(
        np.zeros((150, 1)),
        tr=[1.5, 2.0],
        run_length=[100, 50],
        slice_timing_offset=[0.5, 1.0],
    )
    sf = ds.get_sampling_frame()
    np.testing.assert_allclose(sf.start_time, np.array([0.5, 1.0]))
    # Run 0: frames at 0.5, 2.0, 3.5, ...
    np.testing.assert_allclose(
        sf.block_samples(0)[:3], np.array([0.5, 2.0, 3.5])
    )
    # Run 1: frames at 1.0, 3.0, 5.0, ...
    np.testing.assert_allclose(
        sf.block_samples(1)[:3], np.array([1.0, 3.0, 5.0])
    )
