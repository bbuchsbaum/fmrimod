"""Parity tests for fmrimod vs fitlins-aligned first-level OLS GLM."""

from __future__ import annotations

from dataclasses import asdict

import pytest

from cross_testing.fitlins_parity import (
    DEFAULT_PARITY_THRESHOLDS,
    benchmark_implementations,
    compute_parity_metrics,
    fit_fitlins_reference_ols,
    fit_fmrimod_ols,
    make_synthetic_glm,
)


nilearn = pytest.importorskip("nilearn")


def test_fitlins_reference_outputs_have_expected_shapes():
    X, Y, _, contrast = make_synthetic_glm(
        n_timepoints=120,
        n_regressors=6,
        n_voxels=128,
        noise_sd=1.0,
        seed=7,
    )
    ref = fit_fitlins_reference_ols(X, Y, contrast)

    assert ref["betas"].shape == (X.shape[1], Y.shape[1])
    assert ref["sigma2"].shape == (Y.shape[1],)
    assert ref["t"].shape == (Y.shape[1],)
    assert ref["p"].shape == (Y.shape[1],)


def test_fmrimod_matches_fitlins_reference_parity_contract():
    X, Y, _, contrast = make_synthetic_glm(
        n_timepoints=180,
        n_regressors=8,
        n_voxels=512,
        noise_sd=1.0,
        seed=42,
    )
    candidate = fit_fmrimod_ols(X, Y, contrast)
    reference = fit_fitlins_reference_ols(X, Y, contrast)

    metrics = compute_parity_metrics(
        candidate,
        reference,
        sign_flip_floor=DEFAULT_PARITY_THRESHOLDS.sign_flip_floor,
    )
    failures = metrics.failures(DEFAULT_PARITY_THRESHOLDS)
    payload = {
        "failures": failures,
        "metrics": asdict(metrics),
        "thresholds": asdict(DEFAULT_PARITY_THRESHOLDS),
    }
    assert not failures, f"parity contract failed: {payload}"


def test_benchmark_summary_returns_valid_values():
    X, Y, _, contrast = make_synthetic_glm(
        n_timepoints=120,
        n_regressors=6,
        n_voxels=256,
        noise_sd=1.0,
        seed=11,
    )
    summary = benchmark_implementations(X, Y, contrast, repeats=2, warmup=0)

    assert summary.fmrimod_median_s > 0.0
    assert summary.reference_median_s > 0.0
    assert summary.speedup_vs_reference > 0.0
    assert len(summary.fmrimod_runs_s) == 2
    assert len(summary.reference_runs_s) == 2

