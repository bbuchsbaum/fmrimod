"""Smoke tests for WS01/WS02 core parity matrix runners."""

from __future__ import annotations

import pytest

from cross_testing.core_parity_matrix import (
    build_core_parity_matrix_report,
    run_ws01_design_matrix_parity,
    run_ws02_contrast_parity,
    run_ws03_variance_df_parity,
    run_ws04_run_combination_parity,
    run_ws05_censor_sample_mask_parity,
    run_ws06_lsa_lss_parity_performance,
    run_ws07_rank_deficient_design_parity,
    run_ws08_numeric_precision_parity,
    run_ws09_residual_diagnostic_parity,
    run_ws10_performance_decomposition_parity,
)


nilearn = pytest.importorskip("nilearn")


def test_ws01_design_parity_returns_metrics():
    report = run_ws01_design_matrix_parity(n_scans=120, tr=1.0, seed=11)
    assert report["status"] == "complete"
    assert "metrics" in report
    assert "min_column_corr" in report["metrics"]
    assert "max_scaled_mae" in report["metrics"]


def test_ws02_contrast_parity_returns_metrics():
    report = run_ws02_contrast_parity(
        n_timepoints=120,
        n_regressors=6,
        n_voxels=256,
        noise_sd=1.0,
        seed=12,
    )
    assert report["status"] == "complete"
    assert "metrics" in report
    assert "t_corr" in report["metrics"]
    assert "f_corr" in report["metrics"]


def test_ws03_variance_df_parity_returns_metrics():
    report = run_ws03_variance_df_parity(
        n_timepoints=120,
        n_regressors=6,
        n_voxels=256,
        noise_sd=1.0,
        phi=0.4,
        seed=13,
    )
    assert report["status"] == "complete"
    assert "metrics" in report
    assert "ols_sigma2_corr" in report["metrics"]
    assert "ar1_iter1_sigma2_corr" in report["metrics"]
    assert "ols_df_absdiff" in report["metrics"]


def test_ws04_run_combination_parity_returns_metrics():
    report = run_ws04_run_combination_parity(
        n_timepoints=120,
        n_regressors=6,
        n_voxels=256,
        noise_sd=1.0,
        seed=14,
    )
    assert report["status"] == "complete"
    assert "metrics" in report
    assert "runs_2" in report["metrics"]
    assert "runs_3" in report["metrics"]


def test_ws10_performance_decomposition_returns_stage_metrics():
    report = run_ws10_performance_decomposition_parity(
        n_timepoints=120,
        n_regressors=6,
        n_voxels=256,
        noise_sd=1.0,
        phi=0.4,
        seed=15,
        repeats=2,
        warmup=0,
        design_n_scans=100,
        design_tr=1.0,
        run_combine_runs=3,
    )
    assert report["status"] == "complete"
    assert "metrics" in report
    assert "fit_total_ols" in report["metrics"]
    assert "fit_total_ar1" in report["metrics"]
    assert "contrast_only" in report["metrics"]


def test_ws06_lsa_lss_returns_metrics():
    report = run_ws06_lsa_lss_parity_performance(
        n_timepoints=160,
        n_trials=30,
        n_voxels=256,
        n_confounds=4,
        noise_sd=1.0,
        seed=16,
        repeats=2,
        warmup=0,
        chunk_size=1024,
    )
    assert report["status"] == "complete"
    assert "metrics" in report
    assert "lsa_parity_corr" in report["metrics"]
    assert "lss_parity_corr" in report["metrics"]
    assert "speed" in report["metrics"]


def test_ws05_censor_sample_mask_returns_metrics():
    report = run_ws05_censor_sample_mask_parity(
        n_timepoints=120,
        n_regressors=6,
        n_voxels=256,
        noise_sd=1.0,
        seed=17,
    )
    assert report["status"] == "complete"
    assert "metrics" in report
    assert "fixtures" in report["metrics"]


def test_ws07_rank_deficient_returns_metrics():
    report = run_ws07_rank_deficient_design_parity(
        n_timepoints=120,
        n_regressors=6,
        n_voxels=256,
        noise_sd=1.0,
        seed=18,
    )
    assert report["status"] == "complete"
    assert "metrics" in report
    assert "fixtures" in report["metrics"]


def test_ws08_numeric_precision_returns_metrics():
    report = run_ws08_numeric_precision_parity(
        n_timepoints=120,
        n_regressors=6,
        n_voxels=256,
        noise_sd=1.0,
        seed=19,
    )
    assert report["status"] == "complete"
    assert "metrics" in report
    assert "fixtures" in report["metrics"]


def test_ws09_residual_diagnostic_returns_metrics():
    report = run_ws09_residual_diagnostic_parity(
        n_timepoints=120,
        n_regressors=6,
        n_voxels=256,
        noise_sd=1.0,
        phi=0.4,
        seed=20,
    )
    assert report["status"] == "complete"
    assert "metrics" in report
    assert "fixtures" in report["metrics"]


def test_core_parity_matrix_report_has_all_workstreams():
    report = build_core_parity_matrix_report(
        ws01_n_scans=100,
        ws01_tr=1.0,
        ws01_seed=2,
        ws02_n_timepoints=100,
        ws02_n_regressors=6,
        ws02_n_voxels=128,
        ws02_noise_sd=1.0,
        ws02_seed=3,
    )
    assert report["report_kind"] == "core_parity_matrix"
    assert "workstreams" in report
    assert set(report["workstreams"].keys()) == {
        "ws01",
        "ws02",
        "ws03",
        "ws04",
        "ws05",
        "ws06",
        "ws07",
        "ws08",
        "ws09",
        "ws10",
    }
