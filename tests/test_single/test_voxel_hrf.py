"""Tests for per-voxel HRF estimation and HRF-aware LSS."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from fmrimod.single.voxel_hrf import estimate_voxel_hrf, lss_with_voxel_hrf
from fmrimod.single._types import VoxelHrfResult, SingleTrialResult


@pytest.fixture
def rng():
    return np.random.default_rng(55)


@pytest.fixture
def multi_basis_data(rng):
    """Synthetic multi-basis data for voxel-HRF tests."""
    T, N, K, V = 100, 10, 3, 20
    basis = rng.standard_normal((30, K))
    # Interleaved format: [t1_b1, t1_b2, t1_b3, t2_b1, ...]
    X_trials = rng.standard_normal((T, N * K))
    Y = rng.standard_normal((T, V))
    return Y, X_trials, basis, T, N, K, V


class TestEstimateVoxelHrf:
    def test_basic(self, multi_basis_data):
        Y, X_trials, basis, T, N, K, V = multi_basis_data
        result = estimate_voxel_hrf(Y, X_trials, basis, K=K)
        assert isinstance(result, VoxelHrfResult)
        assert result.coefficients.shape == (K, V)

    def test_positive_peak_on_array_basis_rows(self, multi_basis_data):
        """An ``(L, K)`` array basis is peak-normalized on its own rows.

        fmrilss ``0d97c8b``: the baseline-subtracted waveform has a unit
        positive peak, and ``amplitude_scale`` restores the raw pooled fit.
        """
        Y, X_trials, basis, T, N, K, V = multi_basis_data
        result = estimate_voxel_hrf(Y, X_trials, basis, K=K)
        assert result.normalization == "positive-peak"
        waveforms = (basis - basis[0]) @ result.coefficients
        ok = ~result.degenerate
        assert ok.any()
        assert_allclose(waveforms[:, ok].max(axis=0), 1.0, atol=1e-10)

        A = np.column_stack([X_trials[:, k::K].sum(axis=1) for k in range(K)])
        raw = np.linalg.lstsq(A, Y, rcond=None)[0]
        assert_allclose(result.coefficients * result.amplitude_scale, raw, atol=1e-10)

    def test_infer_K_from_basis(self, multi_basis_data):
        Y, X_trials, basis, T, N, K, V = multi_basis_data
        result = estimate_voxel_hrf(Y, X_trials, basis)
        assert result.coefficients.shape[0] == K

    def test_with_confounds(self, multi_basis_data, rng):
        Y, X_trials, basis, T, N, K, V = multi_basis_data
        confounds = rng.standard_normal((T, 3))
        result = estimate_voxel_hrf(Y, X_trials, basis, K=K, confounds=confounds)
        assert result.coefficients.shape == (K, V)

    def test_bad_ncol_raises(self, rng):
        T, V, K = 50, 10, 3
        Y = rng.standard_normal((T, V))
        X_trials = rng.standard_normal((T, 11))  # 11 not divisible by 3
        basis = rng.standard_normal((20, K))
        with pytest.raises(ValueError, match="not divisible"):
            estimate_voxel_hrf(Y, X_trials, basis, K=K)

    def test_rejects_bad_rows_and_nonfinite_values(self, rng):
        T, V, K = 50, 4, 2
        Y = rng.standard_normal((T, V))
        X_trials = rng.standard_normal((T, 6))
        basis = rng.standard_normal((20, K))

        with pytest.raises(ValueError, match="X_trials has 49 rows"):
            estimate_voxel_hrf(Y, X_trials[:-1], basis, K=K)

        Y_bad = Y.copy()
        Y_bad[0, 0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            estimate_voxel_hrf(Y_bad, X_trials, basis, K=K)

        with pytest.raises(ValueError, match="confounds has 49 rows"):
            estimate_voxel_hrf(
                Y,
                X_trials,
                basis,
                K=K,
                confounds=np.ones((T - 1, 1)),
            )


class TestLssWithVoxelHrf:
    def test_basic(self, multi_basis_data, rng):
        Y, X_trials, basis, T, N, K, V = multi_basis_data
        hrf_result = estimate_voxel_hrf(Y, X_trials, basis, K=K)
        result = lss_with_voxel_hrf(Y, X_trials, hrf_result)
        assert isinstance(result, SingleTrialResult)
        assert result.method == "lss_voxel_hrf"
        assert result.betas.shape == (N, V)

    def test_with_confounds(self, multi_basis_data, rng):
        Y, X_trials, basis, T, N, K, V = multi_basis_data
        confounds = rng.standard_normal((T, 2))
        hrf_result = estimate_voxel_hrf(Y, X_trials, basis, K=K, confounds=confounds)
        result = lss_with_voxel_hrf(Y, X_trials, hrf_result, confounds=confounds)
        assert result.betas.shape == (N, V)

    def test_voxel_mismatch_raises(self, multi_basis_data, rng):
        Y, X_trials, basis, T, N, K, V = multi_basis_data
        hrf_result = estimate_voxel_hrf(Y, X_trials, basis, K=K)
        # Wrong number of voxels
        Y_wrong = rng.standard_normal((T, V + 5))
        with pytest.raises(ValueError, match="voxels"):
            lss_with_voxel_hrf(Y_wrong, X_trials, hrf_result)

    def test_small_chunk(self, multi_basis_data, rng):
        """Test chunked processing with small chunk_size."""
        Y, X_trials, basis, T, N, K, V = multi_basis_data
        hrf_result = estimate_voxel_hrf(Y, X_trials, basis, K=K)
        result_small = lss_with_voxel_hrf(Y, X_trials, hrf_result, chunk_size=5)
        result_big = lss_with_voxel_hrf(Y, X_trials, hrf_result, chunk_size=1000)
        assert_allclose(result_small.betas, result_big.betas, atol=1e-10)

    def test_rejects_malformed_inputs(self, multi_basis_data, rng):
        Y, X_trials, basis, T, N, K, V = multi_basis_data
        hrf_result = estimate_voxel_hrf(Y, X_trials, basis, K=K)

        with pytest.raises(ValueError, match="not divisible"):
            lss_with_voxel_hrf(Y, X_trials[:, :-1], hrf_result)
        with pytest.raises(ValueError, match="chunk_size"):
            lss_with_voxel_hrf(Y, X_trials, hrf_result, chunk_size=0)
        with pytest.raises(ValueError, match="confounds has 99 rows"):
            lss_with_voxel_hrf(Y, X_trials, hrf_result, confounds=np.ones((T - 1, 1)))


def _trial_basis_design(onsets, basis_samples, n):
    """Interleaved ``[t1_b1..t1_bK, t2_b1..]`` design from TR-sampled bases."""
    L, K = basis_samples.shape
    X = np.zeros((n, len(onsets) * K))
    for i, onset in enumerate(onsets):
        stop = min(n, onset + L)
        X[onset:stop, i * K : (i + 1) * K] = basis_samples[: stop - onset]
    return X


def _fine_waveforms(basis):
    """Basis on R's 0.05 s grid over its span, baseline-subtracted."""
    grid = np.arange(0.0, basis.span + 0.025, 0.05)
    H = np.asarray(basis(grid), dtype=np.float64).reshape(grid.size, -1)
    return H - H[0]


class TestVoxelHrfPositivePeakContract:
    """fmrilss ``0d97c8b`` / ``6a2588b`` (``R/voxel_hrf.R:307-370``).

    Shapes are oriented to correlate positively with a reference HRF
    (default canonical SPMG1) and divided by their positive peak, so a
    deactivating voxel keeps negative trial betas. Shapes without a positive
    peak of at least 5% of their largest deflection are flagged
    ``degenerate`` and scaled by that deflection, with one warning.

    Cheap pass disqualified: orienting by ``sign(coef[0])`` passes the
    SPMG basis but fails the FIR fixture, where the first basis column is not
    canonical; peak-normalizing without orientation fails the sign checks.
    """

    def test_deactivating_voxel_keeps_negative_betas(self):
        from fmrimod.hrf import HRF_SPMG2

        rng = np.random.default_rng(3)
        n = 300
        onsets = np.arange(10, 280, 14)
        K = 2
        X = _trial_basis_design(onsets, HRF_SPMG2(np.arange(0.0, 25.0)), n)
        amps = rng.normal(1.0, 0.2, onsets.size)
        shape_pos = np.array([1.0, 0.3])
        Y = np.column_stack(
            [
                sum(amps[i] * X[:, i * K : (i + 1) * K] @ s for i in range(onsets.size))
                for s in (shape_pos, -shape_pos)
            ]
        ) + 0.05 * rng.standard_normal((n, 2))
        conf = np.ones((n, 1))

        est = estimate_voxel_hrf(Y, X, basis=HRF_SPMG2, confounds=conf, K=K)
        betas = lss_with_voxel_hrf(Y, X, est, confounds=conf).betas

        assert est.normalization == "positive-peak"
        assert not est.degenerate.any()
        assert betas[:, 0].mean() > 0
        assert betas[:, 1].mean() < 0
        assert np.corrcoef(betas[:, 0], amps)[0, 1] > 0.5
        assert np.corrcoef(betas[:, 1], amps)[0, 1] < -0.5
        # Both shapes are oriented canonical-positive; only the scale's sign
        # carries the deactivation.
        assert (est.coefficients[0] > 0).all()
        peaks = (_fine_waveforms(HRF_SPMG2) @ est.coefficients).max(axis=0)
        assert_allclose(peaks, 1.0, atol=1e-10)
        assert est.amplitude_scale[0] > 0 > est.amplitude_scale[1]

    def test_fir_degenerate_shapes_are_flagged_once(self):
        """R ``test-lss-rank1.R:220-232`` FIR fixture (good, bad, 0)."""
        from fmrimod.hrf import hrf_fir_generator

        basis = hrf_fir_generator(nbasis=12, span=24)
        good = np.array([0, 0.5, 1, 0.8, 0.4, 0.1, 0, -0.1, -0.1, 0, 0, 0])
        # Negative only in the undershoot window: correlates positively with
        # the canonical HRF but has no positive peak.
        bad = np.array([0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0], dtype=float)
        n = 300
        onsets = np.arange(10, 271, 13)
        K = 12
        X = _trial_basis_design(onsets, basis(np.arange(0.0, 24.0)), n)
        A = np.column_stack([X[:, k::K].sum(axis=1) for k in range(K)])
        Y = np.column_stack([A @ good, A @ bad, np.zeros(n)])

        with pytest.warns(RuntimeWarning) as record:
            est = estimate_voxel_hrf(Y, X, basis=basis, K=K)
        messages = [
            str(w.message)
            for w in record
            if "have no identifiable positive HRF peak" in str(w.message)
        ]
        assert len(messages) == 1
        assert messages[0].startswith("2 of 3 voxel(s)")
        assert est.degenerate.tolist() == [False, True, True]

        H = _fine_waveforms(basis)
        assert_allclose((H @ est.coefficients[:, 0]).max(), 1.0, atol=1e-10)
        assert_allclose(np.abs(H @ est.coefficients[:, 1]).max(), 1.0, atol=1e-10)
        assert_allclose(est.coefficients[:, 2], 0.0)
        assert np.all(np.isfinite(est.amplitude_scale))

        fit = lss_with_voxel_hrf(Y, X, est)
        assert fit.extra.degenerate.tolist() == [False, True, True]
        assert fit.extra.normalization == "positive-peak"

    def test_ref_hrf_sets_the_orientation(self):
        """A reversed reference flips every orientation."""
        from fmrimod.hrf import HRF_SPMG1, HRF_SPMG2, as_hrf

        n = 200
        onsets = np.arange(8, 180, 15)
        X = _trial_basis_design(onsets, HRF_SPMG2(np.arange(0.0, 25.0)), n)
        A = np.column_stack([X[:, k::2].sum(axis=1) for k in range(2)])
        Y = A @ np.array([[1.0, -1.0, 2.0, -0.5], [0.2, 0.1, -0.3, 0.05]])

        flipped = as_hrf(lambda t: -HRF_SPMG1(t), name="neg_spmg1")
        default = estimate_voxel_hrf(Y, X, basis=HRF_SPMG2, K=2)
        reversed_ = estimate_voxel_hrf(Y, X, basis=HRF_SPMG2, K=2, ref_hrf=flipped)
        assert np.sign(default.amplitude_scale).tolist() == [1, -1, 1, -1]
        assert np.sign(reversed_.amplitude_scale).tolist() == [-1, 1, -1, 1]
        peaks = (_fine_waveforms(HRF_SPMG2) @ reversed_.coefficients).max(axis=0)
        assert_allclose(peaks, 1.0, atol=1e-10)

    def test_rejects_basis_without_waveforms(self):
        rng = np.random.default_rng(0)
        with pytest.raises(TypeError, match="basis"):
            estimate_voxel_hrf(
                rng.standard_normal((40, 3)), rng.standard_normal((40, 4)), "spmg2", K=2
            )
