"""Tests for SBHM (Shared-Basis HRF Matching) pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from fmrimod.single._types import SbhmConfig, SingleTrialResult
from fmrimod.single.sbhm.amplitude import sbhm_amplitude
from fmrimod.single.sbhm.library import SbhmLibrary, build_sbhm_library
from fmrimod.single.sbhm.match import SbhmMatchResult, sbhm_match
from fmrimod.single.sbhm.pipeline import sbhm_single_trial
from fmrimod.single.sbhm.prepass import sbhm_prepass


@pytest.fixture
def rng():
    return np.random.default_rng(33)


@pytest.fixture
def library_data(rng):
    """Build a synthetic SBHM library."""
    n_library, n_time = 50, 20
    H = rng.standard_normal((n_time, n_library))
    library = build_sbhm_library(H, r=3)
    return library


class TestBuildSbhmLibrary:
    def test_basic(self, rng):
        H = rng.standard_normal((20, 50))
        lib = build_sbhm_library(H, r=3)
        assert isinstance(lib, SbhmLibrary)
        assert lib.B.shape == (20, 3)
        assert lib.S.shape == (3,)
        assert lib.A.shape == (50, 3)

    def test_r_truncation(self, rng):
        H = rng.standard_normal((30, 100))
        lib = build_sbhm_library(H, r=5)
        assert lib.B.shape[1] == 5
        assert lib.S.shape[0] == 5
        assert lib.A.shape[1] == 5

    def test_normalize(self, rng):
        H = rng.standard_normal((20, 50))
        lib = build_sbhm_library(H, r=3, normalize=True)
        assert lib.B.shape == (20, 3)


class TestSbhmPrepass:
    def test_basic(self, rng):
        T, K, V = 100, 3, 20
        Y = rng.standard_normal((T, V))
        A_agg = rng.standard_normal((T, K))
        beta_bar, G = sbhm_prepass(Y, A_agg)
        assert beta_bar.shape == (K, V)
        assert G.shape == (K, K)

    def test_with_confounds(self, rng):
        T, K, V = 100, 3, 20
        Y = rng.standard_normal((T, V))
        A_agg = rng.standard_normal((T, K))
        confounds = rng.standard_normal((T, 4))
        beta_bar, G = sbhm_prepass(Y, A_agg, confounds=confounds)
        assert beta_bar.shape == (K, V)


class TestSbhmMatch:
    def test_basic(self, rng):
        K, V, n_lib = 3, 20, 50
        beta_bar = rng.standard_normal((K, V))
        S = np.array([10.0, 5.0, 1.0])
        A = rng.standard_normal((n_lib, K))
        result = sbhm_match(beta_bar, S, A)
        assert isinstance(result, SbhmMatchResult)
        assert result.matched_idx.shape == (V,)
        assert result.similarity.shape == (n_lib, V)
        assert result.margin.shape == (V,)
        assert result.alpha_coords.shape == (K, V)

    def test_with_shrinkage(self, rng):
        K, V, n_lib = 3, 20, 50
        beta_bar = rng.standard_normal((K, V))
        S = np.array([10.0, 5.0, 1.0])
        A = rng.standard_normal((n_lib, K))
        result = sbhm_match(beta_bar, S, A, shrink=True)
        assert result.matched_idx.shape == (V,)

    def test_top_k(self, rng):
        K, V, n_lib = 3, 20, 50
        beta_bar = rng.standard_normal((K, V))
        S = np.array([10.0, 5.0, 1.0])
        A = rng.standard_normal((n_lib, K))
        result = sbhm_match(beta_bar, S, A, top_k=3)
        assert result.alpha_coords.shape == (K, V)
        assert result.weights is not None
        assert result.weights.shape == (V, 3)


class TestSbhmAmplitude:
    def test_basic(self, rng):
        T, N, K, V = 100, 10, 3, 20
        Y = rng.standard_normal((T, V))
        X_trials = rng.standard_normal((T, N * K))
        alpha_coords = rng.standard_normal((K, V))
        # Normalise
        norms = np.linalg.norm(alpha_coords, axis=0, keepdims=True)
        alpha_coords = alpha_coords / np.maximum(norms, 1e-12)

        betas = sbhm_amplitude(Y, X_trials, alpha_coords, K=K)
        assert betas.shape == (N, V)

    def test_rejects_uppercase_method(self, rng):
        T, N, K, V = 50, 4, 3, 8
        Y = rng.standard_normal((T, V))
        X_trials = rng.standard_normal((T, N * K))
        alpha_coords = rng.standard_normal((K, V))
        with pytest.raises(ValueError, match="method must be one of"):
            sbhm_amplitude(Y, X_trials, alpha_coords, method="OASIS_VOXEL", K=K)


class TestSbhmConfig:
    def test_accepts_valid_options(self):
        cfg = SbhmConfig(
            r=4,
            shrink=False,
            top_k=2,
            amplitude_method="lss1",
            ridge_mode="absolute",
            ridge_lambda=0.05,
        )
        assert cfg.r == 4
        assert cfg.top_k == 2
        assert cfg.amplitude_method == "lss1"
        assert cfg.ridge_mode == "absolute"

    def test_is_frozen(self):
        cfg = SbhmConfig()
        with pytest.raises(Exception):  # FrozenInstanceError is a dataclasses subclass
            cfg.amplitude_method = "global_ls"  # type: ignore[misc]

    def test_amplitude_method_is_case_sensitive(self):
        with pytest.raises(ValueError, match="amplitude_method must be one of"):
            SbhmConfig(amplitude_method="OASIS_VOXEL")  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"r": 0}, "r must be a positive integer"),
            ({"top_k": 0}, "top_k must be a positive integer"),
            ({"amplitude_method": "bogus"}, "amplitude_method must be one of"),
            ({"ridge_mode": "bogus"}, "ridge_mode must be one of"),
            ({"ridge_lambda": -0.1}, "ridge_lambda must be non-negative"),
        ],
    )
    def test_rejects_invalid_options(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            SbhmConfig(**kwargs)


class TestSbhmPipeline:
    def test_full_pipeline(self, rng, library_data):
        T, N, V = 100, 10, 20
        K = library_data.B.shape[1]  # 3
        X = rng.standard_normal((T, N * K))
        Y = rng.standard_normal((T, V))
        config = SbhmConfig(r=3, amplitude_method="oasis_voxel")
        result = sbhm_single_trial(Y, X, config=config, library=library_data)
        assert isinstance(result, SingleTrialResult)
        assert result.method == "sbhm"
        assert result.betas.shape == (N, V)
        assert "matched_idx" in result.extra
        assert "library" in result.extra

    def test_with_confounds(self, rng, library_data):
        T, N, V = 100, 10, 20
        K = library_data.B.shape[1]
        X = rng.standard_normal((T, N * K))
        Y = rng.standard_normal((T, V))
        confounds = rng.standard_normal((T, 3))
        config = SbhmConfig(r=3)
        result = sbhm_single_trial(Y, X, confounds=confounds,
                                   config=config, library=library_data)
        assert result.betas.shape == (N, V)

    def test_no_library_raises(self, rng):
        T, N, K, V = 100, 10, 3, 20
        X = rng.standard_normal((T, N * K))
        Y = rng.standard_normal((T, V))
        with pytest.raises(ValueError, match="library"):
            sbhm_single_trial(Y, X)

    def test_trial_labels(self, rng, library_data):
        T, N, V = 80, 8, 15
        K = library_data.B.shape[1]
        X = rng.standard_normal((T, N * K))
        Y = rng.standard_normal((T, V))
        labels = [f"trial_{i}" for i in range(N)]
        config = SbhmConfig(r=3)
        result = sbhm_single_trial(Y, X, config=config,
                                   library=library_data, trial_labels=labels)
        assert result.trial_labels == labels
