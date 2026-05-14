"""Tests for the chunkwise GLM engine."""

from __future__ import annotations

import numpy as np
import pytest

from fmrimod.glm.engine import ChunkwiseEngineOptions
from fmrimod.glm.errors import FmriCapabilityError, UnsupportedEngineConfiguration
from fmrimod.glm.fmri_lm import fmri_lm
from fmrimod.model.config import (
    AROptions,
    FmriLmConfig,
    RobustOptions,
    SoftSubspaceOptions,
    VolumeWeightOptions,
)


class _DummyDataset:
    def __init__(self, ys):
        self._ys = [np.asarray(y, dtype=np.float64) for y in ys]
        self.n_timepoints = [y.shape[0] for y in self._ys]

    def get_data(self, run: int):
        return self._ys[run]

    def get_censor(self, run: int):
        return None


class _DummyModel:
    def __init__(self, xs, ys):
        self._xs = [np.asarray(x, dtype=np.float64) for x in xs]
        self.dataset = _DummyDataset(ys)
        self.n_runs = len(self._xs)

    def design_matrix_array(self, run: int):
        return self._xs[run]


def _build_model(seed: int = 99, n_runs: int = 3, n: int = 100, p: int = 5, v: int = 64):
    rng = np.random.default_rng(seed)
    xs = []
    ys = []
    for _ in range(n_runs):
        X = np.column_stack([np.ones(n), rng.standard_normal((n, p - 1))]).astype(np.float64)
        beta = rng.standard_normal((p, v)).astype(np.float64)
        Y = X @ beta + rng.standard_normal((n, v)).astype(np.float64) * 0.25
        xs.append(X)
        ys.append(Y)
    return _DummyModel(xs, ys)


def test_chunkwise_matches_runwise_ols():
    model = _build_model()
    cfg = FmriLmConfig()

    runwise = fmri_lm(model, cfg, engine="runwise")
    chunked = fmri_lm(model, cfg, engine=ChunkwiseEngineOptions(chunk_size=13))

    np.testing.assert_allclose(chunked.betas, runwise.betas, atol=1e-10)
    np.testing.assert_allclose(chunked.sigma, runwise.sigma, atol=1e-10)
    np.testing.assert_allclose(chunked.XtXinv, runwise.XtXinv, atol=1e-10)
    assert chunked.residual_df == runwise.residual_df


def test_chunkwise_parallel_matches_serial():
    model = _build_model(seed=100)
    cfg = FmriLmConfig()

    serial = fmri_lm(model, cfg, engine="chunkwise", chunk_size=11, n_jobs=1)
    parallel = fmri_lm(model, cfg, engine="chunkwise", chunk_size=11, n_jobs=2, blas_threads=1)

    np.testing.assert_allclose(parallel.betas, serial.betas, atol=1e-10)
    np.testing.assert_allclose(parallel.sigma, serial.sigma, atol=1e-10)
    np.testing.assert_allclose(parallel.XtXinv, serial.XtXinv, atol=1e-10)


def test_chunkwise_parallel_single_run_matches_serial():
    model = _build_model(seed=110, n_runs=1, n=140, p=6, v=128)
    cfg = FmriLmConfig()

    serial = fmri_lm(model, cfg, engine="chunkwise", chunk_size=17, n_jobs=1)
    parallel = fmri_lm(model, cfg, engine="chunkwise", chunk_size=17, n_jobs=4, blas_threads=1)

    np.testing.assert_allclose(parallel.betas, serial.betas, atol=1e-10)
    np.testing.assert_allclose(parallel.sigma, serial.sigma, atol=1e-10)
    np.testing.assert_allclose(parallel.XtXinv, serial.XtXinv, atol=1e-10)


def test_chunkwise_matches_runwise_volume_weights():
    model = _build_model(seed=101, n_runs=2, n=60, p=4, v=20)
    weights = np.linspace(0.5, 1.5, num=120, dtype=np.float64)
    cfg = FmriLmConfig(
        volume_weights=VolumeWeightOptions(enabled=True, weights=weights)
    )

    runwise = fmri_lm(model, cfg, engine="runwise")
    chunked = fmri_lm(model, cfg, engine="chunkwise", chunk_size=10)

    np.testing.assert_allclose(chunked.betas, runwise.betas, atol=1e-10)
    np.testing.assert_allclose(chunked.sigma, runwise.sigma, atol=1e-10)
    np.testing.assert_allclose(chunked.XtXinv, runwise.XtXinv, atol=1e-10)
    assert chunked.residual_df == runwise.residual_df


def test_chunkwise_matches_runwise_soft_subspace():
    model = _build_model(seed=102, n_runs=2, n=70, p=5, v=24)
    rng = np.random.default_rng(103)
    nuisance = rng.standard_normal((140, 3)).astype(np.float64)
    cfg = FmriLmConfig(
        soft_subspace=SoftSubspaceOptions(enabled=True, nuisance_matrix=nuisance, lam=0.2)
    )

    runwise = fmri_lm(model, cfg, engine="runwise")
    chunked = fmri_lm(model, cfg, engine="chunkwise", chunk_size=9)

    np.testing.assert_allclose(chunked.betas, runwise.betas, atol=1e-10)
    np.testing.assert_allclose(chunked.sigma, runwise.sigma, atol=1e-10)
    np.testing.assert_allclose(chunked.XtXinv, runwise.XtXinv, atol=1e-10)
    assert chunked.residual_df == runwise.residual_df


def test_chunkwise_ar_raises_typed_capability_error():
    model = _build_model(seed=120, n_runs=1, n=50, p=3, v=8)
    cfg = FmriLmConfig(ar=AROptions(struct="ar1"))

    with pytest.raises(
        UnsupportedEngineConfiguration,
        match='chunkwise.*AR modeling.*use engine="runwise".*ar.struct="iid"',
    ) as excinfo:
        fmri_lm(model, cfg, engine="chunkwise", chunk_size=8)

    err = excinfo.value
    assert isinstance(err, FmriCapabilityError)
    assert isinstance(err, NotImplementedError)
    assert err.engine == "chunkwise"
    assert err.feature == "chunkwise:AR modeling"
    assert err.current_capability == (
        "OLS",
        "censoring",
        "volume weights",
        "soft-subspace",
    )
    assert err.repair == 'use engine="runwise" or set ar.struct="iid"'


def test_chunkwise_robust_raises_typed_capability_error():
    model = _build_model(seed=121, n_runs=1, n=50, p=3, v=8)
    cfg = FmriLmConfig(robust=RobustOptions(type="huber"))

    with pytest.raises(
        UnsupportedEngineConfiguration,
        match='chunkwise.*robust fitting.*use engine="runwise".*robust.type=False',
    ) as excinfo:
        fmri_lm(model, cfg, engine=ChunkwiseEngineOptions(chunk_size=8))

    err = excinfo.value
    assert isinstance(err, FmriCapabilityError)
    assert isinstance(err, NotImplementedError)
    assert err.engine == "chunkwise"
    assert err.feature == "chunkwise:robust fitting"
    assert "OLS" in err.current_capability
    assert err.repair == 'use engine="runwise" or set robust.type=False'
