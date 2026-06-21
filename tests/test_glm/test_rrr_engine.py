"""Integration tests for the reduced-rank GLM engine."""

from __future__ import annotations

import numpy as np

from fmrimod.glm import ReducedRankEngineOptions
from fmrimod.glm.fmri_lm import fmri_lm
from fmrimod.model.config import FmriLmConfig


class _DummyDataset:
    def __init__(self, y: np.ndarray):
        self._y = np.asarray(y, dtype=np.float64)
        self.n_timepoints = [self._y.shape[0]]

    def get_data(self, run: int) -> np.ndarray:
        if run != 0:
            raise IndexError("only one run is available")
        return self._y

    def get_censor(self, run: int):
        if run != 0:
            raise IndexError("only one run is available")
        return None


class _DummyModel:
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        event_columns: tuple[int, ...] | None = None,
        column_names: tuple[str, ...] | None = None,
    ):
        self._x = np.asarray(x, dtype=np.float64)
        self.dataset = _DummyDataset(y)
        self.n_runs = 1
        self._event_columns = event_columns
        self._column_names = column_names

    @property
    def event_column_indices(self) -> tuple[int, ...] | None:
        return self._event_columns

    @property
    def n_event_columns(self) -> int:
        return 0 if self._event_columns is None else len(self._event_columns)

    def design_matrix_array(self, run: int = 0) -> np.ndarray:
        if run != 0:
            raise IndexError("only one run is available")
        return self._x

    def design_columns(self):
        if self._column_names is None:
            return tuple(f"x{i}" for i in range(self._x.shape[1]))
        return self._column_names

    def contrast_weights(self):
        return {}


def test_reduced_rank_options_fit_through_fmri_lm() -> None:
    rng = np.random.default_rng(404)
    n, p, v = 90, 4, 9
    X = np.column_stack([np.ones(n), rng.standard_normal((n, p - 1))])
    beta = rng.standard_normal((p, 1)) @ rng.standard_normal((1, v))
    Y = X @ beta
    model = _DummyModel(X, Y)

    fit = fmri_lm(
        model,
        FmriLmConfig(),
        engine=ReducedRankEngineOptions(rank=1, target="all"),
    )

    assert fit.betas.shape == (p, v)
    assert fit.provenance.solver_path == "ReducedRankEngine"
    assert np.linalg.matrix_rank(X @ fit.betas, tol=1e-8) <= 1


def test_rrr_gls_alias_matches_reduced_rank_engine() -> None:
    rng = np.random.default_rng(405)
    n, p, v = 75, 5, 7
    X = np.column_stack([np.ones(n), rng.standard_normal((n, p - 1))])
    Y = X @ (rng.standard_normal((p, 2)) @ rng.standard_normal((2, v)))
    model = _DummyModel(X, Y)

    typed = fmri_lm(
        model,
        FmriLmConfig(),
        engine=ReducedRankEngineOptions(rank=2, target="all"),
    )
    alias = fmri_lm(
        model,
        FmriLmConfig(),
        engine="rrr_gls",
        rank=2,
        target="all",
    )

    np.testing.assert_allclose(alias.betas, typed.betas, atol=1e-12)
    np.testing.assert_allclose(alias.sigma, typed.sigma, atol=1e-12)


def test_rrr_gls_accepts_ncomp_as_rank_alias() -> None:
    rng = np.random.default_rng(408)
    n, p, v = 70, 4, 6
    X = np.column_stack([np.ones(n), rng.standard_normal((n, p - 1))])
    Y = X @ (rng.standard_normal((p, 2)) @ rng.standard_normal((2, v)))
    model = _DummyModel(X, Y)

    fit = fmri_lm(model, FmriLmConfig(), engine="rrr_gls", ncomp=1, target="all")

    assert np.linalg.matrix_rank(X @ fit.betas, tol=1e-8) <= 1


def test_event_target_keeps_baseline_unrestricted() -> None:
    rng = np.random.default_rng(406)
    n, v = 80, 6
    task = rng.standard_normal((n, 2))
    baseline = np.column_stack([np.ones(n), np.linspace(-1.0, 1.0, n)])
    X = np.column_stack([task, baseline])
    task_beta = np.array([[1.5], [-0.5]]) @ rng.standard_normal((1, v))
    baseline_beta = np.array([[8.0], [3.0]]) @ rng.standard_normal((1, v))
    Y = task @ task_beta + baseline @ baseline_beta
    model = _DummyModel(X, Y, event_columns=(0, 1))

    fit = fmri_lm(model, FmriLmConfig(), engine=ReducedRankEngineOptions(rank=1))

    assert np.linalg.matrix_rank(task @ fit.betas[:2, :], tol=1e-8) <= 1
    np.testing.assert_allclose(fit.betas[2:, :], baseline_beta, atol=1e-10)


def test_named_target_columns_resolve_for_reduced_rank_engine() -> None:
    rng = np.random.default_rng(407)
    n, v = 70, 5
    X = np.column_stack([np.ones(n), rng.standard_normal((n, 3))])
    beta = rng.standard_normal((4, 1)) @ rng.standard_normal((1, v))
    Y = X @ beta
    model = _DummyModel(
        X,
        Y,
        column_names=("intercept", "a", "b", "c"),
    )

    fit = fmri_lm(
        model,
        FmriLmConfig(),
        engine=ReducedRankEngineOptions(rank=1, target_columns=("a", "b")),
    )

    assert fit.betas.shape == (4, v)
