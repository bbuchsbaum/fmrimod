"""Tests for reduced-rank regression primitives."""

from __future__ import annotations

import numpy as np
import pytest

from fmrimod.lowrank.rrr import (
    ReducedRankConfig,
    _block_bootstrap_indices,
    fit_reduced_rank,
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(202)


def _design(rng: np.random.Generator, n: int = 90, p: int = 5) -> np.ndarray:
    return np.column_stack([np.ones(n), rng.standard_normal((n, p - 1))])


def test_fixed_rank_constrains_fitted_signal(rng: np.random.Generator) -> None:
    n, p, v = 100, 5, 12
    X = _design(rng, n=n, p=p)
    left = rng.standard_normal((p, 2))
    right = rng.standard_normal((2, v))
    Y = X @ (left @ right) + 0.01 * rng.standard_normal((n, v))

    rank_two = fit_reduced_rank(X, Y, ReducedRankConfig(rank=2))
    rank_one = fit_reduced_rank(X, Y, ReducedRankConfig(rank=1))

    assert rank_two.rank == 2
    assert np.linalg.matrix_rank(X @ rank_two.betas, tol=1e-8) <= 2
    assert float(np.sum(rank_two.rss)) < float(np.sum(rank_one.rss))


def test_target_columns_leave_nuisance_unrestricted(
    rng: np.random.Generator,
) -> None:
    n, v = 80, 8
    task = rng.standard_normal((n, 2))
    nuisance = np.column_stack([np.ones(n), np.linspace(-1.0, 1.0, n)])
    X = np.column_stack([task, nuisance])
    task_betas = np.array([[2.0], [-1.0]]) @ rng.standard_normal((1, v))
    nuisance_betas = np.array([[10.0], [4.0]]) @ rng.standard_normal((1, v))
    Y = task @ task_betas + nuisance @ nuisance_betas

    result = fit_reduced_rank(
        X,
        Y,
        ReducedRankConfig(rank=1),
        target_columns=(0, 1),
    )

    assert result.target_columns == (0, 1)
    assert result.nuisance_columns == (2, 3)
    assert np.linalg.matrix_rank(task @ result.betas[:2, :], tol=1e-8) <= 1
    np.testing.assert_allclose(result.betas[2:, :], nuisance_betas, atol=1e-10)


def test_energy_rank_selection_picks_smallest_energy_rank(
    rng: np.random.Generator,
) -> None:
    X = _design(rng, n=70, p=4)
    signal = rng.standard_normal((4, 1)) @ rng.standard_normal((1, 9))
    Y = X @ signal

    result = fit_reduced_rank(
        X,
        Y,
        ReducedRankConfig(rank_mode="energy", energy_keep=0.95),
    )

    assert result.rank == 1
    np.testing.assert_allclose(result.betas, signal, atol=1e-10)


def test_rss_budget_rank_selection_uses_extra_rss_budget(
    rng: np.random.Generator,
) -> None:
    X = _design(rng, n=85, p=4)
    beta = rng.standard_normal((4, 2)) @ rng.standard_normal((2, 10))
    Y = X @ beta

    strict = fit_reduced_rank(
        X,
        Y,
        ReducedRankConfig(rank_mode="rss_budget", rss_budget=1e-12),
    )
    loose = fit_reduced_rank(
        X,
        Y,
        ReducedRankConfig(rank_mode="rss_budget", rss_budget=1e9),
    )

    assert strict.rank == 2
    assert loose.rank == 0
    assert float(np.sum(loose.rss)) > float(np.sum(strict.rss))


def test_bootstrap_standard_errors_are_seeded(
    rng: np.random.Generator,
) -> None:
    X = _design(rng, n=50, p=3)
    Y = X @ rng.standard_normal((3, 5)) + 0.1 * rng.standard_normal((50, 5))
    cfg = ReducedRankConfig(
        rank=2,
        se_mode="bootstrap",
        bootstrap_n=6,
        bootstrap_seed=11,
    )

    fit_a = fit_reduced_rank(X, Y, cfg)
    fit_b = fit_reduced_rank(X, Y, cfg)
    fit_c = fit_reduced_rank(
        X,
        Y,
        ReducedRankConfig(
            rank=2,
            se_mode="bootstrap",
            bootstrap_n=6,
            bootstrap_seed=12,
        ),
    )

    assert fit_a.bootstrap_se is not None
    np.testing.assert_allclose(fit_a.bootstrap_se, fit_b.bootstrap_se)
    assert not np.allclose(fit_a.bootstrap_se, fit_c.bootstrap_se)


def test_block_bootstrap_full_series_is_one_contiguous_block() -> None:
    """fmrireg 0.2.0: ``block_size >= n`` is one identity block.

    Cheap pass: modular wrap of a random start (the previous bug).
    """
    rng = np.random.default_rng(0)
    n = 10
    for block_size in (n, n + 3):
        idx = _block_bootstrap_indices(n, block_size, rng)
        np.testing.assert_array_equal(idx, np.arange(n))
        idx2 = _block_bootstrap_indices(n, block_size, np.random.default_rng(99))
        np.testing.assert_array_equal(idx2, np.arange(n))
