"""Residual-bias correction (fmriAR ``acvf_bias.R``, ``test-acvf-bias.R``)."""

from __future__ import annotations

import numpy as np

from fmrimod.ar.acvf import acvf_bias_matrix


def _design(n: int = 90, seed: int = 708) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(-1.0, 1.0, n)
    return np.column_stack([np.ones(n), t, t**2, t**3, rng.standard_normal(n)])


def test_bias_matrix_depends_only_on_the_design_column_space():
    """Rank-deficient and reparameterised designs give the same map (AR-08).

    R builds the map from ``Q[, seq_len(rank)]`` of a pivoted ``qr()``
    (``R/acvf_bias.R:178-179``; ca27f27 tests duplicate and zero columns).
    Cheap pass disqualified: dropping all-zero columns fixes only the zero
    case; a duplicate that is not the last column stays red.
    """
    X = _design()
    n = X.shape[0]
    ref = acvf_bias_matrix(X, max_lag=6)[0]
    variants = {
        "permuted": X[:, [3, 0, 2, 1, 4]],
        "scaled": X * np.array([1.0, 1000.0, 0.1, 7.0, 2.0]),
        "duplicate appended": np.column_stack([X, X[:, 1]]),
        "duplicate at position 2": np.column_stack([X[:, :2], X[:, 1], X[:, 2:]]),
        "zero column at position 1": np.column_stack(
            [X[:, :1], np.zeros(n), X[:, 1:]]
        ),
    }
    for name, Xv in variants.items():
        np.testing.assert_allclose(
            acvf_bias_matrix(Xv, max_lag=6)[0], ref, rtol=0, atol=1e-12, err_msg=name
        )


def _drift_design(n: int, runs: np.ndarray) -> np.ndarray:
    """Run indicators, 6 DCT drift terms per run and a block task (R :549-555)."""
    L = n // 2
    t = np.arange(1, L + 1)
    dct = np.column_stack([np.cos(np.pi * j * (t - 0.5) / L) for j in range(1, 7)])
    ind = np.column_stack([(runs == r).astype(float) for r in (1, 2)])
    drift = np.block([[dct, np.zeros((L, 6))], [np.zeros((L, 6)), dct]])
    task = np.resize(np.repeat([0.0, 1.0], 10), n)
    return np.column_stack([ind, drift, task])


def test_correction_stays_accurate_under_censoring():
    """fmriAR 5f5fc32 + bd4e03f (test-audit-regressions.R:557-582), row AR-05.

    Drift regressors leave the bias map with a near-null direction, and
    censoring lets noise into it: the exact solve returned phi with sd ~0.3
    for a truth of 0.4. Cheap pass disqualified: refusing the correction
    under censoring returns the raw mean (~0.32), which fails the bias
    bound; zeroing the near-null direction (truncated SVD) does not satisfy
    the zero-tail pin and drifts.
    """
    from scipy.signal import lfilter

    from fmrimod.ar.estimation import fit_noise

    rng = np.random.default_rng(9204)
    n = 300
    runs = np.repeat([1, 2], 150)
    X = _drift_design(n, runs)
    raw, cor = [], []
    for _ in range(16):
        E = lfilter([1.0], [1.0, -0.4], rng.standard_normal((n + 200, 30)), axis=0)[200:]
        R = E - X @ np.linalg.lstsq(X, E, rcond=None)[0]
        cen = np.sort(rng.choice(n, 30, replace=False))
        raw.append(fit_noise(R, runs=runs, p=1, censor=cen).phi[0][0])
        cor.append(fit_noise(R, runs=runs, p=1, censor=cen, design=X).phi[0][0])
    raw, cor = np.array(raw), np.array(cor)
    assert abs(cor.mean() - 0.4) < 0.04
    assert cor.std(ddof=1) < 0.08
    assert abs(cor.mean() - 0.4) < abs(raw.mean() - 0.4)


def test_solve_correction_is_exact_for_well_conditioned_maps():
    """Every well-determined direction is solved exactly (R :584-589)."""
    from fmrimod.ar.acvf import _solve_correction

    rng = np.random.default_rng(9205)
    A = np.eye(10) + rng.normal(scale=0.05, size=(10, 10))
    g = rng.standard_normal(10)
    np.testing.assert_allclose(_solve_correction(A, g), np.linalg.solve(A, g),
                               rtol=1e-13, atol=1e-13)


def test_solve_correction_pins_near_null_directions_by_the_zero_tail():
    """Formula check of the pin on a real censored drift-design map (AR-05).

    Noise along the near-null direction (what censoring adds) is divided by
    its singular value in an exact solve; the pin keeps it out.
    """
    from fmrimod.ar.acvf import _solve_correction

    rng = np.random.default_rng(9206)
    n = 300
    runs = np.repeat([1, 2], 150)
    cen = np.sort(rng.choice(n, 30, replace=False))
    A = acvf_bias_matrix(_drift_design(n, runs), runs=runs, censor=cen)[0]
    U, d, Vt = np.linalg.svd(A)
    keep = d > 0.1 * np.median(d)
    assert 0 < np.sum(~keep) < 4  # the drift design leaves isolated null dirs
    truth = 0.4 ** np.arange(A.shape[0])
    g = A @ truth + 0.02 * U[:, ~keep].sum(axis=1)
    x = _solve_correction(A, g)
    np.testing.assert_allclose(x, truth, rtol=0, atol=1e-4)
    assert np.max(np.abs(np.linalg.solve(A, g) - truth)) > 0.3
    # Well-determined directions are solved exactly, and the pinned ones bring
    # the last quarter of the budget to zero in the least-squares sense.
    np.testing.assert_allclose(U[:, keep].T @ (A @ x), U[:, keep].T @ g, atol=1e-10)
    tail = slice(A.shape[0] - int(np.ceil(0.25 * A.shape[0])), A.shape[0])
    np.testing.assert_allclose(Vt[~keep][:, tail] @ x[tail], 0.0, atol=1e-10)


def _ols_resid(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return Y - X @ np.linalg.lstsq(X, Y, rcond=None)[0]


def test_design_correction_rejects_residuals_from_the_wrong_operator():
    """AR-14 (fmriAR ca27f27): raw series and WLS residuals are refused."""
    import pytest

    from fmrimod.ar import DesignResidualMismatchError, fit_noise, noise_acvf

    rng = np.random.default_rng(707)
    n = 120
    X = np.column_stack([_design(n, 707), rng.standard_normal((n, 3))])
    Y = rng.standard_normal((n, 5))
    R = _ols_resid(X, Y)
    fit_noise(R, p=1, design=X)
    fit_noise(1e8 * R, p=1, design=X)

    with pytest.raises(DesignResidualMismatchError, match="not orthogonal"):
        fit_noise(Y, p=1, design=X)
    with pytest.raises(DesignResidualMismatchError, match="matching OLS residuals"):
        noise_acvf(Y, max_lag=5, design=X)
    w = np.linspace(0.5, 1.5, n)
    beta_w = np.linalg.solve(X.T @ (w[:, None] * X), X.T @ (w[:, None] * Y))
    with pytest.raises(DesignResidualMismatchError):
        fit_noise(Y - X @ beta_w, p=1, design=X)
    assert issubclass(DesignResidualMismatchError, ValueError)


def test_design_residual_check_obeys_column_space_invariants():
    """AR-14: rank-deficient, column-mixed, rescaled and tiny residuals pass."""
    from fmrimod.ar.acvf import _validate_design_residuals

    rng = np.random.default_rng(708)
    n = 90
    X = _design(n)
    Xd = np.column_stack([X, X[:, 1], np.zeros(n)])
    R = _ols_resid(Xd, rng.standard_normal((n, 6)))
    R0, X0 = R.copy(), Xd.copy()
    _validate_design_residuals(R, Xd)
    _validate_design_residuals(R.mean(axis=1), Xd)
    _validate_design_residuals(R @ rng.standard_normal((6, 3)), Xd)
    Xe = Xd[:, [2, 0, 4, 1, 3, 5]] * np.array([2, 0.5, 3, 1.5, 0.75, 1])
    _validate_design_residuals(R, Xe)
    _validate_design_residuals(1e-10 * R, Xd)
    _validate_design_residuals(1e-10 * R, 1e-10 * Xd)
    np.testing.assert_array_equal(R, R0)
    np.testing.assert_array_equal(Xd, X0)
