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
