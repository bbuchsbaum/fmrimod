"""``fmri_lm``'s shared AR estimate carries fmriAR's residual-bias correction.

fmrireg (23c275f, 168980d) estimates the shared AR model from the OLS
residuals with ``fmriAR::fit_noise(design = X, correction_max_lag = budget)``,
budget ``max(p, min(25, max(5, 2p + 1), floor(rdf / 3)))``, and holds it fixed
for the GLS solve. Without the correction, projecting out a design with drift
columns biases phi down (the OLS residual operator mixes the autocovariance).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import fmrimod as fm
from fmrimod.ar.estimation import fit_noise
from fmrimod.design import RealizedDesign
from fmrimod.model.config import AROptions, FmriLmConfig


def _design(n: int) -> tuple[np.ndarray, list[str]]:
    t = np.linspace(-1.0, 1.0, n)
    ev1 = (np.sin(np.arange(n) * 2 * np.pi / 20) > 0.6).astype(float)
    ev2 = (np.cos(np.arange(n) * 2 * np.pi / 14) > 0.6).astype(float)
    drift = np.column_stack([t**k for k in range(1, 6)])
    X = np.column_stack([ev1, ev2, np.ones(n), drift])
    return X, ["e1", "e2", "int"] + [f"d{k}" for k in range(1, 6)]


def test_shared_ar1_estimate_is_design_corrected():
    rng = np.random.default_rng(20261024)
    n, V, phi_true, n_rep = 100, 40, 0.4, 40
    X, cols = _design(n)
    rdf = n - np.linalg.matrix_rank(X)
    assert rdf == n - 8
    budget = max(1, min(25, max(5, 2 * 1 + 1), rdf // 3))
    events = pd.DataFrame(
        {"onset": [0.0], "condition": ["A"], "duration": [0.0], "run": [1]}
    )

    fitted, reference = [], []
    for _ in range(n_rep):
        z = rng.standard_normal((n, V))
        E = np.zeros((n, V))
        for i in range(n):
            E[i] = z[i] + (phi_true * E[i - 1] if i else 0.0)
        Y = X @ rng.standard_normal((X.shape[1], V)) + E
        ds = fm.fmri_dataset(Y, tr=2.0, events=events, run_length=[n])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fit = fm.fmri_lm(
                RealizedDesign.from_array(X, cols),
                ds,
                config=FmriLmConfig(ar=AROptions(struct="ar1")),
            )
        fitted.append(float(np.ravel(fit.ar_params)[0]))
        resid = Y - X @ np.linalg.lstsq(X, Y, rcond=None)[0]
        plan = fit_noise(
            resid=resid,
            method="ar",
            p=1,
            p_max=1,
            design=X,
            correction_max_lag=budget,
        )
        reference.append(float(plan.phi[0][0]))

    # The uncorrected pooled Yule-Walker estimate averages about 0.28 here.
    assert abs(np.mean(fitted) - np.mean(reference)) < 0.03
    assert abs(np.mean(fitted) - phi_true) < 0.05
    # Each fit is the corrected estimate itself, not merely close on average.
    np.testing.assert_allclose(fitted, reference, rtol=0, atol=1e-10)


def test_iter_gls_holds_the_corrected_estimate_fixed():
    # R forces iter_gls to 1 for AR: re-estimating from GLS residuals would
    # apply the OLS correction to the wrong residual operator.
    rng = np.random.default_rng(7)
    n, V = 120, 30
    X, cols = _design(n)
    z = rng.standard_normal((n, V))
    E = np.zeros((n, V))
    for i in range(n):
        E[i] = z[i] + (0.5 * E[i - 1] if i else 0.0)
    Y = X @ rng.standard_normal((X.shape[1], V)) + E
    events = pd.DataFrame(
        {"onset": [0.0], "condition": ["A"], "duration": [0.0], "run": [1]}
    )
    ds = fm.fmri_dataset(Y, tr=2.0, events=events, run_length=[n])
    fits = [
        fm.fmri_lm(
            RealizedDesign.from_array(X, cols),
            ds,
            config=FmriLmConfig(ar=AROptions(struct="ar1", iter_gls=k)),
        )
        for k in (1, 3)
    ]
    np.testing.assert_array_equal(fits[0].ar_params, fits[1].ar_params)
    np.testing.assert_array_equal(fits[0].betas, fits[1].betas)
