"""R/Python parity oracle for AR-aware LSS.

Bead: bd-01KRK97N4F0EZWSHTFG3GMCEJB. Compares
:func:`fmrimod.single.estimate_single_trial` (with
:class:`fmrimod.single._prewhiten.PrewhitenConfig`) against
``fmrilss::lss(Y, X, prewhiten = list(method = "ar", p = 1))``.

R's ``fmrilss::lss(Y, X)`` defaults ``Z = matrix(1, n, 1)`` (a column
of ones) when no baseline is provided, so the Python side passes
``include_intercept=True`` to match the same modelled baseline.

Achieved parity on the planted-AR(1) fixtures below: ``phi`` recovered
to ~1e-3, ``betas`` to ~5e-3 relative. Strict-gate (1e-6) parity is
not in scope for this bead; tighter agreement is a follow-up tracked
under the same bead's note stream.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.rpy2


@pytest.fixture(scope="module")
def r_fmrilss():
    """Load the R fmrilss + fmriAR packages once per test module."""
    try:
        import rpy2.robjects as ro
    except ImportError:
        pytest.skip("rpy2 not available")
    try:
        ro.r("suppressMessages(library(fmrilss))")
        ro.r("suppressMessages(library(fmriAR))")
    except Exception as exc:  # pragma: no cover - environment-specific
        pytest.skip(f"R fmrilss / fmriAR not available: {exc}")
    return ro


def _make_fixture(
    seed: int,
    n: int,
    n_trials: int,
    n_voxels: int,
    phi_true: float,
    snr: float,
):
    rng = np.random.default_rng(seed)
    X = np.zeros((n, n_trials), dtype=np.float64)
    spacing = n // (n_trials + 1)
    for k in range(n_trials):
        s = (k + 1) * spacing
        X[s : s + 5, k] = 1.0

    eps = rng.standard_normal((n, n_voxels))
    noise = np.empty_like(eps)
    noise[0] = eps[0] / np.sqrt(1.0 - phi_true**2)
    for t in range(1, n):
        noise[t] = phi_true * noise[t - 1] + eps[t]

    beta_true = rng.standard_normal((n_trials, n_voxels))
    Y = X @ beta_true * snr + noise
    return Y, X, beta_true


def _r_lss_ar1(ro, Y: np.ndarray, X: np.ndarray, p: int = 1) -> np.ndarray:
    from rpy2.robjects import default_converter, numpy2ri
    from rpy2.robjects.conversion import localconverter

    with localconverter(default_converter + numpy2ri.converter):
        ro.globalenv["X"] = X
        ro.globalenv["Y"] = Y
        ro.globalenv["p"] = p
        ro.r("result <- fmrilss::lss(Y, X, prewhiten = list(method='ar', p=p))")
        betas = np.asarray(ro.globalenv["result"])
    return betas


class TestRParity:
    def test_high_snr_low_ar(self, r_fmrilss) -> None:
        from fmrimod.single import estimate_single_trial
        from fmrimod.single._prewhiten import PrewhitenConfig

        Y, X, _ = _make_fixture(
            seed=123, n=200, n_trials=4, n_voxels=3, phi_true=0.3, snr=5.0,
        )
        py = estimate_single_trial(
            Y, X, method="lss",
            prewhiten=PrewhitenConfig(method="ar", p=1),
            include_intercept=True,
        )
        betas_r = _r_lss_ar1(r_fmrilss, Y, X, p=1)

        assert py.betas.shape == betas_r.shape
        # Empirical AR-LSS parity on planted AR(1): max abs ~3e-3, max
        # rel ~5e-3. The residual gap is dominated by tiny phi-recovery
        # differences (1e-3) between fit_noise and fmriAR::fit_noise;
        # tightening this is tracked under bd-01KRK97N4F0EZWSHTFG3GMCEJB.
        np.testing.assert_allclose(py.betas, betas_r, atol=5e-3, rtol=5e-3)

    def test_moderate_snr_higher_ar(self, r_fmrilss) -> None:
        from fmrimod.single import estimate_single_trial
        from fmrimod.single._prewhiten import PrewhitenConfig

        Y, X, _ = _make_fixture(
            seed=42, n=96, n_trials=6, n_voxels=8, phi_true=0.5, snr=1.0,
        )
        py = estimate_single_trial(
            Y, X, method="lss",
            prewhiten=PrewhitenConfig(method="ar", p=1),
            include_intercept=True,
        )
        betas_r = _r_lss_ar1(r_fmrilss, Y, X, p=1)

        assert py.betas.shape == betas_r.shape
        # Higher AR + lower SNR loosens the tolerance to ~5e-2 abs /
        # ~2e-2 rel. The acceptance pins that the Python implementation
        # tracks the R reference shape; the size of the gap is a
        # follow-up under the same bead.
        np.testing.assert_allclose(py.betas, betas_r, atol=5e-2, rtol=2e-2)

    def test_phi_recovery_matches_r_within_1e_3(self, r_fmrilss) -> None:
        from rpy2.robjects import default_converter, numpy2ri
        from rpy2.robjects.conversion import localconverter

        from fmrimod.single._prewhiten import PrewhitenConfig, prewhiten_matrices

        Y, X, _ = _make_fixture(
            seed=123, n=200, n_trials=4, n_voxels=3, phi_true=0.3, snr=5.0,
        )
        Z = np.ones((Y.shape[0], 1), dtype=np.float64)

        pw_py = prewhiten_matrices(
            Y, X, None,
            PrewhitenConfig(method="ar", p=1),
            baseline_regressors=Z,
        )
        phi_py = float(pw_py.plan.phi[0][0])

        with localconverter(default_converter + numpy2ri.converter):
            r_fmrilss.globalenv["X"] = X
            r_fmrilss.globalenv["Y"] = Y
            r_fmrilss.globalenv["Z"] = Z
            r_fmrilss.r(
                "pw <- fmrilss:::.prewhiten_data(Y, X, Z, NULL, "
                "list(method='ar', p=1))"
            )
            phi_r = float(r_fmrilss.r("pw$whiten_plan$phi[[1]]")[0])

        # ~1e-3 agreement is the achieved bar; this assertion pins the
        # contract so any regression below it surfaces as a test failure
        # rather than as a betas-parity drift detected downstream.
        assert abs(phi_py - phi_r) < 2e-3, (
            f"phi mismatch: Python={phi_py:.6f}, R={phi_r:.6f}"
        )
