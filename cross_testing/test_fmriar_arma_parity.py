"""R parity for pooled Hannan-Rissanen ARMA noise models (fmriAR 0.4.2).

Inventory v2 row AR-03 ported fmriAR's ``.hr_arma_pooled`` (``617bd87``,
``a75a5de``): every voxel is a replicate series, lags stay inside contiguous
run/censor segments, and ``sigma2`` is the mean squared exact-whitened
residual. These cases run ``fit_noise`` + ``whiten_apply`` on identical data
in both packages and compare coefficients, ``sigma2`` and the whitened
matrices; the order-selection grid is compared on the private fitters.

``fit_noise(method="arma", p="auto")`` and ``q="auto"`` are not compared:
fmrimod still fixes AR order 2 under ``p="auto"`` and has no ``q="auto"``
(inventory v2 AR-13 and AR-23, both open).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import lfilter

from fmrimod.ar.estimation import fit_noise
from fmrimod.ar.hr_arma import _hr_arma_pooled
from fmrimod.ar.whitening import whiten_apply

from .conftest import _unavailable

pytestmark = pytest.mark.rpy2

# Observed agreement is 1e-13 or better except ARMA(3, 2) with refinement
# (2.5e-11 on coefficients). The assertions use atol=1e-9 (rtol=1e-9 for
# sigma2), which leaves room for BLAS differences.


@pytest.fixture
def r_call(request):
    """Call an fmriAR function on numpy inputs; return plain R objects.

    ``cross_testing.utils`` installs a global numpy/pandas converter that turns
    a classed R list (an ``fmriAR_plan``) into a Python dict, so every call
    runs under rpy2's default converter instead.
    """
    try:
        import rpy2.robjects as ro
        from rpy2.robjects.conversion import localconverter
    except Exception as exc:  # ImportError, or rpy2 present without a usable R
        _unavailable(request.config, f"rpy2/R not available - skipping fmriAR parity: {exc}")
    try:
        with localconverter(ro.default_converter):
            ro.r("suppressMessages(library(fmriAR))")
    except Exception as exc:  # pragma: no cover - environment-specific
        _unavailable(request.config, f"R fmriAR package not available: {exc}")

    def to_r(value):
        if isinstance(value, np.ndarray) and value.ndim == 2:
            a = np.asarray(value, dtype=np.float64)
            return ro.r["matrix"](
                ro.FloatVector(a.ravel(order="F")), nrow=a.shape[0], ncol=a.shape[1]
            )
        if isinstance(value, np.ndarray) and value.dtype == bool:
            return ro.BoolVector(value.tolist())
        if isinstance(value, np.ndarray | range):
            return ro.IntVector([int(v) for v in value])
        if isinstance(value, list):
            return ro.r["list"](*[ro.r["list"](**{k: to_r(v) for k, v in u.items()})
                                  for u in value])
        return ro.NULL if value is None else value

    def to_np(value):
        if isinstance(value, ro.vectors.ListVector):
            return [to_np(v) for v in value]
        a = np.asarray(list(value), dtype=np.float64)
        dim = ro.r["dim"](value)
        if dim is not ro.NULL:
            nrow, ncol = (int(d) for d in dim)
            a = a.reshape(ncol, nrow).T
        return a

    def call(fn, *args, **kwargs):
        with localconverter(ro.default_converter):
            return ro.r(fn)(*[to_r(a) for a in args], **{k: to_r(v) for k, v in kwargs.items()})

    def field(obj, name):
        with localconverter(ro.default_converter):
            return to_np(obj.rx2(name))

    call.field = field
    return call


def _arma_noise(rng, n, V, phi, theta, shared=0.0):
    e = rng.standard_normal((n + 200, V))
    y = lfilter(np.r_[1.0, theta], np.r_[1.0, -np.asarray(phi)], e, axis=0)[200:]
    if shared:
        s = lfilter([1.0], [1.0, -0.9], rng.standard_normal(n + 200))[200:]
        y = y + shared * s[:, None]
    return y


def _dataset(name):
    rng = np.random.default_rng(20261024 if name == "d1" else 7)
    if name == "d1":
        lens, V = [150, 120, 180, 140], 40
        R = np.vstack([_arma_noise(rng, n, V, [0.5, -0.2], [0.4], shared=0.3) for n in lens])
    else:
        lens, V = [200, 160, 220], 25
        R = np.vstack([_arma_noise(rng, n, V, [0.6], [-0.3]) for n in lens])
        R = R * rng.uniform(0.5, 3.0, V)
    n = R.shape[0]
    X = np.column_stack([np.ones(n), rng.standard_normal((n, 2))])
    censor = np.zeros(n, dtype=bool)
    censor[rng.choice(np.arange(5, n - 5), size=12, replace=False)] = True
    censor[40:44] = True
    runs = np.repeat(np.arange(1, len(lens) + 1), lens)
    return R, X, runs, censor


CASES = [
    # (data, p, q, pooling, censor, hr_iter)
    ("d1", 2, 1, "run", False, 0),
    ("d1", 2, 1, "global", False, 0),
    ("d1", 2, 1, "run", True, 0),
    ("d1", 2, 1, "global", True, 0),
    ("d1", 2, 1, "global", False, 1),
    ("d2", 1, 1, "run", False, 0),
    ("d2", 1, 1, "global", True, 0),
    ("d2", 0, 2, "global", False, 0),
    ("d1", 3, 2, "run", True, 2),
]


@pytest.mark.parametrize(("data", "p", "q", "pooling", "use_censor", "hr_iter"), CASES)
def test_fit_noise_arma_matches_fmriar(r_call, data, p, q, pooling, use_censor, hr_iter):
    R, X, runs, censor = _dataset(data)
    cen = censor if use_censor else None

    r_plan = r_call(
        "fmriAR::fit_noise", resid=R, runs=runs, censor=cen, method="arma",
        p=p, q=q, pooling=pooling, hr_iter=hr_iter,
    )
    r_white = r_call("fmriAR::whiten_apply", r_plan, X, R, runs=runs, censor=cen)

    plan = fit_noise(
        resid=R, runs=runs, censor=cen, method="arma", p=p, q=q,
        pooling=pooling, hr_iter=hr_iter,
    )
    white = whiten_apply(plan, X, R, runs=runs, censor=cen)

    r_phi = r_call.field(r_plan, "phi")
    r_theta = r_call.field(r_plan, "theta")
    assert len(plan.phi) == len(r_phi)
    for ours, theirs in zip(plan.phi, r_phi, strict=True):
        np.testing.assert_allclose(ours, theirs, rtol=0, atol=1e-9)
    for ours, theirs in zip(plan.theta, r_theta, strict=True):
        np.testing.assert_allclose(ours, theirs, rtol=0, atol=1e-9)
    np.testing.assert_allclose(
        np.ravel(np.asarray(plan.sigma2, dtype=np.float64)),
        np.ravel(r_call.field(r_plan, "sigma2")),
        rtol=1e-9,
    )
    np.testing.assert_allclose(white.Y, r_call.field(r_white, "Y"), rtol=0, atol=1e-9)
    np.testing.assert_allclose(white.X, r_call.field(r_white, "X"), rtol=0, atol=1e-9)


@pytest.mark.parametrize("data", ["d1", "d2"])
@pytest.mark.parametrize("hr_iter", [0, 1])
def test_pooled_hr_order_grid_matches_fmriar(r_call, data, hr_iter):
    # Two segments per run (a split at frame 37) so selection runs on
    # segment-respecting rows, as under censoring.
    R, _, runs, _ = _dataset(data)
    mats = [R[runs == r] for r in np.unique(runs)]
    starts = np.array([0, 37])

    r_fit = r_call(
        "fmriAR:::.hr_arma_pooled",
        [{"mat": m, "starts0": starts} for m in mats],
        range(5),
        range(3),
        iter=hr_iter,
    )
    fit = _hr_arma_pooled([(m, starts) for m in mats], range(5), range(3), n_iter=hr_iter)

    assert fit.order == tuple(int(v) for v in r_call.field(r_fit, "order"))
    np.testing.assert_allclose(fit.phi, r_call.field(r_fit, "phi"), rtol=0, atol=1e-9)
    np.testing.assert_allclose(fit.theta, r_call.field(r_fit, "theta"), rtol=0, atol=1e-9)
    np.testing.assert_allclose(fit.sigma2, np.ravel(r_call.field(r_fit, "sigma2"))[0], rtol=1e-9)
