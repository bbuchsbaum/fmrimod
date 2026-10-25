"""First-level AR GLS for ``fmri_lm``.

The noise model is estimated by :func:`~fmrimod.ar.estimation.fit_noise` and
applied by :func:`~fmrimod.ar.whitening.whiten_apply`, the API single-trial
prewhitening uses, following fmrireg's runwise AR fit
(``R/fmri_lm_strategies.R`` ``process_run_standard``, ``R/fmrilm.R`` global
AR, ``R/fmri_lm_runwise.R`` ``runwise_lm_voxelwise``):

1. The engine's OLS fit supplies, per run, the matrices it solved (after
   censoring, volume weighting and soft projection) and their residuals.
2. phi is estimated once from those residuals, with fmriAR's residual-bias
   correction for the design that produced them, and held fixed for the GLS
   solve: the correction belongs to the OLS residual operator (fmrireg
   23c275f, 168980d).
3. Each run's matrices are whitened, restarting at the run start and at
   every censoring gap, and solved.

Censored volumes are not observations (fmrimod scrubs them): they are gaps
in the AR estimation and the whitening, never data.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from scipy import linalg

from fmrimod.dataset.data_access import get_run_data
from fmrimod.glm.errors import FmriCapabilityError, UnsupportedEngineConfiguration
from fmrimod.glm.solver import LmResult, Projection, fast_lm_matrix, fast_preproject
from fmrimod.model.config import AROptions, FmriLmConfig

from .acvf import DesignResidualMismatchError
from .estimation import (
    ArEstimationFallbackWarning,
    _quantise_to_noise_pools,
    fit_noise,
)
from .plan import plan_from_phi
from .whitening import whiten_apply

# fmrireg ``.ar_correction_lag_budget``: at most 25 corrected lags.
_MAX_CORRECTION_LAG = 25


@dataclass(frozen=True)
class _RunMatrices:
    """One run as the OLS fit solved it.

    ``X``, ``Y`` and ``resid`` hold the observed volumes only; ``keep`` marks
    them on the run's volume grid (``None`` when nothing was censored).
    """

    X: NDArray[np.float64]
    Y: NDArray[np.float64]
    resid: NDArray[np.float64]
    keep: NDArray[np.bool_] | None
    proj: Projection
    obs_weights: NDArray[np.float64] | None

    @property
    def rank(self) -> int:
        return int(self.proj.rank)

    @property
    def censor(self) -> NDArray[np.bool_] | None:
        """Censoring mask on the volume grid, or ``None``."""
        return None if self.keep is None else ~self.keep

    @property
    def segments(self) -> NDArray[np.intp] | None:
        """Contiguous-segment label of each observed volume, or ``None``."""
        if self.keep is None:
            return None
        idx = np.flatnonzero(self.keep)
        breaks = np.concatenate([[0], (np.diff(idx) > 1).astype(np.intp)])
        return np.cumsum(breaks).astype(np.intp)

    def on_grid(self, mat: NDArray[np.float64]) -> NDArray[np.float64]:
        """Rows of ``mat`` placed on the volume grid, zero where censored.

        A zero row is outside every residual-forming operator and lag
        product, so the grid form describes the scrubbed fit exactly.
        """
        if self.keep is None:
            return mat
        out = np.zeros((self.keep.size, mat.shape[1]), dtype=np.float64)
        out[self.keep] = mat
        return out


def validate_ar_config(model: object, config: FmriLmConfig) -> None:
    """Refuse AR configurations ``fmri_lm`` cannot fit.

    Voxelwise AR follows fmrireg's narrow contract
    (``.validate_builtin_voxelwise_noise``, 3c2a9c2), plus robust fitting,
    which needs one design per run here. ARMA (``q > 0``) follows fmrireg's
    narrow MA contract (``noise_spec``, ``fmri_lm_control``, 3c2a9c2).
    Options no fitting path reads are refused rather than ignored.
    """
    ar = config.ar
    if ar.voxelwise and not ar.enabled:
        raise ValueError(
            "voxelwise AR requires an AR structure; set AROptions(struct='ar1') "
            "or another AR structure"
        )
    if not ar.enabled:
        return
    if ar.method == "afni":
        raise UnsupportedEngineConfiguration(
            engine="runwise",
            feature="AROptions(method='afni')",
            repair=(
                "fmri_lm estimates AR/ARMA models from the data; build an AFNI "
                "plan with fmrimod.ar.afni_restricted_plan and whiten directly"
            ),
            current_capability=("method='ar'", "method='arma'"),
        )
    if ar.parcels is not None:
        raise UnsupportedEngineConfiguration(
            engine="runwise",
            feature="AROptions(parcels=...)",
            repair="drop parcels; fmri_lm does not support parcel pooling",
            current_capability=("pooling='run'", "pooling='global'"),
        )
    if ar.is_arma:
        _validate_arma_config(model, config)
        return
    if ar.pooling == "parcel":
        raise UnsupportedEngineConfiguration(
            engine="runwise",
            feature="AR pooling='parcel'",
            repair="use AROptions(pooling='run') or pooling='global'",
            current_capability=("pooling='run'", "pooling='global'"),
        )
    if config.robust.enabled and config.robust.reestimate_phi and ar.pooling == "global":
        # fmrireg skips the re-estimation when one phi is held fixed across
        # runs; refuse rather than silently ignore the option.
        raise FmriCapabilityError(
            feature="robust AR re-estimation (reestimate_phi=True) with pooling='global'",
            current_capability=("reestimate_phi with pooling='run'",),
            repair="set AROptions(pooling='run') or RobustOptions(reestimate_phi=False)",
        )
    if not ar.voxelwise:
        return
    capability = (
        "pooling='run'",
        "iter_gls=1",
        "no censoring",
        "no volume weights, soft projection or robust fitting",
    )
    blockers = []
    if ar.pooling != "run":
        blockers.append(f"pooling={ar.pooling!r}")
    if ar.iter_gls != 1:
        blockers.append(f"iter_gls={ar.iter_gls}")
    if ar.censor is not None or _dataset_censors(model):
        blockers.append("censoring")
    if config.volume_weights.enabled:
        blockers.append("volume weights")
    if config.soft_subspace.enabled:
        blockers.append("soft subspace projection")
    if config.robust.enabled:
        blockers.append("robust fitting")
    if blockers:
        raise UnsupportedEngineConfiguration(
            engine="runwise",
            feature=f"voxelwise AR with {', '.join(blockers)}",
            repair=(
                "use voxelwise AR with pooling='run', iter_gls=1 and no "
                "censoring, weighting, projection or robust fitting, or set "
                "AROptions(voxelwise=False) for a shared AR model"
            ),
            current_capability=capability,
        )


def _validate_arma_config(model: object, config: FmriLmConfig) -> None:
    """fmrireg's narrow MA contract: runwise, run pooling, no censoring,
    voxelwise covariance, noise pools or robust fitting, ``iter_gls >= 1``."""
    ar = config.ar
    blockers = []
    if ar.pooling != "run":
        blockers.append(f"pooling={ar.pooling!r}")
    if ar.voxelwise:
        blockers.append("voxelwise covariance")
    if ar.censor is not None or _dataset_censors(model):
        blockers.append("censoring")
    if ar.iter_gls < 1:
        blockers.append(f"iter_gls={ar.iter_gls}")
    if ar.noise_pools is not None:
        blockers.append("noise_pools")
    if config.robust.enabled:
        blockers.append("robust fitting")
    if blockers:
        raise UnsupportedEngineConfiguration(
            engine="runwise",
            feature=f"ARMA noise (q > 0) with {', '.join(blockers)}",
            repair=(
                "use ARMA with pooling='run', iter_gls >= 1 and no censoring, "
                "voxelwise covariance, noise pools or robust fitting, or set "
                "q=0 for an AR model"
            ),
            current_capability=(
                "pooling='run'",
                "iter_gls>=1",
                "no censoring",
                "no voxelwise covariance, noise pools or robust fitting",
            ),
        )


def iterative_gls(
    model: object,  # FmriModel
    config: FmriLmConfig,
    initial_fit: dict[str, Any],
) -> tuple[dict[str, Any], NDArray[np.float64]]:
    """Refit the runs of an OLS fit by AR generalised least squares.

    Parameters
    ----------
    model : FmriModel
        The fMRI model.
    config : FmriLmConfig
        Configuration with AR options.
    initial_fit : dict
        Initial OLS fit from ``fit_runwise()``: per-run ``run_X``,
        ``residuals``, ``run_results`` and ``projections``.

    Returns
    -------
    fit_result : dict
        GLS fit. ``run_X`` and ``residuals`` are the whitened design and
        residuals of each run, a matching pair. For voxelwise AR, ``XtXinv``
        holds one matrix per voxel, shape ``(V, p, p)``, and ``run_X`` is
        ``None`` (every voxel has its own whitened design).
    ar_params : NDArray
        The AR coefficients the whitening applied: ``(p,)`` for one run or
        global pooling, ``(n_runs, p)`` for run pooling; voxelwise
        ``(p, V)`` for one run, ``(n_runs, p, V)`` otherwise. Empty when
        ``iter_gls=0``. For an ARMA model the fit also holds the MA
        coefficients under ``"ma_params"``, shaped like ``ar_params``.
    """
    ar_opts = config.ar
    if ar_opts.iter_gls < 1:
        return initial_fit, np.array([], dtype=np.float64)

    runs = _run_matrices(model, initial_fit)
    if ar_opts.is_arma:
        return _iterative_arma_gls(runs, ar_opts, config.solver)
    order = ar_opts.ar_order
    if ar_opts.voxelwise:
        phis = [
            _voxelwise_phi(
                rm, order, ar_opts.noise_pools, getattr(ar_opts, "bin_width", None)
            )
            for rm in runs
        ]
        fit = _solve_voxelwise(runs, phis, ar_opts.exact_first, config.solver)
        params = phis[0] if len(phis) == 1 else np.stack(phis)
        return fit, params

    phis = _shared_phi(runs, ar_opts)
    fit = _solve_shared(runs, phis, ar_opts.exact_first, config.solver)
    single = len(phis) == 1 or ar_opts.pooling == "global"
    return fit, phis[0] if single else np.stack(phis)


def _iterative_arma_gls(
    runs: list[_RunMatrices], ar_opts: AROptions, solver: Any
) -> tuple[dict[str, Any], NDArray[np.float64]]:
    """Runwise ARMA(p, q) GLS (fmrireg ``.iterative_ar_gls_via_fmriAR``).

    Per run, the ARMA model is estimated by fmriAR's pooled Hannan-Rissanen
    regression from the OLS residuals. Up to ``iter_gls - 1`` refinements
    re-estimate it from the unwhitened residuals ``Y - X b_gls`` of the GLS
    fit, stopping once no coefficient moves by more than
    ``convergence_tol``. ARMA has no residual-bias correction, so iterating
    is meaningful here, unlike for AR.
    """
    p, q = ar_opts.ar_order, ar_opts.ma_order
    exact = "ar1" if ar_opts.exact_first else "none"
    phis, thetas = [], []
    for rm in runs:
        finite = np.all(np.isfinite(rm.resid), axis=0)
        if not np.any(finite):
            phis.append(np.zeros(p))
            thetas.append(np.zeros(q))
            continue
        Y, resid = rm.Y[:, finite], rm.resid[:, finite]
        prev: NDArray[np.float64] | None = None
        for it in range(ar_opts.iter_gls):
            plan = fit_noise(
                resid, method="arma", p=p, q=q, pooling="global", exact_first=exact
            )
            phi = np.zeros(p)
            theta = np.zeros(q)
            phi[: len(plan.phi[0])] = plan.phi[0][:p]
            theta[: len(plan.theta[0])] = plan.theta[0][:q]
            params = np.concatenate([phi, theta])
            if prev is not None and np.max(np.abs(params - prev)) <= ar_opts.convergence_tol:
                break
            prev = params
            if it < ar_opts.iter_gls - 1:
                wr = whiten_apply(
                    plan_from_phi(phi, theta=theta, exact_first=ar_opts.exact_first),
                    rm.X,
                    Y,
                )
                assert wr.X is not None and wr.Y is not None
                proj = fast_preproject(wr.X, check_finite=False, method=solver)
                betas = fast_lm_matrix(wr.X, wr.Y, proj, check_finite=False).betas
                resid = Y - rm.X @ betas
        phis.append(phi)
        thetas.append(theta)
    fit = _solve_shared(runs, phis, ar_opts.exact_first, solver, thetas=thetas)
    single = len(runs) == 1
    fit["ma_params"] = thetas[0] if single else np.stack(thetas)
    return fit, phis[0] if single else np.stack(phis)


def reestimate_ar_gls(
    model: object,  # FmriModel
    config: FmriLmConfig,
    initial_fit: dict[str, Any],
    robust_fit: dict[str, Any],
) -> tuple[dict[str, Any], NDArray[np.float64]]:
    """Re-estimate phi after a robust AR fit and solve the GLS again.

    fmrireg ``process_run_ar_robust`` with ``reestimate_phi = TRUE``: phi is
    re-estimated from the *unwhitened* residuals of the robust fit,
    ``Y - X b_robust`` on the run matrices the OLS fit solved (whitened
    residuals carry almost no autocorrelation), without a design correction,
    since these are not OLS residuals. The caller reruns the robust IRLS on
    the returned GLS fit. Run pooling only (see :func:`validate_ar_config`).
    """
    ar_opts = config.ar
    runs = _run_matrices(model, initial_fit)
    order = ar_opts.ar_order
    phis = []
    for rm, robust in zip(runs, robust_fit["run_results"], strict=True):
        resid = rm.Y - rm.X @ robust.betas
        phis.append(_estimate_phi([rm.on_grid(resid)], order, censor=rm.censor))
    fit = _solve_shared(runs, phis, ar_opts.exact_first, config.solver)
    return fit, phis[0] if len(phis) == 1 else np.stack(phis)


# -- Run matrices ------------------------------------------------------------


def _run_matrices(model: object, initial_fit: dict[str, Any]) -> list[_RunMatrices]:
    """The matrices each run's OLS fit solved, with their residuals."""
    out = []
    for r, (X, resid, result, proj) in enumerate(
        zip(
            initial_fit["run_X"],
            initial_fit["residuals"],
            initial_fit["run_results"],
            initial_fit["projections"],
            strict=True,
        )
    ):
        out.append(
            _RunMatrices(
                X=X,
                # The data the OLS fit paired with X (as robust_refit does).
                Y=resid + X @ result.betas,
                resid=resid,
                keep=_kept_volumes(model, r, X.shape[0]),
                proj=proj,
                obs_weights=getattr(result, "obs_weights", None),
            )
        )
    return out


def _dataset_censors(model: object) -> bool:
    """True when the model's dataset censors any volume."""
    dataset = getattr(model, "dataset", None)
    get_censor = getattr(dataset, "get_censor", None)
    if not callable(get_censor):
        return False
    for r in range(int(getattr(model, "n_runs", 0))):
        censor = get_censor(r)
        if censor is not None and np.any(np.asarray(censor)):
            return True
    return False


def _kept_volumes(
    model: object, run: int, n_rows: int
) -> NDArray[np.bool_] | None:
    """Observed volumes of a run (the rows ``fit_run_ols`` kept), or ``None``."""
    dataset = getattr(model, "dataset", None)
    get_censor = getattr(dataset, "get_censor", None)
    censor = get_censor(run) if callable(get_censor) else None
    if censor is None:
        return None
    # fit_run_ols has validated the mask (boolean or 0/1).
    censor = np.asarray(censor).astype(bool)
    if not np.any(censor):
        return None
    keep = ~censor
    if int(keep.sum()) != n_rows:
        raise ValueError(
            f"run {run}: the censor mask keeps {int(keep.sum())} volumes but "
            f"the OLS fit solved {n_rows}"
        )
    return keep


# -- Estimation --------------------------------------------------------------


def _correction_lag_budget(order: int, rdf: int) -> int:
    """fmrireg ``.ar_correction_lag_budget``.

    Projection mixes the whole autocovariance tail into lags ``0..p``: correct
    at least five lags, and ``2p + 1`` for higher orders, keeping about three
    residual df per corrected lag and at most 25.
    """
    desired = max(5, 2 * order + 1)
    supported = max(0, rdf // 3)
    return max(order, min(_MAX_CORRECTION_LAG, desired, supported))


def _estimate_phi(
    resid: Sequence[NDArray[np.float64]],
    order: int,
    *,
    censor: NDArray[np.bool_] | None = None,
    design: NDArray[np.float64] | None = None,
    rdf: int = 0,
) -> NDArray[np.float64]:
    """One AR(``order``) model pooled over voxels (and over runs).

    ``resid`` holds one residual matrix per run, on the volume grid; with
    several runs the model is pooled across them (fmrireg global AR). Only
    voxels whose residuals are finite in every run enter the estimate. An
    estimate that fails falls back to no whitening with a warning, as
    fmrireg ``.estimate_ar_parameters_routed`` does.
    """
    finite = np.logical_and.reduce([np.all(np.isfinite(r), axis=0) for r in resid])
    if not np.any(finite):
        return np.zeros(order, dtype=np.float64)
    if len(resid) == 1 and finite.all():
        stacked = resid[0]  # no copy: (n, V) is large at full-brain size
    else:
        stacked = np.vstack([r[:, finite] for r in resid])
    run_labels = (
        np.repeat(np.arange(len(resid)), [r.shape[0] for r in resid])
        if len(resid) > 1
        else None
    )
    try:
        plan = fit_noise(
            stacked,
            runs=run_labels,
            censor=censor,
            method="ar",
            p=order,
            p_max=order,
            pooling="global",
            design=design,
            correction_max_lag=_correction_lag_budget(order, rdf),
        )
    except DesignResidualMismatchError:
        # The residuals are not this design's OLS residuals: a wiring bug,
        # not a data problem, so it must not degrade to an unwhitened fit.
        raise
    except (ValueError, np.linalg.LinAlgError) as err:
        warnings.warn(
            f"AR({order}) estimation failed ({err}); fitting without AR whitening.",
            ArEstimationFallbackWarning,
            stacklevel=4,
        )
        return np.zeros(order, dtype=np.float64)
    phi = np.zeros(order, dtype=np.float64)
    est = np.asarray(plan.phi[0] if plan.phi else [], dtype=np.float64)[:order]
    if est.size == 0 and order > 0:
        # fit_noise returns no coefficients for zero-variance residuals or
        # too few usable lags: a fallback, not an estimate of zero.
        warnings.warn(
            f"AR({order}) estimation failed (no usable autocovariance); "
            "fitting without AR whitening.",
            ArEstimationFallbackWarning,
            stacklevel=4,
        )
    phi[: est.size] = np.where(np.isfinite(est), est, 0.0)
    return phi


def _shared_phi(runs: list[_RunMatrices], ar_opts: AROptions) -> list[NDArray[np.float64]]:
    """phi per run (``pooling="run"``) or one phi for all runs (``"global"``).

    Each estimate is corrected for the residual-forming operator of the fit
    that produced the residuals. Runs are fitted separately, so for global
    pooling that operator is block diagonal over runs; stacking the designs
    would describe one coefficient vector shared by all runs (168980d).
    """
    order = ar_opts.ar_order
    if ar_opts.pooling == "global" and len(runs) > 1:
        censor = None
        if any(rm.keep is not None for rm in runs):
            censor = np.concatenate(
                [
                    np.zeros(rm.X.shape[0], dtype=bool) if rm.keep is None else rm.censor
                    for rm in runs
                ]
            )
        phi = _estimate_phi(
            [rm.on_grid(rm.resid) for rm in runs],
            order,
            censor=censor,
            design=linalg.block_diag(*(rm.on_grid(rm.X) for rm in runs)),
            rdf=min(rm.X.shape[0] - rm.rank for rm in runs),
        )
        return [phi] * len(runs)
    return [
        _estimate_phi(
            [rm.on_grid(rm.resid)],
            order,
            censor=rm.censor,
            design=rm.on_grid(rm.X),
            rdf=rm.X.shape[0] - rm.rank,
        )
        for rm in runs
    ]


def _voxelwise_yule_walker(
    resid: NDArray[np.float64], order: int, finite: NDArray[np.bool_]
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Vectorised AR(``order``) Yule-Walker for every voxel at once.

    Reproduces :func:`_estimate_phi` on a single voxel with no censoring or
    design correction (the voxelwise path): demeaned series, unbiased
    autocovariance ``sum x_t x_{t-k} / (n - k)``, Yule-Walker via
    Levinson-Durbin. Returns ``(phi, ok)``; ``ok`` is False wherever the
    reference estimator would repair the estimate (autocovariance not
    comfortably positive definite, a partial autocorrelation at or beyond
    the 0.99 clamp, a root near the unit circle) or the series is
    degenerate, and those voxels must use the reference estimator.
    """
    n, V = resid.shape
    phi = np.zeros((order, V), dtype=np.float64)
    ok = finite.copy()
    if order < 1 or n <= order + 1 or not np.any(ok):
        return phi, np.zeros(V, dtype=bool)
    gamma = np.zeros((order + 1, V), dtype=np.float64)
    for start in range(0, V, _SOLVE_CHUNK):
        sl = slice(start, min(start + _SOLVE_CHUNK, V))
        x = np.where(finite[sl], resid[:, sl], 0.0)
        x -= x.mean(axis=0, keepdims=True)
        for k in range(order + 1):
            gamma[k, sl] = np.einsum("tv,tv->v", x[k:], x[: n - k]) / (n - k)
    g0 = gamma[0]
    ok &= np.isfinite(g0) & (g0 > 1e-12 * max(float(np.nanmax(np.abs(g0))), 1e-300))
    safe_g0 = np.where(ok, g0, 1.0)
    rho = gamma / safe_g0
    # Levinson-Durbin, vectorised over voxels.
    a = np.zeros((order, V), dtype=np.float64)
    err = np.ones(V, dtype=np.float64)
    max_kappa = np.zeros(V, dtype=np.float64)
    for k in range(1, order + 1):
        acc = (
            rho[k] - np.einsum("jv,jv->v", a[: k - 1], rho[k - 1 : 0 : -1])
            if k > 1
            else rho[1].copy()
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            kappa = acc / err
        max_kappa = np.maximum(max_kappa, np.abs(np.nan_to_num(kappa, nan=np.inf)))
        if k > 1:
            a[: k - 1] = a[: k - 1] - kappa * a[k - 2 :: -1][: k - 1]
        a[k - 1] = kappa
        err = err * (1.0 - kappa**2)
    # Comfortably inside the 0.99 PACF clamp, and a positive-definite
    # autocovariance (positive prediction error at every order).
    ok &= np.isfinite(max_kappa) & (max_kappa < 0.98) & (err > 1e-4)
    ok &= np.all(np.isfinite(a), axis=0)
    phi[:, ok] = a[:, ok]
    return phi, ok


def _voxelwise_phi(
    rm: _RunMatrices,
    order: int,
    noise_pools: int | None,
    bin_width: float | None = None,
) -> NDArray[np.float64]:
    """phi of every voxel, shape ``(order, V)``.

    fmrireg ``estimate_ar_parameters``: fmriAR on the voxel's OLS residuals,
    with no design correction. ``noise_pools`` quantises the estimates into
    equal-frequency pools (Nilearn's scheme).
    """
    V = rm.resid.shape[1]
    phi = np.zeros((order, V), dtype=np.float64)
    finite = np.all(np.isfinite(rm.resid), axis=0)
    fast_phi, fast_ok = _voxelwise_yule_walker(rm.resid, order, finite)
    phi[:, fast_ok] = fast_phi[:, fast_ok]
    # Voxels near a repair threshold (PD shrink, stationarity clamp) take the
    # reference per-voxel estimator, so results are unchanged.
    for v in np.flatnonzero(finite & ~fast_ok):
        phi[:, v] = _estimate_phi([rm.resid[:, v : v + 1]], order)
    if noise_pools is not None and int(noise_pools) > 1 and np.any(finite):
        phi[:, finite] = _quantise_to_noise_pools(phi[:, finite], int(noise_pools))
    if bin_width is not None:
        # Nilearn's rule: truncate toward zero onto the bin grid.
        phi = np.trunc(phi / bin_width) * bin_width
    return phi


# -- Whitening and solve -----------------------------------------------------


def _whiten(
    rm: _RunMatrices,
    phi: NDArray[np.float64],
    exact_first: bool,
    cols: NDArray[np.intp] | None = None,
    theta: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Whiten a run's design and data, restarting at every censoring gap.

    Voxels with non-finite data (``na_action="propagate"``) stay non-finite.
    """
    Y = rm.Y if cols is None else rm.Y[:, cols]
    plan = plan_from_phi(
        np.asarray(phi, dtype=np.float64),
        theta=None if theta is None else np.asarray(theta, dtype=np.float64),
        exact_first=exact_first,
    )
    finite = np.all(np.isfinite(Y), axis=0)
    wr = whiten_apply(plan, rm.X, Y[:, finite], runs=cast(Any, rm.segments))
    assert wr.X is not None and wr.Y is not None
    if np.all(finite):
        return wr.X, wr.Y
    Yw = np.full(Y.shape, np.nan, dtype=np.float64)
    Yw[:, finite] = wr.Y
    return wr.X, Yw


def _with_observation_df(
    proj: Projection, obs_weights: NDArray[np.float64] | None
) -> Projection:
    """Residual df counting only volumes with a positive weight.

    Whitening is invertible, so the whitened rows still span only the
    weighted volumes (``stats::lm.wfit``, fmrireg 60f8430).
    """
    if obs_weights is None:
        return proj
    dfres = float(np.count_nonzero(np.asarray(obs_weights) > 0) - proj.rank)
    return proj if dfres == proj.dfres else replace(proj, dfres=dfres)


def _solve_shared(
    runs: list[_RunMatrices],
    phis: list[NDArray[np.float64]],
    exact_first: bool,
    solver: Any,
    thetas: list[NDArray[np.float64]] | None = None,
) -> dict[str, Any]:
    """GLS of every run with its phi (and theta), pooled across runs."""
    from fmrimod.glm.strategies import _pool_run_results

    run_results = []
    projections = []
    residuals = []
    run_X = []
    if thetas is None:
        thetas = [np.array([], dtype=np.float64)] * len(runs)
    for rm, phi, theta in zip(runs, phis, thetas, strict=True):
        # The design is whitened and factored once; the data are whitened
        # and solved in column chunks so peak memory stays a few chunks
        # above the output, not several full (n, V) copies.
        V = rm.Y.shape[1]
        Xw, _ = _whiten(rm, phi, exact_first, np.arange(min(V, 1)), theta=theta)
        proj = fast_preproject(Xw, check_finite=False, method=solver)
        proj = _with_observation_df(proj, rm.obs_weights)
        betas = np.empty((Xw.shape[1], V), dtype=np.float64)
        rss = np.empty(V, dtype=np.float64)
        sigma2 = np.empty(V, dtype=np.float64)
        resid = np.empty((Xw.shape[0], V), dtype=np.float64)
        dfres, rank = proj.dfres, proj.rank
        for start in range(0, V, _SOLVE_CHUNK):
            stop = min(start + _SOLVE_CHUNK, V)
            _, Yw = _whiten(
                rm, phi, exact_first, np.arange(start, stop), theta=theta
            )
            chunk = fast_lm_matrix(
                Xw, Yw, proj, return_fitted=True, check_finite=False
            )
            assert chunk.fitted is not None
            betas[:, start:stop] = chunk.betas
            rss[start:stop] = chunk.rss
            sigma2[start:stop] = chunk.sigma2
            # A slice, not an index array: out= must be a view of resid.
            np.subtract(Yw, chunk.fitted, out=resid[:, start:stop])
            dfres, rank = chunk.dfres, chunk.rank
        result = LmResult(
            betas=betas, rss=rss, sigma2=sigma2, dfres=dfres, rank=rank
        )
        result.obs_weights = rm.obs_weights
        run_results.append(result)
        projections.append(proj)
        residuals.append(resid)
        run_X.append(Xw)

    pooled = _pool_run_results(run_results, projections)
    return {
        "betas": pooled["betas"],
        "sigma": pooled["sigma"],
        "dfres": pooled["dfres"],
        "XtXinv": pooled["XtXinv"],
        "projections": projections,
        "run_results": run_results,
        "residuals": residuals,
        "run_X": run_X,
    }


_BATCH_MIN_GROUPS = 32
# Voxel columns whitened and solved together in the shared-AR GLS.
_SOLVE_CHUNK = 16384


def _solve_voxelwise_batched(
    rm: _RunMatrices, phi: NDArray[np.float64]
) -> tuple[NDArray[np.bool_], tuple[NDArray[np.float64], ...]] | None:
    """All voxels' GLS in one batched solve (AR whitening without exact-first).

    Whitening ``Z -> Z + sum_k phi_k op_k(Z)`` is linear in ``phi``, so each
    voxel's whitened Gram matrix and cross-products are quadratic forms in
    ``(1, phi_1, ..., phi_p)`` over a few shared products, and one batched
    ``solve`` replaces a per-voxel whitening + factorisation. Applies to
    full-rank designs and finite voxels; returns ``None`` (or leaves voxels
    out of the ``done`` mask) where the per-voxel path must run instead.
    Linearity is checked on the run's own design before use.
    """
    order, V = phi.shape
    n, p_full = rm.X.shape
    # Columns that are identically zero in this run (other runs' intercepts
    # and drift) get zero coefficients and covariance, as the pseudo-inverse
    # of the per-voxel path gives them; the rest must be full rank.
    live = np.flatnonzero(np.any(rm.X != 0, axis=0))
    p = live.size
    if order == 0 or p == 0 or rm.rank != p:
        return None
    finite = np.all(np.isfinite(rm.Y), axis=0)
    if not np.any(finite):
        return None
    ph = phi[:, finite]
    # A handful of shared coefficients (noise pools) is cheaper per group.
    if np.unique(ph.T, axis=0).shape[0] <= _BATCH_MIN_GROUPS:
        return None
    cols_all = np.flatnonzero(finite)

    def whiten_with(
        coeffs: NDArray[np.float64], Z: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        plan = plan_from_phi(coeffs, exact_first=False)
        wr = whiten_apply(plan, rm.X, Z, runs=cast(Any, rm.segments))
        assert wr.Y is not None
        return wr.Y

    units = []
    for k in range(order):
        unit = np.zeros(order, dtype=np.float64)
        unit[k] = 1.0
        units.append(unit)
    X = rm.X[:, live]
    ops_X = [X] + [whiten_with(unit, X) - X for unit in units]
    # Self-check: the operator must be linear in phi on this run's layout.
    probe = np.linspace(0.3, -0.2, order)
    direct = whiten_with(probe, X)
    combined = X + sum(probe[k] * ops_X[k + 1] for k in range(order))
    if not np.allclose(direct, combined, rtol=1e-12, atol=1e-12):
        return None

    m = order + 1
    cross_X = [[ops_X[j].T @ ops_X[k] for k in range(m)] for j in range(m)]
    nv = cols_all.size
    betas_live = np.empty((p, nv), dtype=np.float64)
    cov_live = np.empty((nv, p, p), dtype=np.float64)
    gram_live = np.empty((nv, p, p), dtype=np.float64)
    rss = np.empty(nv, dtype=np.float64)
    resid = np.empty((n, nv), dtype=np.float64)
    # Voxel columns in chunks: the (n, V) whitened copies per AR lag would
    # otherwise dominate memory at full-brain size.
    for start in range(0, nv, _SOLVE_CHUNK):
        sl = slice(start, min(start + _SOLVE_CHUNK, nv))
        Yc = rm.Y[:, cols_all[sl]]
        ops_Y = [Yc] + [whiten_with(unit, Yc) - Yc for unit in units]
        c = np.vstack([np.ones((1, Yc.shape[1])), ph[:, sl]])
        G = np.zeros((Yc.shape[1], p, p), dtype=np.float64)
        rhs = np.zeros((Yc.shape[1], p), dtype=np.float64)
        for j in range(m):
            for k in range(m):
                w = c[j] * c[k]
                G += w[:, None, None] * cross_X[j][k][None]
                rhs += (w * (ops_X[j].T @ ops_Y[k])).T
        try:
            betas = np.linalg.solve(G, rhs[..., None])[..., 0].T  # (p, chunk)
            cov = np.linalg.inv(G)
        except np.linalg.LinAlgError:
            return None
        # Accumulate in place to keep temporaries to one chunk.
        r = ops_Y[0] - ops_X[0] @ betas
        for j in range(1, m):
            term = ops_X[j] @ betas
            np.subtract(ops_Y[j], term, out=term)
            term *= c[j]
            r += term
        betas_live[:, sl] = betas
        cov_live[sl] = cov
        gram_live[sl] = G
        rss[sl] = np.einsum("tv,tv->v", r, r)
        resid[:, sl] = r
    if p < p_full:
        full_betas = np.zeros((p_full, nv))
        full_betas[live] = betas_live
        full_cov = np.zeros((nv, p_full, p_full))
        full_cov[np.ix_(np.arange(nv), live, live)] = cov_live
        full_gram = np.zeros((nv, p_full, p_full))
        full_gram[np.ix_(np.arange(nv), live, live)] = gram_live
        return finite, (full_betas, rss, full_cov, full_gram, resid)
    return finite, (betas_live, rss, cov_live, gram_live, resid)


def _solve_voxelwise(
    runs: list[_RunMatrices],
    phis: list[NDArray[np.float64]],
    exact_first: bool,
    solver: Any,
) -> dict[str, Any]:
    """Per-voxel GLS: each voxel's design and data whitened with its phi.

    Voxels that share a phi (``noise_pools``) share one solve. Each voxel
    keeps its own ``(X'X)^-1``; the residual df is the run's ``n - rank``,
    since whitening preserves the rank (fmrireg ``runwise_lm_voxelwise``).
    """
    from fmrimod.glm.strategies import _pool_run_results

    run_results = []
    covariances = []
    grams = []
    residuals = []
    for rm, phi in zip(runs, phis, strict=True):
        n, p = rm.X.shape
        V = rm.Y.shape[1]
        batched = None if exact_first else _solve_voxelwise_batched(rm, phi)
        if batched is not None and batched[0].all():
            # Every voxel solved in the batch: use its arrays as they are
            # (at full-brain size a NaN-filled copy costs seconds and GBs).
            done = batched[0]
            betas, rss, cov, gram, resid = batched[1]
        else:
            betas = np.full((p, V), np.nan)
            rss = np.full(V, np.nan)
            cov = np.full((V, p, p), np.nan)
            gram = np.full((V, p, p), np.nan)
            resid = np.full((n, V), np.nan)
            if batched is not None:
                done, (b_betas, b_rss, b_cov, b_gram, b_resid) = batched
                betas[:, done] = b_betas
                rss[done] = b_rss
                cov[done] = b_cov
                gram[done] = b_gram
                resid[:, done] = b_resid
            else:
                done = np.zeros(V, dtype=bool)
        remaining = np.flatnonzero(~done)
        groups, inverse = np.unique(phi[:, remaining].T, axis=0, return_inverse=True)
        inverse = np.asarray(inverse).ravel()
        for g, phi_g in enumerate(groups):
            cols = remaining[inverse == g]
            Xw, Yw = _whiten(rm, phi_g, exact_first, cols)
            proj = fast_preproject(Xw, check_finite=False, method=solver)
            result = fast_lm_matrix(Xw, Yw, proj, return_fitted=True, check_finite=False)
            assert result.fitted is not None
            betas[:, cols] = result.betas
            rss[cols] = result.rss
            cov[cols] = proj.XtXinv
            gram[cols] = Xw.T @ Xw
            resid[:, cols] = Yw - result.fitted
        dfres = float(n - rm.rank)
        sigma2 = rss / dfres if dfres > 0 else np.full(V, np.nan)
        run_results.append(
            LmResult(betas=betas, rss=rss, sigma2=sigma2, dfres=dfres, rank=rm.rank)
        )
        covariances.append(cov)
        grams.append(gram)
        residuals.append(resid)

    # Rank diagnostics come from the unwhitened design (whitening keeps them).
    projections = [rm.proj for rm in runs]
    pooled = _pool_run_results(
        run_results, projections, covariances=covariances, grams=grams
    )
    return {
        "betas": pooled["betas"],
        "sigma": pooled["sigma"],
        "dfres": pooled["dfres"],
        "XtXinv": pooled["XtXinv"],
        "projections": projections,
        "run_results": run_results,
        "residuals": residuals,
        "run_X": None,
    }


def iterative_ar_gls(
    model: object,  # FmriModel
    config: FmriLmConfig,
    initial_fit: dict[str, Any],
) -> tuple[dict[str, Any], object]:
    """Run iterative GLS using plan-based whitening.

    Enhanced version of :func:`iterative_gls` that uses
    :func:`~fmrimod.ar.estimation.fit_noise` and
    :func:`~fmrimod.ar.whitening.whiten_apply` for full ARMA
    support, segment-aware whitening, and convergence checking.

    Parameters
    ----------
    model : FmriModel
        The fMRI model.
    config : FmriLmConfig
        Configuration with AR options.
    initial_fit : dict
        Initial OLS fit result from ``fit_runwise()``.

    Returns
    -------
    fit_result : dict
        Updated fit result with whitened estimates.
    plan : WhiteningPlan
        Final whitening plan.
    """
    from .estimation import fit_noise
    from .whitening import whiten_apply

    ar_opts = config.ar
    n_iter = ar_opts.iter_gls
    n_runs = model.n_runs  # type: ignore[attr-defined]

    # Map config to fit_noise kwargs
    method = getattr(ar_opts, "method", "ar")
    q = getattr(ar_opts, "q", 0)
    p_max_val = getattr(ar_opts, "p_max", 6)
    pooling = getattr(ar_opts, "pooling", "global")
    parcels = getattr(ar_opts, "parcels", None)
    convergence_tol = getattr(ar_opts, "convergence_tol", 5e-3)

    # Determine p argument
    ar_order = ar_opts.ar_order
    p_arg: object = ar_order if ar_order > 0 else "auto"

    # Build run labels
    run_labels = []
    offset = 0
    for r in range(n_runs):
        Y_r = get_run_data(model.dataset, r)  # type: ignore[attr-defined]
        n_r = Y_r.shape[0]
        run_labels.extend([r] * n_r)
        offset += n_r
    runs_arr = np.array(run_labels, dtype=np.intp)

    # Get censor info
    censor_list = []
    offset = 0
    for r in range(n_runs):
        Y_r = get_run_data(model.dataset, r)  # type: ignore[attr-defined]
        n_r = Y_r.shape[0]
        censor_r = None
        if hasattr(model, "dataset") and hasattr(model.dataset, "get_censor"):
            censor_r = model.dataset.get_censor(r)
        if censor_r is not None and np.any(censor_r):
            if censor_r.dtype == bool:
                indices = np.where(censor_r)[0] + offset
            else:
                indices = np.asarray(censor_r, dtype=np.intp) + offset
            censor_list.append(indices)
        offset += n_r
    censor_arr = np.concatenate(censor_list) if censor_list else None

    residuals_list = initial_fit["residuals"]
    current_fit = initial_fit
    prev_plan = None

    for iteration in range(n_iter):
        # Stack residuals across runs
        resid_full = np.vstack(residuals_list)

        # Fit noise model
        plan = fit_noise(
            resid=resid_full,
            runs=runs_arr,
            censor=censor_arr,
            method=method,
            p=p_arg,
            q=q,
            p_max=p_max_val,
            pooling=pooling,
            parcels=parcels,
            exact_first="ar1" if ar_opts.exact_first else "none",
        )

        # Convergence check
        if prev_plan is not None and convergence_tol > 0:
            converged = _check_convergence(prev_plan, plan, convergence_tol)
            if converged:
                break
        prev_plan = plan

        # Whiten and re-fit each run
        run_results = []
        run_projections = []
        new_residuals = []
        ar_residuals = []
        new_run_X = []

        for r in range(n_runs):
            Y_r = get_run_data(model.dataset, r)  # type: ignore[attr-defined]
            X_r = model.design_matrix_array(run=r)  # type: ignore[attr-defined]

            # Create a sub-plan for this run
            run_mask = runs_arr == r
            run_runs = np.ones(int(run_mask.sum()), dtype=np.intp)

            # Get run's phi/theta from the plan
            if plan.pooling == "parcel":
                # For parcel plans, use whiten_apply directly
                wr = whiten_apply(plan, X_r, Y_r, runs=cast(Any, run_runs), parcels=parcels)
                X_w = wr.X
                Y_w = wr.Y
                # For parcel plan, X_w is None; use first parcel's X
                if X_w is None and wr.X_by:
                    X_w = next(iter(wr.X_by.values()))
            else:
                if plan.phi is None:
                    phi_r = np.array([])
                    theta_r = np.array([])
                else:
                    phi_idx = 0 if plan.pooling == "global" else min(r, len(plan.phi) - 1)
                    phi_r = plan.phi[phi_idx] if plan.phi else np.array([])
                    theta_r = (plan.theta[phi_idx]
                               if plan.theta and phi_idx < len(plan.theta)
                               else np.array([]))

                from .whitening import arma_whiten_segments
                seg_starts = np.array([0], dtype=np.intp)
                X_w = arma_whiten_segments(X_r, phi_r, theta_r, cast(Any, seg_starts),
                                           exact_first_ar1=plan.exact_first)
                Y_w = arma_whiten_segments(Y_r, phi_r, theta_r, cast(Any, seg_starts),
                                           exact_first_ar1=plan.exact_first)

            # Re-fit
            assert X_w is not None and Y_w is not None
            proj = fast_preproject(X_w)
            result = fast_lm_matrix(X_w, Y_w, proj, return_fitted=True)

            run_results.append(result)
            run_projections.append(proj)
            new_run_X.append(X_w)

            # Residuals on whitened scale (kept in fit payload)
            if result.fitted is not None:
                new_residuals.append(Y_w - result.fitted)
            else:
                new_residuals.append(Y_w - X_w @ result.betas)
            # Residuals on original scale (used for next AR update)
            ar_residuals.append(Y_r - X_r @ result.betas)

        # Update for next iteration
        residuals_list = ar_residuals

        # Pool results
        from fmrimod.glm.strategies import _pool_run_results

        pooled = _pool_run_results(run_results, run_projections)

        current_fit = {
            "betas": pooled["betas"],
            "sigma": pooled["sigma"],
            "dfres": pooled["dfres"],
            "XtXinv": pooled["XtXinv"],
            "projections": run_projections,
            "run_results": run_results,
            "residuals": new_residuals,
            "run_X": new_run_X,
        }

    return current_fit, plan


def _check_convergence(prev_plan: Any, new_plan: Any, tol: float) -> bool:
    """Check if AR parameters have converged between iterations."""
    try:
        if prev_plan.phi is not None and new_plan.phi is not None:
            for phi_old, phi_new in zip(prev_plan.phi, new_plan.phi):
                if len(phi_old) == 0 and len(phi_new) == 0:
                    continue
                if len(phi_old) != len(phi_new):
                    return False
                if np.max(np.abs(phi_new - phi_old)) >= tol:
                    return False
            return True
        if prev_plan.phi_by_parcel and new_plan.phi_by_parcel:
            for key in prev_plan.phi_by_parcel:
                if key not in new_plan.phi_by_parcel:
                    return False
                old = prev_plan.phi_by_parcel[key]
                new = new_plan.phi_by_parcel[key]
                if len(old) != len(new):
                    return False
                if len(old) > 0 and np.max(np.abs(new - old)) >= tol:
                    return False
            return True
    except Exception:
        pass
    return False
