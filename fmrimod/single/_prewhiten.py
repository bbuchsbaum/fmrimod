"""Prewhitening integration for single-trial estimation.

Bridges :class:`PrewhitenConfig` (mirror of R's ``fmrilss::prewhiten_options``)
to the canonical AR estimation and whitening pipeline in
:mod:`fmrimod.ar`. The same :func:`fmrimod.ar.estimation.fit_noise` +
:func:`fmrimod.ar.whitening.whiten_apply` path used by the GLM solver is
used here, so single-trial prewhitening picks up run-aware pooling,
parcel pooling, ARMA, and exact-first scaling without a parallel
estimator implementation.

Per R's ``fmrilss::.prewhiten_data``, the combined design used to
compute AR residuals is ``cbind(Z, X, Nuisance)`` where ``Z`` is the
baseline / experimental design, ``X`` is the trial regressor matrix,
and ``Nuisance`` is the confound matrix. All four — ``Y``, ``X``,
``Z``, ``Nuisance`` — are whitened with the same plan so the
downstream LSS solve operates on a fully whitened system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..ar.plan import WhiteningPlan


PrewhitenMethod = Literal["ar", "arma", "none"]
PoolingMode = Literal["global", "voxel", "run", "parcel"]
ExactFirstMode = Literal["ar1", "none"]
ArOrder = Union[int, Literal["auto"]]


@dataclass(frozen=True)
class PrewhitenConfig:
    """Prewhitening configuration.

    Mirrors R's ``fmrilss::prewhiten_options()`` (see
    ``~/code/fmrilss/R/options.R``).

    Attributes
    ----------
    method : str
        ``"ar"`` (default), ``"arma"``, or ``"none"``.
    p : int or str
        AR order.  Integer or ``"auto"`` for BIC selection.
    q : int
        MA order (only for ``method="arma"``).
    p_max : int
        Maximum AR order when ``p="auto"``.
    pooling : str
        ``"global"`` (default), ``"voxel"``, ``"run"``, or ``"parcel"``.
    runs : NDArray or None
        Run labels, shape ``(n,)``.
    parcels : NDArray or None
        Parcel labels, shape ``(V,)``.
    exact_first : str
        ``"ar1"`` (default) or ``"none"``.
    """

    method: PrewhitenMethod = "ar"
    p: ArOrder = 1
    q: int = 0
    p_max: int = 6
    pooling: PoolingMode = "global"
    runs: Optional[NDArray[Any]] = None
    parcels: Optional[NDArray[Any]] = None
    exact_first: ExactFirstMode = "ar1"

    def __post_init__(self) -> None:
        if self.method not in {"ar", "arma", "none"}:
            raise ValueError("method must be one of: ar, arma, none")
        if self.pooling not in {"global", "voxel", "run", "parcel"}:
            raise ValueError("pooling must be one of: global, voxel, run, parcel")
        if self.exact_first not in {"ar1", "none"}:
            raise ValueError("exact_first must be one of: ar1, none")
        if self.p != "auto":
            if int(self.p) != self.p or int(self.p) < 0:
                raise ValueError("p must be a non-negative integer or 'auto'")
            object.__setattr__(self, "p", int(self.p))
        if int(self.q) != self.q or self.q < 0:
            raise ValueError("q must be a non-negative integer")
        if int(self.p_max) != self.p_max or self.p_max < 1:
            raise ValueError("p_max must be a positive integer")
        object.__setattr__(self, "q", int(self.q))
        object.__setattr__(self, "p_max", int(self.p_max))


@dataclass(frozen=True)
class PrewhitenResult:
    """Whitened matrices plus the plan that produced them."""

    Y: NDArray[np.float64]
    X: NDArray[np.float64]
    baseline_regressors: Optional[NDArray[np.float64]]
    confounds: Optional[NDArray[np.float64]]
    plan: "WhiteningPlan"


def _resolve_pooling_and_parcels(
    config: PrewhitenConfig,
    n_voxels: int,
) -> Tuple[str, Optional[NDArray[Any]]]:
    """Mirror R's ``pooling='voxel'`` → ``pooling='parcel', parcels=1..V``.

    ``"voxel"`` is the user-facing label; the plan engine speaks
    ``"parcel"`` with one parcel per voxel.
    """

    if config.pooling != "voxel":
        return config.pooling, config.parcels
    if config.parcels is not None:
        # R warns and overrides; mirror that with a typed error so the
        # caller fixes the call site rather than relying on a silent
        # override.
        raise ValueError(
            "PrewhitenConfig.pooling='voxel' is incompatible with explicit "
            "parcels; remove parcels or set pooling='parcel'."
        )
    return "parcel", np.arange(n_voxels, dtype=np.intp)


def _residuals_for_ar_estimation(
    Y: NDArray[np.float64],
    design_full: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute OLS residuals against ``design_full``, augmenting with an
    intercept when the constant column is not already in its span.

    Mirrors R's ``fmrilss::.prewhiten_data`` rank-safe path: if the
    combined design already encodes the mean (e.g. via run-wise dummies),
    no augmentation is needed; otherwise an intercept is added so the
    residuals are mean-zero and AR estimation isn't biased by Y's mean.
    """

    n = Y.shape[0]
    ones = np.ones((n, 1), dtype=np.float64)
    Q, _ = np.linalg.qr(design_full, mode="reduced")
    proj_ones = Q @ (Q.T @ ones)
    resid_ones = ones - proj_ones
    in_span = float(np.linalg.norm(resid_ones)) < 1e-8 * np.sqrt(n)
    aug = design_full if in_span else np.hstack([design_full, ones])
    coef, *_ = np.linalg.lstsq(aug, Y, rcond=None)
    return Y - aug @ coef


def _slice_whitened_design(
    X_full_w: NDArray[np.float64],
    *,
    n_baseline: int,
    n_trial: int,
    has_confounds: bool,
) -> Tuple[
    NDArray[np.float64],
    Optional[NDArray[np.float64]],
    Optional[NDArray[np.float64]],
]:
    """Split the whitened combined design back into (X, Z, Nuisance) blocks."""

    base_w = X_full_w[:, :n_baseline] if n_baseline else None
    trial_start = n_baseline
    trial_end = n_baseline + n_trial
    X_w = X_full_w[:, trial_start:trial_end]
    conf_w = X_full_w[:, trial_end:] if has_confounds else None
    return X_w, base_w, conf_w


def prewhiten_matrices(
    Y: NDArray[np.float64],
    X: NDArray[np.float64],
    confounds: Optional[NDArray[np.float64]],
    config: PrewhitenConfig,
    *,
    baseline_regressors: Optional[NDArray[np.float64]] = None,
) -> PrewhitenResult:
    """Estimate AR parameters and whiten all design / data matrices.

    Parameters
    ----------
    Y : (n, V) array
        Data matrix.
    X : (n, n_trials) array
        Trial regressor matrix.
    confounds : (n, n_nuisance) array or None
        Nuisance regressors (motion, drift, etc.).
    config : PrewhitenConfig
        AR specification mirroring R's ``prewhiten_options``.
    baseline_regressors : (n, n_baseline) array or None
        Baseline / experimental regressors (R's ``Z``). Whitened with the
        same plan as the rest of the design so the downstream LSS
        adjustment is consistent with the whitened Y / X / confounds.

    Returns
    -------
    PrewhitenResult
        Carries the whitened ``Y``, ``X``, ``baseline_regressors``,
        ``confounds``, and the :class:`fmrimod.ar.plan.WhiteningPlan`
        that produced them.
    """

    if config.method == "none":
        raise ValueError(
            "prewhiten_matrices should not be called with method='none'; "
            "the caller is expected to skip prewhitening entirely instead."
        )

    from ..ar.estimation import fit_noise
    from ..ar.whitening import whiten_apply

    Y = np.asarray(Y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    n_baseline = baseline_regressors.shape[1] if baseline_regressors is not None else 0
    n_trial = X.shape[1]
    has_confounds = confounds is not None

    # Compose Z (baseline), X (trial), Nuisance (confounds) the same way
    # R does (``design_full <- cbind(Z, X, Nuisance)``). The combined
    # matrix is used both for AR residual estimation and as the design
    # we whiten so the per-block slicing afterwards is exact.
    blocks: list[NDArray[np.float64]] = []
    if baseline_regressors is not None:
        blocks.append(np.asarray(baseline_regressors, dtype=np.float64))
    blocks.append(X)
    if confounds is not None:
        blocks.append(np.asarray(confounds, dtype=np.float64))
    design_full = np.hstack(blocks)

    pooling, parcels = _resolve_pooling_and_parcels(config, Y.shape[1])
    exact_first = config.exact_first

    # R's ``fmrilss::.prewhiten_data`` augments the residual-estimation
    # design with an intercept when the constant is not already in the
    # span of (baseline + X + confounds). Without this, AR estimation
    # picks up the data's mean as low-frequency AR structure. The
    # augmentation is for *residual computation only* — the whitening
    # operator is a function of phi, not of the design that produced it,
    # so the matrices we hand to ``whiten_apply`` stay un-augmented.
    resid = _residuals_for_ar_estimation(Y, design_full)

    plan = fit_noise(
        resid=resid,
        runs=config.runs,
        method=config.method,
        p=config.p,
        q=config.q,
        p_max=config.p_max,
        exact_first=exact_first,
        pooling=pooling,
        parcels=parcels,
    )

    if plan.pooling == "parcel":
        # Parcel plans whiten Y per-parcel and return a dict-of-X.
        # LSS consumes a single combined design, so demand a non-parcel
        # plan until per-parcel LSS lands (tracked under the same bead).
        raise NotImplementedError(
            "Parcel-pooled prewhitening for LSS is not yet wired; "
            "track follow-up under bd-01KRK97N4F0EZWSHTFG3GMCEJB."
        )

    whitened = whiten_apply(plan, design_full, Y, runs=config.runs)
    if whitened.X is None or whitened.Y is None:  # pragma: no cover - defensive
        raise RuntimeError(
            "whiten_apply returned an incomplete result for a non-parcel plan."
        )

    Y_w = np.asarray(whitened.Y, dtype=np.float64)
    X_w, base_w, conf_w = _slice_whitened_design(
        np.asarray(whitened.X, dtype=np.float64),
        n_baseline=n_baseline,
        n_trial=n_trial,
        has_confounds=has_confounds,
    )

    return PrewhitenResult(
        Y=Y_w,
        X=X_w,
        baseline_regressors=base_w,
        confounds=conf_w,
        plan=plan,
    )
