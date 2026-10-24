"""Prewhitening integration for single-trial estimation.

Bridges :class:`PrewhitenConfig` (mirror of R's ``fmrilss::prewhiten_options``)
to the canonical AR estimation and whitening pipeline in
:mod:`fmrimod.ar`. The same :func:`fmrimod.ar.estimation.fit_noise` +
:func:`fmrimod.ar.whitening.whiten_apply` path used by the GLM solver is
used here, so single-trial prewhitening picks up run-aware pooling,
parcel pooling, ARMA, and exact-first scaling without a parallel
estimator implementation.

Per R's ``fmrilss::.prewhiten_data``, the noise model is fitted to the
OLS residuals of ``cbind(Z, X_model, Nuisance)`` where ``Z`` is the
baseline / experimental design, ``Nuisance`` is the confound matrix and
``X_model`` depends on ``residual_model``: by default one summed trial
regressor (one sum per basis function for multi-basis designs), or the
full trial regressor matrix ``X`` for ``"full"`` / ``"corrected"``. All
four of ``Y``, ``X``, ``Z``, ``Nuisance`` are whitened with the same
plan so the downstream solve operates on a fully whitened system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..ar.plan import WhiteningPlan


PrewhitenMethod = Literal["ar", "arma", "none"]
PoolingMode = Literal["global", "voxel", "run", "parcel"]
ExactFirstMode = Literal["ar1", "none"]
ResidualModel = Literal["aggregate", "full", "corrected"]
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
    design : NDArray or None
        Design whose projection produced the residuals used to estimate
        the noise model. Opt-in residual-autocovariance bias correction
        (fmriAR 0.3.3 / fmrilss ``f851fb0``). Mutually exclusive with
        ``acvf_correction``.
    acvf_correction : NDArray or sequence of NDArray or None
        Cached bias-correction matrix, or list of matrices, from
        :func:`fmrimod.ar.acvf.acvf_bias_matrix`. Mutually exclusive
        with ``design``.
    correction_max_lag : int
        Lag budget used when ``design`` is supplied (default 25).
    residual_model : str or None
        Design whose residuals feed the noise model (fmrilss ``a9d816e``).
        ``"aggregate"`` uses the baseline and confound regressors plus one
        summed trial regressor (one sum per basis function for multi-basis
        designs), which avoids the downward AR bias that many trial columns
        cause in rapid designs. ``"full"`` uses the full trial-wise (LSA)
        design. ``"corrected"`` is ``"full"`` plus the residual-ACVF bias
        correction for that design, built automatically (``method="ar"``
        with global or run pooling only). ``None`` (default) resolves to
        ``"full"`` when ``design`` or ``acvf_correction`` is supplied and to
        ``"aggregate"`` otherwise; see :attr:`resolved_residual_model`.
    """

    method: PrewhitenMethod = "ar"
    p: ArOrder = 1
    q: int = 0
    p_max: int = 6
    pooling: PoolingMode = "global"
    runs: Optional[NDArray[Any]] = None
    parcels: Optional[NDArray[Any]] = None
    exact_first: ExactFirstMode = "ar1"
    design: Optional[NDArray[np.float64]] = None
    acvf_correction: Optional[
        Union[NDArray[np.float64], Sequence[NDArray[np.float64]]]
    ] = None
    correction_max_lag: int = 25
    residual_model: ResidualModel | None = None

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
        if (
            int(self.correction_max_lag) != self.correction_max_lag
            or self.correction_max_lag < 1
        ):
            raise ValueError("correction_max_lag must be a positive integer")
        if self.design is not None and self.acvf_correction is not None:
            raise ValueError("supply either design or acvf_correction, not both")
        if self.design is not None:
            design = np.asarray(self.design, dtype=np.float64)
            if design.ndim != 2:
                raise ValueError("design must be a 2-D matrix")
            object.__setattr__(self, "design", design)
        if self.acvf_correction is not None:
            object.__setattr__(
                self, "acvf_correction", _validate_acvf_correction(self.acvf_correction)
            )
        object.__setattr__(self, "q", int(self.q))
        object.__setattr__(self, "p_max", int(self.p_max))
        object.__setattr__(self, "correction_max_lag", int(self.correction_max_lag))
        # R ``.resolve_prewhiten_options`` (fmrilss ``R/prewhitening.R:155-183``).
        if self.residual_model not in {None, "aggregate", "full", "corrected"}:
            raise ValueError(
                "residual_model must be one of: aggregate, full, corrected"
            )
        explicit_correction = (
            self.design is not None or self.acvf_correction is not None
        )
        if explicit_correction and self.residual_model not in {None, "full"}:
            raise ValueError(
                "residual-bias correction via design/acvf_correction requires "
                "residual_model='full' (residual_model='corrected' builds the "
                "correction automatically; do not also supply design or "
                "acvf_correction)"
            )
        if explicit_correction or self.residual_model == "corrected":
            if self.method != "ar":
                raise ValueError("residual-bias correction requires method='ar'")
            if self.pooling not in {"global", "run"}:
                raise ValueError(
                    "residual-bias correction requires pooling='global' or 'run'"
                )

    @property
    def resolved_residual_model(self) -> ResidualModel:
        """``residual_model`` with the R default applied."""
        if self.residual_model is not None:
            return self.residual_model
        if self.design is not None or self.acvf_correction is not None:
            return "full"
        return "aggregate"


def _validate_acvf_correction(
    correction: Union[NDArray[np.float64], Sequence[NDArray[np.float64]]],
) -> Union[NDArray[np.float64], list[NDArray[np.float64]]]:
    """Mirror R's ``prewhiten$acvf_correction`` square-matrix contract."""

    if isinstance(correction, np.ndarray):
        mat = np.asarray(correction, dtype=np.float64)
        if (
            mat.ndim != 2
            or mat.shape[0] != mat.shape[1]
            or not np.all(np.isfinite(mat))
        ):
            raise ValueError(
                "acvf_correction must be a finite square numeric matrix "
                "or a non-empty list of such matrices"
            )
        return mat
    mats = [np.asarray(item, dtype=np.float64) for item in correction]
    if not mats:
        raise ValueError(
            "acvf_correction must be a finite square numeric matrix "
            "or a non-empty list of such matrices"
        )
    for mat in mats:
        if (
            mat.ndim != 2
            or mat.shape[0] != mat.shape[1]
            or not np.all(np.isfinite(mat))
        ):
            raise ValueError(
                "acvf_correction must contain finite square numeric matrices"
            )
    return mats


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


def aggregate_trials(
    X: NDArray[np.float64], n_basis: int = 1
) -> NDArray[np.float64]:
    """Low-dimensional trial summary for noise-model residuals.

    Mirrors R ``fmrilss:::.aggregate_trials``: one summed regressor for a
    single-basis design, or one sum per basis function for a trial-major
    ``(n, n_trials * n_basis)`` design (``R/oasis_backend.R:245-252``).
    """

    X = np.asarray(X, dtype=np.float64)
    return np.column_stack([X[:, k::n_basis].sum(axis=1) for k in range(n_basis)])


def _with_intercept(design: NDArray[np.float64]) -> NDArray[np.float64]:
    """Append an intercept unless the constant is already in ``design``'s span."""

    n = design.shape[0]
    ones = np.ones((n, 1), dtype=np.float64)
    Q, _ = np.linalg.qr(design, mode="reduced")
    resid_ones = ones - Q @ (Q.T @ ones)
    in_span = float(np.linalg.norm(resid_ones)) < 1e-8 * np.sqrt(n)
    return design if in_span else np.hstack([design, ones])


def _noise_basis(design: NDArray[np.float64]) -> NDArray[np.float64]:
    """Orthonormal basis of the noise-estimation design, intercept included.

    Mirrors R ``fmrilss:::.noise_basis``. Rank-revealing, so collinear
    columns (for example run dummies beside an intercept) are dropped.
    """

    aug = _with_intercept(design)
    U, s, _ = np.linalg.svd(aug, full_matrices=False)
    tol = max(aug.shape) * np.finfo(np.float64).eps * s[0]
    return U[:, s > tol]


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

    aug = _with_intercept(design_full)
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
    noise_trials: NDArray[np.float64] | None = None,
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
    noise_trials : (n, G) array or None
        Trial summary that replaces ``X`` in the noise-model residual
        design when the resolved ``residual_model`` is ``"aggregate"``,
        for example the per-basis sums of a multi-basis design (see
        :func:`aggregate_trials`). Defaults to the row sum of ``X``.

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
    # R does (``cbind(Z, X, Nuisance)``). This combined matrix is the
    # design we whiten, so the per-block slicing afterwards is exact. The
    # noise model sees the same blocks with X replaced by its aggregate
    # summary unless full trial-wise residuals were requested.
    def _compose(trials: NDArray[np.float64]) -> NDArray[np.float64]:
        blocks: list[NDArray[np.float64]] = []
        if baseline_regressors is not None:
            blocks.append(np.asarray(baseline_regressors, dtype=np.float64))
        blocks.append(trials)
        if confounds is not None:
            blocks.append(np.asarray(confounds, dtype=np.float64))
        return np.hstack(blocks)

    design_full = _compose(X)
    residual_model = config.resolved_residual_model
    if residual_model == "aggregate":
        summary = aggregate_trials(X) if noise_trials is None else noise_trials
        noise_design = _compose(np.asarray(summary, dtype=np.float64))
    else:
        noise_design = design_full
    correction_design = config.design
    if residual_model == "corrected":
        # fmriAR corrects the residual autocovariance for projection onto
        # this design; the orthonormal basis spans exactly the projected
        # space, intercept included (R ``.fit_noise_plan``).
        correction_design = _noise_basis(noise_design)

    pooling, parcels = _resolve_pooling_and_parcels(config, Y.shape[1])
    exact_first = config.exact_first

    # R's ``fmrilss::.prewhiten_data`` augments the residual-estimation
    # design with an intercept when the constant is not already in its
    # span. Without this, AR estimation picks up the data's mean as
    # low-frequency AR structure. The augmentation is for *residual
    # computation only* — the whitening operator is a function of phi,
    # not of the design that produced it, so the matrices we hand to
    # ``whiten_apply`` stay un-augmented.
    resid = _residuals_for_ar_estimation(Y, noise_design)

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
        design=correction_design,
        acvf_correction=config.acvf_correction,
        correction_max_lag=config.correction_max_lag,
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
