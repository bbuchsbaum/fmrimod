"""ITEM helper layer for covariance-aware trial decoding.

This module ports the core ITEM preparation and CV helpers from fmrilss
into the Python single-trial stack with typed dataclass contracts.
"""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Optional, Sequence, Union, cast

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, sparse

from .lsa import lsa_single_trial

SolverMethod = Literal["chol", "svd", "pinv"]
VType = Literal["cov", "precision"]
CvMode = Literal["classification", "regression"]
ClassificationMetric = Literal["accuracy", "balanced_accuracy"]
RegressionMetric = Literal["correlation", "rmse"]
MetricName = Union[ClassificationMetric, RegressionMetric]
UOutputMode = Literal["matrix", "by_run"]
TargetLike = Union[NDArray[np.float64], Sequence[str], sparse.spmatrix]
CovarianceLike = Union[
    NDArray[np.float64],
    Mapping[str, NDArray[np.float64]],
    Sequence[NDArray[np.float64]],
    sparse.spmatrix,
]


@dataclass(frozen=True)
class ItemCovarianceBaseResult:
    """Base diagnostics for trial covariance output from :func:`item_compute_u`."""

    diagnostics: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ItemCovarianceMatrixResult(ItemCovarianceBaseResult):
    """Full-matrix ITEM covariance output."""

    U: NDArray[np.float64] = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float64)
    )


@dataclass(frozen=True)
class ItemCovarianceBlockResult(ItemCovarianceBaseResult):
    """Run-block ITEM covariance output."""

    U_by_run: dict[str, NDArray[np.float64]] = field(default_factory=dict)


ItemCovarianceResult = Union[ItemCovarianceMatrixResult, ItemCovarianceBlockResult]


@dataclass
class ItemWeightsResult:
    """Fitted ITEM decoder weights."""

    W_hat: NDArray[np.float64]
    diagnostics: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ItemBundle:
    """Container for aligned ITEM inputs and metadata."""

    Gamma: NDArray[np.float64] | None
    X_t: NDArray[np.float64] | None
    C_transform: NDArray[np.float64] | None
    T_target: NDArray[np.float64] | None
    U: NDArray[np.float64] | None
    U_by_run: dict[str, NDArray[np.float64]] | None
    run_id: NDArray[Any]
    trial_id: NDArray[np.str_]
    trial_hash: str | None
    trial_info: dict[str, NDArray[Any]]
    meta: dict[str, object] = field(default_factory=dict)
    diagnostics: dict[str, object] = field(default_factory=dict)


@dataclass
class ItemFoldSplit:
    """Train/test fold slices for a single held-out run."""

    Gamma_train: NDArray[np.float64]
    Gamma_test: NDArray[np.float64]
    T_train: NDArray[np.float64]
    T_test: NDArray[np.float64]
    U_train: NDArray[np.float64]
    U_test: NDArray[np.float64]
    train_idx: NDArray[np.int_]
    test_idx: NDArray[np.int_]
    train_runs: NDArray[Any]
    test_run: object
    trial_id_train: NDArray[np.str_]
    trial_id_test: NDArray[np.str_]


@dataclass
class ItemFoldMetrics:
    """Per-fold metric summary."""

    fold: int
    test_run: str
    n_train: int
    n_test: int
    metric: float


@dataclass
class ItemPredictions:
    """Trial-level predictions from ITEM CV."""

    T_hat: NDArray[np.float64]
    T_true: NDArray[np.float64]
    predicted_class: NDArray[np.str_] | None = None
    true_class: NDArray[np.str_] | None = None
    class_levels: list[str] | None = None


@dataclass
class ItemCvAggregate:
    """Aggregate CV metric summary."""

    metric: str
    mean: float
    sd: float
    n_folds: int


@dataclass
class ItemCvResult:
    """Cross-validated ITEM decoding result."""

    mode: CvMode
    metric: MetricName
    folds: list[ItemFoldMetrics]
    aggregate: ItemCvAggregate
    predictions: ItemPredictions
    diagnostics: dict[str, object] = field(default_factory=dict)


@dataclass
class _SolveResult:
    value: NDArray[np.float64]
    method: SolverMethod
    warnings: list[str]


@dataclass
class _FoldScore:
    metric: float
    pred_labels: NDArray[np.str_] | None


def item_build_design(
    X_t: NDArray[np.float64],
    T_target: TargetLike | None = None,
    run_id: Sequence[object] | None = None,
    C_transform: NDArray[np.float64] | None = None,
    trial_id: Sequence[object] | None = None,
    trial_hash: str | None = None,
    meta: dict[str, object] | None = None,
    diagnostics: dict[str, object] | None = None,
    validate: bool = True,
) -> ItemBundle:
    """Build and validate trial-wise ITEM metadata."""
    X_t_m = _as_numeric_matrix(X_t, "X_t")
    n_trials = int(X_t_m.shape[1])

    run_id_arr = _as_run_id(run_id, n_trials)
    trial_id_arr = _as_trial_id(trial_id, n_trials)
    target_info = _prepare_targets(T_target, n_trials)
    C_transform_m = _as_matrix_optional(C_transform, "C_transform")

    if C_transform_m is not None and C_transform_m.shape[0] != n_trials:
        raise ValueError(
            f"C_transform must have {n_trials} rows (got {C_transform_m.shape[0]})."
        )

    trial_info: dict[str, NDArray[Any]] = {
        "trial_index": np.arange(1, n_trials + 1, dtype=np.int64),
        "trial_id": trial_id_arr.astype(str),
        "run_id": run_id_arr.copy(),
    }

    base_meta = {
        "target_type": target_info["target_type"],
        "class_levels": target_info["class_levels"],
    }
    if meta is not None:
        base_meta.update(meta)

    bundle = ItemBundle(
        Gamma=None,
        X_t=X_t_m,
        C_transform=C_transform_m,
        T_target=target_info["T_target"],
        U=None,
        U_by_run=None,
        run_id=run_id_arr,
        trial_id=trial_id_arr,
        trial_hash=trial_hash,
        trial_info=trial_info,
        meta=base_meta,
        diagnostics={} if diagnostics is None else dict(diagnostics),
    )

    if validate:
        _validate_bundle(bundle, require_gamma=False, require_u=False, check_hash=False)
    return bundle


def item_compute_u(
    X_t: NDArray[np.float64],
    V: CovarianceLike | None = None,
    v_type: VType = "cov",
    ridge: float = 0.0,
    method: SolverMethod = "chol",
    run_id: Sequence[object] | None = None,
    output: UOutputMode = "matrix",
    tol: float = np.sqrt(np.finfo(np.float64).eps),
) -> ItemCovarianceResult:
    """Compute ITEM trial covariance ``U = (X' V^{-1} X + ridge I)^{-1}``."""
    if v_type not in {"cov", "precision"}:
        raise ValueError("v_type must be 'cov' or 'precision'.")
    if output not in {"matrix", "by_run"}:
        raise ValueError("output must be 'matrix' or 'by_run'.")
    if ridge < 0 or not np.isfinite(ridge):
        raise ValueError("ridge must be a finite non-negative number.")

    X_t_m = _as_numeric_matrix(X_t, "X_t")
    x_vinv = _apply_vinv(V=V, X=X_t_m, v_type=v_type, method=method, tol=tol)

    xt_vinv_x = X_t_m.T @ x_vinv
    if ridge > 0:
        xt_vinv_x = xt_vinv_x + ridge * np.eye(xt_vinv_x.shape[0], dtype=np.float64)

    inv_fit = _safe_solve(
        A=xt_vinv_x,
        B=None,
        method=method,
        tol=tol,
        context="item_compute_u",
    )
    U = 0.5 * (inv_fit.value + inv_fit.value.T)

    diagnostics: dict[str, object] = {
        "rank": int(np.linalg.matrix_rank(xt_vinv_x, tol=tol)),
        "condition_number": _condition_number(xt_vinv_x),
        "solver_path": inv_fit.method,
        "ridge": float(ridge),
        "warnings": list(inv_fit.warnings),
        "v_type": v_type,
        "n_time": int(X_t_m.shape[0]),
        "n_trials": int(X_t_m.shape[1]),
    }

    fallback_warning = _solver_warning(
        context="item_compute_u",
        requested=method,
        used=inv_fit.method,
        warnings_seen=inv_fit.warnings,
    )
    if fallback_warning is not None:
        warnings.warn(fallback_warning, stacklevel=2)

    if output == "by_run":
        run_id_arr = _as_run_id(run_id, X_t_m.shape[1])
        blocks = _split_u_by_run(U, run_id_arr)
        return ItemCovarianceBlockResult(U_by_run=blocks, diagnostics=diagnostics)
    return ItemCovarianceMatrixResult(U=U, diagnostics=diagnostics)


def item_fit(
    Gamma_train: NDArray[np.float64],
    T_train: NDArray[np.float64],
    U_train: CovarianceLike | ItemCovarianceResult,
    ridge: float = 0.0,
    method: SolverMethod = "chol",
    tol: float = np.sqrt(np.finfo(np.float64).eps),
) -> ItemWeightsResult:
    """Fit ITEM decoder weights with GLS."""
    if ridge < 0 or not np.isfinite(ridge):
        raise ValueError("ridge must be a finite non-negative number.")

    Gamma_m = _as_numeric_matrix(Gamma_train, "Gamma_train")
    T_train_m = _as_numeric_matrix(T_train, "T_train")
    U_train_m = _coerce_u_matrix(U_train)

    n_train = int(Gamma_m.shape[0])
    if T_train_m.shape[0] != n_train:
        raise ValueError(f"T_train must have {n_train} rows to match Gamma_train.")
    if U_train_m.shape != (n_train, n_train):
        raise ValueError(
            f"U_train must be {n_train} x {n_train} (got {U_train_m.shape[0]} x {U_train_m.shape[1]})."
        )

    solve_u_gamma = _safe_solve(
        A=U_train_m,
        B=Gamma_m,
        method=method,
        tol=tol,
        context="item_fit(U^{-1}Gamma)",
    )
    solve_u_t = _safe_solve(
        A=U_train_m,
        B=T_train_m,
        method=method,
        tol=tol,
        context="item_fit(U^{-1}T)",
    )

    lhs = Gamma_m.T @ solve_u_gamma.value
    if ridge > 0:
        lhs = lhs + ridge * np.eye(lhs.shape[0], dtype=np.float64)
    rhs = Gamma_m.T @ solve_u_t.value

    fit = _safe_solve(
        A=lhs,
        B=rhs,
        method=method,
        tol=tol,
        context="item_fit(W solve)",
    )

    diagnostics: dict[str, object] = {
        "rank": int(np.linalg.matrix_rank(lhs, tol=tol)),
        "condition_number": _condition_number(lhs),
        "solver_path": fit.method,
        "ridge": float(ridge),
        "warnings": list(
            dict.fromkeys(solve_u_gamma.warnings + solve_u_t.warnings + fit.warnings)
        ),
    }

    warn_msgs = [
        _solver_warning(
            "item_fit(U^{-1}Gamma)", method, solve_u_gamma.method, solve_u_gamma.warnings
        ),
        _solver_warning(
            "item_fit(U^{-1}T)", method, solve_u_t.method, solve_u_t.warnings
        ),
        _solver_warning("item_fit(W solve)", method, fit.method, fit.warnings),
    ]
    warn_msgs_filtered: list[str] = [msg for msg in warn_msgs if msg is not None]
    if warn_msgs_filtered:
        warnings.warn(" ".join(dict.fromkeys(warn_msgs_filtered)), stacklevel=2)

    return ItemWeightsResult(W_hat=fit.value, diagnostics=diagnostics)


def item_predict(
    Gamma_test: NDArray[np.float64],
    W_hat: NDArray[np.float64] | ItemWeightsResult,
) -> NDArray[np.float64]:
    """Predict targets from ITEM weights."""
    Gamma_test_m = _as_numeric_matrix(Gamma_test, "Gamma_test")
    W_hat_m = (
        W_hat.W_hat if isinstance(W_hat, ItemWeightsResult) else _as_numeric_matrix(W_hat, "W_hat")
    )

    if Gamma_test_m.shape[1] != W_hat_m.shape[0]:
        raise ValueError(
            "ncol(Gamma_test) must equal nrow(W_hat); "
            f"got {Gamma_test_m.shape[1]} and {W_hat_m.shape[0]}."
        )
    return Gamma_test_m @ W_hat_m


def item_slice_fold(
    bundle: ItemBundle,
    test_run: object,
    check_hash: bool = False,
) -> ItemFoldSplit:
    """Slice an ITEM bundle into deterministic LOSO train/test components."""
    bundle = _validate_bundle(bundle, require_gamma=True, require_u=True, check_hash=check_hash)

    run_values = _sorted_unique_runs(bundle.run_id)
    if not np.any(bundle.run_id == test_run):
        raise ValueError(f"test_run {test_run!r} not found in run_id.")

    test_idx = np.where(bundle.run_id == test_run)[0]
    train_idx = np.where(bundle.run_id != test_run)[0]

    if test_idx.size == 0 or train_idx.size == 0:
        raise ValueError("Fold split must contain both train and test trials.")

    assert bundle.Gamma is not None
    assert bundle.T_target is not None

    Gamma_train = bundle.Gamma[train_idx, :]
    Gamma_test = bundle.Gamma[test_idx, :]
    T_train = bundle.T_target[train_idx, :]
    T_test = bundle.T_target[test_idx, :]

    if bundle.U is not None:
        U_train = bundle.U[np.ix_(train_idx, train_idx)]
        U_test = bundle.U[np.ix_(test_idx, test_idx)]
    else:
        assert bundle.U_by_run is not None
        U_by_run = _coerce_u_by_run(bundle.U_by_run, bundle.run_id)
        train_runs = run_values[run_values != test_run]
        train_blocks = [U_by_run[str(run)] for run in train_runs]
        U_train = _block_diag(train_blocks)
        U_test = U_by_run[str(test_run)]

    return ItemFoldSplit(
        Gamma_train=Gamma_train,
        Gamma_test=Gamma_test,
        T_train=T_train,
        T_test=T_test,
        U_train=U_train,
        U_test=U_test,
        train_idx=train_idx.astype(np.int64),
        test_idx=test_idx.astype(np.int64),
        train_runs=run_values[run_values != test_run],
        test_run=test_run,
        trial_id_train=bundle.trial_id[train_idx],
        trial_id_test=bundle.trial_id[test_idx],
    )


def item_cv(
    Gamma: ItemBundle | NDArray[np.float64],
    T_target: TargetLike | None = None,
    U: CovarianceLike | ItemCovarianceResult | None = None,
    run_id: Sequence[object] | None = None,
    mode: CvMode = "classification",
    metric: MetricName | None = None,
    ridge: float = 0.0,
    method: SolverMethod = "chol",
    class_levels: Sequence[str] | None = None,
    trial_id: Sequence[object] | None = None,
    trial_hash: str | None = None,
    check_hash: bool = False,
) -> ItemCvResult:
    """Deterministic leave-one-run-out ITEM cross-validation."""
    if mode not in {"classification", "regression"}:
        raise ValueError("mode must be 'classification' or 'regression'.")

    bundle = _prepare_cv_bundle(
        Gamma=Gamma,
        T_target=T_target,
        U=U,
        run_id=run_id,
        mode=mode,
        class_levels=None if class_levels is None else list(class_levels),
        trial_id=trial_id,
        trial_hash=trial_hash,
    )

    bundle = _validate_bundle(bundle, require_gamma=True, require_u=True, check_hash=check_hash)
    folds = _sorted_unique_runs(bundle.run_id)
    if folds.size < 2:
        raise ValueError("item_cv requires at least 2 unique runs for LOSO CV.")

    assert bundle.Gamma is not None
    assert bundle.T_target is not None

    class_levels_list = (
        _class_levels(
            bundle.T_target,
            cast("Optional[Sequence[str]]", bundle.meta.get("class_levels")),
        )
        if mode == "classification"
        else None
    )
    metric_name = _metric_name(mode=mode, metric=metric)

    n_trials = bundle.Gamma.shape[0]
    T_hat_all = np.full(bundle.T_target.shape, np.nan, dtype=np.float64)
    pred_label_all: NDArray[np.str_] | None
    if mode == "classification":
        pred_label_all = np.full(n_trials, "", dtype="<U64")
    else:
        pred_label_all = None

    fold_rows: list[ItemFoldMetrics] = []

    for i, test_run in enumerate(folds, start=1):
        fold = item_slice_fold(bundle, test_run=test_run, check_hash=False)
        W_fit = item_fit(
            Gamma_train=fold.Gamma_train,
            T_train=fold.T_train,
            U_train=fold.U_train,
            ridge=ridge,
            method=method,
        )
        T_hat = item_predict(fold.Gamma_test, W_fit)
        T_hat_all[fold.test_idx, :] = T_hat

        scored = _score_fold(
            T_true=fold.T_test,
            T_hat=T_hat,
            mode=mode,
            metric=metric_name,
            class_levels=class_levels_list,
        )
        if pred_label_all is not None and scored.pred_labels is not None:
            pred_label_all[fold.test_idx] = scored.pred_labels

        fold_rows.append(
            ItemFoldMetrics(
                fold=i,
                test_run=str(test_run),
                n_train=int(fold.Gamma_train.shape[0]),
                n_test=int(fold.Gamma_test.shape[0]),
                metric=float(scored.metric),
            )
        )

    fold_metrics = np.asarray([row.metric for row in fold_rows], dtype=np.float64)
    aggregate = ItemCvAggregate(
        metric=metric_name,
        mean=float(np.nanmean(fold_metrics)),
        sd=float(np.nanstd(fold_metrics, ddof=1)) if fold_metrics.size > 1 else 0.0,
        n_folds=int(fold_metrics.size),
    )

    predictions = ItemPredictions(T_hat=T_hat_all, T_true=bundle.T_target)
    if pred_label_all is not None and class_levels_list is not None:
        predictions.predicted_class = pred_label_all
        predictions.true_class = _true_class(bundle.T_target, class_levels_list)
        predictions.class_levels = class_levels_list

    diagnostics: dict[str, object] = {
        "fold_order": [str(x) for x in folds.tolist()],
        "solver": method,
        "ridge": float(ridge),
    }
    return ItemCvResult(
        mode=mode,
        metric=metric_name,
        folds=fold_rows,
        aggregate=aggregate,
        predictions=predictions,
        diagnostics=diagnostics,
    )


def item_from_lsa(
    Y: NDArray[np.float64],
    X_t: NDArray[np.float64],
    T_target: TargetLike,
    run_id: Sequence[object],
    confounds: NDArray[np.float64] | None = None,
    *,
    nuisance: NDArray[np.float64] | None = None,
    V: CovarianceLike | None = None,
    v_type: VType = "cov",
    ridge: float = 0.0,
    solver: SolverMethod = "chol",
    u_output: UOutputMode = "matrix",
    C_transform: NDArray[np.float64] | None = None,
    trial_id: Sequence[object] | None = None,
    trial_hash: str | None = None,
    meta: dict[str, object] | None = None,
    validate: bool = True,
) -> ItemBundle:
    """Build an ITEM bundle from LS-A estimates."""
    if nuisance is not None and confounds is not None:
        warnings.warn(
            "Both confounds and nuisance were provided; using confounds and ignoring nuisance.",
            stacklevel=2,
        )
    effective_confounds = confounds if confounds is not None else nuisance

    design_bundle = item_build_design(
        X_t=X_t,
        T_target=T_target,
        run_id=run_id,
        C_transform=C_transform,
        trial_id=trial_id,
        trial_hash=trial_hash,
        meta={} if meta is None else dict(meta),
        diagnostics={},
        validate=True,
    )
    assert design_bundle.X_t is not None
    lsa_result = lsa_single_trial(
        Y=np.asarray(Y, dtype=np.float64),
        X=design_bundle.X_t,
        confounds=None if effective_confounds is None else np.asarray(effective_confounds, dtype=np.float64),
        return_se=False,
    )
    Gamma = np.asarray(lsa_result.betas, dtype=np.float64)
    if Gamma.shape[0] != design_bundle.run_id.shape[0]:
        raise ValueError(
            "Alignment mismatch after lsa_single_trial(): "
            f"Gamma has {Gamma.shape[0]} rows but run_id has {design_bundle.run_id.shape[0]}."
        )

    u_res = item_compute_u(
        X_t=design_bundle.X_t,
        V=V,
        v_type=v_type,
        ridge=ridge,
        method=solver,
        run_id=design_bundle.run_id,
        output=u_output,
    )

    meta_out = {
        **design_bundle.meta,
        "lsa_method": "python",
        "u_solver": solver,
        "v_type": v_type,
    }
    diagnostics_out = {
        **design_bundle.diagnostics,
        "lsa": {"method": "python"},
        "u": dict(u_res.diagnostics),
    }
    design_bundle = replace(
        design_bundle,
        Gamma=Gamma,
        U=u_res.U if isinstance(u_res, ItemCovarianceMatrixResult) else None,
        U_by_run=u_res.U_by_run if isinstance(u_res, ItemCovarianceBlockResult) else None,
        meta=meta_out,
        diagnostics=diagnostics_out,
    )

    if validate:
        design_bundle = _validate_bundle(design_bundle, require_gamma=True, require_u=True, check_hash=False)
    return design_bundle


def _prepare_cv_bundle(
    Gamma: ItemBundle | NDArray[np.float64],
    T_target: TargetLike | None,
    U: CovarianceLike | ItemCovarianceResult | None,
    run_id: Sequence[object] | None,
    mode: CvMode,
    class_levels: list[str] | None,
    trial_id: Sequence[object] | None,
    trial_hash: str | None,
) -> ItemBundle:
    if isinstance(Gamma, ItemBundle):
        if T_target is not None or U is not None or run_id is not None:
            warnings.warn(
                "item_cv: Gamma is an ItemBundle; T_target/U/run_id arguments are ignored.",
                stacklevel=2,
            )
        bundle = Gamma
        if mode == "classification" and bundle.T_target is not None:
            lvl = _class_levels(
                bundle.T_target,
                class_levels or cast(
                    "Optional[Sequence[str]]", bundle.meta.get("class_levels")
                ),
            )
            bundle = replace(bundle, meta={**bundle.meta, "class_levels": lvl})
        return bundle

    Gamma_m = _as_numeric_matrix(Gamma, "Gamma")
    n_trials = int(Gamma_m.shape[0])
    if T_target is None:
        raise ValueError("T_target is required when Gamma is not an ItemBundle.")

    target_info = _prepare_targets(T_target, n_trials)
    tmat = target_info["T_target"]
    target_type: str
    class_meta: list[str] | None

    tmat_arr = cast(NDArray[Any], tmat)
    if mode == "classification":
        if tmat_arr.shape[1] < 2:
            raise ValueError("Classification mode requires at least 2 target columns/classes.")
        lvl = _class_levels(
            tmat_arr,
            class_levels or cast(
                "Optional[Sequence[str]]", target_info["class_levels"]
            ),
        )
        target_type = "classification"
        class_meta = lvl
    else:
        if tmat_arr.shape[1] < 1:
            raise ValueError("Regression mode requires at least one target column.")
        target_type = "regression"
        class_meta = None

    run_id_arr = _as_run_id(run_id, n_trials)
    trial_id_arr = _as_trial_id(trial_id, n_trials)

    if U is None:
        raise ValueError("U is required when Gamma is not an ItemBundle.")

    U_matrix: NDArray[np.float64] | None
    U_by_run: dict[str, NDArray[np.float64]] | None
    if isinstance(U, ItemCovarianceMatrixResult):
        U_matrix = U.U
        U_by_run = None
    elif isinstance(U, ItemCovarianceBlockResult):
        U_matrix = None
        U_by_run = U.U_by_run
    elif _is_u_blocks(U):
        U_matrix = None
        U_by_run = _coerce_u_by_run(U, run_id_arr)
    else:
        U_matrix = _as_numeric_matrix(U, "U")
        U_by_run = None

    trial_info: dict[str, NDArray[Any]] = {
        "trial_index": np.arange(1, n_trials + 1, dtype=np.int64),
        "trial_id": trial_id_arr.astype(str),
        "run_id": run_id_arr.copy(),
    }

    return ItemBundle(
        Gamma=Gamma_m,
        X_t=None,
        C_transform=None,
        T_target=tmat,
        U=U_matrix,
        U_by_run=U_by_run,
        run_id=run_id_arr,
        trial_id=trial_id_arr,
        trial_hash=trial_hash,
        trial_info=trial_info,
        meta={"target_type": target_type, "class_levels": class_meta},
        diagnostics={},
    )


def _validate_bundle(
    bundle: ItemBundle,
    require_gamma: bool = True,
    require_u: bool = True,
    check_hash: bool = False,
) -> ItemBundle:
    run_id = bundle.run_id
    if run_id is None:
        raise ValueError("bundle.run_id is required.")
    n_trials = int(len(run_id))
    if n_trials < 2:
        raise ValueError("bundle must contain at least 2 trials.")

    if require_gamma and bundle.Gamma is None:
        raise ValueError("bundle.Gamma is required for this operation.")
    if bundle.Gamma is not None:
        if bundle.Gamma.ndim != 2:
            raise ValueError("bundle.Gamma must be a matrix.")
        if bundle.Gamma.shape[0] != n_trials:
            raise ValueError(
                f"bundle.Gamma has {bundle.Gamma.shape[0]} rows but run_id has length {n_trials}."
            )

    if bundle.T_target is not None:
        if bundle.T_target.ndim != 2:
            raise ValueError("bundle.T_target must be a matrix.")
        if bundle.T_target.shape[0] != n_trials:
            raise ValueError(
                f"bundle.T_target has {bundle.T_target.shape[0]} rows but run_id has length {n_trials}."
            )

    if bundle.U is not None and bundle.U.shape != (n_trials, n_trials):
        raise ValueError(
            f"bundle.U must be {n_trials} x {n_trials} but is {bundle.U.shape[0]} x {bundle.U.shape[1]}."
        )

    if bundle.U_by_run is not None:
        bundle = replace(bundle, U_by_run=_coerce_u_by_run(bundle.U_by_run, run_id))

    if require_u and bundle.U is None and bundle.U_by_run is None:
        raise ValueError("bundle must include either U (full matrix) or U_by_run (run-block map).")

    if bundle.trial_id.shape[0] != n_trials:
        raise ValueError("bundle.trial_id must have length equal to number of trials.")

    if check_hash and bundle.trial_hash is not None:
        _check_trial_hash(bundle.trial_id, bundle.trial_hash)
    return bundle


def _prepare_targets(T_target: TargetLike | None, n_trials: int) -> dict[str, object]:
    if T_target is None:
        return {
            "T_target": np.empty((n_trials, 0), dtype=np.float64),
            "target_type": "unspecified",
            "class_levels": None,
        }

    if sparse.issparse(T_target):
        T_target = cast(Any, T_target).toarray()

    arr = np.asarray(T_target)
    if arr.ndim == 2:
        if arr.shape[0] != n_trials:
            raise ValueError(f"T_target must have {n_trials} rows (got {arr.shape[0]}).")
        if not np.issubdtype(arr.dtype, np.number):
            raise ValueError(
                "T_target matrix must be numeric. For class labels, pass a 1-D label vector."
            )
        return {
            "T_target": np.asarray(arr, dtype=np.float64),
            "target_type": "matrix",
            "class_levels": None,
        }

    if arr.ndim != 1:
        raise ValueError("Unsupported T_target type.")
    if arr.shape[0] != n_trials:
        raise ValueError(f"T_target vector must have length {n_trials}.")

    if np.issubdtype(arr.dtype, np.number):
        return {
            "T_target": np.asarray(arr, dtype=np.float64).reshape(-1, 1),
            "target_type": "regression",
            "class_levels": None,
        }

    labels = arr.astype(str)
    if _contains_missing(arr):
        raise ValueError("T_target labels must not contain missing values.")
    class_levels = sorted(set(labels.tolist()))
    if len(class_levels) < 2:
        raise ValueError("Classification targets must include at least 2 classes.")

    index = {cls: i for i, cls in enumerate(class_levels)}
    out = np.zeros((n_trials, len(class_levels)), dtype=np.float64)
    for i, lab in enumerate(labels.tolist()):
        out[i, index[lab]] = 1.0

    return {
        "T_target": out,
        "target_type": "classification",
        "class_levels": class_levels,
    }


def _as_numeric_matrix(x: object, name: str) -> NDArray[np.float64]:
    if sparse.issparse(x):
        x = cast(Any, x).toarray()
    arr = np.asarray(x)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D numeric matrix.")
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"{name} must be numeric.")
    return np.asarray(arr, dtype=np.float64)


def _as_matrix_optional(x: object | None, name: str) -> NDArray[np.float64] | None:
    if x is None:
        return None
    return _as_numeric_matrix(x, name)


def _as_run_id(run_id: Sequence[object] | None, n_trials: int) -> NDArray[Any]:
    if run_id is None:
        return np.ones(n_trials, dtype=np.int64)
    arr = np.asarray(run_id).reshape(-1)
    if arr.shape[0] != n_trials:
        raise ValueError(f"run_id must have length {n_trials}.")
    if _contains_missing(arr):
        raise ValueError("run_id must not contain missing values.")
    return arr


def _as_trial_id(trial_id: Sequence[object] | None, n_trials: int) -> NDArray[np.str_]:
    if trial_id is None:
        out = [f"Trial_{i + 1}" for i in range(n_trials)]
        return np.asarray(out, dtype=str)
    arr = np.asarray(trial_id).reshape(-1)
    if arr.shape[0] != n_trials:
        raise ValueError(f"trial_id must have length {n_trials}.")
    if _contains_missing(arr):
        raise ValueError("trial_id must not contain missing values.")
    return arr.astype(str)


def _check_trial_hash(trial_id: NDArray[np.str_], trial_hash: str) -> None:
    expected = str(trial_hash)
    actual = _item_simple_hash(trial_id)
    if expected != actual:
        raise ValueError(
            f"Trial hash mismatch: expected '{expected}' but computed '{actual}'."
        )


def _item_simple_hash(x: Sequence[object]) -> str:
    payload = "\x1f".join(np.asarray(x, dtype=str).tolist())
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _coerce_u_by_run(
    U_by_run: Mapping[str, NDArray[np.float64]] | Sequence[NDArray[np.float64]],
    run_id: NDArray[Any],
) -> dict[str, NDArray[np.float64]]:
    run_values = _sorted_unique_runs(run_id)

    if isinstance(U_by_run, Mapping):
        out: dict[str, NDArray[np.float64]] = {}
        for run_value in run_values.tolist():
            key = str(run_value)
            if key not in U_by_run:
                raise ValueError(f"Missing U block for run '{key}'.")
            block = _as_numeric_matrix(U_by_run[key], f"U_by_run['{key}']")
            expected_n = int(np.sum(run_id == run_value))
            if block.shape != (expected_n, expected_n):
                raise ValueError(
                    f"U_by_run block for run '{key}' must be {expected_n} x {expected_n} "
                    f"(got {block.shape[0]} x {block.shape[1]})."
                )
            out[key] = block
        return out

    seq = list(U_by_run)
    if len(seq) != run_values.size:
        raise ValueError(
            "Unnamed U_by_run must have one block per unique run_id "
            f"(expected {run_values.size}, got {len(seq)})."
        )
    out = {}
    for i, run_value in enumerate(run_values.tolist()):
        key = str(run_value)
        block = _as_numeric_matrix(seq[i], f"U_by_run[{i}]")
        expected_n = int(np.sum(run_id == run_value))
        if block.shape != (expected_n, expected_n):
            raise ValueError(
                f"U_by_run block for run '{key}' must be {expected_n} x {expected_n} "
                f"(got {block.shape[0]} x {block.shape[1]})."
            )
        out[key] = block
    return out


def _apply_vinv(
    V: CovarianceLike | None,
    X: NDArray[np.float64],
    v_type: VType,
    method: SolverMethod,
    tol: float,
) -> NDArray[np.float64]:
    if V is None:
        return X
    if _is_block_list(V):
        return _apply_vinv_blocks(
            cast("Sequence[object]", V), X, v_type, method, tol
        )

    if sparse.issparse(V):
        if v_type == "precision":
            return np.asarray(V @ X, dtype=np.float64)
        V = cast(Any, V).toarray()

    V_m = _as_numeric_matrix(V, "V")
    n_time = X.shape[0]
    if V_m.shape != (n_time, n_time):
        raise ValueError(
            f"V must be {n_time} x {n_time} to match n_time, got {V_m.shape[0]} x {V_m.shape[1]}."
        )
    if v_type == "precision":
        return V_m @ X
    return _safe_solve(
        A=V_m,
        B=X,
        method=method,
        tol=tol,
        context="item_compute_u(V solve)",
    ).value


def _apply_vinv_blocks(
    V_blocks: Sequence[object],
    X: NDArray[np.float64],
    v_type: VType,
    method: SolverMethod,
    tol: float,
) -> NDArray[np.float64]:
    if len(V_blocks) == 0:
        raise ValueError("V block list is empty.")

    starts: list[int] = []
    ends: list[int] = []
    blocks: list[NDArray[np.float64]] = []
    offset = 0
    for i, block_in in enumerate(V_blocks):
        if sparse.issparse(block_in):
            block_in = cast(Any, block_in).toarray()
        block = _as_numeric_matrix(block_in, f"V[{i}]")
        if block.shape[0] != block.shape[1]:
            raise ValueError(f"V[{i}] must be square.")
        starts.append(offset)
        ends.append(offset + block.shape[0])
        offset = ends[-1]
        blocks.append(block)

    if offset != X.shape[0]:
        raise ValueError(
            f"Sum of V block sizes ({offset}) must equal nrow(X_t) ({X.shape[0]})."
        )

    out = np.zeros_like(X, dtype=np.float64)
    for i, block in enumerate(blocks):
        sl = slice(starts[i], ends[i])
        Xi = X[sl, :]
        if v_type == "precision":
            out[sl, :] = block @ Xi
        else:
            out[sl, :] = _safe_solve(
                A=block,
                B=Xi,
                method=method,
                tol=tol,
                context=f"item_compute_u(V block {i} solve)",
            ).value
    return out


def _safe_solve(
    A: NDArray[np.float64],
    B: NDArray[np.float64] | None,
    method: SolverMethod,
    tol: float,
    context: str,
) -> _SolveResult:
    methods: list[str] = []
    for candidate_lit in (method, "chol", "svd", "pinv"):
        candidate = cast(str, candidate_lit)
        if candidate not in methods:
            methods.append(candidate)

    warnings_seen: list[str] = []
    for candidate in methods:
        candidate_typed = cast("Literal['chol', 'svd', 'pinv']", candidate)
        try:
            value = _solve_once(A=A, B=B, method=candidate_typed, tol=tol)
            return _SolveResult(value=value, method=candidate_typed, warnings=warnings_seen)
        except Exception:
            warnings_seen.append(f"{context} failed with method '{candidate}'.")
    raise RuntimeError(f"All solver paths failed in {context}.")


def _solve_once(
    A: NDArray[np.float64],
    B: NDArray[np.float64] | None,
    method: SolverMethod,
    tol: float,
) -> NDArray[np.float64]:
    if method == "chol":
        cho = linalg.cho_factor(A, lower=False, check_finite=True)
        d = np.abs(np.diag(cho[0]))
        if d.size == 0 or not np.all(np.isfinite(d)):
            raise linalg.LinAlgError("Invalid Cholesky diagonal.")
        if np.max(d) == 0.0 or np.min(d) <= tol * np.max(d):
            raise linalg.LinAlgError("Near-singular Cholesky factor.")
        if B is None:
            eye = np.eye(A.shape[0], dtype=np.float64)
            return np.asarray(linalg.cho_solve(cho, eye), dtype=np.float64)
        return np.asarray(linalg.cho_solve(cho, B), dtype=np.float64)

    if method == "svd":
        u, s, vt = np.linalg.svd(A, full_matrices=False)
        if s.size == 0:
            raise linalg.LinAlgError("Empty singular value decomposition.")
        s_max = float(np.max(s))
        keep = s > (s_max * tol)
        if not np.any(keep):
            raise linalg.LinAlgError("No stable singular values above tolerance.")
        u_k = u[:, keep]
        vt_k = vt[keep, :]
        inv_s = 1.0 / s[keep]
        if B is None:
            return (vt_k.T * inv_s[np.newaxis, :]) @ u_k.T
        return (vt_k.T * inv_s[np.newaxis, :]) @ (u_k.T @ B)

    if method == "pinv":
        pinv = np.linalg.pinv(A, rcond=tol)
        if B is None:
            return pinv
        return pinv @ B

    raise ValueError(f"Unsupported solve method {method!r}.")


def _condition_number(A: NDArray[np.float64]) -> float:
    try:
        out = float(np.linalg.cond(A))
    except Exception:
        return float("inf")
    if not np.isfinite(out):
        return float("inf")
    return out


def _solver_warning(
    context: str,
    requested: SolverMethod,
    used: SolverMethod,
    warnings_seen: Sequence[str],
) -> str | None:
    changed = requested != used
    has_warnings = len(warnings_seen) > 0
    if not changed and not has_warnings:
        return None

    parts: list[str] = []
    if changed:
        parts.append(f"requested '{requested}' but used '{used}'")
    if has_warnings:
        parts.append(" ".join(dict.fromkeys(warnings_seen)))
    return f"{context}: {'; '.join(parts)}"


def _split_u_by_run(U: NDArray[np.float64], run_id: NDArray[Any]) -> dict[str, NDArray[np.float64]]:
    out: dict[str, NDArray[np.float64]] = {}
    for run_value in _sorted_unique_runs(run_id).tolist():
        idx = np.where(run_id == run_value)[0]
        out[str(run_value)] = U[np.ix_(idx, idx)]
    return out


def _block_diag(blocks: Sequence[NDArray[np.float64]]) -> NDArray[np.float64]:
    if len(blocks) == 0:
        return np.zeros((0, 0), dtype=np.float64)
    mats = [_as_numeric_matrix(block, f"blocks[{i}]") for i, block in enumerate(blocks)]
    return np.asarray(linalg.block_diag(*mats), dtype=np.float64)


def _coerce_u_matrix(
    U: CovarianceLike | ItemCovarianceResult,
) -> NDArray[np.float64]:
    if isinstance(U, ItemCovarianceMatrixResult):
        return _as_numeric_matrix(U.U, "U_train")
    if isinstance(U, ItemCovarianceBlockResult):
        return _block_diag(list(U.U_by_run.values()))
    if _is_u_blocks(U):
        if isinstance(U, Mapping):
            return _block_diag(list(U.values()))
        return _block_diag(list(U))
    return _as_numeric_matrix(U, "U_train")


def _metric_name(mode: CvMode, metric: MetricName | None) -> MetricName:
    if metric is None:
        return "accuracy" if mode == "classification" else "correlation"
    metric_norm = str(metric).lower()
    if mode == "classification" and metric_norm not in {"accuracy", "balanced_accuracy"}:
        raise ValueError("Unsupported classification metric.")
    if mode == "regression" and metric_norm not in {"correlation", "rmse"}:
        raise ValueError("Unsupported regression metric.")
    return metric_norm  # type: ignore[return-value]


def _class_levels(
    T_target: NDArray[np.float64],
    class_levels: Sequence[str] | None,
) -> list[str]:
    if class_levels is not None:
        levels = [str(x) for x in class_levels]
        if len(levels) != T_target.shape[1]:
            raise ValueError(f"class_levels must have length {T_target.shape[1]}.")
        return levels
    return [f"class_{i + 1}" for i in range(T_target.shape[1])]


def _score_fold(
    T_true: NDArray[np.float64],
    T_hat: NDArray[np.float64],
    mode: CvMode,
    metric: MetricName,
    class_levels: list[str] | None,
) -> _FoldScore:
    if mode == "classification":
        assert class_levels is not None
        truth_idx = np.argmax(T_true, axis=1)
        pred_idx = np.argmax(T_hat, axis=1)
        truth_labels = np.asarray([class_levels[i] for i in truth_idx], dtype="<U64")
        pred_labels = np.asarray([class_levels[i] for i in pred_idx], dtype="<U64")

        if metric == "balanced_accuracy":
            recalls: list[float] = []
            for cls in class_levels:
                mask = truth_labels == cls
                if not np.any(mask):
                    recalls.append(np.nan)
                else:
                    recalls.append(float(np.mean(pred_labels[mask] == cls)))
            score = float(np.nanmean(np.asarray(recalls, dtype=np.float64)))
        else:
            score = float(np.mean(pred_labels == truth_labels))
        return _FoldScore(metric=score, pred_labels=pred_labels)

    if metric == "rmse":
        return _FoldScore(
            metric=float(np.sqrt(np.mean((T_hat - T_true) ** 2))),
            pred_labels=None,
        )

    cors: list[float] = []
    for j in range(T_true.shape[1]):
        x = T_true[:, j]
        y = T_hat[:, j]
        if np.std(x) == 0 or np.std(y) == 0:
            cors.append(np.nan)
        else:
            cors.append(float(np.corrcoef(x, y)[0, 1]))
    return _FoldScore(metric=float(np.nanmean(np.asarray(cors))), pred_labels=None)


def _true_class(T_target: NDArray[np.float64], class_levels: list[str]) -> NDArray[np.str_]:
    idx = np.argmax(T_target, axis=1)
    return np.asarray([class_levels[i] for i in idx], dtype="<U64")


def _contains_missing(arr: NDArray[Any]) -> bool:
    if arr.dtype.kind in {"f", "c"}:
        return bool(np.isnan(arr).any())
    for value in arr.reshape(-1).tolist():
        if value is None:
            return True
        if isinstance(value, float) and np.isnan(value):
            return True
    return False


def _sorted_unique_runs(run_id: NDArray[Any]) -> NDArray[Any]:
    unique_vals: list[Any] = []
    seen: set[Any] = set()
    for value in run_id.reshape(-1).tolist():
        if value in seen:
            continue
        seen.add(value)
        unique_vals.append(value)
    try:
        sorted_vals = sorted(unique_vals)
    except TypeError:
        sorted_vals = sorted(unique_vals, key=lambda x: str(x))
    return np.asarray(sorted_vals, dtype=object)


def _is_block_list(value: object) -> bool:
    if isinstance(value, (str, bytes, np.ndarray)):
        return False
    if isinstance(value, Mapping):
        return False
    return isinstance(value, Sequence)


def _is_u_blocks(value: object) -> bool:
    if isinstance(value, (ItemCovarianceMatrixResult, ItemCovarianceBlockResult)):
        return False
    if isinstance(value, (str, bytes, np.ndarray)):
        return False
    return isinstance(value, (Mapping, Sequence))


__all__ = [
    "ItemBundle",
    "ItemCovarianceResult",
    "ItemCovarianceMatrixResult",
    "ItemCovarianceBlockResult",
    "ItemCvAggregate",
    "ItemCvResult",
    "ItemFoldMetrics",
    "ItemFoldSplit",
    "ItemPredictions",
    "ItemWeightsResult",
    "item_build_design",
    "item_compute_u",
    "item_cv",
    "item_fit",
    "item_from_lsa",
    "item_predict",
    "item_slice_fold",
]
