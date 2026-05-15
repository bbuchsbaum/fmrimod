"""LMM-specific helpers used by :mod:`fmrimod.group.reducers`.

Second slice of bd-01KRHTJ9WFSSBZSDGAN4V7PHGS (policy/kernel/registry
split). The helpers here are LMM-specific glue: formula validation,
fixed-effects design construction over the subject x contrast
observation table, the statsmodels ``MixedLM`` fit wrapper, fit-failure
classification, and the result-dataset shape. They depend on pandas,
patsy, and (optionally) statsmodels — heavier than the pure kernels in
:mod:`fmrimod.group._reducers_kernels`, but still scoped to one workflow.

Nothing here is part of the public API.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from patsy import dmatrix

from .dataset import GroupDataset
from .errors import AdapterContractError, UnsupportedGroupFeatureError

_LMM_EXPECTED_FIT_ERRORS = (
    FloatingPointError,
    np.linalg.LinAlgError,
    RuntimeError,
    ValueError,
)
_LMM_NUMERICAL_FAILURE_SIGNATURES = (
    "singular matrix",
    "convergence",
    "could not converge",
    "failed to converge",
    "maximum likelihood optimization",
    "non-positive-definite",
    "not positive definite",
    "positive definite",
    "step size",
)


def _lmm_formula(formula: str) -> str:
    if "|" in formula:
        raise UnsupportedGroupFeatureError(
            "LMM reducers do not support lmer-style random-effects syntax; "
            "choose method='lmm:ri' or method='lmm:ri_slope1' and pass reducer options"
        )
    clean = formula.strip()
    if not clean:
        clean = "~ 1"
    if not clean.startswith("~"):
        raise AdapterContractError("LMM formula must be one-sided, e.g. '~ condition'")
    return clean


def _lmm_observation_data(dataset: GroupDataset) -> pd.DataFrame:
    subject_rows = (
        pd.DataFrame(index=pd.Index(dataset.subjects, name="subject"))
        if dataset.col_data is None
        else dataset.col_data.copy()
    )
    subject_rows = subject_rows.copy()
    subject_rows.index = pd.Index(dataset.subjects, name="subject")
    if "subject" in subject_rows.columns:
        subject_rows = subject_rows.drop(columns=["subject"])

    contrast_rows = (
        pd.DataFrame(index=pd.Index(dataset.contrasts, name="contrast"))
        if dataset.contrast_data is None
        else dataset.contrast_data.copy()
    )
    contrast_rows = contrast_rows.copy()
    contrast_rows.index = pd.Index(dataset.contrasts, name="contrast")
    if "contrast" in contrast_rows.columns:
        contrast_rows = contrast_rows.drop(columns=["contrast"])
    contrast_rows["contrast"] = list(dataset.contrasts)

    rows: list[pd.DataFrame] = []
    for subject in dataset.subjects:
        subj = subject_rows.loc[[subject]].reset_index(drop=True)
        repeated_subj = pd.concat([subj] * dataset.n_contrasts, ignore_index=True)
        block = pd.concat(
            [
                pd.DataFrame({"subject": [subject] * dataset.n_contrasts}),
                repeated_subj,
                contrast_rows.reset_index(drop=True),
            ],
            axis=1,
        )
        rows.append(block)
    obs = pd.concat(rows, ignore_index=True)
    obs["subject"] = pd.Categorical(obs["subject"], categories=list(dataset.subjects))
    obs["contrast"] = pd.Categorical(obs["contrast"], categories=list(dataset.contrasts))
    return obs


def _lmm_fixed_design(
    dataset: GroupDataset,
    *,
    formula: str,
) -> tuple[pd.DataFrame, NDArray[np.float64], list[str]]:
    obs = _lmm_observation_data(dataset)
    design = dmatrix(_lmm_formula(formula), obs, return_type="dataframe")
    X = np.asarray(design, dtype=np.float64)
    if not np.all(np.isfinite(X)):
        raise AdapterContractError("LMM fixed-effects design contains non-finite values")
    return obs, X, list(design.columns)


def _lmm_beta_matrix(dataset: GroupDataset) -> NDArray[np.float64]:
    beta = dataset.assay("beta")
    return beta.reshape(dataset.n_samples, dataset.n_subjects * dataset.n_contrasts).T


def _lmm_not_available(message: str) -> UnsupportedGroupFeatureError:
    return UnsupportedGroupFeatureError(
        f"{message}. Use backend='fmrigds-r' as the explicit R oracle/fallback "
        "during migration, or use theta_mode='voxelwise' for the native "
        "statsmodels first slice where supported."
    )


def _fit_mixedlm_feature(
    y: NDArray[np.float64],
    X: NDArray[np.float64],
    groups: object,
    exog_re: NDArray[np.float64],
    *,
    reml: bool,
    covariance: Literal["diag", "full"] = "full",
) -> Any:
    try:
        from statsmodels.regression.mixed_linear_model import MixedLM, MixedLMParams
    except Exception as exc:  # pragma: no cover - dependency declared by package
        raise UnsupportedGroupFeatureError(
            "native LMM reducers require optional dependency 'statsmodels'"
        ) from exc

    model = MixedLM(endog=y, exog=X, groups=groups, exog_re=exog_re)
    fit_options: dict[str, object] = {}
    if covariance == "diag" and exog_re.shape[1] > 1:
        fit_options["free"] = MixedLMParams.from_components(
            fe_params=np.ones(X.shape[1]),
            cov_re=np.eye(exog_re.shape[1]),
        )
    return model.fit(reml=reml, method="lbfgs", disp=False, **fit_options)


def _is_expected_lmm_fit_failure(exc: BaseException) -> bool:
    if isinstance(exc, (FloatingPointError, np.linalg.LinAlgError)):
        return True
    if isinstance(exc, (RuntimeError, ValueError)):
        message = str(exc).lower()
        return any(sig in message for sig in _LMM_NUMERICAL_FAILURE_SIGNATURES)
    return False


def _lmm_failure_metadata(
    failures: list[tuple[int, BaseException]],
    *,
    attempted_features: int,
) -> dict[str, object]:
    return {
        "fit_attempted_features": int(attempted_features),
        "fit_failed_features": len(failures),
        "fit_failed_feature_indices": tuple(int(idx) for idx, _ in failures),
        "fit_failed_reasons": tuple(
            f"{type(exc).__name__}: {exc}" for _, exc in failures
        ),
    }


def _raise_if_all_lmm_fits_failed(
    *,
    method: str,
    failures: list[tuple[int, BaseException]],
    attempted_features: int,
) -> None:
    if attempted_features == 0 or len(failures) < attempted_features:
        return
    first_idx, first_exc = failures[0]
    raise UnsupportedGroupFeatureError(
        f"native {method} failed for all {attempted_features} finite features; "
        f"first failure at feature {first_idx}: {type(first_exc).__name__}: {first_exc}"
    ) from first_exc


def _lmm_result_dataset(
    dataset: GroupDataset,
    assays: dict[str, NDArray[np.float64]],
    *,
    method: str,
    metadata: dict[str, object],
) -> GroupDataset:
    return GroupDataset(
        assays=assays,
        space=dataset.space,
        subjects=["meta"],
        contrasts=["model"],
        col_data=pd.DataFrame(index=pd.Index(["meta"], name="subject")),
        row_data=dataset.row_data,
        contrast_data=pd.DataFrame(
            {"label": ["model"]},
            index=pd.Index(["model"], name="contrast"),
        ),
        metadata={
            **dict(dataset.metadata),
            "operation": "reduce",
            "reduce_method": method,
            **metadata,
        },
    )


__all__ = [
    "_LMM_EXPECTED_FIT_ERRORS",
    "_LMM_NUMERICAL_FAILURE_SIGNATURES",
    "_lmm_formula",
    "_lmm_observation_data",
    "_lmm_fixed_design",
    "_lmm_beta_matrix",
    "_lmm_not_available",
    "_fit_mixedlm_feature",
    "_is_expected_lmm_fit_failure",
    "_lmm_failure_metadata",
    "_raise_if_all_lmm_fits_failed",
    "_lmm_result_dataset",
]
