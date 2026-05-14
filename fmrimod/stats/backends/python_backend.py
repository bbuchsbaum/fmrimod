"""Native Python parity backend for second-level modeling."""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

from fmrimod.stats.interfaces import GroupFitRequest, GroupFitResult
from fmrimod.stats.meta import (
    FmriMetaResult,
    MetaMethod,
    MetaRobust,
    MetaWeights,
    fmri_meta,
)
from fmrimod.stats.ttest import FmriTTestResult, fmri_ttest


def _as_2d(x: NDArray[np.float64]) -> NDArray[np.float64]:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        return arr[:, np.newaxis]
    if arr.ndim == 2:
        return arr
    raise ValueError("Expected 1-D or 2-D array output from backend")


class PythonParityBackend:
    """Backend that delegates to existing Python parity implementations."""

    name = "python"

    def fit(self, request: GroupFitRequest) -> GroupFitResult:
        if request.model == "meta":
            if request.combine is not None:
                return self._fit_native_group_combine(request)
            if getattr(request.data, "format", None) in ("h5", "nifti", "fmrilm"):
                return self._fit_native_group_meta(request)
            meta_out = fmri_meta(
                data=request.data,
                formula=request.formula,
                method=cast(MetaMethod, request.method),
                robust=cast(MetaRobust, request.robust),
                weights=cast(MetaWeights, request.weights),
                weights_custom=request.weights_custom,
                combine=request.combine,
            )
            return self._from_meta(meta_out, request)

        if request.model == "ttest":
            ttest_out = fmri_ttest(
                data=request.data,
                engine=request.ttest_engine,
                formula=request.formula,
                method=str(request.method),
                weights=str(request.weights),
                weights_custom=request.weights_custom,
            )
            return self._from_ttest(ttest_out, request)

        raise ValueError("model must be one of: meta, ttest")

    def _fit_native_group_combine(self, request: GroupFitRequest) -> GroupFitResult:
        from fmrimod.group import derive, group_dataset_from_group_data
        from fmrimod.group import reduce as group_reduce

        if request.formula.replace(" ", "") not in ("~1", "1"):
            raise NotImplementedError(
                "native group combine currently supports intercept-only formulas"
            )
        if request.robust != "none":
            raise NotImplementedError(
                "native group combine does not support robust meta fitting"
            )

        combine_method = _normalize_combine_method(request.combine)
        dataset = group_dataset_from_group_data(request.data)
        options = _combine_options(request, dataset.n_subjects, combine_method)
        if combine_method == "combine:stouffer" and "z" not in dataset.assays:
            dataset = derive(dataset, "z")
        elif (
            combine_method in ("combine:fisher", "combine:lancaster")
            and "p" not in dataset.assays
        ):
            dataset = derive(dataset, "p")

        reduced = group_reduce(dataset, method=combine_method, **options)
        stat_assay = "z_g" if "z_g" in reduced.assays else "chi2"
        statistic = _flatten_group_assay(reduced, stat_assay)
        p = _flatten_group_assay(reduced, "p_g")
        return GroupFitResult(
            estimate=statistic,
            se=np.full_like(statistic, np.nan),
            statistic=statistic,
            p=p,
            q=None,
            tau2=None,
            predictor_names=["combined"],
            feature_names=_group_feature_names(dataset),
            model="meta",
            method=combine_method,
            formula=request.formula,
            backend=self.name,
            metadata={
                "source": "fmrimod.group",
                "reduce_method": combine_method,
                "source_format": getattr(request.data, "format", None),
                "combine": request.combine,
            },
        )

    def _fit_native_group_meta(self, request: GroupFitRequest) -> GroupFitResult:
        from fmrimod.group import group_dataset_from_group_data
        from fmrimod.group import reduce as group_reduce

        if request.robust != "none":
            raise NotImplementedError(
                "native group Python backend does not yet support robust meta fitting"
            )
        if request.combine is not None:
            raise NotImplementedError(
                "native group Python backend does not yet support combine modes"
            )
        if request.weights != "ivw":
            raise NotImplementedError(
                "native group Python backend currently supports native meta "
                "requests only with inverse-variance weights"
            )

        method = cast(str, request.method)
        reducer_method = {"fe": "meta:fe", "dl": "meta:re"}.get(method)
        if reducer_method is None:
            raise NotImplementedError(
                "native group Python backend currently supports native meta "
                "methods 'fe' and 'dl'"
            )

        dataset = group_dataset_from_group_data(request.data)
        if request.formula.replace(" ", "") not in ("~1", "1"):
            reducer_method = {"fe": "meta:fe_reg", "dl": "meta:re_reg"}[method]
            reduced = group_reduce(dataset, method=reducer_method, formula=request.formula)
            predictor_names = list(reduced.metadata["predictor_names"])
            tau2 = None
            if "tau2" in reduced.assays:
                tau2 = _flatten_group_assay(reduced, "tau2")
            return GroupFitResult(
                estimate=_flatten_regression_assays(reduced, predictor_names, "coef"),
                se=_flatten_regression_assays(reduced, predictor_names, "se_coef"),
                statistic=_flatten_regression_assays(reduced, predictor_names, "z_coef"),
                p=_flatten_regression_assays(reduced, predictor_names, "p_coef"),
                q=None,
                tau2=tau2,
                predictor_names=predictor_names,
                feature_names=_group_feature_names(dataset),
                model="meta",
                method=method,
                formula=request.formula,
                backend=self.name,
                metadata={
                    "source": "fmrimod.group",
                    "reduce_method": reducer_method,
                    "source_format": getattr(request.data, "format", None),
                },
            )

        reduced = group_reduce(dataset, method=reducer_method)
        tau2 = None
        if "tau2" in reduced.assays:
            tau2 = _flatten_group_assay(reduced, "tau2")
        return GroupFitResult(
            estimate=_flatten_group_assay(reduced, "beta_g"),
            se=_flatten_group_assay(reduced, "se_g"),
            statistic=_flatten_group_assay(reduced, "z_g"),
            p=_flatten_group_assay(reduced, "p_g"),
            q=None,
            tau2=tau2,
            predictor_names=["Intercept"],
            feature_names=_group_feature_names(dataset),
            model="meta",
            method=method,
            formula=request.formula,
            backend=self.name,
            metadata={
                "source": "fmrimod.group",
                "reduce_method": reducer_method,
                "source_format": getattr(request.data, "format", None),
            },
        )

    def _from_meta(self, out: FmriMetaResult, request: GroupFitRequest) -> GroupFitResult:
        return GroupFitResult(
            estimate=_as_2d(out.coefficients),
            se=_as_2d(out.se),
            statistic=_as_2d(out.z),
            p=_as_2d(out.p),
            q=None,
            tau2=np.asarray(out.tau2, dtype=np.float64),
            predictor_names=list(out.predictor_names),
            feature_names=list(out.feature_names),
            model="meta",
            method=cast(str, request.method),
            formula=out.formula,
            backend=self.name,
            metadata={
                "source": "fmri_meta",
                "requested_effects": request.effects,
                "requested_tau2": request.tau2,
            },
        )

    def _from_ttest(self, out: FmriTTestResult, request: GroupFitRequest) -> GroupFitResult:
        tau2 = None
        if out.meta_result is not None and hasattr(out.meta_result, "tau2"):
            tau2 = np.asarray(out.meta_result.tau2, dtype=np.float64)
        return GroupFitResult(
            estimate=_as_2d(np.asarray(out.estimate, dtype=np.float64)),
            se=_as_2d(np.asarray(out.se, dtype=np.float64)),
            statistic=_as_2d(np.asarray(out.statistic, dtype=np.float64)),
            p=_as_2d(np.asarray(out.p, dtype=np.float64)),
            q=None,
            tau2=tau2,
            predictor_names=["Intercept"],
            feature_names=list(out.feature_names),
            model="ttest",
            method=cast(str, request.method),
            formula=request.formula,
            backend=self.name,
            metadata={
                "source": "fmri_ttest",
                "engine": out.engine,
            },
        )


def _flatten_group_assay(
    dataset: object,
    assay: str,
) -> NDArray[np.float64]:
    values = np.asarray(dataset.assay(assay)[:, 0, :], dtype=np.float64)
    return values.reshape(dataset.n_samples * dataset.n_contrasts, 1)


def _flatten_regression_assays(
    dataset: object,
    predictor_names: list[str],
    prefix: str,
) -> NDArray[np.float64]:
    cols = [
        _flatten_group_assay(dataset, f"{prefix}:{name}")[:, 0]
        for name in predictor_names
    ]
    return np.column_stack(cols).astype(np.float64, copy=False)


def _normalize_combine_method(combine: str | None) -> str:
    key = "" if combine is None else str(combine).strip().lower()
    aliases = {
        "stouffer": "combine:stouffer",
        "combine:stouffer": "combine:stouffer",
        "fisher": "combine:fisher",
        "combine:fisher": "combine:fisher",
        "lancaster": "combine:lancaster",
        "combine:lancaster": "combine:lancaster",
    }
    try:
        return aliases[key]
    except KeyError as exc:
        raise ValueError(
            "combine must be one of: stouffer, fisher, lancaster"
        ) from exc


def _custom_subject_weights(
    request: GroupFitRequest,
    n_subjects: int,
) -> NDArray[np.float64] | None:
    if request.weights != "custom":
        return None
    if request.weights_custom is None:
        raise ValueError("weights='custom' requires weights_custom")
    weights = np.asarray(request.weights_custom, dtype=np.float64)
    if weights.shape != (n_subjects,):
        raise ValueError("combine custom weights must have shape (n_subjects,)")
    if not np.all(np.isfinite(weights)):
        raise ValueError("combine custom weights must be finite")
    return weights


def _combine_options(
    request: GroupFitRequest,
    n_subjects: int,
    combine_method: str,
) -> dict[str, object]:
    options: dict[str, object] = {}
    merged_options = {**dict(request.backend_options), **dict(request.extra_options)}
    if "min_subjects" in merged_options:
        options["min_subjects"] = int(merged_options["min_subjects"])

    weights = _custom_subject_weights(request, n_subjects)
    if weights is not None and combine_method == "combine:fisher":
        raise ValueError("combine='fisher' does not support custom weights")
    if combine_method == "combine:stouffer" and weights is not None:
        options["weights"] = weights
    if combine_method == "combine:lancaster":
        dfw = merged_options.get("dfw", weights)
        if dfw is None:
            raise ValueError(
                "combine='lancaster' requires dfw in backend_options or extra_options"
            )
        options["dfw"] = np.asarray(dfw, dtype=np.float64)
    return options


def _group_sample_names(dataset: object) -> list[str]:
    from fmrimod.group import SampleLabelSpace, VoxelSpace

    if isinstance(dataset.space, SampleLabelSpace):
        return list(dataset.space.labels)
    if isinstance(dataset.space, VoxelSpace):
        if dataset.space.mask_idx is not None:
            return [f"voxel:{int(idx)}" for idx in dataset.space.mask_idx]
        return [f"voxel:{idx}" for idx in range(dataset.n_samples)]
    return [f"sample{i + 1}" for i in range(dataset.n_samples)]


def _group_feature_names(dataset: object) -> list[str]:
    samples = _group_sample_names(dataset)
    if dataset.n_contrasts == 1:
        return samples
    return [
        f"{sample}:{contrast}"
        for sample in samples
        for contrast in dataset.contrasts
    ]
