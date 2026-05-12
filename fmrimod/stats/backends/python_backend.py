"""Native Python parity backend for second-level modeling."""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

from ..interfaces import GroupFitRequest, GroupFitResult
from ..meta import FmriMetaResult, fmri_meta
from ..ttest import FmriTTestResult, fmri_ttest


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
            out = fmri_meta(
                data=request.data,
                formula=request.formula,
                method=cast(str, request.method),
                robust=request.robust,
                weights=cast(str, request.weights),
                weights_custom=request.weights_custom,
                combine=request.combine,
            )
            return self._from_meta(out, request)

        if request.model == "ttest":
            out = fmri_ttest(
                data=request.data,
                engine=request.ttest_engine,
                formula=request.formula,
                method=cast(str, request.method),
                weights=cast(str, request.weights),
                weights_custom=request.weights_custom,
            )
            return self._from_ttest(out, request)

        raise ValueError("model must be one of: meta, ttest")

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
