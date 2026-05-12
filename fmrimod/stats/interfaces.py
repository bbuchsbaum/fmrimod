"""Canonical second-level interfaces with parity-oriented defaults."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from ..dataset.group_data import GroupData


SecondLevelModel = Literal["meta", "ttest"]
EffectMode = Literal["fixed", "random"]
Tau2Method = Literal["dl", "pm", "reml"]
TTestEngine = Literal["auto", "meta", "classic", "welch"]
WeightMode = Literal["ivw", "equal", "custom"]
CorrectionMethod = Literal["bh", "by", "spatial"]
BackendName = Literal["auto", "python", "fmrigds"]


@dataclass(frozen=True)
class GroupFitRequest:
    """Canonical second-level request.

    Notes
    -----
    This request is intentionally close to R parity semantics.
    Convenience aliases are normalized before backend dispatch.
    """

    data: GroupData
    formula: str = "~ 1"
    model: SecondLevelModel = "meta"
    effects: EffectMode = "random"
    tau2: Tau2Method = "pm"
    method: str | None = None
    ttest_engine: TTestEngine = "auto"
    robust: str = "none"
    combine: str | None = None
    weights: WeightMode | str = "ivw"
    weights_custom: NDArray[np.float64] | None = None
    correction: CorrectionMethod | str | None = None
    alpha: float = 0.05
    group_ids: Sequence[int] | NDArray[np.intp] | None = None
    backend: BackendName | str = "auto"
    backend_options: Mapping[str, Any] = field(default_factory=dict)
    extra_options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GroupFitResult:
    """Canonical second-level result."""

    estimate: NDArray[np.float64]
    se: NDArray[np.float64]
    statistic: NDArray[np.float64]
    p: NDArray[np.float64]
    q: NDArray[np.float64] | None
    tau2: NDArray[np.float64] | None
    predictor_names: list[str]
    feature_names: list[str]
    model: str
    method: str
    formula: str
    backend: str
    metadata: dict[str, Any] = field(default_factory=dict)
