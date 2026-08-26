"""Typed control objects for :func:`fmrimod.group.examine_group`."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, cast

QualityDirection = Literal["low", "high"]
NaAction = Literal["fail", "exclude"]
ReviewStatus = Literal["none", "review", "insufficient"]
ReviewSource = Literal["quality", "surprise", "influence"]


@dataclass(frozen=True)
class SurpriseReview:
    """Absolute surprise gates that can flip ``review_status``."""

    energy_threshold: float = 2.5
    tail_threshold: float = 0.01
    residual_threshold: float = 3.0


@dataclass(frozen=True)
class InfluenceReview:
    """Absolute influence gates that can flip ``review_status``."""

    energy_threshold: float = 1.0
    max_abs_threshold: float = 2.0


@dataclass(frozen=True)
class QualityRule:
    """One named data-validity criterion."""

    direction: QualityDirection
    threshold: float

    def __post_init__(self) -> None:
        if self.direction not in ("low", "high"):
            raise ValueError(
                f"quality direction must be 'low' or 'high', got {self.direction!r}"
            )
        if not math.isfinite(self.threshold):
            raise ValueError("quality threshold must be finite")


@dataclass(frozen=True)
class GeometryControl:
    """Deterministic residual-geometry settings."""

    rank: int = 32
    oversample: int = 8
    cap: float = 8.0
    balance_contrasts: bool = True
    stability_replicates: int = 2

    def __post_init__(self) -> None:
        if int(self.rank) < 1:
            raise ValueError("geometry rank must be a positive integer")
        if int(self.oversample) < 0:
            raise ValueError("geometry oversample must be non-negative")
        if int(self.stability_replicates) < 0:
            raise ValueError("geometry stability_replicates must be non-negative")
        if not math.isfinite(self.cap) or self.cap <= 0:
            raise ValueError("geometry cap must be a positive finite number")
        object.__setattr__(self, "rank", int(self.rank))
        object.__setattr__(self, "oversample", int(self.oversample))
        object.__setattr__(self, "stability_replicates", int(self.stability_replicates))


@dataclass(frozen=True)
class ExaminationTolerance:
    """Numerical tolerances used by diagnostic kernels."""

    rank: float = 1.4901161193847656e-08  # sqrt(eps) on typical IEEE-754
    leverage: float = 1e-8
    degeneracy: float = 1e-12


def _default_quality() -> dict[str, QualityRule]:
    return {"coverage_fraction": QualityRule(direction="low", threshold=0.8)}


@dataclass(frozen=True)
class ExaminationControl:
    """Bounded execution and review settings for group examination.

    Defaults match fmrigds ``examination_control()``. Review status becomes
    ``"review"`` only when an absolute, stability-aware criterion fires.
    """

    block_size: int = 1024
    geometry: GeometryControl = field(default_factory=GeometryControl)
    surprise: SurpriseReview = field(default_factory=SurpriseReview)
    influence: InfluenceReview = field(default_factory=InfluenceReview)
    quality: Mapping[str, QualityRule] = field(default_factory=_default_quality)
    min_stability: float = 0.7
    exact_refit_n: int = 5
    retain_n: int = 5
    tolerance: ExaminationTolerance = field(default_factory=ExaminationTolerance)

    def __post_init__(self) -> None:
        if int(self.block_size) < 1:
            raise ValueError("block_size must be a positive integer")
        if not (0.0 <= float(self.min_stability) <= 1.0):
            raise ValueError("min_stability must be between 0 and 1")
        if int(self.exact_refit_n) < 0:
            raise ValueError("exact_refit_n must be non-negative")
        if int(self.retain_n) < 0:
            raise ValueError("retain_n must be non-negative")
        quality = dict(self.quality)
        object.__setattr__(self, "quality", quality)
        object.__setattr__(self, "block_size", int(self.block_size))
        object.__setattr__(self, "exact_refit_n", int(self.exact_refit_n))
        object.__setattr__(self, "retain_n", int(self.retain_n))
        object.__setattr__(self, "min_stability", float(self.min_stability))


def _check_mapping_keys(
    values: Mapping[str, object], allowed: set[str], *, name: str
) -> None:
    unknown = set(values) - allowed
    if unknown:
        raise TypeError(f"unknown {name} field(s): {', '.join(sorted(unknown))}")


def _geometry_from_mapping(values: Mapping[str, object]) -> GeometryControl:
    _check_mapping_keys(
        values,
        {"rank", "oversample", "cap", "balance_contrasts", "stability_replicates"},
        name="geometry",
    )
    return GeometryControl(
        rank=cast(int, values.get("rank", 32)),
        oversample=cast(int, values.get("oversample", 8)),
        cap=cast(float, values.get("cap", 8.0)),
        balance_contrasts=cast(bool, values.get("balance_contrasts", True)),
        stability_replicates=cast(int, values.get("stability_replicates", 2)),
    )


def _tolerance_from_mapping(
    values: Mapping[str, object],
) -> ExaminationTolerance:
    _check_mapping_keys(
        values, {"rank", "leverage", "degeneracy"}, name="tolerance"
    )
    return ExaminationTolerance(
        rank=cast(float, values.get("rank", ExaminationTolerance.rank)),
        leverage=cast(float, values.get("leverage", ExaminationTolerance.leverage)),
        degeneracy=cast(
            float, values.get("degeneracy", ExaminationTolerance.degeneracy)
        ),
    )


def _surprise_from_mapping(values: Mapping[str, object]) -> SurpriseReview:
    _check_mapping_keys(
        values,
        {"energy_threshold", "tail_threshold", "residual_threshold"},
        name="surprise",
    )
    return SurpriseReview(
        energy_threshold=cast(float, values.get("energy_threshold", 2.5)),
        tail_threshold=cast(float, values.get("tail_threshold", 0.01)),
        residual_threshold=cast(float, values.get("residual_threshold", 3.0)),
    )


def _influence_from_mapping(values: Mapping[str, object]) -> InfluenceReview:
    _check_mapping_keys(
        values, {"energy_threshold", "max_abs_threshold"}, name="influence"
    )
    return InfluenceReview(
        energy_threshold=cast(float, values.get("energy_threshold", 1.0)),
        max_abs_threshold=cast(float, values.get("max_abs_threshold", 2.0)),
    )


def _quality_rule_from_mapping(values: Mapping[str, object]) -> QualityRule:
    _check_mapping_keys(values, {"direction", "threshold"}, name="quality rule")
    if "direction" not in values or "threshold" not in values:
        raise TypeError("quality rule requires direction and threshold")
    return QualityRule(
        direction=cast(QualityDirection, values["direction"]),
        threshold=cast(float, values["threshold"]),
    )


def examination_control(
    *,
    block_size: int | None = None,
    geometry: GeometryControl | Mapping[str, object] | None = None,
    review: Mapping[str, object] | None = None,
    surprise: SurpriseReview | None = None,
    influence: InfluenceReview | None = None,
    quality: Mapping[str, QualityRule | Mapping[str, object]] | None = None,
    min_stability: float | None = None,
    exact_refit_n: int | None = None,
    retain_n: int | None = None,
    tolerance: ExaminationTolerance | Mapping[str, object] | None = None,
) -> ExaminationControl:
    """Build an :class:`ExaminationControl`, merging nested mappings."""
    if geometry is not None and not isinstance(geometry, GeometryControl):
        if not isinstance(geometry, Mapping):
            raise TypeError("geometry must be GeometryControl or a mapping")
        geometry = _geometry_from_mapping(geometry)
    if tolerance is not None and not isinstance(tolerance, ExaminationTolerance):
        if not isinstance(tolerance, Mapping):
            raise TypeError("tolerance must be ExaminationTolerance or a mapping")
        tolerance = _tolerance_from_mapping(tolerance)
    if review is not None:
        if not isinstance(review, Mapping):
            raise TypeError("review must be a mapping")
        if surprise is None and "surprise" in review:
            surprise_spec = review["surprise"]
            if not isinstance(surprise_spec, Mapping):
                raise TypeError("review['surprise'] must be a mapping")
            surprise = _surprise_from_mapping(surprise_spec)
        if influence is None and "influence" in review:
            influence_spec = review["influence"]
            if not isinstance(influence_spec, Mapping):
                raise TypeError("review['influence'] must be a mapping")
            influence = _influence_from_mapping(influence_spec)
        if quality is None and "quality" in review:
            quality_spec = review["quality"]
            if not isinstance(quality_spec, Mapping):
                raise TypeError("review['quality'] must be a mapping")
            quality = cast(
                Mapping[str, QualityRule | Mapping[str, object]], quality_spec
            )
        if min_stability is None and "min_stability" in review:
            min_stability = cast(float, review["min_stability"])
    if quality is not None and not isinstance(quality, Mapping):
        raise TypeError("quality must be a mapping of QualityRule")
    if quality is not None:
        coerced: dict[str, QualityRule] = {}
        for name, spec in quality.items():
            if isinstance(spec, QualityRule):
                coerced[str(name)] = spec
            elif isinstance(spec, Mapping):
                coerced[str(name)] = _quality_rule_from_mapping(spec)
            else:
                raise TypeError(f"quality[{name!r}] must be QualityRule or a mapping")
        quality = coerced
    defaults = ExaminationControl()
    return ExaminationControl(
        block_size=defaults.block_size if block_size is None else block_size,
        geometry=defaults.geometry if geometry is None else geometry,
        surprise=defaults.surprise if surprise is None else surprise,
        influence=defaults.influence if influence is None else influence,
        quality=defaults.quality if quality is None else quality,
        min_stability=(
            defaults.min_stability if min_stability is None else min_stability
        ),
        exact_refit_n=(
            defaults.exact_refit_n if exact_refit_n is None else exact_refit_n
        ),
        retain_n=defaults.retain_n if retain_n is None else retain_n,
        tolerance=defaults.tolerance if tolerance is None else tolerance,
    )
