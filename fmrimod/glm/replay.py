"""Replay and compare two fitted fmrimod GLM results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Sequence

import numpy as np

from fmrimod.spec import SpecDiff, spec_diff

if TYPE_CHECKING:
    from fmrimod.dataset import FmriDataset
    from fmrimod.glm.fmri_lm import FmriLm
    from fmrimod.model.config import FmriLmConfig
    from fmrimod.spec import Spec, Term


@dataclass(frozen=True)
class ReplayContractError(ValueError):
    """Raised when two fits cannot be compared under the replay contract."""

    message: str
    weak_fields: tuple[str, ...] = ()
    repair_path: str | None = None

    def __str__(self) -> str:
        detail = self.message
        if self.weak_fields:
            detail += f" weak_fields={self.weak_fields!r}"
        if self.repair_path:
            detail += f" repair_path={self.repair_path}"
        return detail


@dataclass(frozen=True)
class ContrastDelta:
    """Summary of one contrast statistic delta across two replayed fits."""

    name: str
    stat_type: Literal["t", "F"]
    value_a: float
    value_b: float
    max_abs_delta: float
    median_abs_delta: float
    df_a: float | tuple[object, ...]
    df_b: float | tuple[object, ...]
    df_match: bool


@dataclass(frozen=True)
class ReplayResult:
    """Result of replaying or comparing two fitted GLM objects."""

    diff: "SpecDiff"
    fit_a: "FmriLm"
    fit_b: "FmriLm"
    contrast_deltas: tuple[ContrastDelta, ...]
    dropped_from_a: tuple[str, ...] = ()
    dropped_from_b: tuple[str, ...] = ()

    def explain(self) -> str:
        """Return a compact text summary suitable for board/debug notes."""

        parts = [f"{len(self.contrast_deltas)} contrast(s) compared"]
        if self.contrast_deltas:
            worst = max(self.contrast_deltas, key=lambda delta: delta.max_abs_delta)
            parts.append(
                f"worst={worst.name} max_abs_delta={worst.max_abs_delta:.6g}"
            )
        if self.dropped_from_a:
            parts.append("only in fit_a: " + ", ".join(self.dropped_from_a))
        if self.dropped_from_b:
            parts.append("only in fit_b: " + ", ".join(self.dropped_from_b))
        return "; ".join(parts)


def _df_equal(a: object, b: object) -> bool:
    if isinstance(a, tuple) or isinstance(b, tuple):
        if not isinstance(a, tuple) or not isinstance(b, tuple):
            return False
        if len(a) != len(b):
            return False
        return all(_df_equal(x, y) for x, y in zip(a, b))
    try:
        return bool(np.isclose(float(a), float(b), rtol=0.0, atol=1e-12))
    except (TypeError, ValueError):
        return a == b


def _contrast_delta(name: str, fit_a: "FmriLm", fit_b: "FmriLm") -> ContrastDelta:
    con_a = fit_a.contrasts[name]
    con_b = fit_b.contrasts[name]
    df_match = _df_equal(con_a.df, con_b.df)
    if not df_match:
        raise ReplayContractError(
            message=(
                f"Contrast {name!r} has incompatible degrees of freedom: "
                f"{con_a.df!r} != {con_b.df!r}"
            ),
            weak_fields=(f"contrasts.{name}.df",),
            repair_path="recompute both fits with the same model rank and contrast basis",
        )

    stat_a = np.asarray(con_a.stat, dtype=np.float64)
    stat_b = np.asarray(con_b.stat, dtype=np.float64)
    if stat_a.shape != stat_b.shape:
        raise ReplayContractError(
            message=(
                f"Contrast {name!r} statistic shapes differ: "
                f"{stat_a.shape!r} != {stat_b.shape!r}"
            ),
            weak_fields=(f"contrasts.{name}.stat",),
            repair_path="compare contrasts evaluated over the same response/mask space",
        )
    abs_delta = np.abs(stat_a - stat_b)
    return ContrastDelta(
        name=name,
        stat_type="F" if str(con_a.stat_type).upper() == "F" else "t",
        value_a=float(np.median(stat_a)),
        value_b=float(np.median(stat_b)),
        max_abs_delta=float(np.max(abs_delta)),
        median_abs_delta=float(np.median(abs_delta)),
        df_a=con_a.df,
        df_b=con_b.df,
        df_match=df_match,
    )


def _selected_contrasts(
    fit_a: "FmriLm",
    fit_b: "FmriLm",
    named_contrasts: Sequence[str] | None,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    names_a = set(fit_a.contrasts)
    names_b = set(fit_b.contrasts)
    if named_contrasts is None:
        selected = tuple(sorted(names_a & names_b))
        if not selected:
            raise ReplayContractError(
                message="no compatible contrast intersection between the two fits",
                weak_fields=("contrasts",),
                repair_path=(
                    "compute at least one shared named contrast or pass explicit "
                    "named_contrasts present on both fits"
                ),
            )
        return selected, tuple(sorted(names_a - names_b)), tuple(sorted(names_b - names_a))

    requested = tuple(str(name) for name in named_contrasts)
    absent = [
        name
        for name in requested
        if name not in names_a or name not in names_b
    ]
    if absent:
        raise ReplayContractError(
            message=(
                "explicit named contrasts are absent on one or both fits: "
                + ", ".join(absent)
            ),
            weak_fields=tuple(f"contrasts.{name}" for name in absent),
            repair_path="compute the requested named contrasts on both fits",
        )
    if not requested:
        raise ReplayContractError(
            message="no compatible named contrasts were requested",
            weak_fields=("named_contrasts",),
            repair_path="pass at least one contrast name or omit named_contrasts",
        )
    return requested, (), ()


def _require_complete_provenance(fit: "FmriLm", label: str) -> None:
    provenance = getattr(fit, "provenance", None)
    if provenance is None:
        raise ReplayContractError(
            message=f"{label} is missing FitProvenance",
            weak_fields=(f"{label}.provenance",),
            repair_path="recompute the fit with fmri_lm so replay provenance is populated",
        )
    try:
        provenance.require_complete()
    except Exception as exc:
        raise ReplayContractError(
            message=f"{label} has incomplete FitProvenance: {exc}",
            weak_fields=(f"{label}.provenance",),
            repair_path="recompute the fit with complete seed, AR, and mask provenance",
        ) from exc


def replay_fits(
    fit_a: "FmriLm",
    fit_b: "FmriLm",
    *,
    named_contrasts: Sequence[str] | None = None,
) -> ReplayResult:
    """Compare already-fitted GLM results without recomputing them."""

    _require_complete_provenance(fit_a, "fit_a")
    _require_complete_provenance(fit_b, "fit_b")
    selected, dropped_from_a, dropped_from_b = _selected_contrasts(
        fit_a,
        fit_b,
        named_contrasts,
    )
    deltas = tuple(_contrast_delta(name, fit_a, fit_b) for name in selected)
    spec_a = getattr(getattr(fit_a.model, "event_model", None), "spec", None)
    spec_b = getattr(getattr(fit_b.model, "event_model", None), "spec", None)
    diff = SpecDiff() if spec_a is None or spec_b is None else spec_diff(spec_a, spec_b)
    return ReplayResult(
        diff=diff,
        fit_a=fit_a,
        fit_b=fit_b,
        contrast_deltas=deltas,
        dropped_from_a=dropped_from_a,
        dropped_from_b=dropped_from_b,
    )


def replay(
    spec_a: "Spec | Term | str | Sequence[object]",
    spec_b: "Spec | Term | str | Sequence[object]",
    dataset: "FmriDataset",
    *,
    config: "FmriLmConfig | None" = None,
    named_contrasts: Sequence[str] | None = None,
    **fit_kwargs: object,
) -> ReplayResult:
    """Fit two specs on one dataset and compare their named contrasts."""

    from fmrimod.glm.fmri_lm import fmri_lm

    fit_a = fmri_lm(spec_a, dataset, config=config, **fit_kwargs)
    fit_b = fmri_lm(spec_b, dataset, config=config, **fit_kwargs)
    return replay_fits(fit_a, fit_b, named_contrasts=named_contrasts)
