"""Small generic comparator for fmrimod-vs-reference parity cases."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats


Array = NDArray[np.float64]


@dataclass(frozen=True)
class ParityTolerance:
    """Tolerance envelope for one output array."""

    rtol: float = 1e-6
    atol: float = 1e-9
    min_pearson: float = 0.999
    min_spearman: float = 0.999
    check_allclose: bool = True
    allow_rescale: bool = False
    max_mae: float | None = None
    max_abs: float | None = None


@dataclass(frozen=True)
class Caveat:
    """Declared, reviewable divergence from the reference pipeline."""

    caveat_id: str
    quantity: str
    reason: str
    expected: str
    link: str


@dataclass(frozen=True)
class ColumnMap:
    """Column alignment between candidate and reference design matrices."""

    candidate: str
    reference: str
    caveat_id: str | None = None


@dataclass(frozen=True)
class PipelineOutput:
    """Named arrays and metadata emitted by a parity pipeline."""

    arrays: Mapping[str, Array]
    columns: Sequence[ColumnMap] = field(default_factory=tuple)
    caveats: Sequence[Caveat] = field(default_factory=tuple)


@dataclass(frozen=True)
class ArrayDelta:
    """Comparison summary for one matched output array."""

    name: str
    shape: tuple[int, ...]
    scale: float
    max_abs: float
    mae: float
    pearson_r: float
    spearman_rho: float
    rank_candidate: int
    rank_reference: int
    tolerance: ParityTolerance
    passes: bool


@dataclass(frozen=True)
class ParityCase:
    """Candidate/reference workflow pair plus case-specific tolerances."""

    name: str
    fmrimod_pipeline: Callable[[Any], PipelineOutput]
    reference_pipeline: Callable[[Any], PipelineOutput]
    inputs: Any
    tolerances: Mapping[str, ParityTolerance] = field(default_factory=dict)
    default_tolerance: ParityTolerance = field(default_factory=ParityTolerance)
    declared_caveats: Sequence[Caveat] = field(default_factory=tuple)


@dataclass(frozen=True)
class ParityResult:
    """Full result for one parity case."""

    name: str
    deltas: Mapping[str, ArrayDelta]
    column_alignment: Sequence[ColumnMap]
    caveats: Sequence[Caveat]
    status: str


def _finite_pair(candidate: Array, reference: Array) -> tuple[Array, Array]:
    cand = np.asarray(candidate, dtype=np.float64).ravel()
    ref = np.asarray(reference, dtype=np.float64).ravel()
    mask = np.isfinite(cand) & np.isfinite(ref)
    return cand[mask], ref[mask]


def _safe_pearson(candidate: Array, reference: Array) -> float:
    cand, ref = _finite_pair(candidate, reference)
    if cand.size < 2:
        return 1.0
    cand_std = float(np.std(cand))
    ref_std = float(np.std(ref))
    if cand_std == 0.0 and ref_std == 0.0:
        return 1.0 if np.allclose(cand, ref) else 0.0
    if cand_std == 0.0 or ref_std == 0.0:
        return 0.0
    return float(np.corrcoef(cand, ref)[0, 1])


def _safe_spearman(candidate: Array, reference: Array) -> float:
    cand, ref = _finite_pair(candidate, reference)
    if cand.size < 2:
        return 1.0
    if np.std(cand) == 0.0 and np.std(ref) == 0.0:
        return 1.0 if np.allclose(cand, ref) else 0.0
    rho = sp_stats.spearmanr(cand, ref, nan_policy="omit").correlation
    return 0.0 if not np.isfinite(rho) else float(rho)


def _rank(array: Array) -> int:
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim <= 1:
        return 1
    return int(np.linalg.matrix_rank(arr))


def compare_array(
    name: str,
    candidate: Array,
    reference: Array,
    tolerance: ParityTolerance,
) -> ArrayDelta:
    """Compare one named candidate/reference output pair."""

    cand = np.asarray(candidate, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if cand.shape != ref.shape:
        return ArrayDelta(
            name=name,
            shape=tuple(cand.shape),
            scale=1.0,
            max_abs=float("inf"),
            mae=float("inf"),
            pearson_r=0.0,
            spearman_rho=0.0,
            rank_candidate=_rank(cand),
            rank_reference=_rank(ref),
            tolerance=tolerance,
            passes=False,
        )

    scale = 1.0
    cand_aligned = cand
    if tolerance.allow_rescale:
        cand_vec, ref_vec = _finite_pair(cand, ref)
        denom = float(cand_vec @ cand_vec)
        if denom > 0.0:
            scale = float((cand_vec @ ref_vec) / denom)
            cand_aligned = cand * scale

    diff = np.abs(cand_aligned - ref)
    pearson = _safe_pearson(cand, ref)
    spearman = _safe_spearman(cand, ref)
    allclose = bool(
        np.allclose(cand_aligned, ref, rtol=tolerance.rtol, atol=tolerance.atol)
    )
    mae = float(np.mean(diff)) if diff.size else 0.0
    max_abs = float(np.max(diff)) if diff.size else 0.0
    passes = (
        (allclose or not tolerance.check_allclose)
        and pearson >= tolerance.min_pearson
        and spearman >= tolerance.min_spearman
        and (tolerance.max_mae is None or mae <= tolerance.max_mae)
        and (tolerance.max_abs is None or max_abs <= tolerance.max_abs)
    )

    return ArrayDelta(
        name=name,
        shape=tuple(cand.shape),
        scale=scale,
        max_abs=max_abs,
        mae=mae,
        pearson_r=pearson,
        spearman_rho=spearman,
        rank_candidate=_rank(cand),
        rank_reference=_rank(ref),
        tolerance=tolerance,
        passes=passes,
    )


def run(case: ParityCase) -> ParityResult:
    """Run one parity case and return array deltas plus declared caveats."""

    candidate = case.fmrimod_pipeline(case.inputs)
    reference = case.reference_pipeline(case.inputs)
    common_names = sorted(set(candidate.arrays).intersection(reference.arrays))
    missing = sorted(set(candidate.arrays).symmetric_difference(reference.arrays))
    if missing:
        raise ValueError(f"Unmatched output names for {case.name}: {missing}")

    deltas = {}
    for name in common_names:
        tolerance = case.tolerances.get(name, case.default_tolerance)
        deltas[name] = compare_array(
            name,
            candidate.arrays[name],
            reference.arrays[name],
            tolerance,
        )

    caveats = tuple(case.declared_caveats) + tuple(candidate.caveats) + tuple(
        reference.caveats
    )
    status = "pass" if all(delta.passes for delta in deltas.values()) else "fail"
    if status == "pass" and caveats:
        status = "pass_with_caveats"

    return ParityResult(
        name=case.name,
        deltas=deltas,
        column_alignment=tuple(candidate.columns) or tuple(reference.columns),
        caveats=caveats,
        status=status,
    )
