"""Baseline model implementation for fMRI design matrices.

This module implements baseline models that account for drift, block-wise
intercepts, and nuisance regressors in fMRI time series.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Optional, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from patsy import dmatrix
from scipy.interpolate import BSpline as ScipyBSpline

from ..types import Array, SamplingInfo
from .baseline_term import BaselineTerm

BaselineBasis = Literal['constant', 'poly', 'bs', 'ns']
BaselineIntercept = Literal['runwise', 'global', 'none']
BaselineTermRole = Literal['drift', 'block', 'nuisance']
NuisanceCheckMode = Literal['warn', 'error', 'drop', 'none']

BASELINE_TERM_ORDER: tuple[BaselineTermRole, ...] = ('drift', 'block', 'nuisance')

_BASELINE_BASIS: tuple[BaselineBasis, ...] = ('constant', 'poly', 'bs', 'ns')
_BASELINE_INTERCEPTS: tuple[BaselineIntercept, ...] = ('runwise', 'global', 'none')
_NUISANCE_CHECK_MODES: tuple[NuisanceCheckMode, ...] = (
    'warn',
    'error',
    'drop',
    'none',
)
BasisFunction = Callable[..., NDArray[np.float64]]


def _normalize_basis(basis: str) -> BaselineBasis:
    if basis in _BASELINE_BASIS:
        return cast(BaselineBasis, basis)
    raise ValueError(f"Invalid basis: {basis}")


def _normalize_intercept(intercept: str) -> BaselineIntercept:
    if intercept in _BASELINE_INTERCEPTS:
        return cast(BaselineIntercept, intercept)
    raise ValueError(f"Invalid intercept: {intercept}")


def _normalize_nuisance_check(mode: str) -> NuisanceCheckMode:
    if mode in _NUISANCE_CHECK_MODES:
        return cast(NuisanceCheckMode, mode)
    allowed = ", ".join(_NUISANCE_CHECK_MODES)
    raise ValueError(f"Invalid nuisance_check: {mode!r}; expected one of {allowed}")


@dataclass(frozen=True)
class DuplicatePair:
    """Near-duplicate nuisance-column pair for one block."""

    column: str
    duplicates: str
    correlation: float


@dataclass(frozen=True)
class NuisanceBlockCheck:
    """Per-block nuisance diagnostics."""

    block: int
    baseline_rank: int
    baseline_columns: int
    nuisance_rank: int
    nuisance_columns: int
    rank_with_baseline: int
    columns_with_baseline: int
    non_finite: tuple[str, ...]
    zero_variance: tuple[str, ...]
    duplicate_pairs: tuple[DuplicatePair, ...]
    aliased_columns: tuple[str, ...]
    keep: tuple[bool, ...]


@dataclass(frozen=True)
class NuisanceCheck:
    """Audit report returned by :func:`check_nuisance`."""

    ok: bool
    problems: pd.DataFrame
    by_block: tuple[NuisanceBlockCheck, ...]
    nuisance_list: tuple[pd.DataFrame, ...]


@dataclass(frozen=True)
class CleanedNuisance:
    """Cleaned nuisance matrices and the report used to clean them."""

    nuisance_list: tuple[pd.DataFrame, ...]
    report: NuisanceCheck


@dataclass(frozen=True)
class BaselineModel:
    """Baseline model for fMRI time series.
    
    A baseline model represents low-frequency drift, block-wise intercepts,
    and nuisance regressors that are typically included in fMRI design matrices
    to account for non-neural signal variations.
    
    Parameters
    ----------
    terms : mapping
        Baseline terms keyed by role (drift, block, nuisance)
    drift_spec : BaselineSpec
        Specification for drift modeling
    sampling_frame : SamplingInfo
        Sampling information
    
    Attributes
    ----------
    terms : mapping
        Baseline terms in canonical matrix order
    drift_spec : BaselineSpec
        Drift specification
    sampling_frame : SamplingInfo
        Sampling frame
    
    Examples
    --------
    >>> # Create baseline model with polynomial drift
    >>> bmodel = baseline_model(
    ...     basis='poly',
    ...     degree=3,
    ...     sframe=sampling_frame,
    ...     intercept='runwise'
    ... )
    >>> 
    >>> # Get design matrix
    >>> X_baseline = design_matrix(bmodel)
    """

    terms: Mapping[BaselineTermRole, BaselineTerm | None]
    drift_spec: BaselineSpec
    sampling_frame: SamplingInfo
    nuisance_check: NuisanceCheck | None = None

    def __post_init__(self) -> None:
        """Normalize the term map to the closed baseline role set."""
        normalized = {
            role: self.terms.get(role)
            for role in BASELINE_TERM_ORDER
        }
        object.__setattr__(self, 'terms', MappingProxyType(normalized))
    
    @property
    def design_matrix(self) -> Array:
        """Get full design matrix by combining all terms."""
        matrices: list[Array] = []
        
        for term_name in BASELINE_TERM_ORDER:
            term = self.terms.get(term_name)
            if term is None:
                continue
            mat = term.design_matrix
            if isinstance(mat, pd.DataFrame):
                mat = mat.values
            matrices.append(mat)
        
        if matrices:
            return np.hstack(matrices)
        else:
            # Return empty matrix with correct number of rows
            n_rows = sum(_get_block_lengths(self.sampling_frame))
            return np.zeros((n_rows, 0))

    @property
    def column_names(self) -> list[str]:
        """Return baseline design-matrix column names in matrix order."""
        names: list[str] = []
        for term_name in BASELINE_TERM_ORDER:
            term = self.terms.get(term_name)
            if term is None:
                continue
            mat = term.design_matrix
            if isinstance(mat, pd.DataFrame):
                names.extend(list(mat.columns))
            else:
                n_cols = mat.shape[1] if mat.ndim > 1 else 1
                names.extend([f"{term_name}_{i + 1}" for i in range(n_cols)])
        return names

    @property
    def colnames(self) -> list[str]:
        """Compatibility alias used by generic column extractors."""
        return self.column_names
    
    def __repr__(self) -> str:
        """Rich string representation matching R's print.baseline_model()."""
        lines = ["BaselineModel"]
        lines.append(f"  Basis: {self.drift_spec.basis}, Degree: {self.drift_spec.degree}")
        lines.append(f"  Intercept: {self.drift_spec.intercept}")

        for term_name in BASELINE_TERM_ORDER:
            term = self.terms.get(term_name)
            if term is None:
                continue
            mat = term.design_matrix
            if isinstance(mat, pd.DataFrame):
                cols = list(mat.columns)
            else:
                cols = [f"col_{i}" for i in range(mat.shape[1] if mat.ndim > 1 else 1)]
            n_cols = len(cols)
            preview = ", ".join(cols[:4])
            if n_cols > 4:
                preview += f", ... ({n_cols - 4} more)"
            lines.append(f"  {term_name.capitalize()}: {n_cols} columns [{preview}]")

        dm = self.design_matrix
        lines.append(f"  Design matrix: {dm.shape[0]} x {dm.shape[1]}")

        return "\n".join(lines)


@dataclass(frozen=True)
class BaselineSpec:
    """Specification for baseline drift modeling.
    
    Parameters
    ----------
    degree : int
        Number of basis terms per block
    basis : str
        Type of basis ('constant', 'poly', 'bs', 'ns')
    intercept : str
        Type of intercept ('runwise', 'global', 'none')
    name : str, optional
        Name for the term
    """

    degree: int = 1
    basis: BaselineBasis = 'constant'
    intercept: BaselineIntercept = 'runwise'
    name: str | None = None
    fun: BasisFunction = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate and canonicalize the baseline drift contract."""
        basis = _normalize_basis(self.basis)
        intercept = _normalize_intercept(self.intercept)
        degree = int(self.degree)

        if degree < 1:
            raise ValueError("degree must be >= 1")
        if basis in ('bs', 'ns') and degree < 3:
            raise ValueError(f"'{basis}' basis must have degree >= 3")

        effective_degree = 1 if basis == 'constant' else degree
        name = self.name if self.name is not None else f"baseline_{basis}_{degree}"

        object.__setattr__(self, 'basis', basis)
        object.__setattr__(self, 'intercept', intercept)
        object.__setattr__(self, 'degree', effective_degree)
        object.__setattr__(self, 'name', name)
        object.__setattr__(self, 'fun', self._get_basis_function())

    def _get_basis_function(self) -> BasisFunction:
        """Get the appropriate basis function."""
        if self.basis == 'constant':
            return lambda x, degree=None: np.ones((len(x), 1))
        elif self.basis == 'poly':
            return self._poly_basis
        elif self.basis == 'bs':
            return self._bs_basis
        elif self.basis == 'ns':
            return self._ns_basis
        else:
            raise ValueError(f"Unknown basis type: {self.basis}")
    
    def _poly_basis(self, x: Array, degree: int) -> NDArray[np.float64]:
        """Polynomial basis function."""
        x = np.asarray(x, dtype=float)
        # Create orthogonal polynomial basis.
        # This mirrors the behavior of `stats::poly` by centering each raw
        # power before orthogonalization.
        n = len(x)
        if n == 0:
            return np.empty((n, degree), dtype=float)
        X = np.zeros((n, degree))

        x = x.ravel()
        # Generate polynomial terms
        for i in range(degree):
            power = x ** (i + 1)
            X[:, i] = power - np.mean(power)
        
        # Orthogonalize using QR decomposition
        Q, R = np.linalg.qr(X)
        # Preserve a deterministic sign convention (match reference behavior
        # and keep column orientation stable across runs).
        signs = np.sign(np.diag(R))
        signs[signs == 0] = 1
        Q = Q * signs
        return np.asarray(Q, dtype=np.float64)
    
    def _bs_basis(self, x: Array, degree: int) -> NDArray[np.float64]:
        """B-spline basis function."""
        x = np.asarray(x)
        # Align with R's `splines::bs` usage in fmridesign's baseline
        # specification, where `degree` is the only explicit argument.
        #
        # Patsy requires either `df` or `knots` when using `bs()`, while
        # R's `bs(x, degree = d)` with no explicit knots produces one basis
        # function per degree without interior knots. Passing an empty knots list
        # reproduces that behavior while preserving exact parity.
        return np.asarray(
            dmatrix(
                "0 + bs(x, degree=%d, knots=[], include_intercept=False)"
                % int(degree),
                {"x": x},
            ),
            dtype=np.float64,
        )

    def _ns_basis(self, x: Array, df: int) -> NDArray[np.float64]:
        """Natural spline basis function."""
        x = np.asarray(x)
        if x.ndim != 1:
            x = x.ravel()

        x = x.astype(float)
        n = len(x)

        if n == 0:
            return np.empty((0, df), dtype=float)

        degree = 3

        # Mirroring `ns(x, df=..., intercept=False)` knot logic.
        x_range = [np.min(x), np.max(x)]
        x_no_na = x[~np.isnan(x)]
        if x_no_na.size == 0:
            return np.full((n, df), np.nan, dtype=float)

        n_interior = df - 1
        if n_interior > 0:
            probs = np.linspace(0.0, 1.0, n_interior + 2)[1:-1]
            knots = np.quantile(x_no_na, probs)
        else:
            knots = np.array([], dtype=float)

        # Create cubic B-spline design on extended knots (mirrors R's Aknots).
        a_knots = np.concatenate(
            [
                np.repeat(x_range[0], degree + 1),
                knots,
                np.repeat(x_range[1], degree + 1),
            ]
        )
        n_basis = len(a_knots) - (degree + 1)
        if n_basis <= 2:
            return np.zeros((n, max(df, 0)), dtype=float)

        basis = np.zeros((n, n_basis), dtype=float)
        basis_deriv2 = np.zeros((2, n_basis), dtype=float)
        boundaries = np.array(x_range, dtype=float)

        for i in range(n_basis):
            coef = np.zeros(n_basis, dtype=float)
            coef[i] = 1.0
            spline = ScipyBSpline(a_knots, coef, degree)
            basis[:, i] = spline(x)

            deriv2 = spline.derivative(2)
            basis_deriv2[:, i] = deriv2(boundaries)

        # Remove intercept column for `intercept=False`, as R does.
        basis = basis[:, 1:]
        basis_deriv2 = basis_deriv2[:, 1:]

        # Enforce natural boundary constraints through the same R projection:
        # QR-based null-space transformation and dropping first two constrained
        # coordinates.
        q_const, _ = np.linalg.qr(basis_deriv2.T, mode="complete")
        projected = basis @ q_const[:, 2:]

        if projected.shape[1] != df:
            if projected.shape[1] < df:
                projected = np.pad(
                    projected, ((0, 0), (0, df - projected.shape[1]))
                )
            else:
                projected = projected[:, :df]

        return np.asarray(projected, dtype=np.float64)


def _get_block_lengths(sframe: SamplingInfo) -> list[int]:
    """Return per-block scan counts from a sampling frame."""
    if hasattr(sframe, 'blocklens'):
        return [int(x) for x in sframe.blocklens]
    if hasattr(sframe, 'n_scans'):
        return [int(sframe.n_scans)]
    raise ValueError("Cannot determine block lengths from sampling frame")


def _unique_nuisance_names(names: Sequence[object], n_cols: int) -> list[str]:
    """Normalize nuisance column names like fmridesign's make.unique path."""
    normalized = []
    seen: dict[str, int] = {}
    if len(names) != n_cols:
        names = [f"V{i + 1}" for i in range(n_cols)]

    for i, raw in enumerate(names):
        name = "" if raw is None else str(raw)
        if name == "" or name.lower() == "nan":
            name = f"V{i + 1}"
        count = seen.get(name, 0)
        seen[name] = count + 1
        normalized.append(f"{name}_{count}" if count else name)
    return normalized


def _as_nuisance_matrices(
    nuisance_list: Sequence[Array | pd.DataFrame],
    sframe: SamplingInfo,
) -> list[pd.DataFrame]:
    """Validate nuisance input and return numeric per-block DataFrames."""
    if isinstance(nuisance_list, (str, bytes)) or not isinstance(
        nuisance_list, Sequence
    ):
        raise ValueError("nuisance_list must be a sequence")

    block_lens = _get_block_lengths(sframe)
    if len(nuisance_list) != len(block_lens):
        raise ValueError(
            f"Length of nuisance_list ({len(nuisance_list)}) must equal "
            f"number of blocks ({len(block_lens)})"
        )

    matrices: list[pd.DataFrame] = []
    for i, (nuisance, block_len) in enumerate(zip(nuisance_list, block_lens)):
        if isinstance(nuisance, pd.DataFrame):
            if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in nuisance.dtypes):
                raise ValueError(
                    f"nuisance_list[{i}] must contain only numeric columns"
                )
            mat = nuisance.astype(float).copy()
            mat.columns = _unique_nuisance_names(list(mat.columns), mat.shape[1])
        else:
            arr = np.asarray(nuisance)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            if arr.ndim != 2:
                raise ValueError(
                    f"nuisance_list[{i}] must be a 1-D or 2-D numeric matrix"
                )
            if not np.issubdtype(arr.dtype, np.number) and arr.dtype != bool:
                raise ValueError(
                    f"nuisance_list[{i}] must contain only numeric columns"
                )
            arr = arr.astype(float, copy=False)
            mat = pd.DataFrame(
                arr,
                columns=_unique_nuisance_names([], arr.shape[1]),
            )

        if mat.shape[0] != block_len:
            raise ValueError(
                f"Nuisance matrix {i} has {mat.shape[0]} rows but "
                f"block has {block_len} timepoints"
            )
        matrices.append(mat)

    return matrices


def _qr_rank(matrix: Array, tol: float) -> int:
    """QR/SVD rank helper for nuisance diagnostics."""
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[1] == 0:
        return 0
    return int(np.linalg.matrix_rank(arr, tol=tol))


def _zero_variance_columns(
    mat: NDArray[np.float64],
    tol: float,
) -> NDArray[np.bool_]:
    """Boolean mask of columns with no within-run variance."""
    flags = []
    for j in range(mat.shape[1]):
        col = mat[:, j]
        if np.any(~np.isfinite(col)):
            flags.append(False)
            continue
        scale = max(1.0, float(np.max(np.abs(col)))) if col.size else 1.0
        flags.append(float(np.ptp(col)) <= tol * scale)
    return np.asarray(flags, dtype=bool)


def _duplicate_pairs(
    mat: NDArray[np.float64],
    names: Sequence[str],
    zero_variance: NDArray[np.bool_],
    non_finite: NDArray[np.bool_],
    duplicate_threshold: float,
) -> tuple[DuplicatePair, ...]:
    """Return near-duplicate nuisance-column pairs."""
    keep = np.where(~zero_variance & ~non_finite)[0]
    if len(keep) < 2:
        return ()

    with np.errstate(invalid='ignore', divide='ignore'):
        corr = np.corrcoef(mat[:, keep], rowvar=False)
    corr = np.atleast_2d(corr)
    pairs: list[DuplicatePair] = []
    for left_pos in range(len(keep)):
        for right_pos in range(left_pos + 1, len(keep)):
            r = float(corr[left_pos, right_pos])
            if np.isfinite(r) and abs(r) >= duplicate_threshold:
                pairs.append(
                    DuplicatePair(
                        column=str(names[keep[right_pos]]),
                        duplicates=str(names[keep[left_pos]]),
                        correlation=r,
                    )
                )
    return tuple(pairs)


def _incremental_rank_keep(
    mat: NDArray[np.float64],
    baseline_mat: NDArray[np.float64],
    zero_variance: NDArray[np.bool_],
    non_finite: NDArray[np.bool_],
    tol: float,
) -> NDArray[np.bool_]:
    """Keep nuisance columns that increase rank in column order."""
    keep = np.ones(mat.shape[1], dtype=bool)
    current = baseline_mat
    current_rank = _qr_rank(current, tol)

    for j in range(mat.shape[1]):
        if zero_variance[j] or non_finite[j]:
            keep[j] = False
            continue
        candidate = np.column_stack([current, mat[:, j]])
        candidate_rank = _qr_rank(candidate, tol)
        if candidate_rank > current_rank:
            current = candidate
            current_rank = candidate_rank
        else:
            keep[j] = False
    return keep


def _block_rows(sframe: SamplingInfo) -> list[NDArray[np.int_]]:
    """Return zero-indexed row indices for each block."""
    rows = []
    start = 0
    for block_len in _get_block_lengths(sframe):
        stop = start + block_len
        rows.append(np.arange(start, stop))
        start = stop
    return rows


def _baseline_matrix_for_block(
    baseline_terms: Mapping[BaselineTermRole, BaselineTerm | None],
    rows: NDArray[np.int_],
    tol: float,
) -> NDArray[np.float64]:
    """Extract active baseline columns for one block."""
    matrices = []
    for term in baseline_terms.values():
        if term is None:
            continue
        mat = term.design_matrix
        arr = mat.to_numpy() if isinstance(mat, pd.DataFrame) else np.asarray(mat)
        block = arr[rows, :]
        active = np.array(
            [
                np.any(np.isfinite(block[:, j]) & (np.abs(block[:, j]) > tol))
                for j in range(block.shape[1])
            ],
            dtype=bool,
        )
        if np.any(active):
            matrices.append(block[:, active])
    if not matrices:
        return np.zeros((len(rows), 0), dtype=float)
    return np.column_stack(matrices)


def _problem_row(
    block: int,
    issue: str,
    columns: Sequence[str],
    detail: str,
) -> dict[str, object]:
    """Build one nuisance problem row."""
    return {
        'block': block,
        'issue': issue,
        'columns': ", ".join(str(c) for c in columns),
        'detail': detail,
    }


def _check_nuisance_internal(
    nuisance_list: Sequence[Array | pd.DataFrame],
    sframe: SamplingInfo,
    baseline_terms: Mapping[BaselineTermRole, BaselineTerm | None],
    tol: float,
    duplicate_threshold: float,
) -> NuisanceCheck:
    """Shared nuisance-check implementation."""
    mats = _as_nuisance_matrices(nuisance_list, sframe)
    rows_by_block = _block_rows(sframe)
    by_block: list[NuisanceBlockCheck] = []
    problems: list[dict[str, object]] = []

    for i, mat_df in enumerate(mats):
        names = tuple(str(c) for c in mat_df.columns)
        mat = mat_df.to_numpy(dtype=float)
        baseline_mat = _baseline_matrix_for_block(baseline_terms, rows_by_block[i], tol)

        non_finite = np.any(~np.isfinite(mat), axis=0)
        finite_mat = mat.copy()
        finite_mat[~np.isfinite(finite_mat)] = 0.0
        zero_variance = _zero_variance_columns(finite_mat, tol)
        dup_pairs = _duplicate_pairs(
            finite_mat, names, zero_variance, non_finite, duplicate_threshold
        )

        nuisance_rank = _qr_rank(finite_mat, tol)
        baseline_rank = _qr_rank(baseline_mat, tol)
        combined = np.column_stack([baseline_mat, finite_mat])
        rank_with_baseline = _qr_rank(combined, tol)
        n_with_baseline = baseline_mat.shape[1] + finite_mat.shape[1]
        keep = _incremental_rank_keep(
            finite_mat, baseline_mat, zero_variance, non_finite, tol
        )
        aliased = tuple(
            names[j]
            for j in range(len(names))
            if not keep[j] and not zero_variance[j] and not non_finite[j]
        )

        if np.any(non_finite):
            cols = [names[j] for j in np.where(non_finite)[0]]
            problems.append(
                _problem_row(
                    i + 1,
                    'non_finite',
                    cols,
                    'contains NA, NaN, or infinite values',
                )
            )
        if np.any(zero_variance):
            cols = [names[j] for j in np.where(zero_variance)[0]]
            problems.append(
                _problem_row(
                    i + 1,
                    'zero_variance',
                    cols,
                    'column has no within-run variance',
                )
            )
        if dup_pairs:
            detail = "; ".join(
                f"{pair.column} duplicates {pair.duplicates} (r = {pair.correlation:.6g})"
                for pair in dup_pairs
            )
            problems.append(
                _problem_row(
                    i + 1,
                    'duplicate',
                    [pair.column for pair in dup_pairs],
                    detail,
                )
            )
        if nuisance_rank < finite_mat.shape[1]:
            problems.append(
                _problem_row(
                    i + 1,
                    'rank_deficient_nuisance',
                    names,
                    f"nuisance rank: {nuisance_rank} < {finite_mat.shape[1]} columns",
                )
            )
        if rank_with_baseline < n_with_baseline:
            problems.append(
                _problem_row(
                    i + 1,
                    'rank_deficient_with_baseline',
                    names,
                    (
                        f"rank with baseline terms: {rank_with_baseline} "
                        f"< {n_with_baseline} columns"
                    ),
                )
            )
        if aliased:
            problems.append(
                _problem_row(
                    i + 1,
                    'aliased_columns',
                    aliased,
                    (
                        'columns do not increase rank after baseline and earlier '
                        'nuisance columns'
                    ),
                )
            )

        by_block.append(
            NuisanceBlockCheck(
                block=i + 1,
                baseline_rank=baseline_rank,
                baseline_columns=baseline_mat.shape[1],
                nuisance_rank=nuisance_rank,
                nuisance_columns=finite_mat.shape[1],
                rank_with_baseline=rank_with_baseline,
                columns_with_baseline=n_with_baseline,
                non_finite=tuple(names[j] for j in np.where(non_finite)[0]),
                zero_variance=tuple(names[j] for j in np.where(zero_variance)[0]),
                duplicate_pairs=dup_pairs,
                aliased_columns=aliased,
                keep=tuple(bool(x) for x in keep),
            )
        )

    problems_df = pd.DataFrame(
        problems,
        columns=['block', 'issue', 'columns', 'detail'],
    )
    return NuisanceCheck(
        ok=problems_df.empty,
        problems=problems_df,
        by_block=tuple(by_block),
        nuisance_list=tuple(mats),
    )


def _format_nuisance_check(report: NuisanceCheck, dropped: bool = False) -> str:
    """Format nuisance diagnostics for warnings/errors."""
    if report.ok:
        return "baseline_model(): nuisance_list passed validation."

    lines = ["baseline_model(): nuisance_list has column/rank problems."]
    for block in report.by_block:
        block_lines = []
        if block.non_finite:
            block_lines.append(
                f"  Non-finite columns: {', '.join(block.non_finite)}."
            )
        if block.zero_variance:
            block_lines.append(
                f"  Zero-variance columns: {', '.join(block.zero_variance)}."
            )
        if block.duplicate_pairs:
            dup = "; ".join(
                f"{pair.column} duplicates {pair.duplicates} "
                f"(r = {pair.correlation:.6g})"
                for pair in block.duplicate_pairs
            )
            block_lines.append(f"  Duplicate or near-duplicate columns: {dup}.")
        if block.aliased_columns:
            block_lines.append(
                f"  Aliased columns: {', '.join(block.aliased_columns)}."
            )
        if block.rank_with_baseline < block.columns_with_baseline:
            block_lines.append(
                "  Rank with baseline terms: "
                f"{block.rank_with_baseline} < {block.columns_with_baseline} columns."
            )
        if block_lines:
            lines.append(f"nuisance_list[[{block.block}]]:")
            lines.extend(block_lines)

    if dropped:
        lines.append(
            "Dropped non-finite, zero-variance, and rank-aliased nuisance columns."
        )
    else:
        lines.append(
            'Use nuisance_check = "error" to stop or nuisance_check = "drop" '
            'to remove columns that do not increase rank.'
        )
    return "\n".join(lines)


def _drop_nuisance_columns(report: NuisanceCheck) -> tuple[pd.DataFrame, ...]:
    """Drop columns that did not pass the nuisance rank audit."""
    cleaned = []
    for mat, block in zip(report.nuisance_list, report.by_block):
        keep = list(block.keep)
        cleaned.append(mat.loc[:, keep].copy())
    return tuple(cleaned)


def _baseline_terms_for_nuisance_check(
    basis: BaselineBasis,
    degree: int,
    sframe: SamplingInfo,
    intercept: BaselineIntercept,
) -> dict[BaselineTermRole, BaselineTerm | None]:
    """Construct baseline terms used for nuisance-rank comparison."""
    drift_spec = baseline(degree=degree, basis=basis, intercept=intercept)
    drift_term = _construct_drift_term(drift_spec, sframe)
    block_term = (
        _construct_block_term(sframe, intercept)
        if intercept != 'none' and basis != 'constant'
        else None
    )
    return {'drift': drift_term, 'block': block_term}


def check_nuisance(
    nuisance_list: Sequence[Array | pd.DataFrame],
    sframe: SamplingInfo,
    basis: BaselineBasis = 'constant',
    degree: int = 1,
    intercept: BaselineIntercept = 'runwise',
    tol: Optional[float] = None,
    duplicate_threshold: Optional[float] = None,
) -> NuisanceCheck:
    """Check nuisance regressors for rank and column problems."""
    basis = _normalize_basis(basis)
    intercept = _normalize_intercept(intercept)
    if basis in ('bs', 'ns') and degree < 3:
        raise ValueError(f"'{basis}' basis must have degree >= 3")

    tol_value = float(np.sqrt(np.finfo(float).eps) if tol is None else tol)
    threshold = (
        1.0 - tol_value
        if duplicate_threshold is None
        else float(duplicate_threshold)
    )
    baseline_terms = _baseline_terms_for_nuisance_check(
        basis, degree, sframe, intercept
    )
    return _check_nuisance_internal(
        nuisance_list,
        sframe,
        baseline_terms,
        tol=tol_value,
        duplicate_threshold=threshold,
    )


def clean_nuisance(
    nuisance_list: Sequence[Array | pd.DataFrame],
    sframe: SamplingInfo,
    basis: BaselineBasis = 'constant',
    degree: int = 1,
    intercept: BaselineIntercept = 'runwise',
    tol: Optional[float] = None,
    duplicate_threshold: Optional[float] = None,
) -> CleanedNuisance:
    """Drop nuisance columns that do not increase rank."""
    report = check_nuisance(
        nuisance_list,
        sframe,
        basis=basis,
        degree=degree,
        intercept=intercept,
        tol=tol,
        duplicate_threshold=duplicate_threshold,
    )
    return CleanedNuisance(
        nuisance_list=_drop_nuisance_columns(report),
        report=report,
    )


def baseline_model(
    basis: BaselineBasis = 'constant',
    degree: int = 1,
    sframe: SamplingInfo | None = None,
    intercept: BaselineIntercept = 'runwise',
    nuisance_list: Sequence[Array | pd.DataFrame] | None = None,
    nuisance_check: NuisanceCheckMode = 'warn',
) -> BaselineModel:
    """Construct a baseline model for fMRI time series.
    
    Creates a model for low-frequency drift, block-wise intercepts, and
    nuisance regressors that account for non-neural signal variations.
    
    Parameters
    ----------
    basis : {'constant', 'poly', 'bs', 'ns'}, default='constant'
        Type of basis for drift modeling:
        - 'constant': Constant (intercept only)
        - 'poly': Polynomial basis
        - 'bs': B-spline basis
        - 'ns': Natural spline basis
    degree : int, default=1
        Number of basis functions per block (ignored for 'constant')
    sframe : SamplingInfo
        Sampling frame with block structure information
    intercept : {'runwise', 'global', 'none'}, default='runwise'
        Type of intercept to include:
        - 'runwise': Separate intercept per run/block
        - 'global': Single intercept for all runs
        - 'none': No intercept
    nuisance_list : list of arrays, optional
        List of nuisance regressor matrices, one per block
    nuisance_check : {'warn', 'error', 'drop', 'none'}, default='warn'
        How to handle nuisance columns that are non-finite, zero variance,
        duplicate, or rank-aliased with the baseline design.
    
    Returns
    -------
    BaselineModel
        Constructed baseline model
    
    Examples
    --------
    >>> # Polynomial drift with runwise intercepts
    >>> bmodel = baseline_model(
    ...     basis='poly',
    ...     degree=3,
    ...     sframe=sampling_frame
    ... )
    >>> 
    >>> # B-spline drift with global intercept
    >>> bmodel = baseline_model(
    ...     basis='bs',
    ...     degree=4,
    ...     sframe=sampling_frame,
    ...     intercept='global'
    ... )
    >>> 
    >>> # Add nuisance regressors
    >>> nuisance = [np.random.randn(100, 6), np.random.randn(100, 6)]
    >>> bmodel = baseline_model(
    ...     basis='poly',
    ...     degree=2,
    ...     sframe=sampling_frame,
    ...     nuisance_list=nuisance
    ... )
    """
    if sframe is None:
        raise ValueError("sampling_frame (sframe) must be provided")

    basis = _normalize_basis(basis)
    intercept = _normalize_intercept(intercept)
    nuisance_check = _normalize_nuisance_check(nuisance_check)

    # Validate degree for splines
    if basis in ('bs', 'ns') and degree < 3:
        raise ValueError(f"'{basis}' basis must have degree >= 3")
    
    # Create drift specification
    drift_spec = baseline(degree=degree, basis=basis, intercept=intercept)
    
    # Construct terms
    terms: dict[BaselineTermRole, BaselineTerm | None] = {}
    
    # Always create drift term
    terms['drift'] = _construct_drift_term(drift_spec, sframe)
    
    # Create block term if needed
    if intercept != 'none' and basis != 'constant':
        terms['block'] = _construct_block_term(sframe, intercept)
    else:
        terms['block'] = None

    nuisance_report = None
    effective_nuisance_list = nuisance_list
    if nuisance_list is not None and nuisance_check != 'none':
        tol = float(np.sqrt(np.finfo(float).eps))
        nuisance_report = _check_nuisance_internal(
            nuisance_list,
            sframe,
            {'drift': terms['drift'], 'block': terms['block']},
            tol=tol,
            duplicate_threshold=1.0 - tol,
        )
        if not nuisance_report.ok:
            if nuisance_check == 'error':
                raise ValueError(_format_nuisance_check(nuisance_report))
            if nuisance_check == 'drop':
                effective_nuisance_list = list(_drop_nuisance_columns(nuisance_report))
                warnings.warn(
                    _format_nuisance_check(nuisance_report, dropped=True),
                    UserWarning,
                    stacklevel=2,
                )
            else:
                warnings.warn(
                    _format_nuisance_check(nuisance_report),
                    UserWarning,
                    stacklevel=2,
                )
    
    # Create nuisance term if provided
    if effective_nuisance_list is not None:
        terms['nuisance'] = _make_nuisance_term(effective_nuisance_list, sframe)
    else:
        terms['nuisance'] = None
    
    return BaselineModel(terms, drift_spec, sframe, nuisance_check=nuisance_report)


def baseline(
    degree: int = 1,
    basis: BaselineBasis = 'constant',
    name: Optional[str] = None,
    intercept: BaselineIntercept = 'runwise',
) -> BaselineSpec:
    """Create a baseline specification.
    
    Generates a specification for modeling low-frequency drift in fMRI time series.
    
    Parameters
    ----------
    degree : int, default=1
        Number of basis terms per block (ignored for 'constant')
    basis : {'constant', 'poly', 'bs', 'ns'}, default='constant'
        Type of basis function
    name : str, optional
        Name for the term
    intercept : {'runwise', 'global', 'none'}, default='runwise'
        Type of intercept
    
    Returns
    -------
    BaselineSpec
        Baseline specification
    """
    return BaselineSpec(degree=degree, basis=basis, intercept=intercept, name=name)


def _construct_drift_term(spec: BaselineSpec, sframe: SamplingInfo) -> BaselineTerm:
    """Construct drift term from specification.

    Generates a block-diagonal basis matrix where each block's
    columns are produced by the drift basis function evaluated
    on ``1, 2, ..., block_len``.

    Parameters
    ----------
    spec : BaselineSpec
        Drift specification with basis type and degree.
    sframe : SamplingInfo
        Sampling frame providing block lengths.

    Returns
    -------
    BaselineTerm
        Baseline term with block-diagonal drift regressors.
    """
    block_lens = _get_block_lengths(sframe)
    
    # Generate basis for each block
    matrices = []
    col_indices = []
    row_indices = []
    
    current_col = 0
    current_row = 0
    
    for i, block_len in enumerate(block_lens):
        # Generate time points for this block
        t = np.arange(1, block_len + 1)
        
        # Apply basis function
        if spec.basis == 'ns':
            # Natural splines use 'df' parameter
            block_matrix = spec.fun(t, df=spec.degree)
        elif spec.basis in ('poly', 'bs'):
            block_matrix = spec.fun(t, degree=spec.degree)
        else:
            # Constant
            block_matrix = spec.fun(t)
        
        # Ensure 2D
        if block_matrix.ndim == 1:
            block_matrix = block_matrix.reshape(-1, 1)
        
        matrices.append(block_matrix)
        
        # Track indices
        n_cols = block_matrix.shape[1]
        col_indices.append(list(range(current_col, current_col + n_cols)))
        row_indices.append(list(range(current_row, current_row + block_len)))
        
        current_col += n_cols
        current_row += block_len
    
    # Handle global constant intercept
    if spec.basis == 'constant' and spec.intercept == 'global':
        # Combine into single column
        full_matrix = np.vstack(matrices)
        col_names = [f"base_{spec.basis}"]
        col_indices = [[0] for _ in block_lens]
    else:
        # Block diagonal structure
        full_matrix = _block_diagonal(matrices)
        
        # Generate column names
        col_names = []
        for i, mat in enumerate(matrices):
            for j in range(mat.shape[1]):
                col_names.append(f"base_{spec.basis}{j+1}_block_{i+1}")
    
    # Create DataFrame
    df = pd.DataFrame(full_matrix, columns=col_names)
    
    return BaselineTerm(
        varname=cast(str, spec.name),
        design_matrix=df,
        colind=col_indices,
        rowind=row_indices
    )


def _construct_block_term(
    sframe: SamplingInfo,
    intercept: BaselineIntercept,
) -> BaselineTerm:
    """Construct block intercept term.

    Creates either a single global intercept column or separate
    run-wise intercept columns depending on the ``intercept`` mode.

    Parameters
    ----------
    sframe : SamplingInfo
        Sampling frame providing block lengths.
    intercept : {'runwise', 'global'}
        Intercept strategy.

    Returns
    -------
    BaselineTerm
        Baseline term with intercept columns.
    """
    block_lens = _get_block_lengths(sframe)
    
    n_blocks = len(block_lens)
    total_rows = sum(block_lens)
    
    if intercept == 'global':
        # Single intercept column
        matrix = np.ones((total_rows, 1))
        col_names = ['constant_global']
        col_indices = [[0] for _ in range(n_blocks)]
    else:
        if n_blocks == 1:
            matrix = np.ones((total_rows, 1))
            col_names = ['constant_global']
            col_indices = [[0]]
        else:
            # Runwise intercepts
            matrices = []
            for block_len in block_lens:
                matrices.append(np.ones((block_len, 1)))
            
            matrix = _block_diagonal(matrices)
            col_names = [f'constant_{i+1}' for i in range(n_blocks)]
            col_indices = [[i] for i in range(n_blocks)]
    
    # Create row indices
    row_indices = []
    current_row = 0
    for block_len in block_lens:
        row_indices.append(list(range(current_row, current_row + block_len)))
        current_row += block_len
    
    df = pd.DataFrame(matrix, columns=col_names)
    
    return BaselineTerm(
        varname='block',
        design_matrix=df,
        colind=col_indices,
        rowind=row_indices
    )


def _make_nuisance_term(
    nuisance_list: Sequence[Array | pd.DataFrame],
    sframe: SamplingInfo,
) -> BaselineTerm:
    """Create nuisance term from a list of per-block matrices.

    Arranges the per-block nuisance matrices into a block-diagonal
    structure in the full time-series space.

    Parameters
    ----------
    nuisance_list : list of array-like
        One matrix per block. Each matrix has shape
        ``(block_len, n_nuisance_regressors)``.
    sframe : SamplingInfo
        Sampling frame providing block lengths.

    Returns
    -------
    BaselineTerm
        Baseline term with block-diagonal nuisance regressors.

    Raises
    ------
    ValueError
        If the number of nuisance matrices does not match the
        number of blocks, or if row counts are inconsistent.
    """
    block_lens = _get_block_lengths(sframe)
    matrices = [
        mat.to_numpy(dtype=float)
        for mat in _as_nuisance_matrices(nuisance_list, sframe)
    ]
    
    # Create block diagonal matrix
    full_matrix = _block_diagonal(matrices)
    
    # Generate column names and indices
    col_names = []
    col_indices = []
    current_col = 0
    
    for i, mat in enumerate(matrices):
        n_cols = mat.shape[1]
        block_cols = []
        
        for j in range(n_cols):
            col_names.append(f'nuis_run{i+1}_c{j+1}')
            block_cols.append(current_col)
            current_col += 1
        
        col_indices.append(block_cols)
    
    # Generate row indices
    row_indices = []
    current_row = 0
    
    for block_len in block_lens:
        row_indices.append(list(range(current_row, current_row + block_len)))
        current_row += block_len
    
    df = pd.DataFrame(full_matrix, columns=col_names)
    
    return BaselineTerm(
        varname='nuisance',
        design_matrix=df,
        colind=col_indices,
        rowind=row_indices
    )


def _block_diagonal(matrices: Sequence[Array]) -> Array:
    """Create a block-diagonal matrix from a list of matrices.

    Parameters
    ----------
    matrices : list of array-like
        Sub-matrices to place on the diagonal.

    Returns
    -------
    Array
        Block-diagonal matrix with shape
        ``(sum(nrows), sum(ncols))``.
    """
    if not matrices:
        return np.array([])

    # Get dimensions
    n_rows = sum(mat.shape[0] for mat in matrices)
    n_cols = sum(mat.shape[1] for mat in matrices)

    # Create output matrix
    result = np.zeros((n_rows, n_cols))

    # Fill blocks
    row_offset = 0
    col_offset = 0

    for mat in matrices:
        n_r, n_c = mat.shape
        result[row_offset:row_offset+n_r, col_offset:col_offset+n_c] = mat
        row_offset += n_r
        col_offset += n_c

    return result


def dctbasis(
    n: int,
    p: Optional[int] = None,
    const: bool = False,
) -> NDArray[np.float64]:
    """Discrete Cosine Transform basis matrix.

    Creates a DCT basis matrix for modeling low-frequency drift or other
    smooth trends in fMRI time series. The DCT basis provides an orthogonal
    decomposition similar to Fourier basis but using only cosine functions.

    Parameters
    ----------
    n : int
        Number of time points (rows in output matrix)
    p : int, optional
        Number of basis functions (columns in output matrix).
        If None, defaults to n.
    const : bool, default=False
        If True, prepend a constant (DC) column normalized by (1/n)^0.5.

    Returns
    -------
    np.ndarray
        DCT basis matrix of shape (n, p) if const=False, or (n, p+1) if const=True.
        Each column represents one DCT basis function, orthonormalized.

    Notes
    -----
    The DCT basis is defined as:

    .. math::
        X_{m,k} = \\sqrt{\\frac{2}{n}} \\cos\\left(\\frac{(2m-1)k\\pi}{2n}\\right)

    where m = 1, ..., n are the time points and k = 1, ..., p are the basis indices.

    The constant term (when const=True) is:

    .. math::
        X_{m,0} = \\sqrt{\\frac{1}{n}}

    This is faithful to the R fmridesign implementation in basis.R.

    Examples
    --------
    >>> # Create a 100-point DCT basis with 5 functions
    >>> basis = dctbasis(100, p=5)
    >>> basis.shape
    (100, 5)

    >>> # Include constant term
    >>> basis_const = dctbasis(100, p=5, const=True)
    >>> basis_const.shape
    (100, 6)

    >>> # Verify orthonormality
    >>> import numpy as np
    >>> gram = basis.T @ basis
    >>> np.allclose(gram, np.eye(5))
    True
    """
    if p is None:
        p = n

    m = np.arange(1, n + 1)
    X = np.zeros((n, p))

    for k in range(1, p + 1):
        X[:, k - 1] = (2 / n) ** 0.5 * np.cos(((2 * m - 1) * k * np.pi) / (2 * n))

    if const:
        const_col = np.full((n, 1), (1 / n) ** 0.5)
        X = np.hstack([const_col, X])

    return X
