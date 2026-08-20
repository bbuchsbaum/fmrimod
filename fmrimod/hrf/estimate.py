"""Condition-level smooth FIR HRF estimation.

Ports fmrireg 0.2.0 ``estimate_hrf()``: one shared penalized multiresponse
solve after residualizing baseline/fixed nuisance, with GCV for a single
smoothing parameter. This is not the voxel-aggregate ``lstsq`` helper in
``fmrimod.single.voxel_hrf``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from patsy import bs as patsy_bs
from scipy import linalg
from scipy import stats as sp_stats

from .core import HRF, as_hrf

HrfBasisName = Literal["bspline", "tent"]
LamSpec = float | int | Literal["gcv"]


def _is_2d_numeric_basis(basis: object) -> bool:
    if basis is None or isinstance(basis, (str, bytes, HRF)):
        return False
    try:
        arr = np.asarray(basis)
    except (TypeError, ValueError):
        return False
    return arr.ndim == 2 and arr.size > 0 and np.issubdtype(arr.dtype, np.number)


def _as_design(obj: object) -> NDArray[np.float64]:
    dm = obj.design_matrix
    if callable(dm):
        dm = dm()
    return np.asarray(getattr(dm, "values", dm), dtype=np.float64)


@dataclass(frozen=True)
class HrfBasisSpec:
    """Interior B-spline / tent specification matching R ``.new_hrf_basis_spec``."""

    type: HrfBasisName
    k: int
    span: float
    degree: int
    knots: NDArray[np.float64]


def new_hrf_basis_spec(basis: HrfBasisName, k: int, span: float) -> HrfBasisSpec:
    degree = 3 if basis == "bspline" else 1
    full_df = k + 2
    n_internal = full_df - degree - 1
    if n_internal > 0:
        knots = np.linspace(0.0, float(span), n_internal + 2)[1:-1]
    else:
        knots = np.asarray([], dtype=np.float64)
    return HrfBasisSpec(type=basis, k=k, span=float(span), degree=degree, knots=knots)


def evaluate_hrf_basis(
    time: NDArray[np.float64], spec: HrfBasisSpec
) -> NDArray[np.float64]:
    """Evaluate the free (endpoint-zero) estimation basis at ``time``.

    Mirrors R ``.evaluate_hrf_basis``: ``splines::bs(..., intercept=TRUE)``
    with the first and last columns dropped so the curve is zero at 0 and
    ``span``.
    """
    time = np.asarray(time, dtype=np.float64).reshape(-1)
    out = np.zeros((time.size, spec.k), dtype=np.float64)
    inside = np.isfinite(time) & (time >= 0.0) & (time <= spec.span)
    if not np.any(inside):
        return out
    full = np.asarray(
        patsy_bs(
            time[inside],
            knots=spec.knots.tolist() if spec.knots.size else None,
            degree=spec.degree,
            include_intercept=True,
            lower_bound=0.0,
            upper_bound=float(spec.span),
        ),
        dtype=np.float64,
    )
    if full.shape[1] != spec.k + 2:
        raise RuntimeError(
            f"Internal HRF basis dimension mismatch: got {full.shape[1]} "
            f"columns, expected {spec.k + 2}"
        )
    out[inside, :] = full[:, 1:-1]
    return out


def _as_estimation_hrf(spec: HrfBasisSpec) -> HRF:
    def _eval(t: object) -> NDArray[np.float64]:
        return evaluate_hrf_basis(np.asarray(t, dtype=np.float64), spec)

    return as_hrf(
        _eval,
        name=f"estimate_hrf_{spec.type}",
        nbasis=spec.k,
        span=spec.span,
    )


def _validate_formula(form: str) -> None:
    from fmrimod.formula.parser import FormulaParser

    parsed = FormulaParser().parse(form)
    if not parsed.lhs:
        raise ValueError("form must be a two-sided event-model formula")
    names = [term.function for term in parsed.rhs]
    if "trialwise" in names:
        raise ValueError(
            "estimate_hrf() estimates condition-level hrf() curves; "
            "trialwise() terms are not supported."
        )
    if "hrf" not in names:
        raise ValueError("form must contain at least one condition-level hrf() term")


def _order_event_data(
    data: pd.DataFrame, onset_col: str, block: str | None
) -> pd.DataFrame:
    if onset_col not in data.columns:
        raise ValueError(
            f"The left-hand side of form must name one onset column in the "
            f"event table; '{onset_col}' is missing"
        )
    ordered = data.copy()
    if block is not None and block in ordered.columns:
        return ordered.sort_values([block, onset_col], kind="mergesort")
    return ordered.sort_values(onset_col, kind="mergesort")


def _inject_estimation_hrf(form: str, hrf: HRF) -> list[Any]:
    from fmrimod.formula.parser import parse_formula

    terms = list(parse_formula(form, for_event_model=True))
    for term in terms:
        if getattr(term, "hrf", None) is not None:
            term.with_hrf(hrf)
    return terms


def _hrf_curve_map(
    event_model: object, design: NDArray[np.float64], k: int
) -> dict[str, Any]:
    from fmrimod.design_colmap import design_colmap

    names = [str(n) for n in getattr(event_model, "column_names", [])]
    if len(names) != design.shape[1] or any(not n for n in names):
        raise ValueError("The event design must have non-empty column names")

    colmap = design_colmap(event_model)
    has_semantic = (
        not colmap.empty
        and {"col", "term_tag", "condition", "basis_ix"}.issubset(colmap.columns)
        and len(colmap) == design.shape[1]
    )
    if has_semantic:
        colmap = colmap.sort_values("col")
        term_by_column = colmap["term_tag"].astype(str).to_numpy()
        condition_by_column = colmap["condition"].astype(str).to_numpy()
        curve_id = np.array(
            [
                f"{t}_{c}"
                for t, c in zip(term_by_column, condition_by_column, strict=True)
            ],
            dtype=object,
        )
        basis_index = pd.to_numeric(colmap["basis_ix"], errors="coerce").to_numpy()
        if np.any(pd.isna(basis_index)):
            has_semantic = False
        else:
            basis_index = basis_index.astype(int)
    if not has_semantic:
        import re

        suffix = re.compile(r"_b(\d+)$")
        matched = [suffix.search(n) for n in names]
        if any(m is None for m in matched):
            raise ValueError(
                "Could not map all event-design columns to HRF basis components"
            )
        curve_id = np.array([suffix.sub("", n) for n in names], dtype=object)
        basis_index = np.array(
            [int(cast(re.Match[str], m).group(1)) for m in matched], dtype=int
        )
        term_by_column = np.array(
            [cid.split("_", 1)[0] for cid in curve_id], dtype=object
        )
        condition_by_column = np.array(
            [
                cid[len(str(term)) + 1 :] if cid.startswith(f"{term}_") else cid
                for cid, term in zip(curve_id, term_by_column, strict=True)
            ],
            dtype=object,
        )

    curves = list(dict.fromkeys(curve_id.tolist()))
    raw_indices: list[NDArray[np.intp]] = []
    for curve in curves:
        idx = np.flatnonzero(curve_id == curve)
        observed = np.sort(basis_index[idx])
        if not np.array_equal(observed, np.arange(1, k + 1)):
            raise ValueError(
                f"Curve '{curve}' does not contain exactly basis components "
                f"1 through {k}"
            )
        raw_indices.append(idx[np.argsort(basis_index[idx])])

    order_index = np.concatenate(raw_indices)
    reordered = design[:, order_index]
    ordered_names: list[str] = []
    for curve in curves:
        ordered_names.extend(f"{curve}_b{i:02d}" for i in range(1, k + 1))
    term = []
    condition = []
    for idx in raw_indices:
        first = int(idx[0])
        term.append(str(term_by_column[first]))
        condition.append(str(condition_by_column[first]))
    display = list(condition)
    dup = pd.Series(display).duplicated(keep=False).to_numpy()
    for i, is_dup in enumerate(dup):
        if is_dup:
            display[i] = f"{term[i]}:{condition[i]}"
    display = list(pd.Series(display).astype(str))
    # make.unique equivalent
    seen: dict[str, int] = {}
    unique_display = []
    for name in display:
        count = seen.get(name, 0)
        unique_display.append(name if count == 0 else f"{name}#{count}")
        seen[name] = count + 1

    return {
        "design": reordered,
        "indices": [np.arange(i * k, (i + 1) * k) for i in range(len(curves))],
        "info": pd.DataFrame(
            {"curve": unique_display, "term": term, "condition": condition}
        ),
    }


def _partial_out_nuisance(
    event_design: NDArray[np.float64],
    response: NDArray[np.float64],
    nuisance_design: NDArray[np.float64] | None,
) -> dict[str, Any]:
    if nuisance_design is None or nuisance_design.shape[1] == 0:
        return {
            "event": event_design,
            "response": response,
            "nuisance_rank": 0,
            "nuisance_columns": 0,
        }
    q, r = np.linalg.qr(nuisance_design, mode="reduced")
    diag = np.abs(np.diag(r))
    tol = np.max(diag) * max(nuisance_design.shape) * np.finfo(np.float64).eps
    rank = int(np.sum(diag > tol))
    if rank < nuisance_design.shape[1]:
        import warnings

        warnings.warn(
            f"Nuisance design is rank deficient: retained {rank} of "
            f"{nuisance_design.shape[1]} independent directions.",
            RuntimeWarning,
            stacklevel=3,
        )
    q = q[:, :rank] if rank > 0 else np.zeros((event_design.shape[0], 0))

    def residualize(x: NDArray[np.float64]) -> NDArray[np.float64]:
        if q.shape[1] == 0:
            return x
        return x - q @ (q.T @ x)

    return {
        "event": residualize(event_design),
        "response": residualize(response),
        "nuisance_rank": rank,
        "nuisance_columns": int(nuisance_design.shape[1]),
    }


def _smoothness_penalty(k: int, n_curves: int) -> NDArray[np.float64]:
    order = 2 if k >= 3 else 1
    d = np.diff(np.eye(k), n=order, axis=0)
    block = d.T @ d
    return np.kron(np.eye(n_curves), block)


def _solve_hrf_system(
    xtx: NDArray[np.float64],
    xty: NDArray[np.float64],
    penalty: NDArray[np.float64],
    lam: float,
) -> dict[str, Any] | None:
    a = xtx + float(lam) * penalty
    try:
        factor, lower = linalg.cho_factor(a, overwrite_a=False, check_finite=False)
    except linalg.LinAlgError:
        return None
    inverse = linalg.cho_solve((factor, lower), np.eye(a.shape[0]), check_finite=False)
    coefficients = inverse @ xty
    edf = float(np.trace(inverse @ xtx))
    return {"coefficients": coefficients, "inverse": inverse, "edf": edf}


def _select_lambda(
    xtx: NDArray[np.float64],
    xty: NDArray[np.float64],
    response_ss: NDArray[np.float64],
    penalty: NDArray[np.float64],
    n_effective: int,
    lam: LamSpec,
    lam_grid: Sequence[float],
) -> dict[str, Any]:
    if isinstance(lam, str):
        if lam.lower() != "gcv":
            raise ValueError("lam must be a non-negative number or 'gcv'")
        candidates = np.unique(np.asarray(lam_grid, dtype=np.float64))
    else:
        value = float(lam)
        if not np.isfinite(value) or value < 0:
            raise ValueError("lam must be a non-negative number or 'gcv'")
        candidates = np.asarray([value], dtype=np.float64)
    if (
        candidates.size == 0
        or np.any(~np.isfinite(candidates))
        or np.any(candidates < 0)
    ):
        raise ValueError("lam_grid must contain finite non-negative values")

    scores = np.full(candidates.size, np.inf)
    edf = np.full(candidates.size, np.nan)
    for i, cand in enumerate(candidates):
        solved = _solve_hrf_system(xtx, xty, penalty, float(cand))
        if solved is None:
            continue
        beta = solved["coefficients"]
        fitted_ss = np.sum(beta * (xtx @ beta), axis=0)
        cross_term = np.sum(beta * xty, axis=0)
        rss = np.maximum(response_ss - 2.0 * cross_term + fitted_ss, 0.0)
        denom = 1.0 - solved["edf"] / n_effective
        if np.isfinite(denom) and denom > 0:
            relative_rss = rss / np.maximum(response_ss, np.finfo(np.float64).eps)
            scores[i] = float(np.mean(relative_rss) / denom**2)
            edf[i] = solved["edf"]
    if not np.any(np.isfinite(scores)):
        raise ValueError("No smoothing candidate produced an identifiable HRF fit")
    best = float(np.nanmin(scores))
    tied = np.flatnonzero(np.isfinite(scores) & (scores <= best * (1.0 + 1e-10)))
    pick = int(tied[-1])
    return {
        "lambda": float(candidates[pick]),
        "table": pd.DataFrame({"lambda": candidates, "score": scores, "edf": edf}),
    }


def _response_matrix(dataset: object) -> NDArray[np.float64]:
    if hasattr(dataset, "get_all_data"):
        y = np.asarray(dataset.get_all_data(), dtype=np.float64)
    elif hasattr(dataset, "get_data_matrix"):
        y = np.asarray(dataset.get_data_matrix(), dtype=np.float64)
    else:
        from fmrimod.accessors import get_data_matrix

        y = np.asarray(get_data_matrix(dataset), dtype=np.float64)
    if y.ndim == 1:
        y = y[:, np.newaxis]
    return y


def _prepare_hrf_estimation(
    form: str,
    fixed: str | None,
    block: str | None,
    dataset: object,
    basemod: object,
    basis_spec: HrfBasisSpec,
) -> dict[str, Any]:
    from fmrimod.baseline.baseline_model import baseline_model
    from fmrimod.design.event_model import event_model
    from fmrimod.formula.parser import FormulaParser

    hrf = _as_estimation_hrf(basis_spec)
    parsed = FormulaParser().parse(form)
    onset_col = parsed.lhs
    events = getattr(dataset, "event_table", None)
    if events is None:
        raise ValueError("dataset must expose event_table")
    event_data = _order_event_data(events, onset_col, block)
    sframe = getattr(dataset, "sampling_frame", None)
    if sframe is None:
        raise ValueError("dataset must expose sampling_frame")

    event_mod = event_model(
        _inject_estimation_hrf(form, hrf),
        data=event_data,
        block=block,
        sampling_frame=sframe,
        durations=0.0,
        onset_column=onset_col,
    )
    curve_map = _hrf_curve_map(event_mod, _as_design(event_mod), basis_spec.k)

    if basemod is None:
        baseline_mod = baseline_model("constant", sframe=sframe)
    else:
        baseline_mod = basemod
    baseline_design = _as_design(baseline_mod)

    fixed_mod = None
    fixed_design = None
    if fixed is not None:
        fixed_parsed = FormulaParser().parse(fixed)
        if not fixed_parsed.lhs:
            raise ValueError("fixed must be a two-sided event-model formula")
        fixed_data = _order_event_data(events, fixed_parsed.lhs, block)
        fixed_mod = event_model(
            fixed,
            data=fixed_data,
            block=block,
            sampling_frame=sframe,
            durations=0.0,
            onset_column=fixed_parsed.lhs,
        )
        fixed_design = _as_design(fixed_mod)

    if fixed_design is None:
        nuisance = baseline_design
    else:
        nuisance = np.column_stack([baseline_design, fixed_design])

    return {
        "event_design": curve_map["design"],
        "nuisance_design": nuisance,
        "curve_indices": curve_map["indices"],
        "curve_info": curve_map["info"],
        "event_model": event_mod,
        "fixed_model": fixed_mod,
        "baseline_model": baseline_mod,
    }


@dataclass
class HrfEstimate:
    """Condition-level smooth FIR estimate.

    ``estimate`` / ``std_error`` are ``(time, curve, voxel)``. Use
    :meth:`tidy` and :meth:`predict` for labeled tables and new grids.
    """

    estimate: NDArray[np.float64]
    std_error: NDArray[np.float64]
    lower: NDArray[np.float64] | None
    upper: NDArray[np.float64] | None
    time: NDArray[np.float64]
    curves: list[str]
    voxels: list[str]
    curve_info: pd.DataFrame
    coefficients: NDArray[np.float64]
    sigma: NDArray[np.float64]
    df_residual: float
    edf: float
    lam: float
    gcv: pd.DataFrame
    basis: HrfBasisName
    basis_spec: HrfBasisSpec
    basis_at_time: NDArray[np.float64]
    span: float
    ci_level: float | None
    covariance_unscaled: NDArray[np.float64]
    event_model: object
    fixed_model: object
    baseline_model: object
    event_design: NDArray[np.float64]
    nuisance_design: NDArray[np.float64]
    formula: str
    diagnostics: dict[str, Any]

    @property
    def lambda_(self) -> float:
        return self.lam

    def coef(self) -> NDArray[np.float64]:
        return self.coefficients

    def as_matrix(
        self,
        curve: int | str | None = None,
        what: Literal["estimate", "std.error"] = "estimate",
    ) -> NDArray[np.float64]:
        if curve is None:
            if len(self.curves) != 1:
                raise ValueError(
                    "curve must be supplied when more than one HRF curve was estimated"
                )
            name = self.curves[0]
        elif isinstance(curve, (int, np.integer)):
            idx = int(curve)
            if idx < 0 or idx >= len(self.curves):
                raise IndexError("curve index is out of bounds")
            name = self.curves[idx]
        else:
            name = str(curve)
            if name not in self.curves:
                raise ValueError("curve must identify one fitted HRF curve")
        values = self.estimate if what == "estimate" else self.std_error
        col = self.curves.index(name)
        return np.asarray(values[:, col, :], dtype=np.float64)

    def predict(
        self,
        newdata: Sequence[float] | None = None,
        se_fit: bool = False,
    ) -> NDArray[np.float64] | dict[str, Any]:
        grid = self.time if newdata is None else np.asarray(newdata, dtype=np.float64)
        if (
            grid.size == 0
            or np.any(~np.isfinite(grid))
            or np.any(grid < 0)
            or np.any(grid > self.span)
        ):
            raise ValueError(
                f"newdata must contain finite times within [0, {self.span}]"
            )
        basis = evaluate_hrf_basis(grid, self.basis_spec)
        fit = np.empty(
            (grid.size, len(self.curves), len(self.voxels)), dtype=np.float64
        )
        se = np.empty_like(fit)
        sigma2 = self.sigma**2
        k = self.basis_spec.k
        for curve_i in range(len(self.curves)):
            index = slice(curve_i * k, (curve_i + 1) * k)
            fit[:, curve_i, :] = basis @ self.coefficients[index, :]
            cov = self.covariance_unscaled[index, index]
            var_scale = np.sum((basis @ cov) * basis, axis=1)
            se[:, curve_i, :] = np.sqrt(np.maximum(np.outer(var_scale, sigma2), 0.0))
        if se_fit:
            return {"fit": fit, "se_fit": se, "df": self.df_residual}
        return fit

    def tidy(
        self,
        curve: Sequence[int | str] | int | str | None = None,
        voxel: Sequence[int | str] | int | str | None = None,
    ) -> pd.DataFrame:
        def _index(
            selector: object, labels: Sequence[str], kind: str
        ) -> NDArray[np.intp]:
            if selector is None:
                return np.arange(len(labels))
            values = np.atleast_1d(selector)
            if np.issubdtype(values.dtype, np.number):
                idx = values.astype(int)
            else:
                idx = np.array([labels.index(str(v)) for v in values], dtype=int)
            if np.any(idx < 0) or np.any(idx >= len(labels)):
                raise ValueError(f"{kind} contains unknown or out-of-range values")
            return idx

        curve_index = _index(curve, self.curves, "curve")
        voxel_index = _index(voxel, self.voxels, "voxel")
        info_by_curve = {
            str(row.curve): row for row in self.curve_info.itertuples(index=False)
        }
        rows: list[dict[str, object]] = []
        for ti, t in enumerate(self.time):
            for ci in curve_index:
                curve_name = self.curves[int(ci)]
                meta = info_by_curve[curve_name]
                for vi in voxel_index:
                    rows.append(
                        {
                            "time": float(t),
                            "curve": curve_name,
                            "term": str(meta.term),
                            "condition": str(meta.condition),
                            "voxel": self.voxels[int(vi)],
                            "estimate": float(self.estimate[ti, ci, vi]),
                            "std.error": float(self.std_error[ti, ci, vi]),
                            "lower": (
                                np.nan
                                if self.lower is None
                                else float(self.lower[ti, ci, vi])
                            ),
                            "upper": (
                                np.nan
                                if self.upper is None
                                else float(self.upper[ti, ci, vi])
                            ),
                        }
                    )
        return pd.DataFrame(rows)


def estimate_hrf_fir(
    form: str,
    dataset: object,
    *,
    block: str | None = "run",
    fixed: str | None = None,
    rsam: Sequence[float] = tuple(range(21)),
    basemod: object = None,
    k: int = 8,
    basis: HrfBasisName = "bspline",
    lam: LamSpec = "gcv",
    lam_grid: Sequence[float] | None = None,
    ci_level: float | None = 0.95,
) -> HrfEstimate:
    """Estimate condition-level HRF curves with a shared smooth FIR.

    Parameters
    ----------
    form
        Two-sided event-model formula containing at least one ``hrf()``
        term. ``trialwise()`` is rejected.
    dataset
        Dataset with ``event_table``, ``sampling_frame``, and a data matrix.
    block
        Event-table column naming runs. ``None`` skips block grouping.
    fixed
        Optional nuisance event-model formula.
    rsam
        Strictly increasing post-stimulus times starting at zero.
    basemod
        Optional baseline model. Default is a constant baseline.
    k
        Free basis functions per curve (at least 4 for ``bspline``, 2 for
        ``tent``).
    basis
        ``"bspline"`` (cubic) or ``"tent"`` (piecewise linear).
    lam
        Non-negative penalty or ``"gcv"`` for shared GCV.
    lam_grid
        Candidate penalties when ``lam='gcv'``.
    ci_level
        Confidence level in ``(0, 1)``, or ``None`` to omit intervals.
    """
    if not isinstance(form, str):
        raise TypeError("form must be a two-sided event-model formula string")
    _validate_formula(form)

    rsam_arr = np.asarray(rsam, dtype=np.float64)
    if rsam_arr.size < 2 or np.any(~np.isfinite(rsam_arr)):
        raise ValueError("rsam must contain at least two finite values")
    if abs(float(rsam_arr[0])) > np.sqrt(np.finfo(np.float64).eps):
        raise ValueError("rsam must start at zero")
    if np.any(np.diff(rsam_arr) <= 0):
        raise ValueError("rsam must be strictly increasing")

    if basis not in {"bspline", "tent"}:
        raise ValueError("basis must be 'bspline' or 'tent'")
    k = int(k)
    minimum_k = 4 if basis == "bspline" else 2
    if k < minimum_k:
        raise ValueError(f"k must be at least {minimum_k} for basis = '{basis}'")
    if ci_level is not None and (
        not np.isfinite(float(ci_level)) or not (0.0 < float(ci_level) < 1.0)
    ):
        raise ValueError(
            "ci_level must be None or a number strictly between zero and one"
        )

    if lam_grid is None:
        lam_grid = [0.0, *list(np.power(10.0, np.linspace(-4.0, 4.0, 25)))]

    span = float(np.max(rsam_arr))
    basis_spec = new_hrf_basis_spec(basis, k, span)
    prepared = _prepare_hrf_estimation(form, fixed, block, dataset, basemod, basis_spec)

    response = _response_matrix(dataset)
    if response.shape[0] != prepared["event_design"].shape[0]:
        raise ValueError(
            "Dataset and HRF event design have different numbers of time points"
        )
    if not np.all(np.isfinite(response)):
        raise ValueError("Dataset contains non-finite response values")
    if response.shape[1] < 1:
        raise ValueError("Dataset contains no response columns")

    partial = _partial_out_nuisance(
        prepared["event_design"], response, prepared["nuisance_design"]
    )
    x = partial["event"]
    y = partial["response"]
    event_rank = int(np.linalg.matrix_rank(x))
    if event_rank < x.shape[1]:
        raise ValueError(
            f"Residualized HRF design is rank deficient: rank {event_rank} "
            f"for {x.shape[1]} coefficients; add events, reduce k, or simplify form."
        )
    n_effective = int(x.shape[0] - partial["nuisance_rank"])
    if n_effective <= x.shape[1]:
        raise ValueError("Too few residual degrees of freedom for HRF estimation")

    xtx = x.T @ x
    xty = x.T @ y
    response_ss = np.sum(y**2, axis=0)
    penalty = _smoothness_penalty(k, len(prepared["curve_info"]))
    selected = _select_lambda(
        xtx, xty, response_ss, penalty, n_effective, lam, lam_grid
    )
    solved = _solve_hrf_system(xtx, xty, penalty, selected["lambda"])
    if solved is None:
        raise ValueError(
            "Selected smoothing strength did not produce an identifiable HRF fit"
        )

    coefficients = solved["coefficients"]
    voxel_names = [f"voxel_{i + 1}" for i in range(response.shape[1])]
    fitted_ss = np.sum(coefficients * (xtx @ coefficients), axis=0)
    cross_term = np.sum(coefficients * xty, axis=0)
    rss = np.maximum(response_ss - 2.0 * cross_term + fitted_ss, 0.0)
    df_residual = n_effective - solved["edf"]
    if not np.isfinite(df_residual) or df_residual <= 0:
        raise ValueError("The fitted HRF model has no residual degrees of freedom")
    sigma2 = rss / df_residual
    covariance_unscaled = solved["inverse"] @ xtx @ solved["inverse"]

    basis_at_time = evaluate_hrf_basis(rsam_arr, basis_spec)
    curve_names = list(prepared["curve_info"]["curve"])
    estimate = np.empty(
        (rsam_arr.size, len(curve_names), response.shape[1]), dtype=np.float64
    )
    std_error = np.empty_like(estimate)
    for curve_i, index in enumerate(prepared["curve_indices"]):
        estimate[:, curve_i, :] = basis_at_time @ coefficients[index, :]
        cov = covariance_unscaled[np.ix_(index, index)]
        var_scale = np.sum((basis_at_time @ cov) * basis_at_time, axis=1)
        std_error[:, curve_i, :] = np.sqrt(np.maximum(np.outer(var_scale, sigma2), 0.0))

    lower = upper = None
    if ci_level is not None:
        critical = float(
            sp_stats.t.ppf(1.0 - (1.0 - float(ci_level)) / 2.0, df_residual)
        )
        lower = estimate - critical * std_error
        upper = estimate + critical * std_error

    return HrfEstimate(
        estimate=estimate,
        std_error=std_error,
        lower=lower,
        upper=upper,
        time=rsam_arr,
        curves=curve_names,
        voxels=voxel_names,
        curve_info=prepared["curve_info"],
        coefficients=coefficients,
        sigma=np.sqrt(sigma2),
        df_residual=float(df_residual),
        edf=float(solved["edf"]),
        lam=float(selected["lambda"]),
        gcv=selected["table"],
        basis=basis,
        basis_spec=basis_spec,
        basis_at_time=basis_at_time,
        span=span,
        ci_level=None if ci_level is None else float(ci_level),
        covariance_unscaled=covariance_unscaled,
        event_model=prepared["event_model"],
        fixed_model=prepared["fixed_model"],
        baseline_model=prepared["baseline_model"],
        event_design=prepared["event_design"],
        nuisance_design=prepared["nuisance_design"],
        formula=form,
        diagnostics={
            "algorithm": "shared penalized multiresponse solve",
            "n_time": int(response.shape[0]),
            "n_voxels": int(response.shape[1]),
            "n_curves": len(curve_names),
            "event_rank": event_rank,
            "event_columns": int(x.shape[1]),
            "nuisance_rank": int(partial["nuisance_rank"]),
            "nuisance_columns": int(partial["nuisance_columns"]),
            "condition_number": float(np.linalg.cond(xtx)),
        },
    )


def estimate_hrf(
    Y: NDArray[np.float64] | None = None,
    X_trials: NDArray[np.float64] | None = None,
    basis: object | None = None,
    *,
    confounds: NDArray[np.float64] | None = None,
    K: int | None = None,
    output: Literal["hrf", "coefficients", "result"] = "hrf",
    form: str | None = None,
    fixed: object | None = None,
    block: str | None = None,
    dataset: object = None,
    rsam: Sequence[float] | None = None,
    basemod: object = None,
    k: int = 8,
    lam: LamSpec | None = None,
    lam_grid: Sequence[float] | None = None,
    ci_level: float | None = 0.95,
    **kwargs: object,
) -> NDArray[np.float64] | Any:
    """Dispatch between the smooth-FIR estimator and the voxel-compat helper.

    Formula + dataset calls without a numeric 2-D ``basis`` use the
    fmrireg 0.2.0 smooth FIR. Matrix arguments, or a formula path that
    still supplies a numeric 2-D basis, keep the older
    :func:`~fmrimod.single.voxel_hrf.estimate_hrf` ``lstsq`` helper.
    """
    if "lambda" in kwargs:
        if lam is not None:
            raise TypeError("specify only one of lam= or lambda=")
        lam = cast(LamSpec, kwargs.pop("lambda"))
    extra = {key: value for key, value in kwargs.items() if key != "lambda"}

    compat = (Y is not None and X_trials is not None and basis is not None) or (
        form is not None and dataset is not None and _is_2d_numeric_basis(basis)
    )
    if compat:
        from fmrimod.single.voxel_hrf import estimate_hrf as estimate_hrf_compat

        return estimate_hrf_compat(
            Y,
            X_trials,
            basis,
            confounds=confounds,
            K=K,
            output=output,
            form=form,
            fixed=fixed if isinstance(fixed, np.ndarray) else fixed,
            block=block,
            dataset=dataset,
        )

    if form is None or dataset is None:
        from fmrimod.single.voxel_hrf import estimate_hrf as estimate_hrf_compat

        return estimate_hrf_compat(
            Y,
            X_trials,
            basis,
            confounds=confounds,
            K=K,
            output=output,
            form=form,
            fixed=fixed if isinstance(fixed, np.ndarray) else None,
            block=block,
            dataset=dataset,
        )

    fir_basis: HrfBasisName = "bspline"
    if isinstance(basis, str):
        if basis not in {"bspline", "tent"}:
            raise ValueError("basis must be 'bspline' or 'tent'")
        fir_basis = cast(HrfBasisName, basis)
    if extra:
        unexpected = ", ".join(sorted(extra))
        raise TypeError(f"unexpected keyword argument(s): {unexpected}")
    return estimate_hrf_fir(
        form,
        dataset,
        block="run" if block is None else block,
        fixed=fixed if isinstance(fixed, str) else None,
        rsam=tuple(range(21)) if rsam is None else rsam,
        basemod=basemod,
        k=k,
        basis=fir_basis,
        lam="gcv" if lam is None else lam,
        lam_grid=lam_grid,
        ci_level=ci_level,
    )
