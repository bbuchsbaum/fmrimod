"""Contrast computation on GLM results.

Computes t-statistics, F-statistics, standard errors, and p-values
for linear contrasts of regression coefficients.
"""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import special as sp_special

from fmrimod.glm.spatial import SpatialContext


def weights_payload(
    weights: NDArray[np.float64] | Sequence[Sequence[float]] | Sequence[float],
) -> tuple[tuple[float, ...], ...]:
    """Normalize contrast weights to a tuple-of-tuples for payload equality.

    A t-contrast (1-D) becomes a single-row F-matrix so t and F contrasts
    share one payload schema.
    """
    arr = np.atleast_2d(np.asarray(weights, dtype=np.float64))
    return tuple(tuple(float(v) for v in row) for row in arr)


def provenance_id(fit: object) -> str | None:
    """Return a content-addressed identifier for a fit's :class:`FitProvenance`.

    Hashes :meth:`FitProvenance.to_json` so the value survives serialization
    and process boundaries. Returns ``None`` when the fit exposes no
    provenance to hash (legacy/test paths).
    """
    provenance = getattr(fit, "provenance", None)
    if provenance is None:
        return None
    to_json = getattr(provenance, "to_json", None)
    if not callable(to_json):
        return None
    blob = to_json().encode("utf-8")
    return f"fitprov:sha256:{hashlib.sha256(blob).hexdigest()[:16]}"


def basis_label(fit: object) -> str | None:
    """Return the HRF normalization basis label for a fitted model.

    The payload is intentionally derived from :class:`FitProvenance` so it
    survives serialization and remains independent of Python object identity.
    """
    provenance = getattr(fit, "provenance", None)
    modes = getattr(provenance, "hrf_norm_modes", None)
    if modes is None:
        return None
    labels = tuple("none" if mode is None else str(mode) for mode in modes)
    if not labels:
        return "hrf_norm:none"
    return "hrf_norm:" + ",".join(labels)


def design_id(fit: object) -> str | None:
    """Return a content-addressed identifier for the fitted design matrix."""
    model = getattr(fit, "model", None)
    design_matrix_array = getattr(model, "design_matrix_array", None)
    if not callable(design_matrix_array):
        return None
    try:
        n_runs = int(getattr(model, "n_runs", 1))
        hasher = hashlib.sha256()
        for run in range(n_runs):
            arr = np.asarray(design_matrix_array(run=run), dtype=np.float64)
            hasher.update(str(tuple(arr.shape)).encode("utf-8"))
            hasher.update(np.ascontiguousarray(arr).tobytes())
        design_columns = getattr(fit, "design_columns", None)
        if callable(design_columns):
            columns = design_columns()
            names = getattr(columns, "names", None)
            if names is not None:
                for name in names:
                    hasher.update(str(name).encode("utf-8"))
                    hasher.update(b"\x00")
        return f"design:sha256:{hasher.hexdigest()[:16]}"
    except Exception:
        return None


@dataclass(frozen=True)
class ContrastIntent:
    """Structured record of how a contrast was requested.

    The four trailing fields (``basis_label``, ``weights``, ``design_id``,
    ``provenance_id``) carry the payload-equality invariant agreed on
    ``beat-nilearn-10/post-01KRK7S86HMRS3JNMPX8QQR7CQ``: typed first-level
    intent survives to group inference when these fields plus the legacy
    ``{kind, term, levels}`` triple are byte-equal across the seam. Python
    object identity is intentionally *not* the invariant because it does
    not survive serialization or process boundaries. Producers populate
    each field as the upstream identity becomes available; the release
    receipt gate (filed separately) refuses ``None`` for flagship rows.

    ``weights`` is normalized to a tuple-of-tuples shape so t- and
    F-contrasts share the same payload schema (a t-contrast is a 1-row
    F-matrix); ``provenance_id`` is a content-addressed hash of the
    producing fit's :class:`FitProvenance`; ``design_id`` is a content
    hash of the design matrix the contrast was resolved against;
    ``basis_label`` is the HRF basis identifier the design was built on.
    """

    kind: str
    name: str | None = None
    term: str | None = None
    levels: tuple[str, ...] = ()
    rows: int | None = None
    basis_label: str | None = None
    weights: tuple[tuple[float, ...], ...] | None = None
    design_id: str | None = None
    provenance_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation of the contrast intent."""
        return {
            "kind": self.kind,
            "name": self.name,
            "term": self.term,
            "levels": list(self.levels),
            "rows": self.rows,
            "basis_label": self.basis_label,
            "weights": (
                None if self.weights is None else [list(row) for row in self.weights]
            ),
            "design_id": self.design_id,
            "provenance_id": self.provenance_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ContrastIntent":
        """Reconstruct a :class:`ContrastIntent` from a :meth:`to_dict`
        payload. Missing trailing fields default to ``None`` / ``()`` so
        legacy receipts that pre-date the payload extension round-trip
        without modification.
        """

        raw_weights = payload.get("weights")
        weights: tuple[tuple[float, ...], ...] | None
        if raw_weights is None:
            weights = None
        elif isinstance(raw_weights, Sequence) and not isinstance(
            raw_weights, (str, bytes, bytearray)
        ):
            weight_rows: list[tuple[float, ...]] = []
            for row in raw_weights:
                if not isinstance(row, Sequence) or isinstance(
                    row, (str, bytes, bytearray)
                ):
                    raise TypeError("ContrastIntent weights must be a nested sequence")
                weight_rows.append(tuple(float(value) for value in row))
            weights = tuple(weight_rows)
        else:
            raise TypeError("ContrastIntent weights must be a nested sequence")
        levels_raw = payload.get("levels", ())
        if not isinstance(levels_raw, Sequence) or isinstance(
            levels_raw, (str, bytes, bytearray)
        ):
            raise TypeError("ContrastIntent levels must be a sequence")
        name_raw = payload.get("name")
        term_raw = payload.get("term")
        rows_raw = payload.get("rows")
        basis_raw = payload.get("basis_label")
        design_raw = payload.get("design_id")
        provenance_raw = payload.get("provenance_id")
        return cls(
            kind=str(payload["kind"]),
            name=None if name_raw is None else str(name_raw),
            term=None if term_raw is None else str(term_raw),
            levels=tuple(str(level) for level in levels_raw),
            rows=None if rows_raw is None else int(rows_raw),
            basis_label=None if basis_raw is None else str(basis_raw),
            weights=weights,
            design_id=None if design_raw is None else str(design_raw),
            provenance_id=None if provenance_raw is None else str(provenance_raw),
        )


@dataclass(frozen=True)
class ContrastExplanation:
    """Structured explanation for a fitted contrast result."""

    name: str
    intent: dict[str, Any]
    touched_columns: tuple[str, ...]
    statistic: dict[str, Any]
    design_columns: tuple[dict[str, Any], ...] = ()
    caveats: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready explanation dictionary."""
        return {
            "name": self.name,
            "intent": self.intent,
            "touched_columns": list(self.touched_columns),
            "design_columns": list(self.design_columns),
            "statistic": dict(self.statistic),
            "caveats": list(self.caveats),
        }

    def to_markdown(self) -> str:
        """Render a compact human-readable explanation."""
        family = self.statistic["family"]
        columns = ", ".join(self.touched_columns) or "(none recorded)"
        caveats = ", ".join(self.caveats) or "none"
        return (
            f"### {self.name}\n\n"
            f"- intent: {self.intent.get('kind')}\n"
            f"- statistic: {family}\n"
            f"- touched columns: {columns}\n"
            f"- caveats: {caveats}\n"
        )


def _validate_f_contrast_matrix(
    con_mat: NDArray[np.float64], n_coefficients: int
) -> NDArray[np.float64]:
    """Validate and normalize an F-contrast matrix."""
    con_mat = np.atleast_2d(np.asarray(con_mat, dtype=np.float64))
    if con_mat.shape[0] == 0:
        raise ValueError("F-contrast matrix must have at least one contrast row")
    if con_mat.shape[1] != n_coefficients:
        raise ValueError(
            f"F-contrast matrix has {con_mat.shape[1]} columns but model has "
            f"{n_coefficients} coefficients"
        )
    return con_mat


def _f_quadratic_form_terms(
    con_mat: NDArray[np.float64],
    betas: NDArray[np.float64],
    XtXinv: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """Compute stable F-test quadratic-form numerators."""
    CB = con_mat @ betas  # (k, V)
    cov = con_mat @ XtXinv @ con_mat.T  # (k, k)

    df1 = int(np.linalg.matrix_rank(cov))
    if df1 <= 0:
        return CB, np.zeros(betas.shape[1], dtype=np.float64), 1

    # Use pseudoinverse for rank-deficient or highly ill-conditioned covariance
    # to keep loop and vectorized paths numerically aligned.
    cond = np.linalg.cond(cov)
    use_pinv = (
        df1 < cov.shape[0]
        or not np.isfinite(cond)
        or cond > 1.0 / np.sqrt(np.finfo(np.float64).eps)
    )

    if use_pinv:
        warnings.warn(
            "F-contrast covariance is singular; using pseudoinverse fallback",
            RuntimeWarning,
            stacklevel=3,
        )
        cov_inv = np.linalg.pinv(cov)
        numer = np.sum(CB * (cov_inv @ CB), axis=0)
    else:
        L = np.linalg.cholesky(cov)
        Z = np.linalg.solve(L, CB)
        numer = np.sum(Z ** 2, axis=0)

    return CB, np.maximum(numer, 0.0), df1


def _t_two_sided_pvalue(
    tstat: NDArray[np.float64],
    dfres: float,
) -> NDArray[np.float64]:
    """Compute two-sided p-values for t-statistics."""
    return 2.0 * sp_special.stdtr(dfres, -np.abs(tstat))


def _t_statistic(
    estimate: NDArray[np.float64],
    se: NDArray[np.float64],
    dfres: float,
) -> NDArray[np.float64]:
    """Compute t-statistics without hiding undefined zero-DoF cases."""
    estimate_arr, se_arr = np.broadcast_arrays(
        np.asarray(estimate, dtype=np.float64),
        np.asarray(se, dtype=np.float64),
    )
    tstat = np.full(estimate_arr.shape, np.nan, dtype=np.float64)
    if not np.isfinite(dfres) or dfres <= 0.0:
        return tstat

    valid = np.isfinite(estimate_arr) & np.isfinite(se_arr) & (se_arr > 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(estimate_arr, se_arr, out=tstat, where=valid)

    zero_contrast = (
        np.isfinite(estimate_arr)
        & np.isfinite(se_arr)
        & (se_arr == 0.0)
        & (estimate_arr == 0.0)
    )
    tstat[zero_contrast] = 0.0
    return tstat


def _f_upper_tail_pvalue(
    fstat: NDArray[np.float64],
    df1: float,
    df2: float,
) -> NDArray[np.float64]:
    """Compute upper-tail p-values for F-statistics."""
    return sp_special.fdtrc(df1, df2, fstat)


@dataclass
class ContrastResult:
    """Result of evaluating a contrast on GLM output.

    Attributes
    ----------
    name : str
        Contrast name.
    estimate : NDArray
        Contrast estimate ``c' @ B``, shape ``(k, V)`` for F-tests
        or ``(V,)`` for t-tests.
    stat : NDArray
        Test statistic (t or F), shape ``(V,)``.
    se : NDArray or None
        Standard error, shape ``(V,)`` (t-tests only).
    p_value : NDArray
        Two-sided p-values, shape ``(V,)``.
    df : float or tuple
        Degrees of freedom.  Scalar for t-tests, ``(df1, df2)`` for F-tests.
    stat_type : str
        ``"t"`` or ``"F"``.
    """

    name: str
    estimate: NDArray[np.float64]
    stat: NDArray[np.float64]
    se: NDArray[np.float64] | None
    p_value: NDArray[np.float64]
    df: float | tuple[Any, ...]
    stat_type: str
    spatial: SpatialContext | None = None
    intent: ContrastIntent | dict[str, Any] | None = None
    touched_columns: tuple[str, ...] = ()
    touched_column_details: tuple[dict[str, Any], ...] = ()
    caveats: tuple[str, ...] = ()

    # -- Explanation --

    def explain(self) -> ContrastExplanation:
        """Return structured explanation data for this contrast result."""
        intent = self._intent_dict()
        statistic: dict[str, Any] = {
            "family": self.stat_type,
            "n_voxels": int(np.asarray(self.stat).size),
        }
        if self.stat_type == "F":
            df_num, df_den = self.df
            statistic.update(
                {
                    "df": [float(df_num), float(df_den)],
                    "df_num": float(df_num),
                    "df_den": float(df_den),
                }
            )
        else:
            statistic.update(
                {
                    "df": float(self.df),
                    "df_resid": float(self.df),
                }
            )
        return ContrastExplanation(
            name=self.name,
            intent=intent,
            touched_columns=tuple(self.touched_columns),
            statistic=statistic,
            design_columns=tuple(self.touched_column_details),
            caveats=tuple(self.caveats),
        )

    def summary(self) -> dict[str, Any]:
        """Return a JSON-ready structured summary of this contrast."""
        return self.explain().to_dict()

    def _intent_dict(self) -> dict[str, Any]:
        """Normalize stored intent metadata for public explanation output."""
        if isinstance(self.intent, ContrastIntent):
            return self.intent.to_dict()
        if isinstance(self.intent, dict):
            out = dict(self.intent)
            if isinstance(out.get("levels"), tuple):
                out["levels"] = list(out["levels"])
            return out
        return {
            "kind": "unspecified",
            "name": self.name,
            "term": None,
            "levels": [],
            "rows": None,
        }

    # -- Reverse converters --

    def _vector_for(self, kind: str) -> NDArray[np.float64]:
        """Return a 1-D ``(n_voxels,)`` vector for the requested statistic."""
        if kind in ("stat", "t", "F"):
            return np.asarray(self.stat, dtype=np.float64).ravel()
        if kind in ("estimate", "effect", "effect_size", "beta"):
            est = np.asarray(self.estimate, dtype=np.float64)
            if est.ndim == 2:
                if est.shape[0] != 1:
                    raise ValueError(
                        f"kind={kind!r} requires a 1-row estimate; "
                        f"got shape {est.shape}. Use to_neurovec for multi-row "
                        "F-contrast estimates."
                    )
                est = est.ravel()
            return est
        if kind in ("se", "std_error"):
            if self.se is None:
                raise ValueError(f"kind={kind!r} not available for F-contrasts")
            return np.asarray(self.se, dtype=np.float64).ravel()
        if kind in ("p_value", "p", "pvalue"):
            return np.asarray(self.p_value, dtype=np.float64).ravel()
        raise ValueError(
            f"Unknown kind={kind!r}. Choose from: 'stat', 'estimate', 'se', 'p_value'."
        )

    def _require_spatial(self) -> SpatialContext:
        if self.spatial is None:
            raise ValueError(
                "ContrastResult has no spatial context; call fit.contrast(...) "
                "on a model whose dataset carries a 3-D mask, or attach a "
                "SpatialContext manually."
            )
        return self.spatial

    def to_neurovol(
        self,
        kind: str = "stat",
        *,
        fill: float = 0.0,
        label: str | None = None,
    ) -> Any:
        """Return a ``neuroim.DenseNeuroVol`` for one statistic.

        Parameters
        ----------
        kind : str
            One of ``"stat"`` (default), ``"estimate"``, ``"se"``,
            ``"p_value"``.
        fill : float
            Value placed at non-mask voxels. Defaults to 0.0 so the volume
            writes cleanly to NIfTI; pass ``np.nan`` for diagnostic display.
        label : str, optional
            Volume label. Defaults to ``f"{self.name}.{kind}"``.
        """
        ctx = self._require_spatial()
        vec = self._vector_for(kind)
        return ctx.to_neurovol(vec, label=label or f"{self.name}.{kind}", fill=fill)

    def to_neurovec(
        self,
        kinds: Sequence[str] | None = None,
        *,
        fill: float = 0.0,
    ) -> Any:
        """Return a ``neuroim.DenseNeuroVec`` stacking multiple statistics.

        Parameters
        ----------
        kinds : sequence of str, optional
            Statistic kinds to stack along the 4th dimension. Defaults to
            ``("estimate", "stat", "p_value")`` for t-contrasts and
            ``("stat", "p_value")`` for F-contrasts.
        fill : float
            Out-of-mask fill value (defaults to 0.0).
        """
        import neuroim  # type: ignore[import-untyped]

        ctx = self._require_spatial()
        if kinds is None:
            kinds = (
                ("estimate", "stat", "p_value")
                if self.stat_type == "t"
                else ("stat", "p_value")
            )
        vols: list[NDArray[np.float64]] = []
        for kind in kinds:
            if kind in ("estimate", "effect", "effect_size", "beta"):
                est = np.asarray(self.estimate, dtype=np.float64)
                if est.ndim == 2 and est.shape[0] > 1:
                    vols.extend(ctx.reconstruct(row, fill=fill) for row in est)
                    continue
            vols.append(ctx.reconstruct(self._vector_for(kind), fill=fill))
        stacked = np.stack(vols, axis=-1).astype(np.float64)
        space3d = ctx.to_neuro_space()
        space4d = neuroim.NeuroSpace(
            dim=tuple(int(d) for d in stacked.shape),
            spacing=tuple(float(s) for s in space3d.spacing[:3]) + (1.0,),
            origin=tuple(float(o) for o in space3d.origin[:3]) + (0.0,),
        )
        return neuroim.DenseNeuroVec(stacked, space4d)

    def to_nifti(
        self,
        path: str | Path,
        *,
        kind: str = "stat",
        fill: float = 0.0,
    ) -> Path:
        """Write one statistic to disk as a NIfTI volume.

        Returns the resolved :class:`pathlib.Path`.
        """
        ctx = self._require_spatial()
        return ctx.write_nifti(
            self._vector_for(kind),
            path,
            label=f"{self.name}.{kind}",
            fill=fill,
        )


def contrast_t(
    con_vec: NDArray[np.float64],
    betas: NDArray[np.float64],
    XtXinv: NDArray[np.float64],
    sigma: NDArray[np.float64],
    dfres: float,
    name: str = "t-contrast",
) -> ContrastResult:
    """Compute a t-contrast.

    Parameters
    ----------
    con_vec : NDArray
        Contrast vector, shape ``(p,)``.
    betas : NDArray
        Coefficient matrix, shape ``(p, V)``.
    XtXinv : NDArray
        ``(X'X)^{-1}``, shape ``(p, p)``.
    sigma : NDArray
        Residual standard deviation, shape ``(V,)``.
    dfres : float
        Residual degrees of freedom.
    name : str
        Contrast name.

    Returns
    -------
    ContrastResult
    """
    con_vec = np.asarray(con_vec, dtype=np.float64).ravel()
    p = betas.shape[0]
    if len(con_vec) != p:
        raise ValueError(f"Contrast vector length {len(con_vec)} != {p} coefficients")

    # Estimate: c' B -> (V,)
    estimate = con_vec @ betas

    # Variance of contrast: c' (X'X)^{-1} c * sigma^2
    var_factor = con_vec @ XtXinv @ con_vec  # scalar
    se = sigma * np.sqrt(np.maximum(var_factor, 0.0))

    # t-statistic
    tstat = _t_statistic(estimate, se, dfres)

    # Two-sided p-value
    p_value = _t_two_sided_pvalue(tstat, dfres)

    return ContrastResult(
        name=name,
        estimate=estimate,
        stat=tstat,
        se=se,
        p_value=p_value,
        df=dfres,
        stat_type="t",
    )


def contrast_t_batch(
    con_mat: NDArray[np.float64],
    betas: NDArray[np.float64],
    XtXinv: NDArray[np.float64],
    sigma: NDArray[np.float64],
    dfres: float,
    names: list[str] | None = None,
    name_prefix: str = "t-contrast",
) -> list[ContrastResult]:
    """Compute multiple t-contrasts in one vectorized pass.

    Parameters
    ----------
    con_mat : NDArray
        Contrast matrix, shape ``(k, p)``.
    betas : NDArray
        Coefficient matrix, shape ``(p, V)``.
    XtXinv : NDArray
        ``(X'X)^{-1}``, shape ``(p, p)``.
    sigma : NDArray
        Residual standard deviation, shape ``(V,)``.
    dfres : float
        Residual degrees of freedom.
    names : list[str], optional
        Names for each contrast row.
    name_prefix : str
        Prefix used when ``names`` is not provided.

    Returns
    -------
    list[ContrastResult]
        One result per contrast row.
    """
    con_mat = np.atleast_2d(np.asarray(con_mat, dtype=np.float64))
    if con_mat.shape[0] == 0:
        return []
    p = betas.shape[0]
    if con_mat.shape[1] != p:
        raise ValueError(
            f"Contrast matrix has {con_mat.shape[1]} columns but model has "
            f"{p} coefficients"
        )
    if names is not None and len(names) != con_mat.shape[0]:
        raise ValueError("names length must match number of contrast rows")

    # Estimates: C B -> (k, V)
    estimates = con_mat @ betas

    # Per-contrast variance factors: diag(C XtXinv C')
    c_xtx = con_mat @ XtXinv
    var_factors = np.sum(c_xtx * con_mat, axis=1)
    se = np.sqrt(np.maximum(var_factors, 0.0))[:, np.newaxis] * sigma[np.newaxis, :]

    tstat = _t_statistic(estimates, se, dfres)
    p_value = _t_two_sided_pvalue(tstat, dfres)

    if names is None:
        names = [f"{name_prefix}[{i}]" for i in range(con_mat.shape[0])]

    return [
        ContrastResult(
            name=names[i],
            estimate=estimates[i],
            stat=tstat[i],
            se=se[i],
            p_value=p_value[i],
            df=dfres,
            stat_type="t",
        )
        for i in range(con_mat.shape[0])
    ]


def contrast_f(
    con_mat: NDArray[np.float64],
    betas: NDArray[np.float64],
    XtXinv: NDArray[np.float64],
    sigma: NDArray[np.float64],
    dfres: float,
    name: str = "F-contrast",
) -> ContrastResult:
    """Compute an F-contrast.

    Parameters
    ----------
    con_mat : NDArray
        Contrast matrix, shape ``(k, p)`` where ``k`` is the number
        of linear constraints.
    betas : NDArray
        Coefficient matrix, shape ``(p, V)``.
    XtXinv : NDArray
        ``(X'X)^{-1}``, shape ``(p, p)``.
    sigma : NDArray
        Residual standard deviation, shape ``(V,)``.
    dfres : float
        Residual degrees of freedom.
    name : str
        Contrast name.

    Returns
    -------
    ContrastResult
    """
    con_mat = _validate_f_contrast_matrix(con_mat, betas.shape[0])

    # Fast-path: one-row F-contrast is equivalent to squared t-statistic.
    if con_mat.shape[0] == 1:
        con_vec = con_mat[0]
        estimate = con_vec @ betas  # (V,)
        var_factor = con_vec @ XtXinv @ con_vec
        se = sigma * np.sqrt(np.maximum(var_factor, 0.0))
        tstat = _t_statistic(estimate, se, dfres)
        fstat = np.maximum(tstat * tstat, 0.0)
        p_value = _t_two_sided_pvalue(tstat, dfres)
        return ContrastResult(
            name=name,
            estimate=estimate[np.newaxis, :],
            stat=fstat,
            se=None,
            p_value=p_value,
            df=(1.0, dfres),
            stat_type="F",
        )

    CB, numer, df1 = _f_quadratic_form_terms(con_mat, betas, XtXinv)
    sigma2 = np.square(sigma, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        fstat = np.where(
            sigma2 > 1e-15,
            numer / (df1 * sigma2),
            0.0,
        )

    p_value = _f_upper_tail_pvalue(fstat, float(df1), float(dfres))

    return ContrastResult(
        name=name,
        estimate=CB,
        stat=fstat,
        se=None,
        p_value=p_value,
        df=(float(df1), dfres),
        stat_type="F",
    )


def contrast_f_vectorized(
    con_mat: NDArray[np.float64],
    betas: NDArray[np.float64],
    XtXinv: NDArray[np.float64],
    sigma: NDArray[np.float64],
    dfres: float,
    name: str = "F-contrast",
) -> ContrastResult:
    """Compatibility wrapper for F-contrasts.

    Delegates to :func:`contrast_f` to keep numerics identical across
    near-singular and rank-deficient paths.
    """
    return contrast_f(con_mat, betas, XtXinv, sigma, dfres, name=name)
