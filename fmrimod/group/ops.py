"""Eager operations for native group-analysis datasets."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Callable, Literal, Optional, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import special as sp_special
from scipy import stats as sp_stats

from fmrimod.stats.inference import fdr_correction, p_to_z, z_to_p

from .dataset import GroupDataset
from .errors import (
    AdapterContractError,
    GroupRegistryError,
    UnsupportedGroupFeatureError,
)
from .io import write_hdf5
from .progress import ProgressCallback, ProgressReporter, emit_progress
from .registry import posthoc_registry, reducer_registry


def _normalize_axis_index(
    index: Sequence[int] | NDArray[np.bool_] | None,
    *,
    length: int,
) -> NDArray[np.intp]:
    if index is None:
        return np.arange(length, dtype=np.intp)
    arr = np.asarray(index)
    if arr.dtype == np.bool_:
        if arr.ndim != 1 or arr.shape[0] != length:
            raise AdapterContractError("logical index length must match axis length")
        return np.flatnonzero(arr).astype(np.intp, copy=False)
    idx = np.asarray(arr, dtype=np.intp).reshape(-1)
    if idx.size == 0:
        raise AdapterContractError("axis index must be non-empty")
    if np.any(idx < 0) or np.any(idx >= length):
        raise AdapterContractError("axis index contains values out of range")
    return idx


# Scoped/strict divergence (bd-01KRNN0H73CCYGFJSJ30JPVFTW): the cast
# below is REQUIRED under scoped mypy (--follow-imports=skip) -- the
# frame source type is opaque Any there so frame.iloc[idx].copy() is Any
# and removing the cast raises no-any-return against Optional[DataFrame].
# Full-strict resolves it and flags the cast redundant-cast. Verified
# empirically (scoped goes red without it). Scoped is the epic gate so
# the cast stays; same divergence shape as events/basis.py:160.
def _subset_frame(
    frame: Optional[pd.DataFrame], idx: NDArray[np.intp]
) -> Optional[pd.DataFrame]:
    if frame is None:
        return None
    return cast(pd.DataFrame, frame.iloc[idx].copy())


def subset(
    dataset: GroupDataset,
    *,
    sample: Sequence[int] | NDArray[np.bool_] | None = None,
    subject: Sequence[int] | NDArray[np.bool_] | None = None,
    contrast: Sequence[int] | NDArray[np.bool_] | None = None,
) -> GroupDataset:
    """Return a dataset restricted to sample, subject, and/or contrast axes."""
    sample_idx = _normalize_axis_index(sample, length=dataset.n_samples)
    subject_idx = _normalize_axis_index(subject, length=dataset.n_subjects)
    contrast_idx = _normalize_axis_index(contrast, length=dataset.n_contrasts)

    assays = {
        name: cast(NDArray[np.float64], arr)[
            np.ix_(sample_idx, subject_idx, contrast_idx)
        ]
        for name, arr in dataset.assays.items()
    }
    return GroupDataset(
        assays=assays,
        space=dataset.space.subset(sample_idx),
        subjects=[dataset.subjects[int(i)] for i in subject_idx],
        contrasts=[dataset.contrasts[int(i)] for i in contrast_idx],
        col_data=_subset_frame(dataset.col_data, subject_idx),
        row_data=_subset_frame(dataset.row_data, sample_idx),
        contrast_data=_subset_frame(dataset.contrast_data, contrast_idx),
        metadata={**dict(dataset.metadata), "operation": "subset"},
    )


def mask(
    dataset: GroupDataset,
    sample: Sequence[int] | NDArray[np.bool_],
) -> GroupDataset:
    """Alias for sample-axis subsetting."""
    return subset(dataset, sample=sample)


def _require_assay(dataset: GroupDataset, name: str) -> NDArray[np.float64]:
    if name not in dataset.assays:
        raise AdapterContractError(f"derive requires assay '{name}'")
    return dataset.assay(name)


def _z_from_t(
    t: NDArray[np.float64], df: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Signed z with the same one-sided tail probability as ``t``.

    Works on the log scale (fmrigds ``R/derive-stats.R``): ``isf(sf(|t|))``
    loses all precision once the tail underflows relative to 1, for
    example t = 12 with df = 100.
    """
    with np.errstate(invalid="ignore"):
        log_tail = sp_stats.t.logcdf(-np.abs(t), df)
    return cast("NDArray[np.float64]", -sp_special.ndtri_exp(log_tail) * np.sign(t))


def _signed_z_from_p(
    p: NDArray[np.float64], beta: NDArray[np.float64] | None
) -> NDArray[np.float64]:
    if beta is None:
        raise AdapterContractError(
            "derive cannot sign z from a two-sided p without the effect sign ('beta')"
        )
    return p_to_z(p) * np.sign(beta)


def derive(
    dataset: GroupDataset,
    what: str | Iterable[str],
) -> GroupDataset:
    """Derive common statistics eagerly and return a new dataset.

    Supported derivations are ``"var"``, ``"se"``, ``"t"``, ``"z"``,
    ``"p"`` and ``"z_from_p"``. ``"t"`` is ``beta / se``. ``"z"`` and
    ``"p"`` come from ``{t, df}`` through the t distribution when both are
    present, otherwise from ``beta / se`` (or an existing ``z``); a ``t``
    without ``df`` is never treated as a z. ``"z_from_p"`` signs the
    two-sided z by ``beta``.
    """
    requested = [what] if isinstance(what, str) else list(what)
    if not requested:
        raise AdapterContractError("derive requires at least one target statistic")

    assays: dict[str, NDArray[np.float64]] = {
        name: np.array(arr, copy=True) for name, arr in dataset.assays.items()
    }

    def need(name: str) -> NDArray[np.float64]:
        if name not in assays:
            raise AdapterContractError(f"derive requires assay '{name}'")
        return assays[name]

    def has_wald() -> bool:
        return "beta" in assays and ("se" in assays or "var" in assays)

    def wald_z() -> NDArray[np.float64]:
        se_arr = assays.get("se")
        if se_arr is None:
            se_arr = np.sqrt(need("var"))
            assays["se"] = se_arr
        return need("beta") / se_arr

    def has_t_df() -> bool:
        return "t" in assays and "df" in assays

    for target in requested:
        if target == "var":
            if "var" not in assays:
                se = need("se")
                assays["var"] = se * se
        elif target == "se":
            if "se" not in assays:
                assays["se"] = np.sqrt(need("var"))
        elif target == "t":
            if "t" not in assays:
                assays["t"] = wald_z()
        elif target == "z":
            if "z" not in assays:
                if has_t_df():
                    assays["z"] = _z_from_t(assays["t"], assays["df"])
                elif has_wald():
                    assays["z"] = wald_z()
                elif "p" in assays:
                    assays["z"] = _signed_z_from_p(assays["p"], assays.get("beta"))
                else:
                    raise AdapterContractError(
                        "derive 'z' requires {t, df}, {beta, se or var} or {p, beta}"
                    )
        elif target == "p":
            if "p" not in assays:
                if has_t_df():
                    assays["p"] = 2.0 * sp_stats.t.sf(np.abs(assays["t"]), assays["df"])
                elif "z" in assays:
                    assays["p"] = z_to_p(assays["z"])
                elif has_wald():
                    assays["z"] = wald_z()
                    assays["p"] = z_to_p(assays["z"])
                elif "t" in assays:
                    raise AdapterContractError(
                        "derive 'p' from 't' requires a 'df' assay; a t statistic "
                        "is not a z score"
                    )
                else:
                    raise AdapterContractError(
                        "derive 'p' requires {t, df}, z or {beta, se or var}"
                    )
        elif target == "z_from_p":
            assays["z"] = _signed_z_from_p(need("p"), assays.get("beta"))
        else:
            raise UnsupportedGroupFeatureError(
                f"derive target '{target}' is not supported by native group ops"
            )

    return dataset.with_assays(
        assays,
        metadata={"operation": "derive", "derived": tuple(requested)},
    )


def _fdr_posthoc_column(
    p_values: NDArray[np.float64],
    *,
    alpha: float,
    method: Literal["bh", "by"],
) -> NDArray[np.float64]:
    _, q_values = fdr_correction(p_values, alpha=alpha, method=method)
    return np.asarray(q_values, dtype=np.float64)


def posthoc_bh(
    p_values: NDArray[np.float64],
    *,
    alpha: float = 0.05,
) -> NDArray[np.float64]:
    """Benjamini-Hochberg FDR correction for one p-value vector."""
    return _fdr_posthoc_column(p_values, alpha=alpha, method="bh")


def posthoc_by(
    p_values: NDArray[np.float64],
    *,
    alpha: float = 0.05,
) -> NDArray[np.float64]:
    """Benjamini-Yekutieli FDR correction for one p-value vector."""
    return _fdr_posthoc_column(p_values, alpha=alpha, method="by")


def register_core_posthoc(*, overwrite: bool = True) -> None:
    """Register built-in native posthoc methods."""
    for name, function, description in (
        ("bh", posthoc_bh, "Benjamini-Hochberg FDR correction"),
        ("fdr:bh", posthoc_bh, "Benjamini-Hochberg FDR correction"),
        ("by", posthoc_by, "Benjamini-Yekutieli FDR correction"),
        ("fdr:by", posthoc_by, "Benjamini-Yekutieli FDR correction"),
    ):
        posthoc_registry.register(
            name,
            function,
            description=description,
            overwrite=overwrite,
        )


def posthoc(
    dataset: GroupDataset,
    method: str = "bh",
    *,
    alpha: float = 0.05,
) -> GroupDataset:
    """Apply columnwise FDR correction to the ``p`` assay and add ``q``."""
    try:
        posthoc_fn = posthoc_registry.get(method)
    except GroupRegistryError as exc:
        raise UnsupportedGroupFeatureError(
            "posthoc currently supports only registered methods: "
            + ", ".join(posthoc_registry.list_names())
        ) from exc
    p = _require_assay(dataset, "p")
    q = np.empty_like(p)
    for subject_idx in range(p.shape[1]):
        for contrast_idx in range(p.shape[2]):
            q_col = posthoc_fn(
                p[:, subject_idx, contrast_idx],
                alpha=alpha,
            )
            q[:, subject_idx, contrast_idx] = q_col

    assays = {name: np.array(arr, copy=True) for name, arr in dataset.assays.items()}
    assays["q"] = q
    return dataset.with_assays(
        assays,
        metadata={"operation": "posthoc", "posthoc": method, "alpha": float(alpha)},
    )


def reduce(
    dataset: GroupDataset,
    method: str = "meta:fe",
    *,
    progress: ProgressCallback | ProgressReporter | None = None,
    **options: object,
) -> GroupDataset:
    """Run a native reducer from the reducer registry."""
    emit_progress(
        progress,
        "start",
        method=method,
        message=f"Starting group reducer {method}",
        completed=0,
        total=1,
    )
    try:
        try:
            reducer = reducer_registry.get(method)
        except GroupRegistryError as exc:
            raise UnsupportedGroupFeatureError(str(exc)) from exc
        reducer_fn = cast(Callable[..., GroupDataset], reducer)
        out = reducer_fn(dataset, **options)
    except Exception as exc:
        emit_progress(
            progress,
            "error",
            method=method,
            message=str(exc),
            completed=0,
            total=1,
            error_type=type(exc).__name__,
        )
        raise
    emit_progress(
        progress,
        "done",
        method=method,
        message=f"Finished group reducer {method}",
        completed=1,
        total=1,
    )
    return out


def write_out(
    dataset: GroupDataset,
    path: str | Path,
    *,
    output_format: str | None = None,
    **kwargs: object,
) -> Path:
    """Write a native group dataset to disk."""
    resolved = output_format
    if "format" in kwargs:
        if output_format is not None:
            raise AdapterContractError("Use only one of output_format or format")
        resolved = str(kwargs.pop("format"))
    if resolved is None:
        suffixes = "".join(Path(path).suffixes).lower()
        resolved = "h5" if suffixes.endswith((".h5", ".hdf5")) else ""
    if resolved in ("h5", "hdf5"):
        return write_hdf5(dataset, path)
    raise UnsupportedGroupFeatureError(
        "native group write_out currently supports only format='h5'"
    )


register_core_posthoc()
