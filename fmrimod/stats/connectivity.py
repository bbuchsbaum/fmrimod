"""Seed-based connectivity helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence, Union, cast

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from .inference import r_to_z

if TYPE_CHECKING:
    from fmrimod.glm.fmri_lm import FmriLm

ColumnSelector = Union[int, str]
TargetSelector = Union[ColumnSelector, Sequence[ColumnSelector], slice, None]


def _as_signal_matrix(
    signals: ArrayLike | pd.DataFrame,
) -> tuple[NDArray[np.float64], list[str] | None]:
    if isinstance(signals, pd.DataFrame):
        arr = signals.to_numpy(dtype=np.float64)
        return arr, [str(col) for col in signals.columns]
    arr = np.asarray(signals, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("signals must be a 2-D array or DataFrame")
    return arr, None


def _column_index(
    selector: ColumnSelector,
    names: Sequence[str] | None,
    n_col: int,
) -> int:
    if isinstance(selector, str):
        if names is None:
            raise TypeError("string selectors require a DataFrame input")
        try:
            return list(names).index(selector)
        except ValueError as exc:
            raise KeyError(f"unknown signal column {selector!r}") from exc
    index = int(selector)
    if index < 0:
        index += n_col
    if index < 0 or index >= n_col:
        raise IndexError(f"signal column index {selector!r} out of range")
    return index


def _target_indices(
    targets: TargetSelector,
    seed_index: int,
    names: Sequence[str] | None,
    n_col: int,
) -> list[int]:
    if targets is None:
        return [idx for idx in range(n_col) if idx != seed_index]
    if isinstance(targets, slice):
        return list(range(n_col))[targets]
    if isinstance(targets, (int, str)):
        return [_column_index(targets, names, n_col)]
    return [_column_index(target, names, n_col) for target in targets]


def _default_target_names(
    target_indices: Sequence[int],
    names: Sequence[str] | None,
    target_names: Sequence[str] | None,
) -> list[str]:
    if target_names is not None:
        if len(target_names) != len(target_indices):
            raise ValueError("target_names must match the number of targets")
        return [str(name) for name in target_names]
    if names is not None:
        return [str(names[idx]) for idx in target_indices]
    return [f"target_{idx + 1}" for idx in range(len(target_indices))]


def _split_seed_targets(
    signals: ArrayLike | pd.DataFrame,
    seed: ColumnSelector,
    targets: TargetSelector,
    target_names: Sequence[str] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[str]]:
    arr, names = _as_signal_matrix(signals)
    if arr.shape[1] < 2:
        raise ValueError("signals must contain at least one seed and one target")
    seed_index = _column_index(seed, names, arr.shape[1])
    target_indices = _target_indices(targets, seed_index, names, arr.shape[1])
    if not target_indices:
        raise ValueError("at least one target signal is required")
    target_label = _default_target_names(target_indices, names, target_names)
    return arr[:, seed_index], arr[:, target_indices], target_label


def _fisher_columns(
    r: NDArray[np.float64],
    n: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if n <= 3:
        raise ValueError("n must be > 3 for Fisher Z variance")
    z = np.full(r.shape, np.nan, dtype=np.float64)
    z_var = np.full(r.shape, 1.0 / (float(n) - 3.0), dtype=np.float64)
    finite = np.isfinite(r)
    interior = finite & (np.abs(r) < 1.0)
    if np.any(interior):
        z[interior], _ = r_to_z(r[interior], float(n))
    z[finite & (r >= 1.0)] = np.inf
    z[finite & (r <= -1.0)] = -np.inf
    return z, z_var


def seed_target_correlation(
    signals: ArrayLike | pd.DataFrame,
    seed: ColumnSelector = 0,
    targets: TargetSelector = None,
    *,
    target_names: Sequence[str] | None = None,
    fisher: bool = True,
) -> pd.DataFrame:
    """Estimate seed-to-target Pearson correlations.

    ``signals`` should already be cleaned/residualized when nuisance control is
    needed. Pass a DataFrame to select columns by name and preserve target
    labels; pass an array for the compact ROI-matrix case.
    """
    seed_vec, target_mat, target_label = _split_seed_targets(
        signals,
        seed,
        targets,
        target_names,
    )
    sc = seed_vec - np.mean(seed_vec)
    tc = target_mat - np.mean(target_mat, axis=0)
    denom = np.sqrt(np.sum(sc**2)) * np.sqrt(np.sum(tc**2, axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.asarray((sc @ tc) / denom, dtype=np.float64)
    r = np.where(denom > 0.0, r, np.nan)

    out = pd.DataFrame({"target": target_label, "r": r, "n": int(seed_vec.shape[0])})
    if fisher:
        z, z_var = _fisher_columns(r, int(seed_vec.shape[0]))
        out["fisher_z"] = z
        out["z_variance"] = z_var
    return out


@dataclass(frozen=True)
class SeedTargetLmResult:
    """Result wrapper for a seed-regressor GLM connectivity fit."""

    fit: FmriLm
    target_names: tuple[str, ...]
    seed_column: str = "seed"

    def to_frame(self) -> pd.DataFrame:
        """Return seed-regressor estimates, t-statistics, and p-values."""
        from fmrimod.accessors import p_values

        seed_row = self.fit.n_coefficients - 1
        p = cast("NDArray[np.float64]", p_values(self.fit, type="estimates"))
        return pd.DataFrame(
            {
                "target": list(self.target_names),
                "estimate": self.fit.coef()[seed_row, :],
                "se": self.fit.se()[seed_row, :],
                "t": self.fit.tstat()[seed_row, :],
                "p": p[seed_row, :],
                "df": float(self.fit.residual_df),
            }
        )


def seed_target_lm(
    signals: ArrayLike | pd.DataFrame,
    seed: ColumnSelector = 0,
    targets: TargetSelector = None,
    *,
    confounds: ArrayLike | pd.DataFrame | None = None,
    target_names: Sequence[str] | None = None,
    add_intercept: bool = True,
) -> SeedTargetLmResult:
    """Fit targets on a seed regressor using the normal ``fmri_lm`` machinery.

    This is the fmrireg-style connectivity path: the seed time series is a
    sampled BOLD covariate, not an HRF-convolved event. Use ``confounds`` for
    nuisance columns when you want adjusted seed effects without pre-cleaning.
    """
    from fmrimod.glm.matrix import fit_glm_from_matrix

    seed_vec, target_mat, target_label = _split_seed_targets(
        signals,
        seed,
        targets,
        target_names,
    )
    parts: list[NDArray[np.float64]] = []
    columns: list[str] = []
    if add_intercept:
        parts.append(np.ones((seed_vec.shape[0], 1), dtype=np.float64))
        columns.append("intercept")
    if confounds is not None:
        conf_arr = (
            confounds.to_numpy(dtype=np.float64)
            if isinstance(confounds, pd.DataFrame)
            else np.asarray(confounds, dtype=np.float64)
        )
        if conf_arr.ndim == 1:
            conf_arr = conf_arr[:, np.newaxis]
        if conf_arr.ndim != 2:
            raise ValueError("confounds must be a 1-D or 2-D array/DataFrame")
        if conf_arr.shape[0] != seed_vec.shape[0]:
            raise ValueError("confounds and signals must have the same number of rows")
        parts.append(conf_arr)
        if isinstance(confounds, pd.DataFrame):
            columns.extend([str(col) for col in confounds.columns])
        else:
            columns.extend([f"confound_{idx}" for idx in range(conf_arr.shape[1])])
    parts.append(seed_vec[:, np.newaxis])
    columns.append("seed")
    design = pd.DataFrame(np.column_stack(parts), columns=columns)
    fit = fit_glm_from_matrix(design, target_mat, model=design)
    return SeedTargetLmResult(
        fit=fit,
        target_names=tuple(target_label),
        seed_column="seed",
    )
