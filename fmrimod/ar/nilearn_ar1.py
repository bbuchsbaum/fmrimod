"""Nilearn-compatible AR(1) GLM fit with binned voxelwise coefficients.

Reproduces Nilearn's ``run_glm(..., noise_model='ar1')`` convention:

1. Fit OLS, get residuals.
2. Estimate per-voxel AR(1) coefficient via Yule-Walker.
3. **Bin** the per-voxel coefficients onto a fixed-width grid (default
   ``0.01``, matching Nilearn's ``bins=100``).
4. Group voxels by their binned coefficient; for each unique bin, whiten
   the design and the associated voxel block with a single phi value, then
   re-fit OLS on the whitened pair.
5. Optionally repeat (Cochrane-Orcutt-style) for ``iter_gls > 1``.

The whitened-then-grouped pass is exact Nilearn parity at the AR(1) level
once the bin width is matched. The binning step is what was historically
documented as the ``fitlins-ar1-coefficient-binning`` caveat — exposing it
as a first-class option in :class:`fmrimod.model.AROptions` and a
top-level :func:`fmrimod.ar1_nilearn` helper removes the caveat from
parity reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy import special as sp_special

from .estimation import estimate_ar
from .whitening import ar_whiten_matrix
from ..glm.solver import fast_lm_matrix, fast_preproject


Array = NDArray[np.float64]

# Nilearn's ``run_glm(..., noise_model='ar1', bins=100)`` corresponds to a
# 0.01 bin width.
DEFAULT_BIN_WIDTH = 0.01


@dataclass(frozen=True)
class Ar1NilearnConfig:
    """Configuration for :func:`ar1_nilearn`.

    Parameters
    ----------
    iter_gls
        Number of Cochrane-Orcutt iterations. ``1`` matches Nilearn's
        ``run_glm(noise_model='ar1')`` exactly.
    voxelwise
        Estimate per-voxel AR(1) coefficients. Must be ``True`` when
        binning is requested.
    coefficient_bin_width
        Width of the AR(1) coefficient bin (Nilearn uses ``0.01``). Set
        to ``None`` to disable binning and use raw voxelwise coefficients.
    exact_first_ar1
        Apply the first-observation Toeplitz correction during whitening.
    """

    iter_gls: int = 1
    voxelwise: bool = True
    coefficient_bin_width: Optional[float] = DEFAULT_BIN_WIDTH
    exact_first_ar1: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.iter_gls, bool)
            or not isinstance(self.iter_gls, int)
            or self.iter_gls < 1
        ):
            raise ValueError("iter_gls must be an integer >= 1")
        if not isinstance(self.voxelwise, bool):
            raise ValueError("voxelwise must be a boolean")
        if not isinstance(self.exact_first_ar1, bool):
            raise ValueError("exact_first_ar1 must be a boolean")
        if self.coefficient_bin_width is not None:
            width = float(self.coefficient_bin_width)
            if not np.isfinite(width) or width <= 0.0:
                raise ValueError("coefficient_bin_width must be positive")
            if not self.voxelwise:
                raise ValueError(
                    "coefficient_bin_width requires voxelwise AR coefficients"
                )


def bin_ar1_coefficients(
    phi: Array, bin_width: Optional[float] = DEFAULT_BIN_WIDTH
) -> Array:
    """Round AR(1) coefficients toward zero onto a fixed-width grid.

    Mirrors Nilearn's ``run_glm(..., noise_model='ar1', bins=N)`` step where
    each voxel's estimated phi is rounded onto a discrete grid of width
    ``1 / bins`` (e.g. ``bins=100`` → ``bin_width=0.01``).

    Parameters
    ----------
    phi
        AR(1) coefficient vector or scalar.
    bin_width
        Bin width. ``None`` passes the input through unchanged.

    Returns
    -------
    NDArray
        Binned coefficients with the same shape as ``phi``.
    """
    phi_arr = np.asarray(phi, dtype=np.float64)
    if bin_width is None:
        return phi_arr.copy()
    width = float(bin_width)
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("bin_width must be positive")
    return np.trunc(phi_arr / width) * width


def _t_and_p(
    estimate: Array,
    variance: Array,
    dfres: float,
) -> tuple[Array, Array]:
    with np.errstate(divide="ignore", invalid="ignore"):
        t_vals = np.where(variance > 1e-30, estimate / np.sqrt(variance), 0.0)
    p_vals = 2.0 * sp_special.stdtr(dfres, -np.abs(t_vals))
    return t_vals, p_vals


def ar1_nilearn(
    X: Array,
    Y: Array,
    contrast: Optional[Array] = None,
    *,
    iter_gls: int = 1,
    coefficient_bin_width: Optional[float] = DEFAULT_BIN_WIDTH,
    exact_first_ar1: bool = False,
    config: Optional[Ar1NilearnConfig] = None,
) -> dict:
    """Nilearn-compatible AR(1) GLM fit with binned voxelwise coefficients.

    Parameters
    ----------
    X
        Design matrix, shape ``(n_t, n_p)``.
    Y
        Data matrix, shape ``(n_t, n_v)``.
    contrast
        Optional contrast vector for a t-statistic. If ``None``, only
        ``betas`` and ``sigma2`` are returned.
    iter_gls
        Number of Cochrane-Orcutt iterations (default ``1`` — exact Nilearn
        parity).
    coefficient_bin_width
        Width of the phi binning grid (default ``0.01``).
    exact_first_ar1
        Apply Toeplitz first-observation correction during whitening.

    Returns
    -------
    dict
        Keys: ``"betas"`` ``(n_p, n_v)``, ``"sigma2"`` ``(n_v,)``,
        ``"phi"`` ``(n_v,)`` (binned), and — when ``contrast`` is given —
        ``"effect"``, ``"variance"``, ``"t"``, ``"p"`` (each shape ``(n_v,)``).

    Examples
    --------
    >>> from fmrimod.ar import ar1_nilearn
    >>> out = ar1_nilearn(X, Y, contrast=c, coefficient_bin_width=0.01)
    >>> out["t"].shape
    (n_voxels,)
    """
    if config is None:
        config = Ar1NilearnConfig(
            iter_gls=iter_gls,
            voxelwise=True,
            coefficient_bin_width=coefficient_bin_width,
            exact_first_ar1=exact_first_ar1,
        )

    X_base = np.asarray(X, dtype=np.float64)
    Y_base = np.asarray(Y, dtype=np.float64)
    n_p, n_v = X_base.shape[1], Y_base.shape[1]

    con = None if contrast is None else np.asarray(contrast, dtype=np.float64).ravel()

    def _grouped_fit(phi_vec: Array) -> dict:
        phi_bins = bin_ar1_coefficients(
            np.asarray(phi_vec, dtype=np.float64).reshape(-1),
            config.coefficient_bin_width,
        )
        betas = np.zeros((n_p, n_v), dtype=np.float64)
        sigma2 = np.zeros(n_v, dtype=np.float64)
        effect = np.zeros(n_v, dtype=np.float64) if con is not None else None
        variance = np.zeros(n_v, dtype=np.float64) if con is not None else None

        for ar1_bin in np.unique(phi_bins):
            cols = phi_bins == ar1_bin
            X_w, Y_w = ar_whiten_matrix(
                X_base,
                Y_base[:, cols],
                np.array([ar1_bin], dtype=np.float64),
                exact_first_ar1=config.exact_first_ar1,
            )
            proj = fast_preproject(X_w, check_finite=False)
            fit = fast_lm_matrix(X_w, Y_w, proj, return_fitted=False, check_finite=False)
            betas[:, cols] = np.asarray(fit.betas, dtype=np.float64)
            sigma2[cols] = np.asarray(fit.sigma2, dtype=np.float64)
            if con is not None:
                est = con @ np.asarray(fit.betas, dtype=np.float64)
                var_factor = float(con @ np.asarray(proj.XtXinv, dtype=np.float64) @ con)
                effect[cols] = est
                variance[cols] = max(var_factor, 0.0) * np.asarray(fit.sigma2, dtype=np.float64)

        out = {"betas": betas, "sigma2": sigma2, "phi": phi_bins}
        if con is not None:
            t_vals, p_vals = _t_and_p(effect, variance, dfres=float(n_p))  # dfres set below
            out["effect"] = effect
            out["variance"] = variance
            out["t"] = t_vals
            out["p"] = p_vals
        return out

    # Initial OLS fit to seed residuals.
    proj_ols = fast_preproject(X_base, check_finite=False)
    fit_ols = fast_lm_matrix(X_base, Y_base, proj_ols, return_fitted=True, check_finite=False)
    beta_for_residual = np.asarray(fit_ols.betas, dtype=np.float64)
    dfres = float(proj_ols.dfres)

    outputs: dict = {}
    for _ in range(config.iter_gls):
        residuals = Y_base - X_base @ beta_for_residual
        phi = estimate_ar(residuals, order=1, voxelwise=True)
        outputs = _grouped_fit(phi)
        beta_for_residual = outputs["betas"]

    # Recompute t/p with the correct dfres (n_t - n_p).
    if con is not None:
        t_vals, p_vals = _t_and_p(outputs["effect"], outputs["variance"], dfres=dfres)
        outputs["t"] = t_vals
        outputs["p"] = p_vals

    return outputs
