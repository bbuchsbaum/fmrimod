"""Bootstrap confidence intervals for GLM parameters.

Supports residual, case, and wild bootstrap with block structure
that respects temporal dependencies and run boundaries in fMRI data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats

from .solver import fast_preproject, fast_lm_matrix


class BootstrapMethod(str, Enum):
    RESIDUAL = "residual"
    CASE = "case"
    WILD = "wild"


@dataclass
class BootstrapResult:
    """Result of bootstrap inference on a GLM.

    Attributes
    ----------
    boot_betas : NDArray, shape ``(n_boot, p, V)``
        Bootstrap distribution of coefficients.
    beta_ci : NDArray, shape ``(2, p, V)``
        Lower and upper confidence bounds for betas.
    beta_se : NDArray, shape ``(p, V)``
        Bootstrap standard errors.
    contrast_ci : dict
        Per-contrast confidence intervals (if contrasts supplied).
    observed_betas : NDArray, shape ``(p, V)``
        Original point estimates.
    confidence : float
        Confidence level used.
    n_boot : int
        Number of bootstrap iterations completed.
    method : str
        Bootstrap method used.
    """

    boot_betas: NDArray[np.float64]
    beta_ci: NDArray[np.float64]
    beta_se: NDArray[np.float64]
    contrast_ci: dict = field(default_factory=dict)
    observed_betas: Optional[NDArray[np.float64]] = None
    confidence: float = 0.95
    n_boot: int = 0
    method: str = "residual"


# -- Block construction -------------------------------------------------------

def create_blocks(
    n: int,
    block_size: int,
    run_boundaries: Optional[Sequence[int]] = None,
) -> List[NDArray[np.intp]]:
    """Partition time indices into blocks respecting run boundaries.

    Parameters
    ----------
    n : int
        Total number of time points.
    block_size : int
        Target block size.
    run_boundaries : sequence of int, optional
        Indices where new runs begin (e.g. ``[0, 200, 400]``).
        If ``None``, treats all data as one run.

    Returns
    -------
    list of NDArray[int]
        Each element is an array of contiguous indices forming one block.
    """
    if not isinstance(block_size, (int, np.integer)) or int(block_size) <= 0:
        raise ValueError("block_size must be a positive integer")
    block_size = int(block_size)

    if run_boundaries is None:
        run_boundaries = [0]
    boundaries = [int(b) for b in run_boundaries]
    if not boundaries or boundaries[0] != 0:
        raise ValueError("run_boundaries must start at 0")
    if any(b < 0 or b >= n for b in boundaries):
        raise ValueError("run_boundaries entries must be in [0, n)")
    if any(boundaries[i] >= boundaries[i + 1] for i in range(len(boundaries) - 1)):
        raise ValueError("run_boundaries must be strictly increasing")

    boundaries = boundaries + [n]

    blocks: List[NDArray[np.intp]] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        run_indices = np.arange(start, end, dtype=np.intp)
        # Split this run's indices into blocks
        for i in range(0, len(run_indices), block_size):
            blocks.append(run_indices[i : i + block_size])
    return blocks


# -- Resampling schemes -------------------------------------------------------

def _resample_residual(
    X: NDArray[np.float64],
    fitted: NDArray[np.float64],
    residuals: NDArray[np.float64],
    blocks: List[NDArray[np.intp]],
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Residual bootstrap: resample blocks of residuals, add to fitted."""
    n = X.shape[0]
    new_idx = _sample_block_indices(blocks, n, rng)
    Y_star = fitted + residuals[new_idx]
    X_star = X  # design unchanged in residual bootstrap
    return X_star, Y_star


def _resample_case(
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    blocks: List[NDArray[np.intp]],
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Case bootstrap: resample blocks of (X, Y) pairs."""
    n = X.shape[0]
    new_idx = _sample_block_indices(blocks, n, rng)
    return X[new_idx], Y[new_idx]


def _sample_block_indices(
    blocks: List[NDArray[np.intp]],
    n: int,
    rng: np.random.Generator,
) -> NDArray[np.intp]:
    """Draw block-bootstrap indices, guaranteeing exactly ``n`` rows."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not blocks:
        raise ValueError("blocks must be non-empty")

    total = 0
    picks: List[NDArray[np.intp]] = []
    while total < n:
        block = blocks[int(rng.integers(0, len(blocks)))]
        picks.append(block)
        total += int(block.shape[0])

    return np.concatenate(picks)[:n]


def _resample_wild(
    X: NDArray[np.float64],
    fitted: NDArray[np.float64],
    residuals: NDArray[np.float64],
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Wild bootstrap: Rademacher weights on residuals."""
    signs = rng.choice([-1.0, 1.0], size=(residuals.shape[0], 1))
    Y_star = fitted + signs * residuals
    return X, Y_star


# -- BCa confidence intervals -------------------------------------------------

def _bca_ci(
    boot_dist: NDArray[np.float64],
    observed: float,
    confidence: float,
) -> tuple[float, float]:
    """Bias-corrected and accelerated confidence interval (scalar).

    Parameters
    ----------
    boot_dist : NDArray, shape ``(n_boot,)``
        Bootstrap distribution for one parameter.
    observed : float
        Original point estimate.
    confidence : float
        Confidence level (e.g. 0.95).

    Returns
    -------
    (lower, upper)
    """
    n_boot = len(boot_dist)
    alpha = 1.0 - confidence

    # Bias correction factor z0
    prop_below = np.mean(boot_dist < observed)
    prop_below = np.clip(prop_below, 1e-10, 1.0 - 1e-10)
    z0 = sp_stats.norm.ppf(prop_below)

    # Acceleration factor (jackknife)
    mean_all = np.mean(boot_dist)
    diffs = mean_all - boot_dist
    a_hat = np.sum(diffs**3) / (6.0 * np.sum(diffs**2) ** 1.5 + 1e-15)

    # Adjusted percentiles
    z_alpha_lo = sp_stats.norm.ppf(alpha / 2.0)
    z_alpha_hi = sp_stats.norm.ppf(1.0 - alpha / 2.0)

    def _adjusted_pct(z_alpha: float) -> float:
        num = z0 + z_alpha
        denom = 1.0 - a_hat * num
        if abs(denom) < 1e-15:
            return sp_stats.norm.cdf(z0 + z_alpha)
        return sp_stats.norm.cdf(z0 + num / denom)

    p_lo = _adjusted_pct(z_alpha_lo)
    p_hi = _adjusted_pct(z_alpha_hi)

    lo = np.percentile(boot_dist, 100.0 * np.clip(p_lo, 0, 1))
    hi = np.percentile(boot_dist, 100.0 * np.clip(p_hi, 0, 1))
    return float(lo), float(hi)


# -- Main entry point ---------------------------------------------------------

def bootstrap_glm(
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    n_boot: int = 1000,
    method: Union[str, BootstrapMethod] = "residual",
    block_size: int = 10,
    confidence: float = 0.95,
    run_boundaries: Optional[Sequence[int]] = None,
    contrasts: Optional[dict[str, NDArray[np.float64]]] = None,
    use_bca: bool = True,
    seed: Optional[int] = None,
) -> BootstrapResult:
    """Bootstrap confidence intervals for a GLM.

    Parameters
    ----------
    X : NDArray, shape ``(n, p)``
        Design matrix.
    Y : NDArray, shape ``(n, V)``
        Data matrix.
    n_boot : int
        Number of bootstrap iterations.
    method : str
        ``"residual"``, ``"case"``, or ``"wild"``.
    block_size : int
        Block size for temporal resampling.
    confidence : float
        Confidence level (e.g. 0.95).
    run_boundaries : sequence of int, optional
        Time indices where new runs start.
    contrasts : dict, optional
        Named contrast vectors ``{name: (p,) array}``.
    use_bca : bool
        Use BCa adjustment for confidence intervals.
    seed : int, optional
        Random seed.

    Returns
    -------
    BootstrapResult
    """
    if not isinstance(n_boot, (int, np.integer)) or int(n_boot) <= 0:
        raise ValueError("n_boot must be a positive integer")
    n_boot = int(n_boot)

    if not isinstance(block_size, (int, np.integer)) or int(block_size) <= 0:
        raise ValueError("block_size must be a positive integer")
    block_size = int(block_size)

    try:
        confidence = float(confidence)
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence must be between 0 and 1") from exc
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be between 0 and 1")

    method = BootstrapMethod(method)
    rng = np.random.default_rng(seed)

    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, np.newaxis]
    n, p = X.shape
    V = Y.shape[1]

    # Original fit
    proj = fast_preproject(X)
    orig = fast_lm_matrix(X, Y, proj, return_fitted=True)
    fitted = orig.fitted
    residuals = Y - fitted

    blocks = create_blocks(n, block_size, run_boundaries)

    # Bootstrap iterations
    boot_betas = np.empty((n_boot, p, V), dtype=np.float64)
    for b in range(n_boot):
        if method is BootstrapMethod.RESIDUAL:
            X_star, Y_star = _resample_residual(X, fitted, residuals, blocks, rng)
        elif method is BootstrapMethod.CASE:
            X_star, Y_star = _resample_case(X, Y, blocks, rng)
        elif method is BootstrapMethod.WILD:
            X_star, Y_star = _resample_wild(X, fitted, residuals, rng)
        else:
            raise ValueError(f"Unknown method: {method}")

        proj_star = fast_preproject(X_star)
        result_star = fast_lm_matrix(X_star, Y_star, proj_star)
        boot_betas[b] = result_star.betas

    # Standard errors
    beta_se = np.std(boot_betas, axis=0, ddof=1)

    # Confidence intervals
    alpha = 1.0 - confidence
    if use_bca:
        beta_ci = np.empty((2, p, V), dtype=np.float64)
        for j in range(p):
            for v in range(V):
                lo, hi = _bca_ci(boot_betas[:, j, v], orig.betas[j, v], confidence)
                beta_ci[0, j, v] = lo
                beta_ci[1, j, v] = hi
    else:
        beta_ci = np.stack([
            np.percentile(boot_betas, 100 * alpha / 2, axis=0),
            np.percentile(boot_betas, 100 * (1 - alpha / 2), axis=0),
        ])

    # Contrast CIs
    contrast_ci: dict = {}
    if contrasts:
        for name, cvec in contrasts.items():
            cvec = np.asarray(cvec, dtype=np.float64)
            # boot contrast estimates: (n_boot, V)
            boot_con = np.einsum("j,bjv->bv", cvec, boot_betas)
            obs_con = cvec @ orig.betas
            if use_bca:
                lo_arr = np.empty(V)
                hi_arr = np.empty(V)
                for v in range(V):
                    lo_arr[v], hi_arr[v] = _bca_ci(
                        boot_con[:, v], obs_con[v], confidence,
                    )
                contrast_ci[name] = np.stack([lo_arr, hi_arr])
            else:
                contrast_ci[name] = np.stack([
                    np.percentile(boot_con, 100 * alpha / 2, axis=0),
                    np.percentile(boot_con, 100 * (1 - alpha / 2), axis=0),
                ])

    return BootstrapResult(
        boot_betas=boot_betas,
        beta_ci=beta_ci,
        beta_se=beta_se,
        contrast_ci=contrast_ci,
        observed_betas=orig.betas,
        confidence=confidence,
        n_boot=n_boot,
        method=method.value,
    )
