"""OASIS closed-form single-trial solver.

Reformulates per-trial LSS as a single batched matrix operation.
Supports single-basis (K=1) and multi-basis (K>1) HRFs, optional
ridge regularisation, and per-trial standard errors.

The algorithm precomputes residualised cross-products and solves
each trial's 2(K) x 2(K) normal equation system without forming
per-trial design matrices.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
from numpy.typing import NDArray
from scipy import linalg

from ._types import OasisConfig, OasisExtras, SingleTrialMethod, SingleTrialResult

# =========================================================================
# K = 1 path
# =========================================================================

def _oasis_precompute_design_k1(
    X_trials: NDArray[np.float64],
    Q: NDArray[np.float64],
) -> dict[str, Any]:
    """Precompute residualised design terms for single-basis OASIS.

    Parameters
    ----------
    X_trials : (T, N)
    Q : (T, p) thin QR factor of nuisance, or empty (T, 0).

    Returns
    -------
    dict with keys: A, s_all, d, alpha, s, s_all_norm2, Q
    """
    T, N = X_trials.shape

    # Residualise trial regressors
    A = X_trials.copy()
    if Q.shape[1] > 0:
        A -= Q @ (Q.T @ A)

    s_all = A.sum(axis=1)                    # (T,)
    s_all_norm2 = float(s_all @ s_all)       # scalar

    d = np.einsum("ij,ij->j", A, A)          # (N,) ||a_j||^2
    aTs = A.T @ s_all                         # (N,)
    alpha = aTs - d                           # (N,) c_j'b_j
    s = s_all_norm2 + d - 2.0 * aTs          # (N,) ||b_j||^2

    return dict(
        A=A, s_all=s_all, d=d, alpha=alpha, s=s,
        s_all_norm2=s_all_norm2, Q=Q,
    )


def _oasis_products_blocked_k1(
    A: NDArray[np.float64],
    s_all: NDArray[np.float64],
    Q: NDArray[np.float64],
    Y: NDArray[np.float64],
    block_cols: int = 4096,
) -> dict[str, Any]:
    """Compute blocked A'(RY) and s_all'(RY) products.

    Returns
    -------
    dict with N_Y (N, V), S_Y (V,), RY_norm2 (V,)
    """
    N = A.shape[1]
    V = Y.shape[1]
    N_Y = np.empty((N, V), dtype=np.float64)
    S_Y = np.empty(V, dtype=np.float64)
    RY_norm2 = np.empty(V, dtype=np.float64)

    for start in range(0, V, block_cols):
        end = min(start + block_cols, V)
        Yb = Y[:, start:end].copy()
        if Q.shape[1] > 0:
            Yb -= Q @ (Q.T @ Yb)
        N_Y[:, start:end] = A.T @ Yb
        S_Y[start:end] = s_all @ Yb
        RY_norm2[start:end] = np.einsum("ij,ij->j", Yb, Yb)

    return dict(N_Y=N_Y, S_Y=S_Y, RY_norm2=RY_norm2)


def _oasis_betas_k1(
    N_Y: NDArray[np.float64],
    S_Y: NDArray[np.float64],
    d: NDArray[np.float64],
    alpha: NDArray[np.float64],
    s: NDArray[np.float64],
    ridge_x: float = 0.0,
    ridge_b: float = 0.0,
    eps: float = 1e-12,
) -> NDArray[np.float64]:
    """Closed-form K=1 OASIS betas.

    Parameters
    ----------
    N_Y : (N, V) — A' @ R @ Y
    S_Y : (V,)  — s_all' @ R @ Y
    d, alpha, s : (N,) — design scalars per trial
    ridge_x, ridge_b : ridge regularisation

    Returns
    -------
    (N, V) trial beta estimates
    """
    N, V = N_Y.shape
    dj = d + ridge_x                           # (N,)
    ej = s + ridge_b                           # (N,)
    cj = alpha                                 # (N,)

    denom = dj * ej - cj ** 2                  # (N,)
    denom = np.maximum(denom, eps)

    # n2 = S_Y - N_Y[j, :] for each j
    # beta_j = (ej * n1 - cj * n2) / denom
    #        = (ej * N_Y[j] - cj * (S_Y - N_Y[j])) / denom
    #        = ((ej + cj) * N_Y[j] - cj * S_Y) / denom
    coeff_n1 = (ej + cj) / denom               # (N,)
    coeff_sy = cj / denom                       # (N,)

    B = coeff_n1[:, np.newaxis] * N_Y - coeff_sy[:, np.newaxis] * S_Y[np.newaxis, :]
    return B


def _oasis_betas_gammas_k1(
    N_Y: NDArray[np.float64],
    S_Y: NDArray[np.float64],
    d: NDArray[np.float64],
    alpha: NDArray[np.float64],
    s: NDArray[np.float64],
    ridge_x: float = 0.0,
    ridge_b: float = 0.0,
    eps: float = 1e-12,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Betas and gammas (aggregator coefficients) for SE computation."""
    N, V = N_Y.shape
    dj = d + ridge_x
    ej = s + ridge_b
    cj = alpha

    denom = np.maximum(dj * ej - cj ** 2, eps)

    # beta_j  = (ej * n1 - cj * n2) / denom
    # gamma_j = (dj * n2 - cj * n1) / denom
    n2 = S_Y[np.newaxis, :] - N_Y              # (N, V)

    betas = (ej[:, np.newaxis] * N_Y - cj[:, np.newaxis] * n2) / denom[:, np.newaxis]
    gammas = (dj[:, np.newaxis] * n2 - cj[:, np.newaxis] * N_Y) / denom[:, np.newaxis]

    return betas, gammas


def _oasis_se_k1(
    d: NDArray[np.float64],
    alpha: NDArray[np.float64],
    s: NDArray[np.float64],
    ridge_x: float,
    ridge_b: float,
    RY_norm2: NDArray[np.float64],
    betas: NDArray[np.float64],
    gammas: NDArray[np.float64],
    N_Y: NDArray[np.float64],
    S_Y: NDArray[np.float64],
    dof: float,
    eps: float = 1e-12,
) -> NDArray[np.float64]:
    """Standard errors from the 2x2 inverse Gram diagonal."""
    N, V = betas.shape
    dj = d + ridge_x
    ej = s + ridge_b
    cj = alpha

    n2 = S_Y[np.newaxis, :] - N_Y

    # SSE = ||RY||^2 - 2*(beta*n1 + gamma*n2) + beta^2*d + gamma^2*s + 2*beta*gamma*c
    SSE = (
        RY_norm2[np.newaxis, :]
        - 2.0 * (betas * N_Y + gammas * n2)
        + betas ** 2 * d[:, np.newaxis]
        + gammas ** 2 * s[:, np.newaxis]
        + 2.0 * betas * gammas * alpha[:, np.newaxis]
    )
    sigma2 = np.maximum(SSE / max(dof, 1.0), 0.0)

    # (G^{-1})_{11} = e_j / det(G)
    denom = np.maximum(dj * ej - cj ** 2, eps)
    g11 = ej / denom                           # (N,)

    return np.sqrt(sigma2 * g11[:, np.newaxis])


# =========================================================================
# K > 1 path
# =========================================================================

def _oasis_precompute_design_kn(
    X_trials: NDArray[np.float64],
    Q: NDArray[np.float64],
    K: int,
) -> dict[str, Any]:
    """Precompute residualised design terms for multi-basis OASIS.

    X_trials columns are interleaved: [t1_b1, t1_b2, ..., t1_bK, t2_b1, ...].
    """
    T, NK = X_trials.shape
    N = NK // K

    # Residualise
    A = X_trials.copy()
    if Q.shape[1] > 0:
        A -= Q @ (Q.T @ A)

    # S = sum of all trial blocks → (T, K)
    S = np.zeros((T, K), dtype=np.float64)
    for k in range(K):
        S[:, k] = A[:, k::K].sum(axis=1)

    # Per-trial Gram blocks
    D = np.empty((K, K, N), dtype=np.float64)   # trial Gram D_j
    C = np.empty((K, K, N), dtype=np.float64)   # cross C_j = a_j' b_j
    E = np.empty((K, K, N), dtype=np.float64)   # aggregator Gram E_j

    StS = S.T @ S                               # (K, K)

    for j in range(N):
        cols = slice(j * K, (j + 1) * K)
        Aj = A[:, cols]                          # (T, K)
        D[:, :, j] = Aj.T @ Aj
        AjS = Aj.T @ S                           # (K, K)
        C[:, :, j] = AjS - D[:, :, j]
        E[:, :, j] = StS - AjS - AjS.T + D[:, :, j]

    return dict(A=A, S=S, Q=Q, D=D, C=C, E=E, K=K, N=N)


def _oasis_products_blocked_kn(
    A: NDArray[np.float64],
    S: NDArray[np.float64],
    Q: NDArray[np.float64],
    Y: NDArray[np.float64],
    K: int,
    block_cols: int = 4096,
) -> dict[str, Any]:
    """Blocked products for multi-basis OASIS."""
    NK = A.shape[1]
    V = Y.shape[1]

    N1 = np.empty((NK, V), dtype=np.float64)    # A' R Y
    SY = np.empty((K, V), dtype=np.float64)      # S' R Y
    RY_norm2 = np.empty(V, dtype=np.float64)

    for start in range(0, V, block_cols):
        end = min(start + block_cols, V)
        Yb = Y[:, start:end].copy()
        if Q.shape[1] > 0:
            Yb -= Q @ (Q.T @ Yb)
        N1[:, start:end] = A.T @ Yb
        SY[:, start:end] = S.T @ Yb
        RY_norm2[start:end] = np.einsum("ij,ij->j", Yb, Yb)

    return dict(N1=N1, SY=SY, RY_norm2=RY_norm2)


def _oasis_betas_kn(
    D: NDArray[np.float64],
    C: NDArray[np.float64],
    E: NDArray[np.float64],
    N1: NDArray[np.float64],
    SY: NDArray[np.float64],
    K: int,
    ridge_x: float = 0.0,
    ridge_b: float = 0.0,
) -> NDArray[np.float64]:
    """Multi-basis OASIS betas via per-trial 2K x 2K block solve."""
    NK, V = N1.shape
    N = NK // K
    Ik = np.eye(K, dtype=np.float64)

    betas = np.empty((N * K, V), dtype=np.float64)

    for j in range(N):
        cols = slice(j * K, (j + 1) * K)
        Dj = D[:, :, j] + ridge_x * Ik         # (K, K)
        Ej = E[:, :, j] + ridge_b * Ik         # (K, K)
        Cj = C[:, :, j]                         # (K, K)

        # 2K x 2K system: [[D, C], [C', E]] @ [beta; gamma] = [n1; n2]
        G = np.block([
            [Dj, Cj],
            [Cj.T, Ej],
        ])

        n1 = N1[cols, :]                        # (K, V)
        n2 = SY - n1                            # (K, V)
        rhs = np.vstack([n1, n2])                # (2K, V)

        try:
            L = linalg.cho_factor(G)
            sol = linalg.cho_solve(L, rhs)
        except linalg.LinAlgError:
            sol = np.linalg.lstsq(G, rhs, rcond=None)[0]

        betas[cols, :] = sol[:K, :]

    return betas


# =========================================================================
# Ridge resolution
# =========================================================================

def _resolve_ridge(
    pre: dict[str, Any],
    config: OasisConfig,
) -> tuple[float, float]:
    """Convert fractional ridge to absolute values."""
    if config.ridge_mode == "none":
        return 0.0, 0.0
    if config.ridge_mode == "absolute":
        return config.ridge_x, config.ridge_b

    # Fractional: scale by mean design energy
    K = config.K
    if K == 1:
        mx = float(np.mean(pre["d"]))
        mb = float(np.mean(pre["s"]))
    else:
        D = pre["D"]
        E = pre["E"]
        N = D.shape[2]
        mx = float(np.mean([np.mean(np.diag(D[:, :, j])) for j in range(N)]))
        mb = float(np.mean([np.mean(np.diag(E[:, :, j])) for j in range(N)]))
    return config.ridge_x * mx, config.ridge_b * mb


# =========================================================================
# Public API
# =========================================================================

def oasis_single_trial(
    Y: NDArray[np.float64],
    X: NDArray[np.float64],
    confounds: Optional[NDArray[np.float64]] = None,
    config: Optional[OasisConfig] = None,
    trial_labels: Optional[List[str]] = None,
) -> SingleTrialResult:
    """OASIS closed-form single-trial estimation.

    Parameters
    ----------
    Y : NDArray, shape ``(n, V)``
        Data matrix.
    X : NDArray, shape ``(n, n_trials)`` or ``(n, n_trials * K)``
        Trial regressor matrix.
    confounds : NDArray, shape ``(n, q)``, optional
        Nuisance regressors.
    config : OasisConfig, optional
        Solver configuration.
    trial_labels : list of str, optional

    Returns
    -------
    SingleTrialResult
    """
    if config is None:
        config = OasisConfig()

    Y = np.asarray(Y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, np.newaxis]
    if Y.ndim != 2:
        raise ValueError("Y must be a 1-D or 2-D matrix")
    if X.ndim != 2:
        raise ValueError("X must be a 2-D matrix")
    if not np.all(np.isfinite(Y)):
        raise ValueError("Y must contain only finite values")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite values")

    n = Y.shape[0]
    if X.shape[0] != n:
        raise ValueError(f"Y has {n} timepoints, X has {X.shape[0]}.")
    K = config.K
    NK = X.shape[1]
    if NK % K != 0:
        raise ValueError(
            f"ncol(X)={NK} is not divisible by K={K}"
        )
    N = NK // K
    if trial_labels is not None and len(trial_labels) != N:
        raise ValueError(
            f"trial_labels has length {len(trial_labels)}, "
            f"expected {N}."
        )

    # Nuisance QR
    if confounds is not None:
        confounds = np.asarray(confounds, dtype=np.float64)
        if confounds.ndim == 1:
            confounds = confounds[:, np.newaxis]
        if confounds.ndim != 2:
            raise ValueError("confounds must be a 1-D or 2-D matrix")
        if confounds.shape[0] != n:
            raise ValueError(f"confounds has {confounds.shape[0]} rows, expected {n}.")
        if not np.all(np.isfinite(confounds)):
            raise ValueError("confounds must contain only finite values")
        Q, _ = np.linalg.qr(confounds, mode="reduced")
    else:
        Q = np.empty((n, 0), dtype=np.float64)

    if K == 1:
        pre = _oasis_precompute_design_k1(X, Q)
        mats = _oasis_products_blocked_k1(
            pre["A"], pre["s_all"], pre["Q"], Y,
            block_cols=config.block_cols,
        )
        ridge_x, ridge_b = _resolve_ridge(pre, config)

        if config.return_se:
            betas, gammas = _oasis_betas_gammas_k1(
                mats["N_Y"], mats["S_Y"],
                pre["d"], pre["alpha"], pre["s"],
                ridge_x=ridge_x, ridge_b=ridge_b,
            )
            nuis_rank = Q.shape[1]
            dof = max(1.0, n - nuis_rank - 2.0)
            se = _oasis_se_k1(
                pre["d"], pre["alpha"], pre["s"],
                ridge_x, ridge_b,
                mats["RY_norm2"], betas, gammas,
                mats["N_Y"], mats["S_Y"], dof,
            )
        else:
            betas = _oasis_betas_k1(
                mats["N_Y"], mats["S_Y"],
                pre["d"], pre["alpha"], pre["s"],
                ridge_x=ridge_x, ridge_b=ridge_b,
            )
            se = None
            dof = max(1.0, float(n - Q.shape[1] - 2))
    else:
        pre = _oasis_precompute_design_kn(X, Q, K)
        mats = _oasis_products_blocked_kn(
            pre["A"], pre["S"], pre["Q"], Y, K,
            block_cols=config.block_cols,
        )
        ridge_x, ridge_b = _resolve_ridge(pre, config)
        betas = _oasis_betas_kn(
            pre["D"], pre["C"], pre["E"],
            mats["N1"], mats["SY"], K,
            ridge_x=ridge_x, ridge_b=ridge_b,
        )
        se = None  # TODO: multi-basis SE
        nuis_rank = Q.shape[1]
        dof = max(1.0, float(n - nuis_rank - 2 * K))

    return SingleTrialResult(
        betas=betas,
        method=SingleTrialMethod.OASIS,
        trial_labels=list(trial_labels) if trial_labels is not None else None,
        residual_df=dof,
        se=se,
        extra=OasisExtras(K=K),
    )
