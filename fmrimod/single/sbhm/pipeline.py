"""SBHM pipeline: orchestrate library → prepass → match → amplitude."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from .._types import SbhmConfig, SingleTrialMethod, SingleTrialResult
from .amplitude import sbhm_amplitude
from .library import SbhmLibrary
from .match import sbhm_match
from .prepass import sbhm_prepass


def sbhm_single_trial(
    Y: NDArray[np.float64],
    X: NDArray[np.float64],
    confounds: Optional[NDArray[np.float64]] = None,
    config: Optional[SbhmConfig] = None,
    trial_labels: Optional[list[Any]] = None,
    library: Optional[SbhmLibrary] = None,
) -> SingleTrialResult:
    """SBHM single-trial estimation pipeline.

    Orchestrates the full SBHM workflow:
    1. Build library (if not provided)
    2. Prepass: aggregate regression
    3. Match: find best HRF per voxel
    4. Amplitude: single-trial estimation with matched HRF

    Parameters
    ----------
    Y : NDArray, shape ``(T, V)``
        Data matrix (time x voxels).
    X : NDArray, shape ``(T, N*K)``
        Interleaved trial regressors: [t1_b1, t1_b2, ..., t1_bK, t2_b1, ...].
    confounds : NDArray, shape ``(T, q)``, optional
        Nuisance regressors.
    config : SbhmConfig, optional
        SBHM configuration. If not provided, uses defaults.
    trial_labels : list of str, optional
        Labels for each trial.
    library : SbhmLibrary, optional
        Pre-built library. If not provided, one must be constructed externally
        or passed via config.

    Returns
    -------
    SingleTrialResult
        Result with:
        - ``betas``: shape ``(N, V)``, single-trial amplitudes
        - ``method``: ``"sbhm"``
        - ``extra``: dict with SBHM-specific info (matched_idx, margin, etc.)

    Notes
    -----
    The SBHM pipeline requires a library of candidate HRF shapes. This can be
    provided explicitly via the ``library`` parameter, or constructed from a
    basis (not implemented in this function; must be done externally).

    The aggregate regressor matrix ``A_agg`` is constructed by summing trial
    regressors per basis column::

        A_agg[:, k] = sum_n X[:, n*K + k]

    Examples
    --------
    >>> import numpy as np
    >>> from fmrimod.single.sbhm import sbhm_single_trial, build_sbhm_library
    >>> from fmrimod.single import SbhmConfig
    >>> # Build library
    >>> library_H = np.random.randn(20, 50)
    >>> library = build_sbhm_library(library_H, r=3)
    >>> # Generate data
    >>> Y = np.random.randn(100, 500)
    >>> X = np.random.randn(100, 60)  # 20 trials x 3 basis
    >>> config = SbhmConfig(r=3, amplitude_method="oasis_voxel")
    >>> result = sbhm_single_trial(Y, X, config=config, library=library)
    >>> result.betas.shape
    (20, 500)
    """
    if config is None:
        config = SbhmConfig()

    Y = np.asarray(Y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)

    if Y.ndim == 1:
        Y = Y[:, np.newaxis]

    T, V = Y.shape
    NK = X.shape[1]

    if X.shape[0] != T:
        raise ValueError(f"Y has {T} timepoints, X has {X.shape[0]}.")

    if library is None:
        raise ValueError(
            "SBHM requires a library. Pass via 'library' parameter or "
            "construct externally using build_sbhm_library()."
        )

    # Infer K from library
    K = library.B.shape[1]
    if NK % K != 0:
        raise ValueError(
            f"X has {NK} columns, not divisible by library rank K={K}"
        )
    N = NK // K

    # Build aggregate regressor: sum per basis column
    A_agg = np.zeros((T, K), dtype=np.float64)
    for k in range(K):
        A_agg[:, k] = X[:, k::K].sum(axis=1)

    # Step 1: Prepass
    beta_bar, G = sbhm_prepass(Y, A_agg, confounds=confounds)

    # Step 2: Match
    match_result = sbhm_match(
        beta_bar=beta_bar,
        S=library.S,
        A=library.A,
        shrink=config.shrink,
        top_k=config.top_k,
    )

    # Step 3: Amplitude estimation
    betas = sbhm_amplitude(
        Y=Y,
        X_trials=X,
        alpha_coords=match_result.alpha_coords,
        confounds=confounds,
        method=config.amplitude_method,
        ridge_x=config.ridge_lambda,
        ridge_b=config.ridge_lambda,
        K=K,
    )

    # Residual DOF
    nuis_rank = confounds.shape[1] if confounds is not None else 0
    dof = max(1.0, T - nuis_rank - 2.0)

    # Extra info
    extra = {
        "matched_idx": match_result.matched_idx,
        "margin": match_result.margin,
        "alpha_coords": match_result.alpha_coords,
        "similarity": match_result.similarity,
        "K": K,
        "library": library,
    }
    if match_result.weights is not None:
        extra["weights"] = match_result.weights

    return SingleTrialResult(
        betas=betas,
        method=SingleTrialMethod.SBHM,
        trial_labels=list(trial_labels) if trial_labels is not None else None,
        residual_df=dof,
        se=None,
        extra=extra,
    )
