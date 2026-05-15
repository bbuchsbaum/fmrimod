"""Beta extraction: OLS and LSS (Least Squares Separate) estimation.

.. deprecated::
    This module is a backward-compatibility shim.  New code should use
    :mod:`fmrimod.single` instead, which provides vectorised LSS, OASIS,
    SBHM, and mixed-model solvers.

Provides trial-wise beta estimation for single-trial analyses (e.g.
representational similarity analysis, decoding).

LSS (Mumford et al., 2012) estimates each trial's activation by
fitting a separate GLM where the trial of interest has its own
regressor and all other trials are collapsed into a single "nuisance"
regressor.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Sequence, cast

import numpy as np
from numpy.typing import NDArray

from ..single._types import SingleTrialResult


class BetaMethod(str, Enum):
    OLS = "ols"
    LSS = "lss"


@dataclass
class BetaResult:
    """Result of trial-wise beta extraction.

    .. deprecated::
        Use :class:`fmrimod.single.SingleTrialResult` instead.

    Attributes
    ----------
    betas : NDArray, shape ``(n_trials, V)``
        Per-trial beta estimates.
    method : str
        Estimation method used.
    trial_labels : list or None
        Labels identifying each trial.
    residual_df : float
        Residual degrees of freedom (from last/representative fit).
    """

    betas: NDArray[np.float64]
    method: str
    trial_labels: Optional[list[Any]] = None
    residual_df: float = 0.0

    @classmethod
    def from_single_trial_result(cls, result: SingleTrialResult) -> "BetaResult":
        """Convert a :class:`SingleTrialResult` to a legacy :class:`BetaResult`."""
        return cls(
            betas=result.betas,
            method=result.method,
            trial_labels=result.trial_labels,
            residual_df=result.residual_df,
        )


def _build_trial_regressors(
    onsets: NDArray[np.float64],
    durations: NDArray[np.float64],
    hrf_kernel: NDArray[np.float64],
    n_time: int,
    tr: float,
) -> NDArray[np.float64]:
    """Build per-trial regressors by convolving boxcar with HRF.

    Parameters
    ----------
    onsets : NDArray, shape ``(n_trials,)``
        Trial onset times in seconds.
    durations : NDArray, shape ``(n_trials,)``
        Trial durations in seconds.
    hrf_kernel : NDArray, shape ``(n_hrf,)``
        Sampled HRF kernel at TR resolution.
    n_time : int
        Number of TRs.
    tr : float
        Repetition time in seconds.

    Returns
    -------
    NDArray, shape ``(n_time, n_trials)``
    """
    n_trials = len(onsets)
    regressors = np.zeros((n_time, n_trials), dtype=np.float64)

    for i in range(n_trials):
        # Build boxcar for this trial
        boxcar = np.zeros(n_time, dtype=np.float64)
        start_tr = int(round(onsets[i] / tr))
        dur_trs = max(1, int(round(durations[i] / tr)))
        end_tr = min(start_tr + dur_trs, n_time)
        if 0 <= start_tr < n_time:
            boxcar[start_tr:end_tr] = 1.0

        # Convolve with HRF and truncate
        conv = np.convolve(boxcar, hrf_kernel)[:n_time]
        regressors[:, i] = conv

    return regressors


def estimate_betas_ols(
    trial_regressors: NDArray[np.float64],
    Y: NDArray[np.float64],
    confounds: Optional[NDArray[np.float64]] = None,
    baseline_regressors: Optional[NDArray[np.float64]] = None,
    include_intercept: bool = False,
) -> BetaResult:
    """Estimate trial-wise betas via standard OLS.

    Delegates to :func:`fmrimod.single.lsa_single_trial`.
    """
    from ..single.lsa import lsa_single_trial

    result = lsa_single_trial(
        Y,
        trial_regressors,
        confounds=confounds,
        baseline_regressors=baseline_regressors,
        include_intercept=include_intercept,
    )
    br = BetaResult.from_single_trial_result(result)
    br.method = "ols"  # preserve legacy method name
    return br


def estimate_betas_lss(
    trial_regressors: NDArray[np.float64],
    Y: NDArray[np.float64],
    confounds: Optional[NDArray[np.float64]] = None,
    nuisance_projector: Optional[object] = None,
    chunk_size: Optional[int] = None,
    baseline_regressors: Optional[NDArray[np.float64]] = None,
    include_intercept: bool = False,
) -> BetaResult:
    """Estimate trial-wise betas via LSS (Least Squares Separate).

    Delegates to :func:`fmrimod.single.lss_single_trial` (vectorised).

    References
    ----------
    Mumford, J. A., Turner, B. O., Ashby, F. G., & Poldrack, R. A.
    (2012). Deconvolving BOLD activation in event-related designs for
    multivoxel pattern classification analyses. *NeuroImage*, 59(3),
    2636-2643.
    """
    from ..single.lss import lss_single_trial

    result = lss_single_trial(
        Y,
        trial_regressors,
        confounds=confounds,
        nuisance_projector=cast(Any, nuisance_projector),
        chunk_size=chunk_size,
        baseline_regressors=baseline_regressors,
        include_intercept=include_intercept,
    )
    return BetaResult.from_single_trial_result(result)


def estimate_betas(
    trial_regressors: NDArray[np.float64],
    Y: NDArray[np.float64],
    method: str = "lss",
    confounds: Optional[NDArray[np.float64]] = None,
    nuisance_projector: Optional[object] = None,
    chunk_size: Optional[int] = None,
    trial_labels: Optional[Sequence[str]] = None,
    baseline_regressors: Optional[NDArray[np.float64]] = None,
    include_intercept: bool = False,
) -> BetaResult:
    """Estimate trial-wise betas.

    .. deprecated::
        Use :func:`fmrimod.single.estimate_single_trial` instead.

    Dispatcher for OLS and LSS beta estimation.

    Parameters
    ----------
    trial_regressors : NDArray, shape ``(n_time, n_trials)``
        Per-trial design columns (already convolved with HRF).
    Y : NDArray, shape ``(n_time, V)``
        Data matrix.
    method : str
        ``"ols"`` or ``"lss"`` (default).
    confounds : NDArray, optional
        Nuisance regressors.
    nuisance_projector : object, optional
        Precomputed nuisance projector for LSS.
    chunk_size : int, optional
        Voxel chunk size for beta-only LSS solves.
    trial_labels : sequence of str, optional
        Labels for each trial.
    baseline_regressors : NDArray, optional
        Baseline or experimental regressors included in every LSS model.
    include_intercept : bool
        Add an intercept to the LSS adjustment design.

    Returns
    -------
    BetaResult
    """
    method_enum = BetaMethod(method)
    if method_enum is BetaMethod.OLS:
        result = estimate_betas_ols(
            trial_regressors,
            Y,
            confounds,
            baseline_regressors=baseline_regressors,
            include_intercept=include_intercept,
        )
    elif method_enum is BetaMethod.LSS:
        result = estimate_betas_lss(
            trial_regressors,
            Y,
            confounds=confounds,
            nuisance_projector=nuisance_projector,
            chunk_size=chunk_size,
            baseline_regressors=baseline_regressors,
            include_intercept=include_intercept,
        )
    else:
        raise ValueError(f"Unknown method: {method!r}")

    if trial_labels is not None:
        result.trial_labels = list(trial_labels)
    return result
