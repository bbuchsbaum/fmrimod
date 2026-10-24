"""Per-voxel HRF estimation and HRF-aware LSS.

Ports R ``fmrilss`` voxel_hrf.R: estimate voxel-specific HRF shapes
from a multi-basis GLM, then run LSS with per-voxel trial regressors
reconstructed from the estimated shapes.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal, Optional, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._project import project_nuisance
from ._types import (
    SingleTrialMethod,
    SingleTrialResult,
    VoxelHrfExtras,
    VoxelHrfResult,
)
from .lss import _lss_beta_vec

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

    from fmrimod.hrf import HRF
    from ..sampling import SamplingFrame

# R ``.normalize_voxel_hrf_coefficients`` defaults (fmrilss ``0d97c8b``).
_WAVEFORM_PRECISION = 0.05
_PEAK_TOLERANCE = 0.05


@runtime_checkable
class HrfEstimationDataset(Protocol):
    """Dataset surface required by formula-based HRF estimation."""

    n_runs: int
    event_table: pd.DataFrame
    sampling_frame: SamplingFrame

    def get_all_data(self) -> NDArray[np.float64]: ...


def _sample_hrf(
    hrf: Callable[[NDArray[np.float64]], ArrayLike], times: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Evaluate an HRF (or callable of time) as a ``(len(times), K)`` matrix."""
    return np.asarray(hrf(times), dtype=np.float64).reshape(times.size, -1)


def _voxel_hrf_waveforms(
    basis: object, ref_hrf: HRF | ArrayLike | None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Basis and reference waveforms, baseline-subtracted.

    Mirrors R ``.voxhrf_waveforms``: an fmrimod HRF basis is evaluated on a
    0.05 s grid over its span. An ``(L, K)`` array has no time axis, so it
    is used on its own rows and the reference is sampled at 1 s row spacing
    unless ``ref_hrf`` is itself a length-``L`` waveform on those rows.
    """
    from fmrimod.hrf import HRF, HRF_SPMG1

    if isinstance(basis, HRF):
        times = np.arange(
            0.0, float(basis.span) + _WAVEFORM_PRECISION / 2, _WAVEFORM_PRECISION
        )
        H = _sample_hrf(basis, times)
    else:
        try:
            H = np.asarray(basis, dtype=np.float64)
        except (TypeError, ValueError):
            H = np.empty((0, 0))
        if H.ndim == 1:
            H = H[:, np.newaxis]
        if H.ndim != 2 or H.shape[0] < 2:
            raise TypeError(
                "basis must be an fmrimod HRF or an (L, K) array of basis "
                "samples; its waveforms define the HRF shape normalization"
            )
        times = np.arange(H.shape[0], dtype=np.float64)

    if ref_hrf is None:
        ref_hrf = HRF_SPMG1
    if callable(ref_hrf):
        ref = _sample_hrf(ref_hrf, times)[:, 0]
    else:
        ref = np.asarray(ref_hrf, dtype=np.float64).reshape(-1)
        if ref.size != times.size:
            raise ValueError(
                f"ref_hrf samples must match the {times.size} basis waveform rows"
            )
    if not (np.all(np.isfinite(H)) and np.all(np.isfinite(ref))):
        raise ValueError("basis and ref_hrf waveforms must be finite")
    return H - H[0], ref - ref[0]


def _normalize_voxel_hrf_coefficients(
    coefficients: NDArray[np.float64],
    waveform_basis: NDArray[np.float64],
    ref: NDArray[np.float64],
    peak_tolerance: float = _PEAK_TOLERANCE,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Orient and peak-normalize voxel HRF coefficients.

    Mirrors R ``.normalize_voxel_hrf_coefficients`` (fmrilss ``0d97c8b``,
    ``6a2588b``): each voxel's waveform is oriented to correlate positively
    with ``ref`` and divided by its positive peak. A waveform whose positive
    peak is below ``peak_tolerance`` times its largest absolute deflection is
    *degenerate*: it is scaled by that deflection instead (left unscaled when
    identically zero) and flagged rather than failing the fit.

    Returns the normalized coefficients, the signed scale removed from each
    voxel (``amplitude_scale``) and the ``degenerate`` flags.
    """
    waveforms = waveform_basis @ coefficients
    orientation = np.sign(ref @ waveforms)
    orientation[~np.isfinite(orientation) | (orientation == 0)] = 1.0
    peak = (waveforms * orientation).max(axis=0)
    amp = np.abs(waveforms).max(axis=0)
    degenerate = ~np.isfinite(peak) | ~(peak >= peak_tolerance * amp) | (amp <= 0)
    peak = np.where(degenerate, amp, peak)
    peak = np.where(np.isfinite(peak) & (peak > 0), peak, 1.0)
    scale = orientation * peak
    return coefficients / scale, scale, degenerate


def _warn_degenerate_hrf(degenerate: NDArray[np.bool_]) -> None:
    """Warn once about voxels without an identifiable positive HRF peak."""
    n_bad = int(np.sum(degenerate))
    if n_bad:
        warnings.warn(
            f"{n_bad} of {degenerate.size} voxel(s) have no identifiable "
            "positive HRF peak; their shapes are scaled by the largest absolute "
            "response and flagged in `degenerate`, so their amplitudes are not "
            "in peak-response units",
            RuntimeWarning,
            stacklevel=3,
        )


def estimate_voxel_hrf(
    Y: NDArray[np.float64],
    X_trials: NDArray[np.float64],
    basis: object,
    confounds: Optional[NDArray[np.float64]] = None,
    K: Optional[int] = None,
    ref_hrf: HRF | ArrayLike | None = None,
) -> VoxelHrfResult:
    """Estimate per-voxel HRF via multi-basis aggregate GLM.

    Parameters
    ----------
    Y : NDArray, shape ``(T, V)``
        fMRI data.
    X_trials : NDArray, shape ``(T, N*K)``
        Trial design with K basis functions per trial, interleaved:
        ``[t1_b1, t1_b2, ..., t1_bK, t2_b1, ...]``.
    basis : HRF or NDArray
        HRF basis whose waveforms define the shape normalization: an
        :class:`fmrimod.hrf.HRF` (evaluated on a 0.05 s grid over its
        span, as in R), or an ``(L, K)`` array of basis samples, which is
        normalized on its own rows.
    confounds : NDArray, shape ``(T, q)``, optional
        Nuisance regressors to project out.
    K : int, optional
        Number of basis functions per trial.  Inferred from *basis*
        if omitted.
    ref_hrf : HRF or NDArray, optional
        Reference HRF that orients each voxel's shape (fmrilss
        ``0d97c8b``). Defaults to the canonical
        :data:`fmrimod.hrf.HRF_SPMG1`. For an array *basis* it is sampled
        at 1 s row spacing, or it may be given as a length-``L`` waveform
        on the basis rows.

    Returns
    -------
    VoxelHrfResult
        ``coefficients`` has shape ``(K, V)``. Each voxel's waveform is
        oriented to correlate positively with *ref_hrf* and has a unit
        positive peak, so a deactivating voxel keeps negative trial
        amplitudes. ``amplitude_scale`` holds the signed scale removed from
        the raw pooled fit. Voxels whose positive peak is below 5% of their
        largest absolute deflection are flagged in ``degenerate`` and scaled
        by that deflection; one ``RuntimeWarning`` reports how many.
    """
    Y = np.asarray(Y, dtype=np.float64)
    X_trials = np.asarray(X_trials, dtype=np.float64)
    if Y.ndim != 2:
        raise ValueError("Y must be a 2-D matrix")
    if X_trials.ndim != 2:
        raise ValueError("X_trials must be a 2-D matrix")
    T = Y.shape[0]
    if X_trials.shape[0] != T:
        raise ValueError(f"X_trials has {X_trials.shape[0]} rows, expected {T}")
    if not np.all(np.isfinite(Y)):
        raise ValueError("Y must contain only finite values")
    if not np.all(np.isfinite(X_trials)):
        raise ValueError("X_trials must contain only finite values")

    waveform_basis, ref = _voxel_hrf_waveforms(basis, ref_hrf)
    if K is None:
        K = waveform_basis.shape[1]

    NK = X_trials.shape[1]
    if K < 1:
        raise ValueError("K must be >= 1")
    if NK % K != 0:
        raise ValueError(f"ncol(X_trials)={NK} not divisible by K={K}")
    if waveform_basis.shape[1] != K:
        raise ValueError(
            f"basis has {waveform_basis.shape[1]} functions, expected K={K}"
        )

    # Aggregate per-basis regressors: A[:, k] = sum of all trial-k columns
    A = np.zeros((T, K), dtype=np.float64)
    for k in range(K):
        A[:, k] = X_trials[:, k::K].sum(axis=1)

    # Project out confounds
    if confounds is not None:
        confounds = np.asarray(confounds, dtype=np.float64)
        if confounds.ndim == 1:
            confounds = confounds[:, np.newaxis]
        if confounds.ndim != 2:
            raise ValueError("confounds must be a 1-D or 2-D matrix")
        if confounds.shape[0] != T:
            raise ValueError(f"confounds has {confounds.shape[0]} rows, expected {T}")
        Y_c, A_c = project_nuisance(confounds, Y, A)
    else:
        Y_c, A_c = Y, A

    # Solve A'A @ W = A'Y via least squares → W (K, V)
    W, _, _, _ = np.linalg.lstsq(A_c, Y_c, rcond=None)

    coefficients, scale, degenerate = _normalize_voxel_hrf_coefficients(
        W, waveform_basis, ref
    )
    _warn_degenerate_hrf(degenerate)

    return VoxelHrfResult(
        coefficients=coefficients,
        basis=basis,
        conditions=[],
        amplitude_scale=scale,
        degenerate=degenerate,
        normalization="positive-peak",
    )


def _parse_onset_column(form: object) -> str:
    """Extract onset column name from a formula-like string."""
    if not isinstance(form, str):
        raise TypeError("form must be a string when using dataset-based estimate_hrf")
    if "~" not in form:
        raise ValueError("form must contain '~' (for example: 'onset ~ hrf(condition)')")
    lhs = form.split("~", 1)[0].strip()
    if lhs == "":
        raise ValueError("form must include a left-hand onset column")
    return lhs


def _build_trial_basis_design(
    onsets: NDArray[np.float64],
    basis: NDArray[np.float64],
    n_timepoints: int,
    tr: float,
) -> NDArray[np.float64]:
    """Build interleaved trial-basis design matrix from onset times."""
    onsets = np.asarray(onsets, dtype=np.float64).reshape(-1)
    B = np.asarray(basis, dtype=np.float64)
    if B.ndim != 2:
        raise ValueError("basis must be 2-D for dataset-based estimate_hrf")
    L, K = B.shape
    n_trials = int(onsets.shape[0])
    X_trials = np.zeros((n_timepoints, n_trials * K), dtype=np.float64)
    onset_idx = np.rint(onsets / float(tr)).astype(int)

    for i, idx0 in enumerate(onset_idx):
        if idx0 < 0 or idx0 >= n_timepoints:
            continue
        max_len = min(L, n_timepoints - idx0)
        for k in range(K):
            col = i * K + k
            X_trials[idx0 : idx0 + max_len, col] = B[:max_len, k]
    return X_trials


def estimate_hrf(
    Y: NDArray[np.float64] | None = None,
    X_trials: NDArray[np.float64] | None = None,
    basis: NDArray[np.float64] | object | None = None,
    *,
    confounds: Optional[NDArray[np.float64]] = None,
    K: Optional[int] = None,
    output: Literal["hrf", "coefficients", "result"] = "hrf",
    form: Optional[str] = None,
    fixed: Optional[NDArray[np.float64]] = None,
    block: Optional[str] = None,
    dataset: HrfEstimationDataset | None = None,
    ref_hrf: HRF | ArrayLike | None = None,
) -> NDArray[np.float64] | VoxelHrfResult:
    """Estimate HRF using matrix inputs or a limited formula+dataset path.

    This parity helper provides the ``estimate_hrf`` entry point from fmrireg.
    Supported workflows:
    - Matrix mode: provide ``Y``, ``X_trials``, and ``basis`` directly.
    - Formula+dataset mode (initial subset): provide ``form``, ``dataset``,
      and a 2-D ``basis``. Requires a single-run dataset with an ``event_table``
      onset column referenced by ``form``.

    Parameters
    ----------
    Y : NDArray, shape ``(T, V)``
        Data matrix.
    X_trials : NDArray, shape ``(T, N*K)``
        Trial design matrix with interleaved basis columns.
    basis : object
        Basis representation. If array-like ``(L, K)``, ``output="hrf"``
        returns reconstructed HRFs with shape ``(L, V)``.
    confounds : NDArray, optional
        Nuisance regressors.
    K : int, optional
        Number of basis functions per trial.
    output : {"hrf", "coefficients", "result"}, default "hrf"
        Output mode:
        - ``"hrf"``: reconstructed HRF matrix when possible, else coefficients.
        - ``"coefficients"``: return basis coefficients ``(K, V)``.
        - ``"result"``: return :class:`VoxelHrfResult`.
    form, fixed, block, dataset : Any, optional
        Compatibility arguments for formula+dataset mode. Current limitations:
        - ``dataset`` must expose ``get_all_data()``, ``sampling_frame``,
          ``event_table``, and ``n_runs``.
        - Only single-run datasets are supported.
        - ``fixed`` may be passed as a numeric confound matrix; formula-based
          fixed-effects construction is not yet implemented.
        - ``block`` is accepted for signature compatibility but not used yet.
    ref_hrf : HRF or NDArray, optional
        Orientation reference forwarded to :func:`estimate_voxel_hrf`. In
        formula+dataset mode the TR-sampled basis rows are aligned with a
        reference sampled at the dataset TR.

    Returns
    -------
    NDArray or VoxelHrfResult
        Estimated HRFs, coefficients, or full result container.
    """
    if dataset is not None or form is not None:
        if dataset is None or form is None:
            raise ValueError("dataset mode requires both form and dataset")
        if basis is None:
            raise ValueError("dataset mode requires basis")
        if not isinstance(dataset, HrfEstimationDataset):
            raise TypeError(
                "dataset mode requires an object with get_all_data(), "
                "event_table, sampling_frame, and n_runs"
            )
        if int(dataset.n_runs) != 1:
            raise NotImplementedError("dataset mode currently supports single-run datasets only")

        onset_col = _parse_onset_column(form)
        events = dataset.event_table
        if events is None:
            raise ValueError("dataset mode requires dataset.event_table")
        if onset_col not in events.columns:
            raise ValueError(f"onset column '{onset_col}' not found in dataset.event_table")

        Y = np.asarray(dataset.get_all_data(), dtype=np.float64)
        tr = float(dataset.sampling_frame.TR)
        basis_2d = np.asarray(basis, dtype=np.float64)
        X_trials = _build_trial_basis_design(
            onsets=np.asarray(events[onset_col], dtype=np.float64),
            basis=basis_2d,
            n_timepoints=Y.shape[0],
            tr=tr,
        )
        if ref_hrf is None or callable(ref_hrf):
            from fmrimod.hrf import HRF_SPMG1

            ref_times = np.arange(basis_2d.shape[0], dtype=np.float64) * tr
            reference = HRF_SPMG1 if ref_hrf is None else ref_hrf
            ref_hrf = _sample_hrf(reference, ref_times)[:, 0]
        if fixed is not None:
            if isinstance(fixed, np.ndarray):
                confounds = np.asarray(fixed, dtype=np.float64)
            else:
                raise NotImplementedError(
                    "dataset mode currently supports fixed as a numeric confound matrix only"
                )
    else:
        if Y is None or X_trials is None or basis is None:
            raise ValueError("estimate_hrf requires Y, X_trials, and basis")

    result = estimate_voxel_hrf(
        Y=np.asarray(Y, dtype=np.float64),
        X_trials=np.asarray(X_trials, dtype=np.float64),
        basis=basis,
        confounds=None if confounds is None else np.asarray(confounds, dtype=np.float64),
        K=K,
        ref_hrf=ref_hrf,
    )

    if output == "result":
        return result
    if output == "coefficients":
        return result.coefficients
    if output != "hrf":
        raise ValueError("output must be one of: 'hrf', 'coefficients', 'result'")

    basis_array = np.asarray(basis)
    if basis_array.ndim == 2 and basis_array.shape[1] == result.coefficients.shape[0]:
        return np.asarray(basis_array, dtype=np.float64) @ result.coefficients
    return result.coefficients


def lss_with_voxel_hrf(
    Y: NDArray[np.float64],
    X_trials: NDArray[np.float64],
    hrf_result: VoxelHrfResult,
    confounds: Optional[NDArray[np.float64]] = None,
    chunk_size: int = 5000,
) -> SingleTrialResult:
    """LSS with per-voxel HRF shapes.

    For each voxel, reconstructs scalar trial regressors from the
    multi-basis columns weighted by the estimated HRF coefficients,
    then runs vectorised LSS.

    Parameters
    ----------
    Y : NDArray, shape ``(T, V)``
    X_trials : NDArray, shape ``(T, N*K)``
        Same interleaved multi-basis format as :func:`estimate_voxel_hrf`.
    hrf_result : VoxelHrfResult
        From :func:`estimate_voxel_hrf`.
    confounds : NDArray, shape ``(T, q)``, optional
    chunk_size : int
        Voxels per chunk.

    Returns
    -------
    SingleTrialResult
        ``extra`` is a :class:`VoxelHrfExtras` repeating the per-voxel
        ``degenerate`` flags and ``normalization`` of *hrf_result*: with
        positive-peak shapes, betas are peak-response amplitudes except for
        degenerate voxels.
    """
    Y = np.asarray(Y, dtype=np.float64)
    X_trials = np.asarray(X_trials, dtype=np.float64)
    if Y.ndim != 2:
        raise ValueError("Y must be a 2-D matrix")
    if X_trials.ndim != 2:
        raise ValueError("X_trials must be a 2-D matrix")
    T, V = Y.shape
    K = hrf_result.coefficients.shape[0]
    NK = X_trials.shape[1]
    if X_trials.shape[0] != T:
        raise ValueError(f"X_trials has {X_trials.shape[0]} rows, expected {T}")
    if K < 1:
        raise ValueError("hrf_result must have at least one basis coefficient")
    if NK % K != 0:
        raise ValueError(f"ncol(X_trials)={NK} not divisible by K={K}")
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    if not np.all(np.isfinite(Y)):
        raise ValueError("Y must contain only finite values")
    if not np.all(np.isfinite(X_trials)):
        raise ValueError("X_trials must contain only finite values")
    N = NK // K

    if hrf_result.coefficients.shape[1] != V:
        raise ValueError(
            f"HRF result has {hrf_result.coefficients.shape[1]} voxels, "
            f"Y has {V}"
        )

    # Project out confounds
    if confounds is not None:
        confounds = np.asarray(confounds, dtype=np.float64)
        if confounds.ndim == 1:
            confounds = confounds[:, np.newaxis]
        if confounds.ndim != 2:
            raise ValueError("confounds must be a 1-D or 2-D matrix")
        if confounds.shape[0] != T:
            raise ValueError(f"confounds has {confounds.shape[0]} rows, expected {T}")
        Y_c, X_c = project_nuisance(confounds, Y, X_trials)
    else:
        Y_c, X_c = Y, X_trials

    betas = np.empty((N, V), dtype=np.float64)

    # Process voxels in chunks
    for start_v in range(0, V, chunk_size):
        end_v = min(start_v + chunk_size, V)
        n_v = end_v - start_v
        W_chunk = hrf_result.coefficients[:, start_v:end_v]  # (K, n_v)

        # Build per-voxel scalar trial regressors: (T, N, n_v)
        # For trial n, voxel v: x_nv = sum_k W[k,v] * X_c[:, n*K+k]
        X_vox = np.zeros((T, N, n_v), dtype=np.float64)
        for n in range(N):
            trial_cols = X_c[:, n * K : (n + 1) * K]  # (T, K)
            X_vox[:, n, :] = trial_cols @ W_chunk      # (T, n_v)

        Y_chunk = Y_c[:, start_v:end_v]                # (T, n_v)

        # Run vectorised LSS per voxel (each has different X)
        for v_local in range(n_v):
            C_v = X_vox[:, :, v_local]                  # (T, N)
            y_v = Y_chunk[:, v_local : v_local + 1]     # (T, 1)
            b_v = _lss_beta_vec(C_v, y_v)               # (N, 1)
            betas[:, start_v + v_local] = b_v.ravel()

    return SingleTrialResult(
        betas=betas,
        method=SingleTrialMethod.LSS_VOXEL_HRF,
        residual_df=float(T - 2),
        extra=VoxelHrfExtras(
            degenerate=hrf_result.degenerate,
            normalization=hrf_result.normalization,
        ),
    )
