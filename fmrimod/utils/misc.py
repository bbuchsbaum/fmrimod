"""Miscellaneous utility functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, List, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..hrf.core import HRF

if TYPE_CHECKING:
    from ..regressor import Regressor
else:
    Regressor = Any


def recycle_or_error(
    x: ArrayLike,
    n: int,
    name: str
) -> NDArray[np.float64]:
    """Recycle scalar to array or validate array length.

    Args:
        x: Scalar or array-like values
        n: Expected length
        name: Parameter name for error messages

    Returns:
        Array of length n

    Raises:
        ValueError: If array length doesn't match n
    """
    x = np.asarray(x, dtype=np.float64)

    if x.ndim == 0 or len(x) == 1:
        # Scalar or single element - recycle
        return np.full(n, x.item() if x.ndim == 0 else x[0])
    elif len(x) == n:
        return x
    else:
        raise ValueError(
            f"`{name}` must have length 1 or {n}, not {len(x)}"
        )


def list_available_hrfs(details: bool = False) -> Union[List[str], List[dict[str, Any]]]:
    """List all available HRF types.

    Args:
        details: If True, return detailed metadata for each registered HRF.

    Returns:
        List of HRF names or metadata dictionaries
    """
    from ..hrf.registry import list_available_hrfs as registry_list_available_hrfs
    return registry_list_available_hrfs(details=details)


def single_trial_regressor(
    onset: float,
    hrf: Union[str, "HRF"] = "spmg1",
    duration: float = 0.0,
    amplitude: float = 1.0,
    span: Optional[float] = None
) -> "Regressor":
    """Create a single trial regressor.

    Convenience wrapper for creating a regressor with a single event.

    Args:
        onset: Event onset in seconds (scalar)
        hrf: HRF specification
        duration: Event duration (scalar)
        amplitude: Event amplitude (scalar)
        span: Temporal window (if None, uses HRF default)

    Returns:
        Regressor object
    """
    from ..regressor import regressor

    # Validate scalar inputs
    if not np.isscalar(onset):
        raise ValueError("onset must be a scalar")
    if not np.isscalar(duration):
        raise ValueError("duration must be a scalar")
    if not np.isscalar(amplitude):
        raise ValueError("amplitude must be a scalar")

    if span is None:
        return regressor(
            onsets=[onset],
            hrf=hrf,
            duration=[duration],
            amplitude=[amplitude]
        )
    else:
        return regressor(
            onsets=[onset],
            hrf=hrf,
            duration=[duration],
            amplitude=[amplitude],
            span=span
        )


def hrf_toeplitz(
    hrf: Union[Callable, "HRF"],
    time: ArrayLike,
    length: int,
    sparse: bool = False
) -> NDArray[np.float64]:
    """Create Toeplitz matrix for HRF convolution.

    Args:
        hrf: HRF function or object
        time: Time points for HRF evaluation
        length: Length of output matrix
        sparse: Whether to return sparse matrix

    Returns:
        Toeplitz matrix for convolution
    """
    from scipy.linalg import toeplitz

    # Evaluate HRF
    if callable(hrf):
        hrf_values = hrf(time)
    else:
        hrf_values = hrf(time)

    # Flatten in column-major order, matching R's matrix-to-vector coercion.
    hrf_values = np.asarray(hrf_values)
    hrf_values = hrf_values.reshape(-1, order='F')

    if length <= 0:
        raise ValueError("length must be positive")
    if len(hrf_values) > length:
        raise ValueError(
            "invalid times argument: length must be >= number of HRF samples"
        )

    # Create Toeplitz matrix
    col = np.pad(
        hrf_values,
        (0, length - len(hrf_values)),
        mode='constant',
        constant_values=0.0,
    )

    # First row: first HRF value followed by zeros
    row = np.zeros(length)
    row[0] = hrf_values[0]

    matrix = toeplitz(col, row)

    if sparse:
        from scipy.sparse import csr_matrix
        return csr_matrix(matrix)

    return matrix
