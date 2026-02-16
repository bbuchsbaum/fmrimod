"""BOLD signal simulation from events and HRF."""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray


def simulate_bold(
    design_matrix: NDArray[np.float64],
    betas: NDArray[np.float64],
    noise_sd: float = 1.0,
    ar_coeffs: Optional[NDArray[np.float64]] = None,
    n_voxels: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> NDArray[np.float64]:
    """Simulate BOLD time series from a design matrix and true betas.

    Parameters
    ----------
    design_matrix : NDArray
        Design matrix, shape ``(n, p)``.
    betas : NDArray
        True coefficients, shape ``(p,)`` or ``(p, V)``.
    noise_sd : float
        Standard deviation of noise.
    ar_coeffs : NDArray, optional
        AR noise coefficients for temporally correlated noise.
    n_voxels : int
        Number of voxels to simulate (if betas is 1-D).
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    NDArray
        Simulated data, shape ``(n, V)``.
    """
    if rng is None:
        rng = np.random.default_rng()

    X = np.asarray(design_matrix, dtype=np.float64)
    B = np.asarray(betas, dtype=np.float64)
    n, p = X.shape

    if B.ndim == 1:
        B = np.tile(B[:, np.newaxis], (1, n_voxels))

    V = B.shape[1]

    # Signal
    signal = X @ B  # (n, V)

    # Noise
    if ar_coeffs is not None and len(ar_coeffs) > 0:
        from .noise import ar_noise

        noise = ar_noise(n, V, ar_coeffs, sd=noise_sd, rng=rng)
    else:
        noise = rng.normal(0, noise_sd, size=(n, V))

    return signal + noise
