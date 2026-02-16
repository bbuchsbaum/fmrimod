"""Noise generation for fMRI simulation."""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray


def white_noise(
    n: int,
    V: int = 1,
    sd: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> NDArray[np.float64]:
    """Generate white Gaussian noise.

    Parameters
    ----------
    n : int
        Number of timepoints.
    V : int
        Number of voxels.
    sd : float
        Standard deviation.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    NDArray
        Noise matrix, shape ``(n, V)``.
    """
    if rng is None:
        rng = np.random.default_rng()
    return rng.normal(0, sd, size=(n, V))


def ar_noise(
    n: int,
    V: int = 1,
    phi: Optional[NDArray[np.float64]] = None,
    sd: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> NDArray[np.float64]:
    """Generate AR(p) noise.

    Parameters
    ----------
    n : int
        Number of timepoints.
    V : int
        Number of voxels.
    phi : NDArray, optional
        AR coefficients.  If *None*, generates white noise.
    sd : float
        Innovation standard deviation.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    NDArray
        AR noise, shape ``(n, V)``.
    """
    if rng is None:
        rng = np.random.default_rng()

    if phi is None or len(phi) == 0:
        return white_noise(n, V, sd, rng)

    phi = np.asarray(phi, dtype=np.float64).ravel()
    p = len(phi)

    innovations = rng.normal(0, sd, size=(n, V))
    noise = np.zeros((n, V), dtype=np.float64)

    for t in range(n):
        noise[t] = innovations[t]
        for k in range(min(t, p)):
            noise[t] += phi[k] * noise[t - k - 1]

    return noise
