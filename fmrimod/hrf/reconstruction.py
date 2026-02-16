"""Reconstruction matrix for HRF basis."""

from __future__ import annotations

from typing import Union
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .core import HRF
from ..sampling import SamplingFrame


def reconstruction_matrix(
    hrf: HRF,
    sframe: Union[SamplingFrame, ArrayLike],
    **kwargs
) -> NDArray[np.float64]:
    """Generate reconstruction matrix for HRF basis.
    
    Returns a matrix Phi that converts basis coefficients into a 
    sampled HRF shape. Each column represents one basis function
    evaluated at the sampling times.
    
    Args:
        hrf: HRF object with basis functions
        sframe: SamplingFrame object or array of time points
        **kwargs: Additional arguments (for compatibility)
        
    Returns:
        Matrix with shape (n_samples, nbasis) where each column
        is a basis function evaluated at the sample times
        
    Examples:
        >>> # Get reconstruction matrix for SPMG3
        >>> from fmrimod import get_hrf, SamplingFrame
        >>> hrf = get_hrf("spmg3")
        >>> sf = SamplingFrame(blocklens=[100], tr=2.0)
        >>> times = sf.samples()
        >>> R = reconstruction_matrix(hrf, times)
        >>> R.shape
        (100, 3)
        >>> 
        >>> # Use with coefficients
        >>> coefs = np.array([1.0, 0.5, -0.2])
        >>> hrf_shape = R @ coefs  # Reconstructed HRF
    """
    # Get time points
    if isinstance(sframe, SamplingFrame):
        times = sframe.samples  # This is a property, not a method
    else:
        times = np.asarray(sframe)
    
    # Evaluate HRF basis at time points
    basis_values = hrf(times)
    
    # Ensure 2D matrix
    if hrf.nbasis == 1:
        if basis_values.ndim == 1:
            basis_values = basis_values.reshape(-1, 1)
    
    return basis_values