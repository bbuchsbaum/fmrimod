"""Penalty matrix functions for HRF regularization."""

from __future__ import annotations

from typing import Optional
import numpy as np
from numpy.typing import NDArray

from .core import HRF


def penalty_matrix(hrf: HRF, order: int = 2, **kwargs) -> NDArray[np.float64]:
    """Generate penalty matrix for regularizing HRF basis coefficients.
    
    The penalty matrix encodes shape priors that discourage implausible
    or overly wiggly HRF estimates. Different HRF types use different
    penalty structures.
    
    Args:
        hrf: HRF object
        order: Order of the penalty (default: 2)
        **kwargs: Additional arguments for specific HRF types
        
    Returns:
        Symmetric positive definite penalty matrix of shape (nbasis, nbasis)
    """
    # Get HRF type from name or class
    hrf_type = getattr(hrf, 'name', '').upper()
    
    # Dispatch to specific penalty matrix functions
    if hrf_type in ['SPMG2', 'SPM2']:
        return _penalty_spmg2(hrf, order, **kwargs)
    elif hrf_type in ['SPMG3', 'SPM3']:
        return _penalty_spmg3(hrf, order, **kwargs)
    elif hrf_type in ['BSPLINE', 'BS']:
        return _penalty_roughness(hrf, order)
    elif hrf_type in ['FIR', 'TENT']:
        return _penalty_roughness(hrf, order)
    elif hrf_type == 'FOURIER':
        return _penalty_fourier(hrf, order)
    elif hrf_type == 'DAGUERRE':
        return _penalty_daguerre(hrf, order)
    else:
        # Default: identity matrix (ridge penalty)
        return np.eye(hrf.nbasis)


def _penalty_roughness(hrf: HRF, order: int = 2) -> NDArray[np.float64]:
    """Roughness penalty based on discrete derivatives.
    
    Used for FIR, B-spline, and tent basis functions.
    """
    nb = hrf.nbasis
    
    if nb <= 1:
        return np.eye(nb)
    elif nb > order:
        # Create difference matrix
        D = np.diff(np.eye(nb), n=order, axis=0)
        # Return D'D
        return D.T @ D
    else:
        return np.eye(nb)


def _penalty_spmg2(hrf: HRF, order: int = 2, shrink_deriv: float = 2.0, **kwargs) -> NDArray[np.float64]:
    """Penalty for SPM with temporal derivative.
    
    Canonical term is not penalized, derivative is shrunk.
    """
    nb = hrf.nbasis
    R = np.diag(np.ones(nb))
    
    if nb >= 1:
        R[0, 0] = 0  # Don't penalize canonical
    if nb >= 2:
        R[1, 1] = shrink_deriv  # Shrink temporal derivative
    
    return R


def _penalty_spmg3(hrf: HRF, order: int = 2, shrink_deriv: float = 2.0, **kwargs) -> NDArray[np.float64]:
    """Penalty for SPM with temporal and dispersion derivatives.
    
    Canonical term is not penalized, derivatives are shrunk.
    """
    nb = hrf.nbasis
    R = np.diag(np.ones(nb))
    
    if nb >= 1:
        R[0, 0] = 0  # Don't penalize canonical
    if nb >= 2:
        R[1, 1] = shrink_deriv  # Shrink temporal derivative
    if nb >= 3:
        R[2, 2] = shrink_deriv  # Shrink dispersion derivative
    
    return R


def _penalty_fourier(hrf: HRF, order: int = 2) -> NDArray[np.float64]:
    """Penalty for Fourier basis.
    
    Higher frequencies are penalized more.
    """
    nb = hrf.nbasis
    # Frequency indices (1, 1, 2, 2, 3, 3, ...)
    freqs = np.ceil(np.arange(1, nb + 1) / 2)
    # Penalty proportional to frequency^order
    return np.diag(freqs**order)


def _penalty_daguerre(hrf: HRF, order: int = 2) -> NDArray[np.float64]:
    """Penalty for Daguerre basis.
    
    Higher order terms are penalized more.
    """
    nb = hrf.nbasis
    # Penalty increases with basis index
    return np.diag((np.arange(nb))**2)