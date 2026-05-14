"""Integration with fmrimod HRF functions.

This module provides seamless integration between fmrimod and fmrimod,
allowing users to use all HRF functions from fmrimod within fmrimod's
formula syntax and design matrix construction.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ._warnings import call_safely, suppress_fmrimod_warnings

try:
    with suppress_fmrimod_warnings():
        from . import hrf as _hrf_pkg
        from .hrf import registry as _hrf_registry
        from .utils import misc as _utils_misc
except ImportError as err:
    raise ImportError(
        "fmrimod HRF subpackage is required for HRF functionality."
    ) from err

from .types import Array, HRFProtocol


class PyFMRIHRF(HRFProtocol):
    """Wrapper for fmrimod HRF functions.
    
    This class wraps fmrimod HRF objects to implement the HRFProtocol
    interface used throughout fmrimod.
    
    Parameters
    ----------
    hrf_spec : str or fmrimod.HRF
        HRF specification. Can be:
        - String name of HRF (e.g., "spm", "gamma", "SPMG1")
        - fmrimod HRF object
        - Dictionary with 'name' and optional parameters
    tr : float, optional
        Repetition time in seconds. Required for some HRFs.
    **kwargs
        Additional parameters passed to HRF constructor
    
    Attributes
    ----------
    hrf : fmrimod.HRF
        The wrapped HRF object
    name : str
        Name of the HRF
    nbasis : int
        Number of basis functions
    
    Examples
    --------
    >>> # From string name
    >>> hrf = PyFMRIHRF("spm")
    >>> 
    >>> # With parameters
    >>> hrf = PyFMRIHRF("gamma", shape=6, scale=1)
    >>> 
    >>> # From fmrimod object
    >>> spm_hrf = fmrimod.SPM_CANONICAL
    >>> hrf = PyFMRIHRF(spm_hrf)
    """
    
    def __init__(
        self,
        hrf_spec: Union[str, Any, Dict[str, Any]],
        tr: Optional[float] = None,
        **kwargs
    ):
        """Initialize PyFMRIHRF wrapper."""
        if isinstance(hrf_spec, str):
            # Get HRF by name
            self._name = hrf_spec
            request_kwargs = dict(kwargs)
            if tr is not None:
                request_kwargs["tr"] = tr
            self._hrf = self._get_hrf_by_name(hrf_spec, **request_kwargs)
        elif isinstance(hrf_spec, dict):
            # HRF specification with parameters
            params = dict(hrf_spec)
            name = params.pop('name', 'spm')
            self._name = name
            params = {**params, **kwargs}
            if tr is not None:
                params['tr'] = tr
            self._hrf = self._get_hrf_by_name(name, **params)
        elif hasattr(hrf_spec, 'evaluate'):
            # Already an HRF object
            self._hrf = hrf_spec
            self._name = getattr(hrf_spec, 'name', 'custom')
        else:
            raise TypeError(
                f"hrf_spec must be str, dict, or HRF object, got {type(hrf_spec)}"
            )
    
    def _get_hrf_by_name(self, name: str, **kwargs) -> Any:
        """Get fmrimod HRF by name.
        
        Parameters
        ----------
        name : str
            HRF name
        **kwargs
            Parameters for HRF
        
        Returns
        -------
        fmrimod.HRF
            HRF object
        """
        # Map common aliases
        name_map = {
            'spm': 'spmg1',
            'spm_canonical': 'spmg1',
            'canonical': 'spmg1',
            'spm_time': 'spmg2',
            'spm_time_derivative': 'spmg2',
            'spm_dispersion': 'spmg3',
            'spm_time_dispersion': 'spmg3',
        }
        
        # Normalize name
        name_lower = name.lower()
        if name_lower in name_map:
            name = name_map[name_lower]
        else:
            name = name_lower

        # Remove null kwargs to avoid triggering fmrimod warnings for ignored values.
        kwargs = {key: value for key, value in kwargs.items() if value is not None}

        # Get available HRFs
        available = _utils_misc.list_available_hrfs()
        available_lower = {avail.lower() for avail in available}

        # Try exact match first
        if name in available:
            return call_safely(_hrf_registry.get_hrf, name, **kwargs)

        # Try case-insensitive match
        if name in available_lower:
            for avail in available:
                if avail.lower() == name:
                    return call_safely(_hrf_registry.get_hrf, avail, **kwargs)

        if name != name_lower and name_lower in available_lower:
            for avail in available:
                if avail.lower() == name_lower:
                    return call_safely(_hrf_registry.get_hrf, avail, **kwargs)
        
        # If not found, raise error with suggestions
        raise ValueError(
            f"Unknown HRF '{name}'. Available HRFs: {', '.join(available)}"
        )
    
    @property
    def name(self) -> str:
        """Name of the HRF."""
        return self._name
    
    @property
    def nbasis(self) -> int:
        """Number of basis functions."""
        if hasattr(self._hrf, 'nbasis'):
            return self._hrf.nbasis
        elif hasattr(self._hrf, 'n_basis'):
            return self._hrf.n_basis
        else:
            # Default to 1 for simple HRFs
            return 1
    
    def evaluate(self, t: Union[float, Array]) -> Array:
        """Evaluate HRF at given time points.
        
        Parameters
        ----------
        t : float or array-like
            Time points in seconds
        
        Returns
        -------
        array-like
            HRF values at time points. Shape is (len(t),) for single basis
            or (len(t), nbasis) for multiple bases.
        """
        t = np.asarray(t)
        scalar_input = t.ndim == 0
        t = np.atleast_1d(t)
        
        # Call fmrimod evaluate
        if hasattr(self._hrf, 'evaluate'):
            result = self._hrf.evaluate(t)
        elif callable(self._hrf):
            # Some HRFs are directly callable
            result = self._hrf(t)
        else:
            raise AttributeError(f"HRF {self.name} has no evaluate method")
        
        # Handle output shape based on nbasis
        if self.nbasis == 1:
            # Single basis function - return 1D array
            result = result.ravel()
            if scalar_input:
                result = float(result[0]) if result.size else float("nan")
        else:
            # Multiple basis functions - ensure 2D
            result = np.atleast_2d(result)
            # Ensure correct orientation (time x basis)
            if result.shape[0] == self.nbasis and result.shape[1] == len(t):
                result = result.T
            
            # If single time point was passed, squeeze the time dimension
            if scalar_input and result.shape[0] == 1:
                result = result.squeeze(axis=0)
        
        return result
    
    def __repr__(self) -> str:
        """String representation."""
        return f"PyFMRIHRF(name='{self.name}', nbasis={self.nbasis})"


def get_fmrimod(
    hrf_spec: Union[str, Any, Dict[str, Any]],
    tr: Optional[float] = None,
    **kwargs
) -> PyFMRIHRF:
    """Get PyFMRIHRF wrapper for HRF specification.
    
    This is a convenience function for creating PyFMRIHRF objects.
    
    Parameters
    ----------
    hrf_spec : str, dict, or HRF
        HRF specification
    tr : float, optional
        Repetition time
    **kwargs
        Additional parameters
    
    Returns
    -------
    PyFMRIHRF
        Wrapped HRF object
    
    Examples
    --------
    >>> hrf = get_fmrimod("spm")
    >>> hrf = get_fmrimod("gamma", shape=6)
    >>> hrf = get_fmrimod({"name": "SPMG2"})
    """
    return PyFMRIHRF(hrf_spec, tr=tr, **kwargs)


def list_hrfs() -> List[str]:
    """List all available HRF functions.

    Returns
    -------
    list of str
        Available HRF names

    Examples
    --------
    >>> hrfs = list_hrfs()
    >>> print(hrfs[:5])
    ['SPMG1', 'SPMG2', 'SPMG3', 'bspline', 'fir']
    """
    return _utils_misc.list_available_hrfs()


def create_hrf_basis(
    name: str,
    n_basis: Optional[int] = None,
    length: Optional[float] = None,
    tr: Optional[float] = None,
    **kwargs
) -> PyFMRIHRF:
    """Create HRF basis set.
    
    Parameters
    ----------
    name : str
        Basis type: "fir", "bspline", "fourier", etc.
    n_basis : int, optional
        Number of basis functions
    length : float, optional
        Length of HRF in seconds
    tr : float, optional
        Repetition time
    **kwargs
        Additional parameters for basis
    
    Returns
    -------
    PyFMRIHRF
        HRF basis set
    
    Examples
    --------
    >>> # FIR basis with 10 time points
    >>> fir = create_hrf_basis("fir", n_basis=10, length=20)
    >>> 
    >>> # B-spline basis
    >>> bspline = create_hrf_basis("bspline", n_basis=5, length=20)
    """
    # Build parameters for generic and special-cased basis creation.
    params = {}
    if n_basis is not None:
        params['N'] = n_basis
    if length is not None:
        params['span'] = length
    if tr is not None:
        params['tr'] = tr
    params.update(kwargs)

    # Prefer generator-backed basis constructors when supported so that user-provided
    # basis-size and span settings are honored.
    basis_name = name.lower()
    basis_generators = {
        'fir': 'fir_generator',
        'bspline': 'bspline_generator',
        'fourier': 'fourier_generator',
        'daguerre': 'daguerre_generator',
    }
    if basis_name in basis_generators:
        gen_name = basis_generators[basis_name]
        generator = getattr(_hrf_pkg, gen_name, None)
        if generator is not None:
            signature = inspect.signature(generator)
            gen_kwargs = {
                key: value
                for key, value in params.items()
                if key in signature.parameters or any(
                    p.kind == inspect.Parameter.VAR_KEYWORD
                    for p in signature.parameters.values()
                )
            }
            if gen_kwargs and "name" not in gen_kwargs:
                gen_kwargs["name"] = basis_name
            hrf_obj = generator(**gen_kwargs)
            return PyFMRIHRF(hrf_obj)

    return PyFMRIHRF(name, **params)


# Register all fmrimod HRFs with fmrimod's registry
def register_fmrimod_hrfs():
    """Register all fmrimod HRFs with fmrimod.
    
    This function is called on module import to make all fmrimod
    HRFs available through fmrimod's get_hrf() function.
    """
    from .hrf.registry import _HRF_REGISTRY, register_hrf
    
    # Get all available HRFs
    available = list_hrfs()
    
    # Register each one
    for hrf_name in available:
        # Create a factory function for this HRF
        def make_hrf_factory(name):
            def factory(**kwargs):
                return PyFMRIHRF(name, **kwargs)
            return factory
        
        # Register only names not already owned by the canonical registry.
        if hrf_name.lower() not in _HRF_REGISTRY:
            register_hrf(hrf_name, make_hrf_factory(hrf_name))
        if (
            hrf_name.lower().startswith('spmg')
            and hrf_name.upper().lower() not in _HRF_REGISTRY
        ):
            register_hrf(hrf_name.upper(), make_hrf_factory(hrf_name))
    
    # Also register common aliases
    aliases = {
        'spm': 'SPMG1',
        'spm_canonical': 'SPMG1',
        'canonical': 'SPMG1',
        'spm_time': 'SPMG2',
        'spm_time_derivative': 'SPMG2',
        'spm_dispersion': 'SPMG3',
        'spm_time_dispersion': 'SPMG3',
    }
    
    for alias, target in aliases.items():
        if alias.lower() not in _HRF_REGISTRY:
            register_hrf(alias, make_hrf_factory(target))


# Auto-register on import
register_fmrimod_hrfs()


__all__ = [
    'PyFMRIHRF',
    'get_fmrimod',
    'list_hrfs',
    'create_hrf_basis',
    'register_fmrimod_hrfs',
]
