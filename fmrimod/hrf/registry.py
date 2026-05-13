"""HRF registry for managing available HRF types."""

from __future__ import annotations

import inspect
import logging
from typing import Callable, Dict, List, Optional, Union

from .core import HRF

logger = logging.getLogger(__name__)
from .library import PREDEFINED_HRFS

# Global HRF registry
_HRF_REGISTRY: Dict[str, Union[HRF, Callable]] = {}

# Initialize with predefined HRFs
_HRF_REGISTRY.update(PREDEFINED_HRFS)


def _add_to_registry(
    name: str,
    hrf: Union[HRF, Callable],
    aliases: Optional[List[str]] = None
) -> None:
    """Add an HRF to the registry.
    
    Args:
        name: Primary name for the HRF
        hrf: HRF object or callable that returns an HRF
        aliases: Optional list of alternative names
    """
    # Add primary name
    _HRF_REGISTRY[name.lower()] = hrf
    
    # Add aliases
    if aliases:
        for alias in aliases:
            _HRF_REGISTRY[alias.lower()] = hrf


def get_hrf(
    name: str,
    lag: float = 0.0,
    width: float = 0.0,
    summate: bool = True,
    normalize: bool = False,
    block_width: Optional[float] = None,
    **kwargs,
) -> HRF:
    """Get an HRF from the registry by name.

    Args:
        name: Name of the HRF (case-insensitive)
        lag: Time shift to apply to the HRF (in seconds)
        width: Block width in seconds. For generators that define a ``width``
            argument (e.g., ``boxcar``, ``weighted``), this value is routed to
            the generator unless ``block_width`` is explicitly provided.
        summate: Summation mode when applying block decoration.
        normalize: If True, normalize the HRF to unit peak amplitude.
        block_width: Explicit block width decorator override. When provided,
            this value is used for decoration even if the generator accepts a
            ``width`` argument.
        **kwargs: Additional parameters passed to HRF generators

    Returns:
        HRF object

    Raises:
        ValueError: If HRF name not found in registry
    """
    name_lower = name.lower()

    if name_lower not in _HRF_REGISTRY:
        available = list_available_hrfs()
        raise ValueError(
            f"HRF '{name}' not found in registry. "
            f"Available HRFs: {', '.join(available)}"
        )

    hrf_or_gen = _HRF_REGISTRY[name_lower]
    kwargs_local = dict(kwargs)

    # If it's already an HRF-like object (HRF or duck-typed), return it
    _is_hrf_obj = isinstance(hrf_or_gen, HRF) or (
        hasattr(hrf_or_gen, 'evaluate') and hasattr(hrf_or_gen, 'name')
        and hasattr(hrf_or_gen, 'nbasis') and not inspect.isfunction(hrf_or_gen)
    )
    if _is_hrf_obj:
        if kwargs_local:
            # Warn that parameters are ignored for pre-defined HRFs
            import warnings
            warnings.warn(
                f"Parameters {kwargs_local} ignored for pre-defined HRF '{name}'",
                UserWarning
            )
        result = hrf_or_gen
    else:
        # Otherwise it's a generator function. Route width correctly when the
        # generator itself takes a width parameter.
        sig = inspect.signature(hrf_or_gen)
        params = sig.parameters
        accepts_var_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )

        decorator_width = width if block_width is None else block_width
        if (
            block_width is None
            and width != 0
            and "width" in params
            and "width" not in kwargs_local
        ):
            kwargs_local["width"] = width
            decorator_width = 0.0

        # Match R behavior: only pass args accepted by the generator.
        if accepts_var_kwargs:
            gen_kwargs = kwargs_local
        else:
            gen_kwargs = {k: v for k, v in kwargs_local.items() if k in params}

        result = hrf_or_gen(**gen_kwargs)

        width = decorator_width

    # Apply decorator parameters in R order: block -> lag -> normalize
    if width != 0:
        if width < 0:
            raise ValueError("width must be non-negative")
        from .decorators import block_hrf
        result = block_hrf(result, width=width, summate=summate)

    if lag != 0:
        from .decorators import lag_hrf
        result = lag_hrf(result, lag=lag)

    if normalize:
        from .normalization import normalize as _normalize

        result = _normalize(result, "unit_peak_per_basis")

    return result


def list_available_hrfs(details: bool = False) -> Union[List[str], List[dict]]:
    """List registered HRF names and optional metadata.

    Args:
        details: If True, return list of dicts with metadata.

    Returns:
        If details=False: Sorted list of registered names.
        If details=True: List of dictionaries with metadata per registered name.
    """
    # Hide internal generator registry entries from public listing.
    names = sorted(name for name in _HRF_REGISTRY.keys() if not name.startswith("hrf_"))

    if not details:
        return names

    # First, find canonical names (first alphabetical name for each unique HRF).
    id_to_primary = {}
    for name in names:
        hrf = _HRF_REGISTRY[name]
        hrf_id = id(hrf)
        if hrf_id not in id_to_primary:
            id_to_primary[hrf_id] = name

    result = []
    for name in names:
        hrf = _HRF_REGISTRY[name]
        hrf_id = id(hrf)
        is_hrf_obj = isinstance(hrf, HRF)
        hrf_type = "object" if is_hrf_obj else "generator"

        nbasis_default: Optional[Union[int, float]] = None
        if is_hrf_obj:
            nbasis_default = hrf.nbasis
        else:
            # Try to inspect generator defaults for basis count.
            try:
                sig = inspect.signature(hrf)
                for param_name in ["n_basis", "nbasis"]:
                    if param_name in sig.parameters:
                        default = sig.parameters[param_name].default
                        if default != inspect.Parameter.empty:
                            nbasis_default = default
                            break
            except (ValueError, TypeError):
                pass

        is_alias = id_to_primary[hrf_id] != name

        result.append({
            "name": name,
            "type": hrf_type,
            "nbasis_default": nbasis_default,
            "is_alias": is_alias,
            "description": f"{name} HRF ({'alias' if is_alias else hrf_type})",
        })

    return result


def register_hrf(
    name: str,
    hrf: Union[HRF, Callable],
    aliases: Optional[List[str]] = None,
    force: bool = False
) -> None:
    """Register a custom HRF in the global registry.
    
    Args:
        name: Name for the HRF
        hrf: HRF object or callable that returns an HRF
        aliases: Optional list of alternative names
        force: If True, overwrite existing entries
        
    Raises:
        ValueError: If name already exists and force=False
    """
    name_lower = name.lower()
    
    # Check for existing entries
    if not force:
        existing = []
        if name_lower in _HRF_REGISTRY:
            existing.append(name)
        if aliases:
            for alias in aliases:
                if alias.lower() in _HRF_REGISTRY:
                    existing.append(alias)
        
        if existing:
            raise ValueError(
                f"HRF names already exist in registry: {', '.join(existing)}. "
                f"Use force=True to overwrite."
            )
    
    _add_to_registry(name, hrf, aliases)


def remove_hrf(name: str) -> None:
    """Remove an HRF from the registry.
    
    Args:
        name: Name of the HRF to remove
        
    Raises:
        ValueError: If HRF not found in registry
    """
    name_lower = name.lower()
    
    if name_lower not in _HRF_REGISTRY:
        raise ValueError(f"HRF '{name}' not found in registry")
    
    # Find all aliases pointing to the same HRF
    hrf_to_remove = _HRF_REGISTRY[name_lower]
    names_to_remove = [
        key for key, value in _HRF_REGISTRY.items()
        if value is hrf_to_remove
    ]
    
    # Remove all references
    for key in names_to_remove:
        del _HRF_REGISTRY[key]


def clear_registry(keep_predefined: bool = True) -> None:
    """Clear the HRF registry.
    
    Args:
        keep_predefined: If True, keep pre-defined HRFs and only clear custom ones
    """
    if keep_predefined:
        _HRF_REGISTRY.clear()
        _HRF_REGISTRY.update(PREDEFINED_HRFS)
    else:
        _HRF_REGISTRY.clear()


# Register HRF generator functions
def _register_generators():
    """Register HRF generator functions with the registry."""
    try:
        from .generators import (
            boxcar_generator,
            bspline_generator,
            daguerre_generator,
            fir_generator,
            fourier_generator,
            gamma_generator,
            tent_generator,
            weighted_generator,
        )

        # HRF generators
        _add_to_registry("hrf_gamma", gamma_generator)

        # Basis generators
        _add_to_registry("hrf_bspline", bspline_generator)
        _add_to_registry("hrf_fir", fir_generator)
        _add_to_registry("hrf_fourier", fourier_generator)
        _add_to_registry("hrf_daguerre", daguerre_generator)
        _add_to_registry("daguerre", daguerre_generator)

        # New generators
        _add_to_registry("hrf_tent", tent_generator)
        _add_to_registry("tent", tent_generator)
        _add_to_registry("hrf_boxcar", boxcar_generator)
        _add_to_registry("boxcar", boxcar_generator)
        _add_to_registry("hrf_weighted", weighted_generator)
        _add_to_registry("weighted", weighted_generator)
    except ImportError:
        logger.debug("HRF generators not yet available for registration")


# Register generators on module import
try:
    _register_generators()
except ImportError:
    logger.debug("HRF generators module not yet available")
