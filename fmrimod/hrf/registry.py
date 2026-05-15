"""HRF registry for managing available HRF types."""

from __future__ import annotations

import inspect
import logging
import math
from typing import Any, Callable, Dict, List, Optional, Union, cast

from ..types import HRFProtocol
from .aliases import HRFName, _normalize_hrf_name
from .core import HRF

logger = logging.getLogger(__name__)
from .library import PREDEFINED_HRFS

HRFEntry = Union[HRF, Callable[..., HRF]]

# Global HRF registry
_HRF_REGISTRY: Dict[str, HRFEntry] = {}

# Initialize with predefined HRFs
_HRF_REGISTRY.update(cast(Dict[str, HRFEntry], PREDEFINED_HRFS))


def _callable_params(func: Callable[..., object]) -> tuple[set[str], bool]:
    sig = inspect.signature(func)
    params = sig.parameters
    accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    return set(params), accepts_var_kwargs


def _validate_kwargs(kind: str, kwargs: dict[str, object], accepted: set[str]) -> None:
    unknown = sorted(set(kwargs) - accepted)
    if unknown:
        raise ValueError(
            f"Unknown parameter(s) for HRF {kind!r}: {', '.join(unknown)}. "
            f"Accepted parameters: {', '.join(sorted(accepted))}."
        )


def _normalize_generator_kwargs(kind: str, kwargs: dict[str, object]) -> dict[str, object]:
    """Normalize legacy parameter aliases before constructor validation."""
    out = dict(kwargs)
    if kind == "gamma" and "scale" in out:
        if "rate" in out:
            raise ValueError("Use either gamma scale or rate, not both.")
        try:
            scale = float(cast(Any, out.pop("scale")))
        except (TypeError, ValueError):
            raise ValueError("gamma scale must be numeric") from None
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("gamma scale must be finite and > 0")
        out["rate"] = 1.0 / scale
    return out


def _instantiate_predefined_hrf(
    kind: str,
    hrf: HRF,
    kwargs: dict[str, object],
) -> HRF:
    cls = type(hrf)
    accepted, accepts_var_kwargs = _callable_params(cls)
    ctor_kwargs = _normalize_generator_kwargs(kind, kwargs)
    if (
        "n_basis" in ctor_kwargs
        and "nbasis" in accepted
        and "n_basis" not in accepted
    ):
        ctor_kwargs["nbasis"] = ctor_kwargs.pop("n_basis")
    if not accepts_var_kwargs:
        _validate_kwargs(kind, ctor_kwargs, accepted)
    return cls(**cast(Any, ctor_kwargs))


def _add_to_registry(
    name: str,
    hrf: HRFEntry,
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
    name: Union[HRFName, str],
    lag: float = 0.0,
    width: float = 0.0,
    summate: bool = True,
    normalize: bool = False,
    block_width: Optional[float] = None,
    **kwargs: object,
) -> HRF:
    """Get an HRF from the registry by name.

    Args:
        name: Canonical HRF name (typed via :data:`HRFName`) or an
            accepted alias (``"spm"``, ``"hrf_bspline"``, ...). Aliases
            are collapsed through :func:`aliases._normalize_hrf_name`
            before registry lookup.
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
        ValueError: If HRF name not found in registry or alias table.
    """
    try:
        canonical = _normalize_hrf_name(name)
    except ValueError:
        canonical = cast(Any, str(name).strip().lower())

    if canonical not in _HRF_REGISTRY:
        available = [str(x) for x in list_available_hrfs()]
        raise ValueError(
            f"HRF {name!r} (normalized to {canonical!r}) not found in registry. "
            f"Available HRFs: {', '.join(available)}"
        )

    hrf_or_gen = _HRF_REGISTRY[canonical]
    kwargs_local = _normalize_generator_kwargs(canonical, dict(kwargs))

    generator_key = f"hrf_{canonical}"
    if kwargs_local and generator_key in _HRF_REGISTRY:
        maybe_generator = _HRF_REGISTRY[generator_key]
        if callable(maybe_generator) and not isinstance(maybe_generator, HRF):
            hrf_or_gen = maybe_generator

    # If it's already an HRF-like object (HRF subclass or HRFProtocol-compatible),
    # return it. The ``not inspect.isfunction`` guard rejects plain factory
    # functions that happen to carry name/nbasis/evaluate attributes; only
    # generator callables registered via the ``hrf_<name>`` slot should hit
    # the generator branch below.
    _is_hrf_obj = isinstance(hrf_or_gen, HRF) or (
        isinstance(hrf_or_gen, HRFProtocol)
        and not inspect.isfunction(hrf_or_gen)
    )
    result: HRF
    if _is_hrf_obj:
        if kwargs_local:
            if not isinstance(hrf_or_gen, HRF):
                raise ValueError(
                    f"Parameters {kwargs_local} cannot be applied to non-HRF "
                    f"object {name!r}."
                )
            result = _instantiate_predefined_hrf(canonical, hrf_or_gen, kwargs_local)
        else:
            result = cast(HRF, hrf_or_gen)
    else:
        # Otherwise it's a generator function. Route width correctly when the
        # generator itself takes a width parameter.
        generator = cast(Callable[..., HRF], hrf_or_gen)
        accepted, accepts_var_kwargs = _callable_params(generator)

        decorator_width = width if block_width is None else block_width
        if (
            block_width is None
            and width != 0
            and "width" in accepted
            and "width" not in kwargs_local
        ):
            kwargs_local["width"] = width
            decorator_width = 0.0

        # Match R behavior: only pass args accepted by the generator.
        if accepts_var_kwargs:
            gen_kwargs = kwargs_local
        else:
            _validate_kwargs(canonical, kwargs_local, accepted)
            gen_kwargs = kwargs_local

        result = generator(**gen_kwargs)

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


def list_available_hrfs(
    details: bool = False,
) -> Union[List[str], List[dict[str, object]]]:
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

    result: List[dict[str, object]] = []
    for name in names:
        hrf = _HRF_REGISTRY[name]
        hrf_id = id(hrf)
        nbasis_default: Optional[Union[int, float]] = None
        if isinstance(hrf, HRF):
            hrf_type = "object"
            nbasis_default = hrf.nbasis
        else:
            hrf_type = "generator"
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
    hrf: HRFEntry,
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
        _HRF_REGISTRY.update(cast(Dict[str, HRFEntry], PREDEFINED_HRFS))
    else:
        _HRF_REGISTRY.clear()


# Register HRF generator functions
def _register_generators() -> None:
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
