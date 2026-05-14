"""Extension registry for custom HRF specification types.

Allows external packages to register custom HRF spec types that integrate
with the fmrimod formula and convolution pipeline.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class HRFSpecExtension:
    """Registration info for an external HRF spec type."""
    spec_class: str
    package: str
    convolved_class: Optional[str] = None
    requires_external_processing: bool = False
    formula_functions: Optional[List[str]] = None
    registered_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# Module-level private registry
_registry: Dict[str, HRFSpecExtension] = {}


def _validate_single_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or value == "":
        raise TypeError(f"{name} must be a single non-empty string")
    return value


def _coerce_formula_functions(
    formula_functions: Optional[Any],
) -> Optional[List[str]]:
    if formula_functions is None:
        return None
    if isinstance(formula_functions, str):
        return [formula_functions]
    if not isinstance(formula_functions, (list, tuple)):
        raise TypeError("formula_functions must be a character vector")
    if not all(isinstance(fn, str) and fn for fn in formula_functions):
        raise TypeError("formula_functions must be a character vector")
    return list(formula_functions)


def _candidate_spec_classes(spec_class_or_object: Any) -> List[str]:
    if isinstance(spec_class_or_object, str):
        return [spec_class_or_object]
    explicit = getattr(spec_class_or_object, "__fmrimod_hrfspec_classes__", None)
    if explicit is not None:
        if isinstance(explicit, str):
            return [explicit]
        return [str(cls) for cls in explicit]
    cls = spec_class_or_object.__class__
    return [base.__name__ for base in cls.__mro__]


def register_hrfspec_extension(
    spec_class: str,
    package: str,
    convolved_class: Optional[str] = None,
    requires_external_processing: bool = False,
    formula_functions: Optional[List[str]] = None,
) -> None:
    """Register an external HRF specification type.

    This function allows external packages to register custom HRF spec types
    that can be used in the fmrimod formula and convolution pipeline.

    Parameters
    ----------
    spec_class : str
        Name of the HRF spec class being registered
    package : str
        Name of the package providing the extension
    convolved_class : str, optional
        Name of the corresponding convolved class
    requires_external_processing : bool, default=False
        Whether external processing is required
    formula_functions : list of str, optional
        Names of formula functions provided by this extension

    Examples
    --------
    >>> register_hrfspec_extension(
    ...     spec_class='CustomHRF',
    ...     package='my_package',
    ...     convolved_class='ConvolvedCustomHRF',
    ...     formula_functions=['custom_hrf']
    ... )
    """
    spec_class = _validate_single_string(spec_class, "spec_class")
    package = _validate_single_string(package, "package")
    if convolved_class is not None:
        convolved_class = _validate_single_string(convolved_class, "convolved_class")
    formula_functions = _coerce_formula_functions(formula_functions)

    _registry[spec_class] = HRFSpecExtension(
        spec_class=spec_class,
        package=package,
        convolved_class=convolved_class,
        requires_external_processing=requires_external_processing,
        formula_functions=formula_functions,
    )


def is_external_hrfspec(spec_class: Any) -> bool:
    """Check if a spec class is registered as external.

    Parameters
    ----------
    spec_class : str
        Name of the spec class to check

    Returns
    -------
    bool
        True if the spec class is registered as external

    Examples
    --------
    >>> is_external_hrfspec('CustomHRF')
    False
    >>> register_hrfspec_extension('CustomHRF', 'my_package')
    >>> is_external_hrfspec('CustomHRF')
    True
    """
    return any(cls in _registry for cls in _candidate_spec_classes(spec_class))


def get_external_hrfspec_info(spec_class: Any) -> Optional[HRFSpecExtension]:
    """Get registration info for an external spec class.

    Parameters
    ----------
    spec_class : str
        Name of the spec class

    Returns
    -------
    HRFSpecExtension or None
        Registration info if found, None otherwise

    Examples
    --------
    >>> info = get_external_hrfspec_info('CustomHRF')
    >>> if info:
    ...     print(info.package)
    my_package
    """
    for cls in _candidate_spec_classes(spec_class):
        if cls in _registry:
            return _registry[cls]
    return None


def list_external_hrfspecs() -> List[str]:
    """List all registered external spec classes.

    Returns
    -------
    list of str
        Names of all registered spec classes

    Examples
    --------
    >>> list_external_hrfspecs()
    ['CustomHRF', 'AnotherHRF']
    """
    return list(_registry.keys())


def requires_external_processing(spec_class: Any) -> bool:
    """Check if external processing is required for a spec class.

    Parameters
    ----------
    spec_class : str
        Name of the spec class

    Returns
    -------
    bool
        True if external processing is required, False otherwise

    Examples
    --------
    >>> requires_external_processing('CustomHRF')
    False
    """
    ext = get_external_hrfspec_info(spec_class)
    return ext.requires_external_processing if ext else False


def get_external_hrfspec_functions(spec_class: Any) -> Optional[List[str]]:
    """Get formula function names for a registered spec class.

    Parameters
    ----------
    spec_class : str
        Name of the spec class

    Returns
    -------
    list of str or None
        Formula function names if registered, None otherwise

    Examples
    --------
    >>> get_external_hrfspec_functions('CustomHRF')
    ['custom_hrf']
    """
    ext = get_external_hrfspec_info(spec_class)
    if ext is None:
        return None
    if ext.formula_functions is None and ext.spec_class == "afni_hrfspec":
        return ["afni_hrf"]
    return ext.formula_functions


def get_all_external_hrf_functions() -> List[str]:
    """Get all registered formula function names across all extensions.

    Returns
    -------
    list of str
        All formula function names from all registered extensions

    Examples
    --------
    >>> get_all_external_hrf_functions()
    ['custom_hrf', 'another_hrf']
    """
    funcs = []
    seen = set()
    for spec_class in _registry:
        for fn in get_external_hrfspec_functions(spec_class) or []:
            if fn not in seen:
                funcs.append(fn)
                seen.add(fn)
    return funcs
