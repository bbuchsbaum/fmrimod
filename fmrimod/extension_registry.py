"""Extension registry for custom HRF specification types.

Allows external packages to register custom HRF spec types that integrate
with the fmrimod formula and convolution pipeline.
"""

from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field


@dataclass
class HRFSpecExtension:
    """Registration info for an external HRF spec type."""
    spec_class: str
    package: str
    convolved_class: Optional[str] = None
    requires_external_processing: bool = False
    formula_functions: Optional[List[str]] = None


# Module-level private registry
_registry: Dict[str, HRFSpecExtension] = {}


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
    _registry[spec_class] = HRFSpecExtension(
        spec_class=spec_class,
        package=package,
        convolved_class=convolved_class,
        requires_external_processing=requires_external_processing,
        formula_functions=formula_functions,
    )


def is_external_hrfspec(spec_class: str) -> bool:
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
    return spec_class in _registry


def get_external_hrfspec_info(spec_class: str) -> Optional[HRFSpecExtension]:
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
    return _registry.get(spec_class)


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


def requires_external_processing(spec_class: str) -> bool:
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
    ext = _registry.get(spec_class)
    return ext.requires_external_processing if ext else False


def get_external_hrfspec_functions(spec_class: str) -> Optional[List[str]]:
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
    ext = _registry.get(spec_class)
    return ext.formula_functions if ext else None


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
    for ext in _registry.values():
        if ext.formula_functions:
            funcs.extend(ext.formula_functions)
    return funcs
