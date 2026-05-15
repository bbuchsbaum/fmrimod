"""Basis-function filtering for contrast weight computation.

Allows contrasts to target specific basis functions when using multi-basis
HRFs (e.g., select only the canonical HRF, not temporal/dispersion derivatives).
"""

import re
import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np


def filter_basis(
    condnames: List[str],
    basis: Optional[Union[str, List[int]]] = None,
    nbasis: int = 1,
    contrast_name: str = "",
) -> List[str]:
    """Filter condition names by basis function indices.

    Condition names with multi-basis HRFs have _b01, _b02, ... suffixes.
    This function selects only the requested basis function indices.

    Parameters
    ----------
    condnames : list of str
        Condition names (may include _b## suffixes)
    basis : None, "all", or list of int
        Which basis functions to keep:
        - None or "all": keep all
        - List of ints: keep only these basis indices (1-indexed)
    nbasis : int
        Total number of basis functions
    contrast_name : str
        Name of contrast (for warning messages)

    Returns
    -------
    list of str
        Filtered condition names
    """
    if basis is None or basis == "all" or nbasis <= 1:
        return condnames

    if isinstance(basis, (int, np.integer)):
        basis = [basis]

    # Build regex pattern for matching basis suffixes
    # Basis indices are 1-indexed in the suffix: _b01, _b02, etc.
    kept = []
    for name in condnames:
        match = re.search(r'_b(\d+)$', name)
        if match:
            basis_idx = int(match.group(1))
            if basis_idx in basis:
                kept.append(name)
        else:
            # No basis suffix - keep if basis=[1] (canonical)
            if 1 in basis:
                kept.append(name)

    if not kept:
        warnings.warn(
            f"Contrast '{contrast_name}': basis filtering removed all conditions. "
            f"Requested basis indices {basis} but none matched in {condnames[:5]}..."
        )

    return kept


def apply_basis_filter(
    weights_mat: np.ndarray,
    condnames: List[str],
    basis_spec: Optional[Union[str, List[int]]] = None,
    nbasis: int = 1,
    contrast_name: str = "",
    basis_weights: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Apply basis filtering to contrast weight matrix.

    Parameters
    ----------
    weights_mat : ndarray
        Weight matrix (n_conditions x n_contrasts)
    condnames : list of str
        Condition names (row names of weights_mat)
    basis_spec : None, "all", or list of int
        Which basis functions to keep
    nbasis : int
        Number of basis functions detected
    contrast_name : str
        Name of contrast (for messages)
    basis_weights : ndarray, optional
        Weights for selected basis functions (normalized to sum to 1)

    Returns
    -------
    dict
        Keys: 'weights' (filtered matrix), 'condnames' (filtered names), 'nbasis' (int)
    """
    # No filtering needed for single-basis HRFs
    if nbasis <= 1:
        if basis_spec is not None and basis_spec != "all":
            warnings.warn(
                f"Contrast '{contrast_name}': basis filtering requested but "
                f"term has no multi-basis HRF (nbasis={nbasis}). Ignoring.",
            )
        return {
            'weights': weights_mat,
            'condnames': condnames,
            'nbasis': nbasis,
        }

    # No filtering requested
    if basis_spec is None or basis_spec == "all":
        return {
            'weights': weights_mat,
            'condnames': condnames,
            'nbasis': nbasis,
        }

    # Filter condition names
    filtered_names = filter_basis(condnames, basis_spec, nbasis, contrast_name)

    # Zero out weights for non-selected conditions
    result_weights = weights_mat.copy()
    for i, name in enumerate(condnames):
        if name not in filtered_names:
            result_weights[i, :] = 0

    # Apply basis_weights if provided
    if basis_weights is not None:
        if isinstance(basis_spec, list):
            n_selected = len(basis_spec)
        else:
            n_selected = nbasis

        if len(basis_weights) != n_selected:
            raise ValueError(
                f"Contrast '{contrast_name}': basis_weights length ({len(basis_weights)}) "
                f"must match number of selected basis functions ({n_selected})"
            )

        # Normalize to sum to 1
        weight_sum = np.sum(basis_weights)
        TOLERANCE = 1e-8
        if abs(weight_sum - 1.0) > TOLERANCE:
            warnings.warn(
                f"Contrast '{contrast_name}': basis_weights sum to {weight_sum:.6f}, "
                f"normalizing to sum to 1.0"
            )
            basis_weights = basis_weights / weight_sum

        # Apply weights to selected basis functions
        for i, name in enumerate(condnames):
            if name in filtered_names:
                match = re.search(r'_b(\d+)$', name)
                if match:
                    basis_idx = int(match.group(1))
                    if isinstance(basis_spec, list):
                        weight_idx = basis_spec.index(basis_idx)
                        result_weights[i, :] *= basis_weights[weight_idx]

    return {
        'weights': result_weights,
        'condnames': filtered_names,
        'nbasis': nbasis,
    }
