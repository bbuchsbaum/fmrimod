"""Basis subsetting functionality.

This module provides functions for creating subsets of basis functions.
"""

from typing import List, Optional, Union
import numpy as np
from numpy.typing import ArrayLike

from ..types import Array
from .base import ParametricBasis


class SubBasis(ParametricBasis):
    """A subset of basis functions from a parent basis.
    
    This class wraps an existing basis and provides access to only
    a subset of its basis functions.
    
    Parameters
    ----------
    parent : ParametricBasis
        The parent basis to subset
    indices : list of int or slice
        Indices of basis functions to include
    name : str, optional
        Name for the sub-basis. If None, uses parent name with suffix.
    
    Examples
    --------
    >>> # Create a polynomial basis and take a subset
    >>> poly = Poly(degree=5)
    >>> sub = SubBasis(poly, [0, 2, 4])  # Take 1st, 3rd, 5th basis functions
    >>> 
    >>> # Or using sub_basis function
    >>> sub = sub_basis(poly, [0, 2, 4])
    """
    
    def __init__(
        self,
        parent: ParametricBasis,
        indices: Union[List[int], slice],
        name: Optional[str] = None
    ):
        """Initialize a subset of an existing basis.

        Parameters
        ----------
        parent : ParametricBasis
            The parent basis to subset.
        indices : list of int or slice
            Indices of basis functions to include. Negative
            indices are supported (e.g., ``[-1]`` selects the
            last basis function).
        name : str, optional
            Name for the sub-basis. If None, auto-generated
            from the parent name and selected indices.

        Raises
        ------
        IndexError
            If any index is out of range for the parent basis.
        ValueError
            If no basis functions are selected.
        """
        self.parent = parent
        
        # Convert slice to list of indices
        if isinstance(indices, slice):
            indices = list(range(*indices.indices(parent.n_basis)))
        
        # Validate indices
        self.indices = []
        for idx in indices:
            if idx < 0:
                # Handle negative indexing
                idx = parent.n_basis + idx
            if idx < 0 or idx >= parent.n_basis:
                raise IndexError(
                    f"Index {idx} out of range for basis with "
                    f"{parent.n_basis} functions"
                )
            self.indices.append(idx)
        
        if not self.indices:
            raise ValueError("Must select at least one basis function")
        
        # Set name
        if name is None:
            name = f"{parent.name}_sub[{','.join(map(str, self.indices))}]"
        
        super().__init__(name)
    
    @property
    def n_basis(self) -> int:
        """Number of basis functions in subset."""
        return len(self.indices)
    
    @property
    def basis_names(self) -> List[str]:
        """Names for each basis function in subset."""
        parent_names = self.parent.basis_names
        return [parent_names[i] for i in self.indices]
    
    def evaluate(self, x: ArrayLike) -> Array:
        """Evaluate subset of basis functions at x.
        
        Parameters
        ----------
        x : array-like
            Values at which to evaluate basis functions
        
        Returns
        -------
        Array
            Basis function values for subset, shape (len(x), n_basis)
        """
        # Get full basis evaluation
        full_basis = self.parent.evaluate(x)
        
        # Return only selected columns
        return full_basis[:, self.indices]
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SubBasis(parent={self.parent.name}, "
            f"indices={self.indices}, n_basis={self.n_basis})"
        )


def sub_basis(
    basis: ParametricBasis,
    indices: Union[List[int], slice],
    name: Optional[str] = None
) -> SubBasis:
    """Create a subset of basis functions.
    
    This function creates a new basis object that contains only
    a subset of the original basis functions.
    
    Parameters
    ----------
    basis : ParametricBasis
        The basis to subset
    indices : list of int or slice
        Indices of basis functions to include. Can be:
        - List of integers: [0, 2, 4] selects 1st, 3rd, and 5th
        - Slice: slice(1, 4) selects 2nd through 4th
        - Negative indices are supported: [-1] selects last
    name : str, optional
        Name for the sub-basis
    
    Returns
    -------
    SubBasis
        A new basis containing only the selected functions
    
    Examples
    --------
    >>> from fmrimod.basis import Poly, BSpline
    >>> 
    >>> # Select specific polynomial terms
    >>> poly = Poly(degree=5)
    >>> linear_cubic = sub_basis(poly, [1, 3])  # x and x^3 terms
    >>> 
    >>> # Select a range of spline basis functions
    >>> spline = BSpline(df=10)
    >>> middle_splines = sub_basis(spline, slice(3, 7))
    >>> 
    >>> # Select all but the intercept
    >>> no_intercept = sub_basis(poly, slice(1, None))
    >>> 
    >>> # Use negative indexing
    >>> last_three = sub_basis(poly, [-3, -2, -1])
    """
    return SubBasis(basis, indices, name)