"""Base classes for basis functions."""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, cast

import numpy as np
from numpy.typing import ArrayLike

from ..types import Array


class ParametricBasis(ABC):
    """Abstract base class for parametric basis functions.
    
    All basis functions should inherit from this class and implement
    the required abstract methods.
    
    Attributes
    ----------
    name : str
        Name of the basis function
    """
    
    def __init__(self, name: Optional[str] = None):
        """Initialize basis function.
        
        Parameters
        ----------
        name : str, optional
            Name for the basis function. If None, uses class name.
        """
        self.name = name or self.__class__.__name__.lower()
    
    @property
    @abstractmethod
    def n_basis(self) -> int:
        """Number of basis functions."""
        pass
    
    @property
    def basis_names(self) -> List[str]:
        """Names for each basis function.
        
        Returns
        -------
        list of str
            Names for each basis column
        """
        return [f"{self.name}{i+1}" for i in range(self.n_basis)]
    
    @abstractmethod
    def evaluate(self, x: ArrayLike) -> Array:
        """Evaluate basis functions at x.
        
        Parameters
        ----------
        x : array-like
            Values at which to evaluate basis functions
        
        Returns
        -------
        Array
            Basis function values, shape (len(x), n_basis)
        """
        pass
    
    def get_range(self, x: ArrayLike) -> Tuple[float, float]:
        """Get the range of x values.
        
        Parameters
        ----------
        x : array-like
            Input values
        
        Returns
        -------
        tuple of float
            (min, max) of x values
        """
        x = np.asarray(x)
        return float(np.min(x)), float(np.max(x))
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name='{self.name}', n_basis={self.n_basis})"
    
    def __call__(self, x: ArrayLike) -> Array:
        """Convenience method for evaluation.
        
        Parameters
        ----------
        x : array-like
            Values at which to evaluate basis functions
        
        Returns
        -------
        Array
            Basis function values
        """
        return self.evaluate(x)
    
    def predict(self, x: ArrayLike, coef: Optional[ArrayLike] = None) -> Array:
        """Predict values using basis functions and coefficients.
        
        This method evaluates the basis functions at the given points
        and optionally multiplies by coefficients to produce predictions.
        
        Parameters
        ----------
        x : array-like
            Values at which to evaluate basis functions
        coef : array-like, optional
            Coefficients for each basis function. If None, returns
            the basis matrix itself (same as evaluate()).
        
        Returns
        -------
        Array
            If coef is None: Basis matrix, shape (len(x), n_basis)
            If coef provided: Predictions, shape (len(x),)
        
        Examples
        --------
        >>> basis = Poly(degree=2)
        >>> x = np.array([0, 1, 2, 3])
        >>> 
        >>> # Get basis matrix
        >>> basis.predict(x)
        array([[1., 0., 0.],
               [1., 1., 1.],
               [1., 2., 4.],
               [1., 3., 9.]])
        >>> 
        >>> # Get predictions with coefficients
        >>> coef = np.array([1.0, 2.0, 0.5])
        >>> basis.predict(x, coef)
        array([1. , 3.5, 7. , 11.5])
        """
        # Evaluate basis functions
        basis_matrix = self.evaluate(x)
        
        if coef is None:
            # Return basis matrix
            return basis_matrix
        
        # Convert coefficients to array
        coef = np.asarray(coef)
        
        # Check coefficient dimensions
        if coef.ndim == 1:
            if len(coef) != self.n_basis:
                raise ValueError(
                    f"Number of coefficients ({len(coef)}) must match "
                    f"number of basis functions ({self.n_basis})"
                )
            # Matrix multiplication for predictions
            return cast(Array, basis_matrix @ coef)
        elif coef.ndim == 2:
            if coef.shape[0] != self.n_basis:
                raise ValueError(
                    f"First dimension of coefficients ({coef.shape[0]}) must match "
                    f"number of basis functions ({self.n_basis})"
                )
            # Matrix multiplication for multiple sets of predictions
            return cast(Array, basis_matrix @ coef)
        else:
            raise ValueError(
                f"Coefficients must be 1D or 2D array, got {coef.ndim}D"
            )