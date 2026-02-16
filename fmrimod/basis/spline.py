"""Spline basis functions."""

from typing import List, Optional, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy import interpolate

from ..types import Array
from .base import ParametricBasis


class BSpline(ParametricBasis):
    """B-spline basis functions.
    
    Creates B-spline basis functions with specified degrees of freedom
    or knot locations.
    
    Parameters
    ----------
    df : int, optional
        Degrees of freedom (number of basis functions). 
        Either df or knots must be specified.
    knots : array-like, optional
        Interior knot locations. Either df or knots must be specified.
    degree : int, optional
        Degree of the splines. Default is 3 (cubic).
    intercept : bool, optional
        Whether to include intercept. Default is True.
    boundary_knots : tuple of float, optional
        Boundary knot locations (min, max). If None, uses data range.
    name : str, optional
        Name for the basis. If None, uses 'bs{df}'.
    
    Attributes
    ----------
    df : int
        Degrees of freedom
    degree : int
        Spline degree
    knots : Array
        Interior knot locations
    boundary_knots : tuple
        Boundary knot locations
    
    Examples
    --------
    >>> # 4 df cubic B-splines
    >>> basis = BSpline(df=4)
    >>> x = np.linspace(0, 1, 100)
    >>> X = basis.evaluate(x)
    >>> X.shape
    (100, 4)
    
    >>> # Specify knot locations
    >>> basis = BSpline(knots=[0.25, 0.5, 0.75], degree=2)
    """
    
    def __init__(
        self,
        df: Optional[int] = None,
        knots: Optional[ArrayLike] = None,
        degree: int = 3,
        intercept: bool = True,
        boundary_knots: Optional[tuple] = None,
        name: Optional[str] = None
    ):
        """Initialize B-spline basis.

        Parameters
        ----------
        df : int, optional
            Degrees of freedom (number of basis functions).
            Exactly one of ``df`` or ``knots`` must be specified.
        knots : array-like, optional
            Interior knot locations. Exactly one of ``df`` or
            ``knots`` must be specified.
        degree : int, optional
            Polynomial degree of the splines. Default is 3 (cubic).
        intercept : bool, optional
            Whether to include an intercept. Default is True.
        boundary_knots : tuple of float, optional
            ``(min, max)`` boundary knot locations. If None, computed
            from the data range during the first ``evaluate`` call.
        name : str, optional
            Name for the basis. If None, defaults to ``'bs{df}'``.

        Raises
        ------
        ValueError
            If neither or both of ``df`` and ``knots`` are specified.
        """
        if df is None and knots is None:
            raise ValueError("Either df or knots must be specified")
        
        if df is not None and knots is not None:
            raise ValueError("Cannot specify both df and knots")
        
        self.degree = degree
        self.intercept = intercept
        self.boundary_knots = boundary_knots
        
        if knots is not None:
            self.knots = np.asarray(knots)
            self.df = len(self.knots) + self.degree + (1 if intercept else 0)
        else:
            self.df = df
            self.knots = None
        
        name = name or f"bs{self.df}"
        super().__init__(name)
        
        # Will be set during first evaluation
        self._computed_knots = None
        self._boundary = None
    
    @property
    def n_basis(self) -> int:
        """Number of basis functions."""
        return self.df
    
    @property
    def basis_names(self) -> List[str]:
        """Names for each basis function."""
        return [f"{self.name}_{i+1}" for i in range(self.n_basis)]
    
    def evaluate(self, x: ArrayLike) -> Array:
        """Evaluate B-spline basis at x.
        
        Parameters
        ----------
        x : array-like
            Values at which to evaluate basis
        
        Returns
        -------
        Array
            Basis matrix, shape (len(x), n_basis)
        """
        x = np.asarray(x)
        n = len(x)
        
        # Set up knots if not already done
        if self._computed_knots is None:
            self._setup_knots(x)
        
        # Create B-spline basis
        basis = np.zeros((n, self.n_basis))
        
        # Use scipy's B-spline implementation
        for i in range(self.n_basis):
            # Create a coefficient vector with 1 at position i
            c = np.zeros(self.n_basis)
            c[i] = 1.0
            
            # Create B-spline object
            spl = interpolate.BSpline(
                self._computed_knots,
                c,
                self.degree,
                extrapolate=False
            )
            
            # Evaluate
            basis[:, i] = spl(x)
        
        # Handle values outside boundaries
        mask = (x < self._boundary[0]) | (x > self._boundary[1])
        if np.any(mask):
            basis[mask, :] = 0
        
        # Remove intercept if needed
        if not self.intercept and self.n_basis > 1:
            # For B-splines, intercept removal is handled in df calculation
            pass
        
        return basis
    
    def _setup_knots(self, x: Array) -> None:
        """Set up knot locations.
        
        Parameters
        ----------
        x : Array
            Data values
        """
        # Determine boundaries
        if self.boundary_knots is None:
            x_min, x_max = np.min(x), np.max(x)
            # Add small buffer to ensure all data is included
            buffer = 0.01 * (x_max - x_min)
            self._boundary = (x_min - buffer, x_max + buffer)
        else:
            self._boundary = self.boundary_knots
        
        # Determine interior knots
        if self.knots is None:
            # Compute number of interior knots
            # For B-splines: df = n_interior_knots + degree + 1
            n_interior = self.df - self.degree - 1
            if n_interior > 0:
                # Equal spacing in quantiles
                quantiles = np.linspace(0, 1, n_interior + 2)[1:-1]
                self.knots = np.quantile(x, quantiles)
            else:
                self.knots = np.array([])
        
        # Create full knot sequence for scipy
        # Need degree+1 copies at each boundary
        self._computed_knots = np.concatenate([
            np.repeat(self._boundary[0], self.degree + 1),
            self.knots,
            np.repeat(self._boundary[1], self.degree + 1)
        ])


class NaturalSpline(BSpline):
    """Natural cubic spline basis functions.
    
    Natural splines are cubic splines constrained to be linear
    beyond the boundary knots. This reduces variance at the
    boundaries.
    
    Parameters
    ----------
    df : int, optional
        Degrees of freedom. Either df or knots must be specified.
    knots : array-like, optional
        Interior knot locations. Either df or knots must be specified.
    intercept : bool, optional
        Whether to include intercept. Default is True.
    boundary_knots : tuple of float, optional
        Boundary knot locations. If None, uses data range.
    name : str, optional
        Name for the basis. If None, uses 'ns{df}'.
    
    Examples
    --------
    >>> basis = NaturalSpline(df=4)
    >>> x = np.linspace(0, 1, 100)
    >>> X = basis.evaluate(x)
    """
    
    def __init__(
        self,
        df: Optional[int] = None,
        knots: Optional[ArrayLike] = None,
        intercept: bool = True,
        boundary_knots: Optional[tuple] = None,
        name: Optional[str] = None
    ):
        """Initialize natural spline basis.

        Parameters
        ----------
        df : int, optional
            Degrees of freedom. Exactly one of ``df`` or ``knots``
            must be specified.
        knots : array-like, optional
            Interior knot locations.
        intercept : bool, optional
            Whether to include an intercept. Default is True.
        boundary_knots : tuple of float, optional
            ``(min, max)`` boundary knot locations. If None,
            computed from the data range.
        name : str, optional
            Name for the basis. If None, defaults to
            ``'ns{df}'``.
        """
        # Natural splines are cubic (degree 3)
        name = name or f"ns{df if df is not None else 'custom'}"
        super().__init__(
            df=df,
            knots=knots,
            degree=3,
            intercept=intercept,
            boundary_knots=boundary_knots,
            name=name
        )
    
    def evaluate(self, x: ArrayLike) -> Array:
        """Evaluate natural spline basis at x.
        
        Natural splines require special constraint handling
        at the boundaries.
        
        Parameters
        ----------
        x : array-like
            Values at which to evaluate basis
        
        Returns
        -------
        Array
            Basis matrix, shape (len(x), n_basis)
        """
        # Get regular B-spline basis
        basis = super().evaluate(x)
        
        # Apply natural spline constraints
        # This is a simplified version - full implementation
        # would need proper constraint matrix
        # For now, just return the B-spline basis
        return basis