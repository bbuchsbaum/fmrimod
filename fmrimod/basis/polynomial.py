"""Polynomial basis functions."""

from typing import Any, List, Optional

import numpy as np
from numpy.typing import ArrayLike

from ..types import Array
from .base import ParametricBasis


class Poly(ParametricBasis):
    """Polynomial basis functions.
    
    Creates polynomial basis functions up to a specified degree.
    Can optionally include or exclude the intercept term.
    
    Parameters
    ----------
    degree : int
        Maximum polynomial degree
    intercept : bool, optional
        Whether to include intercept (constant) term. Default is True.
    raw : bool, optional
        If True, use raw polynomials. If False, use orthogonal
        polynomials. Default is False.
    name : str, optional
        Name for the basis. If None, uses 'poly{degree}'.
    
    Attributes
    ----------
    degree : int
        Maximum polynomial degree
    intercept : bool
        Whether intercept is included
    raw : bool
        Whether using raw polynomials
    
    Examples
    --------
    >>> # Linear basis with intercept
    >>> basis = Poly(degree=1)
    >>> x = np.array([0, 1, 2, 3])
    >>> basis.evaluate(x)
    array([[1., 0.],
           [1., 1.],
           [1., 2.],
           [1., 3.]])
    
    >>> # Quadratic without intercept
    >>> basis = Poly(degree=2, intercept=False)
    >>> basis.n_basis
    2
    """
    
    def __init__(
        self,
        *args: Any,
        degree: Optional[int] = None,
        intercept: Optional[bool] = None,
        raw: bool = False,
        name: Optional[str] = None,
    ):
        """Initialize polynomial basis.

        Parameters
        ----------
        degree : int
            Maximum polynomial degree (must be >= 1).
        *args
            Optional positional compatibility arguments. Supports:
            ``Poly(3)`` and ``Poly(x, degree=3)``.
        intercept : bool, optional
            Whether to include the constant (degree-0) term.
            Default is True.
        raw : bool, optional
            If True, use raw (non-orthogonal) polynomials.
            If False, use QR-orthogonalized polynomials for
            better numerical stability. Default is False.
        name : str, optional
            Name for the basis set. If None, defaults to
            ``'poly{degree}'``.

        Raises
        ------
        ValueError
            If ``degree`` < 1.
        """
        compat_x = None
        if len(args) > 1:
            raise TypeError("Poly accepts at most one positional argument")

        if len(args) == 1:
            arg0 = args[0]
            if np.isscalar(arg0) and degree is None:
                degree = int(arg0)
            elif degree is not None:
                compat_x = np.asarray(arg0)
            else:
                raise TypeError(
                    "When passing data as first positional argument, "
                    "you must provide degree=..."
                )

        if degree is None:
            raise TypeError("degree must be provided")

        if degree < 1:
            raise ValueError("Degree must be at least 1")
        
        # Idiomatic constructor defaults to intercept=True, but legacy
        # convenience mode Poly(x, degree=...) follows R's poly() default
        # (no intercept).
        if intercept is None:
            intercept = False if compat_x is not None else True

        self.degree = degree
        self.intercept = intercept
        self.raw = raw
        self.y = None
        
        name = name or f"poly{degree}"
        super().__init__(name)

        # Compatibility mode: Poly(x, degree=...)
        if compat_x is not None:
            self.y = self.evaluate(compat_x)
    
    @property
    def n_basis(self) -> int:
        """Number of basis functions."""
        return self.degree + (1 if self.intercept else 0)
    
    @property
    def basis_names(self) -> List[str]:
        """Names for each basis function."""
        names = []
        if self.intercept:
            names.append(f"{self.name}_intercept")
        for d in range(1, self.degree + 1):
            names.append(f"{self.name}_{d}")
        return names
    
    def evaluate(self, x: ArrayLike) -> Array:
        """Evaluate polynomial basis at x.
        
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
        
        if self.raw:
            # Raw polynomials
            basis = np.zeros((n, self.n_basis))
            col = 0
            
            if self.intercept:
                basis[:, col] = 1.0
                col += 1
            
            for d in range(1, self.degree + 1):
                basis[:, col] = x ** d
                col += 1
        else:
            # Orthogonal polynomials
            basis = self._orthogonal_polynomials(x)
        
        self.y = basis
        return basis
    
    def _orthogonal_polynomials(self, x: Array) -> Array:
        """Generate orthogonal polynomials.
        
        Uses QR decomposition to create orthogonal polynomial basis.
        
        Parameters
        ----------
        x : Array
            Input values
        
        Returns
        -------
        Array
            Orthogonal polynomial basis
        """
        x = np.asarray(x, dtype=float).ravel()
        n = len(x)
        if n == 0:
            return np.empty((0, self.n_basis), dtype=float)

        # Mirror R's `poly` implementation by center each power before
        # orthogonalization instead of pre-scaling by sample standard deviation.
        raw_basis = np.zeros((n, self.n_basis))
        col = 0

        if self.intercept:
            raw_basis[:, col] = 1.0
            col += 1

        # Center each power before orthogonalization.
        # For power 1 this becomes the usual x-centered column.
        for d in range(1, self.degree + 1):
            power = x ** d
            raw_basis[:, col] = power - np.mean(power)
            col += 1

        # QR decomposition for orthogonalization
        Q, R = np.linalg.qr(raw_basis)
        
        # Ensure consistent sign
        signs = np.sign(np.diag(R))
        signs[signs == 0] = 1
        Q = Q * signs
        
        return Q


class NaturalPoly(Poly):
    """Natural polynomial basis (centered at mean).
    
    Like Poly but automatically centers x values at their mean
    before computing polynomials. This can improve numerical
    stability and interpretability.
    
    Parameters
    ----------
    degree : int
        Maximum polynomial degree
    intercept : bool, optional
        Whether to include intercept term. Default is True.
    name : str, optional
        Name for the basis. If None, uses 'npoly{degree}'.
    
    Examples
    --------
    >>> basis = NaturalPoly(degree=2)
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> # Polynomials will be computed on (x - mean(x))
    """
    
    def __init__(
        self,
        degree: int,
        intercept: bool = True,
        name: Optional[str] = None
    ):
        """Initialize natural polynomial basis.

        Parameters
        ----------
        degree : int
            Maximum polynomial degree.
        intercept : bool, optional
            Whether to include the constant term. Default is True.
        name : str, optional
            Name for the basis. If None, defaults to
            ``'npoly{degree}'``.
        """
        name = name or f"npoly{degree}"
        # Natural polynomials are always raw (not orthogonal)
        super().__init__(degree=degree, intercept=intercept, raw=True, name=name)
        self._x_mean = None
    
    def evaluate(self, x: ArrayLike) -> Array:
        """Evaluate natural polynomial basis at x.
        
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
        
        # Center x at its mean
        self._x_mean = np.mean(x)
        x_centered = x - self._x_mean
        
        # Use parent's raw polynomial evaluation
        return super().evaluate(x_centered)
