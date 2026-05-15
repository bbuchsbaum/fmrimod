"""Polynomial basis functions."""

from typing import List, Optional, cast

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
        Maximum polynomial degree (must be >= 1).
    intercept : bool, optional
        Whether to include intercept (constant) term. Default is True.
    raw : bool, optional
        If True, use raw polynomials. If False, use QR-orthogonalized
        polynomials for better numerical stability. Default is False.
    name : str, optional
        Name for the basis. If None, uses ``'poly{degree}'``.

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

    See Also
    --------
    Poly.from_data : R-style construction that evaluates on data
        and defaults to ``intercept=False``.
    """

    def __init__(
        self,
        degree: int,
        intercept: bool = True,
        raw: bool = False,
        name: Optional[str] = None,
    ) -> None:
        if degree < 1:
            raise ValueError("Degree must be at least 1")

        self.degree = degree
        self.intercept = intercept
        self.raw = raw
        self.y: Optional[Array] = None

        super().__init__(name or f"poly{degree}")

    @classmethod
    def from_data(
        cls,
        x: ArrayLike,
        degree: int,
        intercept: bool = False,
        raw: bool = False,
        name: Optional[str] = None,
    ) -> "Poly":
        """Construct a :class:`Poly` and evaluate it on ``x``.

        Mirrors R's ``poly(x, degree)`` ergonomics: constructs the basis
        and exposes the evaluated matrix on ``.y``. The default
        ``intercept=False`` matches R's ``poly()`` (raw=FALSE) behavior.
        """
        basis = cls(degree=degree, intercept=intercept, raw=raw, name=name)
        basis.y = basis.evaluate(x)
        return basis
    
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

        return cast(Array, Q)


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
        self._x_mean: Optional[float] = None
    
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
        self._x_mean = float(np.mean(x))
        x_centered = x - self._x_mean
        
        # Use parent's raw polynomial evaluation
        return super().evaluate(x_centered)
