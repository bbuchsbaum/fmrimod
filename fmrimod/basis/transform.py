"""Transformation basis functions."""

from typing import List, Optional

import numpy as np
from numpy.typing import ArrayLike

from ..types import Array
from .base import ParametricBasis


class Scale(ParametricBasis):
    """Scaling transformation (z-score).

    Centers and scales values to have mean 0 and standard deviation 1.

    Parameters
    ----------
    center : bool, optional
        Whether to center (subtract mean). Default is True.
    scale : bool, optional
        Whether to scale (divide by SD). Default is True.
    name : str, optional
        Name for the transformation. Default is ``'scale'``.

    Examples
    --------
    >>> transform = Scale()
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> z = transform.evaluate(x)
    >>> np.mean(z), np.std(z)
    (0.0, 1.0)

    See Also
    --------
    Scale.from_data : R-style construction that evaluates on data
        with sample-SD (``ddof=1``) scaling.
    """

    def __init__(
        self,
        center: bool = True,
        scale: bool = True,
        name: Optional[str] = None,
    ) -> None:
        self.center = center
        self.scale = scale
        super().__init__(name or "scale")

        # Statistics computed during evaluation
        self._mean: Optional[float] = None
        self._std: Optional[float] = None
        self.y: Optional[Array] = None
        self._compat_r_scale = False

    @classmethod
    def from_data(
        cls,
        x: ArrayLike,
        center: bool = True,
        scale: bool = True,
        name: Optional[str] = None,
    ) -> "Scale":
        """Construct a :class:`Scale` and evaluate it on ``x``.

        Mirrors R's ``scale(x)`` ergonomics: constructs the transform,
        evaluates on the supplied data, and stores the result on
        ``.y``. The sample standard deviation (``ddof=1``) is used so
        the output matches R's ``scale()``.
        """
        basis = cls(center=center, scale=scale, name=name)
        basis._compat_r_scale = True
        basis.y = basis.evaluate(x)
        return basis
    
    @property
    def n_basis(self) -> int:
        """Number of basis functions (always 1 for scaling)."""
        return 1
    
    @property
    def basis_names(self) -> List[str]:
        """Names for basis functions."""
        return [self.name]
    
    def evaluate(self, x: ArrayLike) -> Array:
        """Apply scaling transformation.
        
        Parameters
        ----------
        x : array-like
            Values to transform
        
        Returns
        -------
        Array
            Transformed values, shape (len(x), 1)
        """
        x = np.asarray(x)
        if x.size == 0:
            self._mean = np.nan
            self._std = np.nan
            self.y = x.reshape(-1, 1)
            return self.y
        result = x.copy().reshape(-1, 1)
        
        # Compute statistics
        self._mean = float(np.mean(x))
        ddof = 1 if self._compat_r_scale and len(x) > 1 else 0
        self._std = float(np.std(x, ddof=ddof))

        # Apply transformations
        if self.center:
            result = result - self._mean

        if self.scale and self._std is not None and self._std > 0:
            result = result / self._std
        
        self.y = result
        return result


class ScaleWithin(ParametricBasis):
    """Scaling transformation within groups.
    
    Centers and scales values within each group separately.
    
    Parameters
    ----------
    groups : array-like
        Group indicators for each observation
    center : bool, optional
        Whether to center within groups. Default is True.
    scale : bool, optional
        Whether to scale within groups. Default is True.
    name : str, optional
        Name for the transformation. Default is 'scalewithin'.
    
    Examples
    --------
    >>> groups = ['A', 'A', 'B', 'B', 'B']
    >>> transform = ScaleWithin(groups)
    >>> x = np.array([1, 2, 10, 20, 30])
    >>> z = transform.evaluate(x)
    """
    
    def __init__(
        self,
        groups: ArrayLike,
        center: bool = True,
        scale: bool = True,
        name: Optional[str] = None
    ):
        """Initialize within-group scaling transformation.

        Parameters
        ----------
        groups : array-like
            Group indicators for each observation. Length must
            match the data passed to :meth:`evaluate`.
        center : bool, optional
            Whether to subtract the group mean. Default is True.
        scale : bool, optional
            Whether to divide by the group standard deviation.
            Default is True.
        name : str, optional
            Name for the transformation. Default is
            ``'scalewithin'``.
        """
        self.groups = np.asarray(groups)
        self.center = center
        self.scale = scale
        name = name or "scalewithin"
        super().__init__(name)
        
        # Group statistics
        self._group_stats: dict[object, dict[str, object]] = {}
    
    @property
    def n_basis(self) -> int:
        """Number of basis functions (always 1)."""
        return 1
    
    @property
    def basis_names(self) -> List[str]:
        """Names for basis functions."""
        return [self.name]
    
    def evaluate(self, x: ArrayLike) -> Array:
        """Apply group-wise scaling transformation.
        
        Parameters
        ----------
        x : array-like
            Values to transform
        
        Returns
        -------
        Array
            Transformed values, shape (len(x), 1)
        """
        x = np.asarray(x)
        if len(x) != len(self.groups):
            raise ValueError(
                f"Length mismatch: {len(x)} values but {len(self.groups)} groups"
            )
        
        result = np.zeros_like(x, dtype=float).reshape(-1, 1)
        
        # Process each group
        unique_groups = np.unique(self.groups)
        for group in unique_groups:
            mask = self.groups == group
            group_data = x[mask]
            
            # Compute group statistics
            group_mean = np.mean(group_data)
            group_std = np.std(group_data)
            
            self._group_stats[group] = {
                'mean': group_mean,
                'std': group_std,
                'n': len(group_data)
            }
            
            # Transform group data
            transformed = group_data.copy()
            
            if self.center:
                transformed = transformed - group_mean
            
            if self.scale and group_std > 0:
                transformed = transformed / group_std
            
            result[mask, 0] = transformed
        
        return result


class RobustScale(ParametricBasis):
    """Robust scaling transformation.
    
    Centers at median and scales by MAD (median absolute deviation)
    or IQR (interquartile range).
    
    Parameters
    ----------
    center : bool, optional
        Whether to center at median. Default is True.
    scale : {'mad', 'iqr'}, optional
        Scaling method. Default is 'mad'.
    constant : float, optional
        Constant for MAD scaling (1.4826 for normal). Default is 1.4826.
    name : str, optional
        Name for the transformation. Default is 'robustscale'.
    
    Examples
    --------
    >>> transform = RobustScale(scale='mad')
    >>> x = np.array([1, 2, 3, 4, 100])  # 100 is outlier
    >>> z = transform.evaluate(x)
    """
    
    def __init__(
        self,
        center: bool = True,
        scale: str = "mad",
        constant: float = 1.4826,
        name: Optional[str] = None,
    ) -> None:
        """Initialize robust scaling transformation.

        Parameters
        ----------
        center : bool, optional
            Whether to subtract the median. Default is True.
        scale : {'mad', 'iqr'}
            Scaling method. ``'mad'`` uses the median absolute
            deviation; ``'iqr'`` uses the interquartile range.
            Default is ``'mad'``.
        constant : float, optional
            Consistency constant for MAD scaling. The default
            1.4826 makes MAD consistent with the standard
            deviation for normal distributions.
        name : str, optional
            Name for the transformation. Default is
            ``'robustscale'``.

        Raises
        ------
        ValueError
            If ``scale`` is not ``'mad'`` or ``'iqr'``.
        """
        if scale not in ('mad', 'iqr'):
            raise ValueError("scale must be 'mad' or 'iqr'")

        self.center = center
        self.scale = scale
        self.constant = constant

        super().__init__(name or "robustscale")

        self._median: Optional[float] = None
        self._scale_factor: Optional[float] = None
        self.y: Optional[Array] = None

    @classmethod
    def from_data(
        cls,
        x: ArrayLike,
        center: bool = True,
        scale: str = "mad",
        constant: float = 1.4826,
        name: Optional[str] = None,
    ) -> "RobustScale":
        """Construct a :class:`RobustScale` and evaluate it on ``x``.

        Mirrors R's ``robustscale(x)`` ergonomics: constructs the
        transform, evaluates on the supplied data, and stores the
        result on ``.y``.
        """
        basis = cls(center=center, scale=scale, constant=constant, name=name)
        basis.y = basis.evaluate(x)
        return basis
    
    @property
    def n_basis(self) -> int:
        """Number of basis functions (always 1)."""
        return 1
    
    @property
    def basis_names(self) -> List[str]:
        """Names for basis functions."""
        return [self.name]
    
    def evaluate(self, x: ArrayLike) -> Array:
        """Apply robust scaling transformation.
        
        Parameters
        ----------
        x : array-like
            Values to transform
        
        Returns
        -------
        Array
            Transformed values, shape (len(x), 1)
        """
        x = np.asarray(x)
        result = x.copy().reshape(-1, 1)
        
        # Compute robust statistics
        self._median = float(np.median(x))

        if self.scale == 'mad':
            # Median absolute deviation
            mad = float(np.median(np.abs(x - self._median)))
            self._scale_factor = mad * self.constant
        else:  # iqr
            # Interquartile range
            quartiles = np.asarray(np.percentile(x, [25, 75]), dtype=np.float64)
            q1 = float(quartiles[0])
            q3 = float(quartiles[1])
            self._scale_factor = q3 - q1

        # Apply transformations
        if self.center:
            result = result - self._median

        if self._scale_factor is not None and self._scale_factor > 0:
            result = result / self._scale_factor
        
        self.y = result
        return result


class Ident(ParametricBasis):
    """Identity transformation (no change).
    
    Returns input values unchanged. Useful as a placeholder
    or for explicit no-transformation in pipelines.
    
    Parameters
    ----------
    name : str, optional
        Name for the transformation. Default is 'ident'.
    
    Examples
    --------
    >>> transform = Ident()
    >>> x = np.array([1, 2, 3])
    >>> y = transform.evaluate(x)
    >>> np.array_equal(x, y.ravel())
    True
    """
    
    def __init__(self, name: Optional[str] = None):
        """Initialize identity transformation.

        Parameters
        ----------
        name : str, optional
            Name for the transformation. Default is ``'ident'``.
        """
        name = name or "ident"
        super().__init__(name)
    
    @property
    def n_basis(self) -> int:
        """Number of basis functions (always 1)."""
        return 1
    
    @property
    def basis_names(self) -> List[str]:
        """Names for basis functions."""
        return [self.name]
    
    def evaluate(self, x: ArrayLike) -> Array:
        """Apply identity transformation (no change).
        
        Parameters
        ----------
        x : array-like
            Values to transform
        
        Returns
        -------
        Array
            Input values unchanged, shape (len(x), 1)
        """
        x = np.asarray(x)
        return x.reshape(-1, 1)


class Standardized(ParametricBasis):
    """Standardized transformation.
    
    Transforms input to have zero mean and unit variance.
    This is equivalent to z-score standardization but as
    a basis function transform.
    
    Parameters
    ----------
    center : bool, optional
        Whether to center (subtract mean). Default is True.
    scale : bool, optional
        Whether to scale (divide by standard deviation). Default is True.
    name : str, optional
        Name for the basis. Default is 'standardized'.
    
    Examples
    --------
    >>> std = Standardized()
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> z = std.evaluate(x)
    >>> np.mean(z), np.std(z)
    (0.0, 1.0)
    >>> 
    >>> # Only centering, no scaling
    >>> centered = Standardized(scale=False)
    >>> y = centered.evaluate(x)
    >>> np.mean(y), np.std(y)
    (0.0, 1.41421356)
    """
    
    def __init__(
        self,
        center: bool = True,
        scale: bool = True,
        name: Optional[str] = None
    ):
        """Initialize standardized transformation.

        Parameters
        ----------
        center : bool, optional
            Whether to subtract the mean. Default is True.
        scale : bool, optional
            Whether to divide by the standard deviation.
            Default is True.
        name : str, optional
            Name for the basis. Default is ``'standardized'``.
        """
        self.center = center
        self.scale = scale
        self._mean: Optional[float] = None
        self._std: Optional[float] = None
        super().__init__(name or "standardized")
    
    @property
    def n_basis(self) -> int:
        """Number of basis functions (always 1)."""
        return 1
    
    @property
    def basis_names(self) -> List[str]:
        """Names for basis functions."""
        return [self.name]
    
    def fit(self, x: ArrayLike) -> "Standardized":
        """Fit the standardization parameters.
        
        Parameters
        ----------
        x : array-like
            Data to compute mean and standard deviation from
        
        Returns
        -------
        self
            For method chaining
        """
        x = np.asarray(x)
        
        if self.center:
            self._mean = float(np.mean(x))
        else:
            self._mean = 0.0

        if self.scale:
            self._std = float(np.std(x))
            if self._std == 0:
                self._std = 1.0
        else:
            self._std = 1.0
        
        return self
    
    def evaluate(self, x: ArrayLike) -> Array:
        """Apply standardization transform.
        
        If fit() has not been called, fits on the input data.
        
        Parameters
        ----------
        x : array-like
            Input values
        
        Returns
        -------
        Array
            Standardized values as column vector
        """
        x = np.asarray(x)
        
        # Auto-fit if not already fitted
        if self._mean is None:
            self.fit(x)
        
        # Apply transformation
        result = (x - self._mean) / self._std
        return result.reshape(-1, 1)
