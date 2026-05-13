"""Core HRF classes and functionality."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, Optional, Sequence, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from .normalization import NormMode


HrfParamValue = Union[int, float, str, bool, Sequence[float]]
"""Value types accepted in :attr:`HRF.params`.

Every HRF subclass populates ``params`` with primitive scalars (kind
parameters like ``shape``, ``rate``, ``n_basis``), short string flags
(``method``), boolean toggles, or short numeric sequences (``weights``,
``times``). The alias names this contract so the signature stays out of
``Any``-typed territory."""


@dataclass
class HRF(ABC):
    """Base class for Hemodynamic Response Functions.

    An HRF object represents a hemodynamic response function that can be evaluated
    at arbitrary time points. HRFs can have one or more basis functions.

    Attributes:
        name: Name of the HRF
        nbasis: Number of basis functions
        span: Temporal span in seconds
        params: Dictionary of HRF parameters
        param_names: List of parameter names
    """
    name: str
    nbasis: int = 1
    span: float = 24.0
    params: Dict[str, HrfParamValue] = field(default_factory=dict)
    param_names: Optional[list[str]] = None
    
    def __post_init__(self) -> None:
        """Initialize parameter names from params if not provided."""
        if self.param_names is None and self.params:
            self.param_names = list(self.params.keys())
    
    @abstractmethod
    def __call__(self, t: ArrayLike) -> NDArray[np.float64]:
        """Evaluate the HRF at time points t.
        
        Args:
            t: Time points at which to evaluate the HRF
            
        Returns:
            Array of HRF values. Shape is (len(t),) for single basis,
            or (len(t), nbasis) for multi-basis HRFs.
        """
        pass
    
    def evaluate(
        self,
        grid: ArrayLike,
        duration: float = 0,
        precision: float = 0.1,
        summate: bool = True,
        normalize: bool | None = None,
    ) -> NDArray[np.float64]:
        """Evaluate HRF with optional block duration.

        Block convolution delegates to :class:`BlockedHRF` so the
        quadrature lives in exactly one place. Normalization is a
        separate transformation; call :meth:`normalize` first.

        Args:
            grid: Time points at which to evaluate
            duration: Duration of block/sustained stimulus in seconds
            precision: Temporal precision for convolution in seconds
            summate: If True, responses accumulate; if False, averaged
            normalize: Retired. Use ``hrf.normalize(mode).evaluate(...)``.

        Returns:
            Evaluated HRF values at grid points
        """
        grid = np.asarray(grid)

        if len(grid) == 0:
            raise ValueError("grid must contain at least one time point")
        if np.any(np.isnan(grid)):
            raise ValueError("grid cannot contain NaN values")
        if precision <= 0:
            raise ValueError("precision must be positive")
        if normalize is not None:
            raise ValueError(
                "HRF.evaluate(normalize=...) is retired; "
                "use normalize(hrf, mode) before evaluate()."
            )

        if duration < precision:
            result = self(grid)
        else:
            from .decorators import BlockedHRF

            blocked = BlockedHRF(
                base=self,
                width=float(duration),
                precision=precision,
                summate=summate,
            )
            result = blocked(grid)

        if len(grid) == 1 and self.nbasis > 1:
            result = result.reshape(1, -1)

        return result
    
    def from_coefficients(self, coefficients: ArrayLike) -> Callable[[ArrayLike], NDArray[np.float64]]:
        """Create a function that evaluates HRF with given coefficients.
        
        Args:
            coefficients: Coefficients for linear combination of basis functions
            
        Returns:
            Function that evaluates the weighted HRF
        """
        coefficients = np.asarray(coefficients)
        
        if len(coefficients) != self.nbasis:
            raise ValueError(
                f"Number of coefficients ({len(coefficients)}) must match "
                f"number of basis functions ({self.nbasis})"
            )
        
        def weighted_hrf(t: ArrayLike) -> NDArray[np.float64]:
            """Evaluate weighted combination of basis functions."""
            basis_values = self(t)
            
            if self.nbasis == 1:
                return coefficients[0] * basis_values
            else:
                # Matrix multiplication for multi-basis case
                return basis_values @ coefficients
        
        return weighted_hrf
    
    def plot(self, time=None, normalize: bool = False, show_peak: bool = True, ax=None, **kwargs):
        """Plot this HRF.  See :func:`fmrimod.plotting.plot_hrf`."""
        from ..plotting import plot_hrf
        return plot_hrf(self, time=time, normalize=normalize, show_peak=show_peak, ax=ax, **kwargs)

    # -- Fluent decorators (return new HRF, mirror fmrihrf's gen_hrf chain) --

    def lag(self, dt: float) -> "HRF":
        """Return a new HRF lagged by ``dt`` seconds.

        Mirrors R's :func:`lag_hrf`.  Equivalent to
        :func:`fmrimod.hrf.lag_hrf(self, dt) <fmrimod.hrf.lag_hrf>`.
        """
        from .decorators import lag_hrf

        return lag_hrf(self, dt)

    def block(
        self,
        width: float,
        precision: float = 0.1,
        half_life: float = float("inf"),
        summate: bool = True,
        normalize: bool = False,
    ) -> "HRF":
        """Return a new HRF convolved with a boxcar of duration ``width``.

        Mirrors R's :func:`block_hrf`.
        """
        from .decorators import block_hrf

        return block_hrf(
            self,
            width=width,
            precision=precision,
            half_life=half_life,
            summate=summate,
            normalize=normalize,
        )

    def normalize(self, mode: "NormMode" = "unit_peak") -> "HRF":
        """Return a new HRF normalized via :func:`fmrimod.hrf.normalize`."""
        from .normalization import normalize

        return normalize(self, mode)

    # -- Composition --

    def __add__(self, other: "HRF") -> "HRF":
        """Column-bind two HRFs into a multi-basis HRF.

        Equivalent to :func:`fmrimod.hrf.bind_basis(self, other)
        <fmrimod.hrf.bind_basis>`.  Lets users write
        ``HRF_SPMG1 + HRF_GAMMA`` to assemble a 2-basis set.
        """
        if not isinstance(other, HRF):
            return NotImplemented
        return bind_basis(self, other)

    def __str__(self) -> str:
        """String representation of HRF."""
        params_str = ""
        if self.params:
            param_items = [f"{k}={v}" for k, v in self.params.items()]
            params_str = f", {', '.join(param_items)}"
        return f"HRF(name='{self.name}', nbasis={self.nbasis}, span={self.span}{params_str})"

    def __repr__(self) -> str:
        """Detailed representation of HRF."""
        return self.__str__()


class FunctionHRF(HRF):
    """HRF created from a callable function.
    
    This class wraps a regular Python function to create an HRF object.
    """
    
    def __init__(
        self,
        func: Callable[[ArrayLike], NDArray[np.float64]],
        name: Optional[str] = None,
        nbasis: int = 1,
        span: float = 24.0,
        params: Optional[Dict[str, HrfParamValue]] = None,
        param_names: Optional[list[str]] = None,
    ):
        """Initialize FunctionHRF.
        
        Args:
            func: Function that evaluates the HRF
            name: Name of the HRF (defaults to function name)
            nbasis: Number of basis functions
            span: Temporal span in seconds
            params: Dictionary of HRF parameters
            param_names: List of parameter names
        """
        if name is None:
            name = getattr(func, '__name__', 'custom_hrf')
        
        super().__init__(
            name=name,
            nbasis=nbasis,
            span=span,
            params=params or {},
            param_names=param_names,
        )
        self.func = func
    
    def __call__(self, t: ArrayLike) -> NDArray[np.float64]:
        """Evaluate the HRF at time points t."""
        return self.func(t)


def as_hrf(
    func: Callable[[ArrayLike], NDArray[np.float64]],
    name: Optional[str] = None,
    nbasis: int = 1,
    span: float = 24.0,
    params: Optional[Dict[str, Any]] = None,
) -> HRF:
    """Convert a function to an HRF object.
    
    Args:
        func: Function that evaluates the HRF
        name: Name of the HRF (defaults to function name)
        nbasis: Number of basis functions
        span: Temporal span in seconds
        params: Dictionary of HRF parameters
        
    Returns:
        HRF object wrapping the function
    """
    # Extract parameter names from params dict
    param_names = list(params.keys()) if params else None
    
    return FunctionHRF(
        func=func,
        name=name,
        nbasis=nbasis,
        span=span,
        params=params,
        param_names=param_names,
    )


@dataclass
class BoundBasisHRF(HRF):
    """HRF formed by column-binding multiple HRF components.

    ``components`` is the structural seam preserved across the
    composition: downstream code can introspect which HRFs were bound
    instead of pattern-matching on a generated name. See bead
    ``bd-01KRGCYXE7CQTT86MW7FRR309A`` for the migration away from the
    earlier ``FunctionHRF``-wrapped form.
    """

    name: str = ""
    nbasis: int = 1
    span: float = 24.0
    components: tuple[HRF, ...] = ()

    def __post_init__(self) -> None:
        if not self.components:
            raise ValueError("BoundBasisHRF requires at least one component")
        # Derive identity / span / nbasis from the components rather than
        # accepting whatever the parent dataclass defaults handed us.
        self.name = " + ".join(c.name for c in self.components)
        self.nbasis = sum(c.nbasis for c in self.components)
        self.span = max(c.span for c in self.components)
        combined_params: Dict[str, HrfParamValue] = {}
        for c in self.components:
            for key, value in c.params.items():
                combined_params[f"{c.name}_{key}"] = value
        self.params = combined_params
        self.param_names = list(combined_params.keys()) if combined_params else None

    def __call__(self, t: ArrayLike) -> NDArray[np.float64]:
        t = np.asarray(t)
        results = []
        for component in self.components:
            result = component(t)
            if component.nbasis == 1 and result.ndim == 1:
                result = result.reshape(-1, 1)
            results.append(result)
        return np.hstack(results)


def bind_basis(*hrfs: HRF) -> HRF:
    """Combine multiple HRF objects into a single multi-basis HRF.

    Args:
        *hrfs: HRF objects to combine

    Returns:
        Combined HRF with concatenated basis functions. A single input is
        returned unchanged; two or more inputs produce a :class:`BoundBasisHRF`
        whose ``components`` tuple preserves the originals for structural
        introspection.
    """
    if len(hrfs) == 0:
        raise ValueError("At least one HRF must be provided")

    if len(hrfs) == 1:
        return hrfs[0]

    return BoundBasisHRF(components=tuple(hrfs))


def hrf_from_coefficients(hrf: HRF, coefficients: ArrayLike) -> HRF:
    """Create a new HRF by linearly combining basis functions.
    
    Takes an HRF with multiple basis functions and creates a new HRF
    that is a linear combination of those basis functions.
    
    Args:
        hrf: HRF object with multiple basis functions
        coefficients: Coefficients for linear combination
        
    Returns:
        New HRF object
        
    Examples:
        >>> # Get a 3-basis HRF
        >>> hrf = get_hrf("spmg3")
        >>> 
        >>> # Create custom combination
        >>> coefs = [1.0, 0.5, -0.2]  # Main + 0.5*deriv1 - 0.2*deriv2
        >>> custom_hrf = hrf_from_coefficients(hrf, coefs)
    """
    coefficients = np.asarray(coefficients, dtype=np.float64)
    
    if len(coefficients) != hrf.nbasis:
        raise ValueError(
            f"Number of coefficients ({len(coefficients)}) must match "
            f"number of basis functions ({hrf.nbasis})"
        )
    
    # Create function that evaluates the linear combination
    def combined_func(t: ArrayLike) -> NDArray[np.float64]:
        # Evaluate base HRF
        basis_values = hrf(t)
        
        # Handle single basis case
        if hrf.nbasis == 1:
            return coefficients[0] * basis_values
        
        # Multiple basis - ensure 2D
        if basis_values.ndim == 1:
            basis_values = basis_values.reshape(-1, 1)
        
        # Linear combination
        result = basis_values @ coefficients
        
        return result
    
    # Create new HRF
    return FunctionHRF(
        func=combined_func,
        name=f"{hrf.name}_combined",
        nbasis=1,
        span=hrf.span,
        params={"coefficients": coefficients.tolist()},
        param_names=["coefficients"]
    )
