"""Core HRF classes and functionality."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence, Union, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from .normalization import NormMode


HrfParamValue = Union[int, float, str, bool, Sequence[float]]
"""Value type for HRF parameter keyword arguments.

Used by the HRF factories (:func:`gen_hrf`, :func:`get_hrf`,
:func:`make_hrf`) and the SPM helpers to keep ``**kwargs`` parameter
values (primitive scalars like ``shape``/``rate``/``n_basis``, short
string flags like ``method``, boolean toggles, or short numeric
sequences) out of ``Any``-typed territory. HRF parameters themselves
now live as typed dataclass fields on each subclass, not a dict."""


@dataclass
class HRF(ABC):
    """Base class for Hemodynamic Response Functions.

    An HRF object represents a hemodynamic response function that can be evaluated
    at arbitrary time points. HRFs can have one or more basis functions.

    Attributes:
        name: Name of the HRF
        nbasis: Number of basis functions
        span: Temporal span in seconds

    Concrete subclasses declare their parameters as typed dataclass
    fields (e.g. ``GammaHRF.shape``); the type system is the parameter
    contract — there is no ``params`` dict.
    """
    name: str
    nbasis: int = 1
    span: float = 24.0

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
                return cast("NDArray[np.float64]", coefficients[0] * basis_values)
            else:
                # Matrix multiplication for multi-basis case
                return cast("NDArray[np.float64]", basis_values @ coefficients)
        
        return weighted_hrf
    
    def plot(
        self,
        time: Any = None,
        normalize: bool = False,
        show_peak: bool = True,
        ax: Any = None,
        **kwargs: object,
    ) -> Any:
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

    _STR_SKIP_FIELDS = frozenset({"name", "nbasis", "span"})

    def __str__(self) -> str:
        """String representation of HRF.

        Parameter fields are read from the typed dataclass fields of the
        concrete subclass (the type system is the parameter contract),
        not a back-compat ``params`` dict. Base identity fields and
        fields that hold HRF instances (decorator ``base`` /
        ``components``) are skipped to keep the rendering flat.
        """
        param_items: list[str] = []
        for f in fields(self):
            if f.name in self._STR_SKIP_FIELDS:
                continue
            value = getattr(self, f.name, None)
            if isinstance(value, HRF):
                continue
            if isinstance(value, (tuple, list)) and any(
                isinstance(item, HRF) for item in value
            ):
                continue
            param_items.append(f"{f.name}={value}")
        params_str = f", {', '.join(param_items)}" if param_items else ""
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
    ):
        """Initialize FunctionHRF.

        Args:
            func: Function that evaluates the HRF
            name: Name of the HRF (defaults to function name)
            nbasis: Number of basis functions
            span: Temporal span in seconds
        """
        if name is None:
            name = getattr(func, '__name__', 'custom_hrf')

        super().__init__(
            name=name,
            nbasis=nbasis,
            span=span,
        )
        self.func = func
    
    def __call__(self, t: ArrayLike) -> NDArray[np.float64]:
        """Evaluate the HRF at time points t."""
        return self.func(t)


def _with_metadata(
    hrf: HRF,
    *,
    name: Optional[str] = None,
    span: Optional[float] = None,
) -> HRF:
    """Return ``hrf`` with display metadata updated.

    Typed HRF instances keep their concrete class via ``dataclasses.replace``.
    ``FunctionHRF`` remains the fallback only for raw-callable adapters.
    """
    if name is None and span is None:
        return hrf

    next_name = hrf.name if name is None else name
    next_span = hrf.span if span is None else span

    if is_dataclass(hrf) and not isinstance(hrf, FunctionHRF):
        updated = replace(hrf)
        updated.name = next_name
        updated.span = next_span
        return updated

    return FunctionHRF(
        func=hrf,
        name=next_name,
        nbasis=hrf.nbasis,
        span=next_span,
    )


def as_hrf(
    func: Callable[[ArrayLike], NDArray[np.float64]],
    name: Optional[str] = None,
    nbasis: int = 1,
    span: float = 24.0,
) -> HRF:
    """Convert a function to an HRF object.

    Args:
        func: Function that evaluates the HRF
        name: Name of the HRF (defaults to function name)
        nbasis: Number of basis functions
        span: Temporal span in seconds

    Returns:
        HRF object wrapping the function
    """
    return FunctionHRF(
        func=func,
        name=name,
        nbasis=nbasis,
        span=span,
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


@dataclass(init=False)
class CoefficientHRF(HRF):
    """Single-basis HRF formed from coefficients over another HRF basis."""

    base: HRF  # type: ignore[misc]
    coefficients: tuple[float, ...]  # type: ignore[misc]

    def __init__(
        self,
        base: HRF,
        coefficients: ArrayLike,
        name: Optional[str] = None,
    ) -> None:
        coefs = tuple(float(x) for x in np.asarray(coefficients, dtype=np.float64))
        if len(coefs) != base.nbasis:
            raise ValueError(
                f"Number of coefficients ({len(coefs)}) must match "
                f"number of basis functions ({base.nbasis})"
            )
        self.base = base
        self.coefficients = coefs
        self.name = name or f"{base.name}_combined"
        self.nbasis = 1
        self.span = base.span

    def __call__(self, t: ArrayLike) -> NDArray[np.float64]:
        basis_values = self.base(t)
        coefs = np.asarray(self.coefficients, dtype=np.float64)
        if self.base.nbasis == 1:
            return cast("NDArray[np.float64]", coefs[0] * basis_values)
        if basis_values.ndim == 1:
            basis_values = basis_values.reshape(-1, 1)
        return basis_values @ coefs


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
    return CoefficientHRF(hrf, coefficients)
