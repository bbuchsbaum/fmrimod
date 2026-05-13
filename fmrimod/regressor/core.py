"""Core Regressor classes for fMRI event modeling."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple, Union, cast

import numpy as np
import scipy
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csr_matrix

from ..hrf import HRF
from ..hrf.generators import make_hrf
from .convolution import ConvolutionMethod, convolve_hrf, validate_convolution_method
from .neural_input import neural_input_core

# The factory input type. Strings and callables are coerced to HRF via
# make_hrf() in _coerce_hrf(); the stored Regressor.hrf field is strictly
# typed below (Union[HRF, List[HRF]]).
HRFSpec = Union[HRF, str, Callable[..., object]]
HRFLike = Union[HRFSpec, List[HRFSpec]]


def _coerce_one_hrf(hrf: HRFSpec) -> HRF:
    if isinstance(hrf, HRF):
        return hrf
    return make_hrf(cast(Union[str, dict[str, Any], HRF], hrf), lag=0)


def _coerce_hrf(hrf: HRFLike) -> Union[HRF, List[HRF]]:
    """Normalize a user-supplied HRF spec to a typed HRF or list of HRFs.

    Strings and callables are resolved through :func:`make_hrf`; HRF values
    pass through unchanged. List inputs are coerced element-wise.
    """
    if isinstance(hrf, list):
        return [_coerce_one_hrf(h) for h in hrf]
    return _coerce_one_hrf(hrf)


def _recycle_or_error(
    values: ArrayLike, 
    target_length: int, 
    name: str
) -> NDArray[np.float64]:
    """Recycle scalar to array or validate array length.
    
    Args:
        values: Scalar or array-like values
        target_length: Expected length
        name: Parameter name for error messages
        
    Returns:
        Array of target_length
        
    Raises:
        ValueError: If array length doesn't match target_length
    """
    values = np.asarray(values, dtype=np.float64)
    
    if values.ndim == 0:  # Scalar
        return np.full(target_length, values.item())
    elif len(values) == target_length:
        return values
    else:
        raise ValueError(
            f"Length of {name} ({len(values)}) must match number of onsets ({target_length})"
        )


@dataclass(frozen=True)
class Regressor:
    """Represents event-related regressors for fMRI modeling.

    Instances are immutable after construction; consumers should build new
    regressors via the :func:`regressor` factory or :meth:`shift` rather
    than mutating fields in place.

    The *hrf* field may be a single :class:`HRF` object (applied to every
    event) **or** a list of HRF objects for trial-varying designs.  When a
    list is given its length must be 1 (recycled) or equal to the number of
    onsets.

    Attributes:
        onsets: Event onset times in seconds
        hrf: Hemodynamic response function (single HRF or list of HRFs)
        duration: Event durations in seconds
        amplitude: Event amplitudes/scaling factors
        span: HRF temporal span in seconds
        summate: Whether overlapping HRF responses should sum
        filtered_all: Whether all events were filtered out (zero/NA amplitudes)
    """

    onsets: NDArray[np.float64]
    hrf: Union[HRF, List[HRF]]
    duration: NDArray[np.float64] = field(default_factory=lambda: np.array([0.0]))
    amplitude: NDArray[np.float64] = field(default_factory=lambda: np.array([1.0]))
    # ``None`` means "derive span from the HRF"; a numeric value is honored as
    # the user-supplied override. After ``__post_init__`` ``self.span`` is
    # always a positive finite float.
    span: Optional[float] = None
    summate: bool = True
    filtered_all: bool = field(default=False, init=False)

    # --- properties for trial-varying HRFs ---

    @property
    def hrf_is_list(self) -> bool:
        """True when *hrf* is a list of per-event HRFs."""
        return isinstance(self.hrf, list)

    @property
    def nbasis(self) -> int:
        """Number of HRF basis functions (must be consistent across list)."""
        hrf = self.hrf
        if isinstance(hrf, list):
            return hrf[0].nbasis if len(hrf) > 0 else 1
        return hrf.nbasis

    def __post_init__(self) -> None:
        """Validate and process inputs after initialization."""
        _set = object.__setattr__

        # Convert inputs to numpy arrays
        _set(self, "onsets", np.asarray(self.onsets, dtype=np.float64))
        _set(self, "duration", np.asarray(self.duration, dtype=np.float64))
        _set(self, "amplitude", np.asarray(self.amplitude, dtype=np.float64))

        # Handle single NA onset (represents zero events)
        if len(self.onsets) == 1 and np.isnan(self.onsets[0]):
            _set(self, "onsets", np.array([], dtype=np.float64))

        # Validate explicit span; None means "derive from HRF later".
        if self.span is not None:
            if not np.isfinite(self.span) or self.span <= 0:
                raise ValueError("span must be a positive, finite number")

        n_onsets = len(self.onsets)

        # Validate onsets: must be finite and non-negative
        if n_onsets > 0:
            if np.any(~np.isfinite(self.onsets)):
                raise ValueError("onsets must contain finite numeric values")
            if np.any(self.onsets < 0):
                raise ValueError("onsets must be non-negative")

        # Recycle duration and amplitude before validation
        if n_onsets > 0:
            _set(self, "duration", _recycle_or_error(self.duration, n_onsets, "duration"))
            _set(self, "amplitude", _recycle_or_error(self.amplitude, n_onsets, "amplitude"))

        # Validate duration after recycling (ensures 1-d array)
        if self.duration.size > 0:
            if np.any(~np.isfinite(self.duration)):
                raise ValueError("duration must contain finite numeric values")
            if np.any(self.duration < 0):
                raise ValueError("duration cannot be negative")

        # --- Handle list-of-HRFs (trial-varying) ---
        hrf_value = self.hrf
        _hrf_is_list = isinstance(hrf_value, list)
        if isinstance(hrf_value, list):
            for i, h in enumerate(hrf_value):
                if not isinstance(h, HRF):
                    raise TypeError(
                        f"Regressor.hrf list element {i} must be an HRF; "
                        f"got {type(h).__name__}. Use the regressor() factory "
                        "to coerce strings or callables."
                    )
            # Recycle length-1 list
            if len(hrf_value) == 1 and n_onsets > 0:
                hrf_value = hrf_value * n_onsets
                _set(self, "hrf", hrf_value)
            elif len(hrf_value) != n_onsets and n_onsets > 0:
                raise ValueError(
                    f"`hrf` list must have length 1 or {n_onsets}, "
                    f"got {len(hrf_value)}."
                )
            # Validate consistent nbasis
            if len(hrf_value) > 0:
                nb0 = hrf_value[0].nbasis
                for i, h in enumerate(hrf_value):
                    if h.nbasis != nb0:
                        raise ValueError(
                            f"All HRFs in list must have the same nbasis. "
                            f"Element 0 has {nb0}, element {i} has {h.nbasis}."
                        )

        # Filter events based on non-zero and non-NA amplitude
        if n_onsets > 0:
            keep_indices = np.where((self.amplitude != 0) & ~np.isnan(self.amplitude))[0]
            filtered_some = len(keep_indices) < n_onsets
            _set(self, "filtered_all", len(keep_indices) == 0)

            if filtered_some:
                _set(self, "onsets", self.onsets[keep_indices])
                _set(self, "duration", self.duration[keep_indices])
                _set(self, "amplitude", self.amplitude[keep_indices])
                if _hrf_is_list:
                    hrf_list = cast(List[HRF], self.hrf)
                    _set(self, "hrf", [hrf_list[i] for i in keep_indices])
        else:
            _set(self, "filtered_all", True)

        # Ensure HRF is valid (single-HRF case)
        if not _hrf_is_list and not isinstance(self.hrf, HRF):
            raise TypeError(
                f"Regressor.hrf must be an HRF or list of HRFs; "
                f"got {type(self.hrf).__name__}. Use the regressor() factory "
                "to coerce strings or callables."
            )

        # Resolve span: explicit value wins; otherwise derive from the HRF.
        if self.span is None:
            if _hrf_is_list:
                hrf_list = cast(List[HRF], self.hrf)
                derived = (
                    max(h.span for h in hrf_list) if len(hrf_list) > 0 else 40.0
                )
            else:
                hrf_single = cast(HRF, self.hrf)
                derived = (
                    hrf_single.span
                    if getattr(hrf_single, "span", None) is not None
                    else 40.0
                )
            span_value = float(derived)
            if not np.isfinite(span_value) or span_value <= 0:
                raise ValueError(
                    "Derived span from HRF is not a positive, finite number; "
                    "supply span= explicitly."
                )
            _set(self, "span", span_value)
    
    def neural_input(
        self, 
        start: float = 0.0,
        end: Optional[float] = None,
        resolution: float = 0.33
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Generate neural input time series.
        
        Args:
            start: Start time in seconds
            end: End time in seconds (if None, auto-determined)
            resolution: Time resolution in seconds
            
        Returns:
            Tuple of (time_points, neural_input_values)
        """
        if end is None:
            if len(self.onsets) > 0:
                end = np.max(self.onsets + self.duration) + 10.0
            else:
                end = start + cast(float, self.span)
        
        return neural_input_core(
            self.onsets, 
            self.duration, 
            self.amplitude,
            start, 
            end, 
            resolution
        )

    def evaluate(
        self,
        grid: ArrayLike,
        precision: float = 0.33,
        method: ConvolutionMethod = "conv",
        sparse: bool = False,
        normalize: bool = False,
    ) -> Union[NDArray[np.float64], scipy.sparse.csr_matrix]:
        """Evaluate the regressor at specified time points.

        Args:
            grid: Time points at which to evaluate (seconds)
            precision: Temporal precision for convolution (seconds)
            method: Evaluation method ('conv', 'fft', 'direct')
            sparse: Whether to return sparse matrix
            normalize: Whether to normalize each column to unit peak

        Returns:
            Evaluated regressor values at grid points
        """
        grid = np.asarray(grid, dtype=np.float64)

        # Validate inputs
        if len(grid) == 0 or np.any(np.isnan(grid)):
            raise ValueError("grid must be a non-empty numeric vector with no NA values")
        if not np.isfinite(precision) or precision <= 0:
            raise ValueError("precision must be a positive numeric value")

        # Sort grid if needed
        if not np.all(grid[:-1] <= grid[1:]):
            warnings.warn("Input grid is unsorted. Sorting grid for evaluation.")
            grid = np.sort(grid)

        method = validate_convolution_method(method)
        nb = self.nbasis

        # If no events, return zeros
        if len(self.onsets) == 0 or self.filtered_all:
            result = np.zeros((len(grid), nb))
            if nb == 1:
                result = result.ravel()
            if sparse:
                return csr_matrix(result if result.ndim == 2 else result[:, np.newaxis])
            return result

        # Filter events based on grid boundaries
        onset_min = grid[0] - self.span
        onset_max = grid[-1]
        keep = (self.onsets >= onset_min) & (self.onsets <= onset_max)

        if not np.any(keep):
            result = np.zeros((len(grid), nb))
            if nb == 1:
                result = result.ravel()
            if sparse:
                return csr_matrix(result if result.ndim == 2 else result[:, np.newaxis])
            return result

        # --- Trial-varying HRF path (per-event loop) ---
        hrf_value = self.hrf
        span = cast(float, self.span)
        if isinstance(hrf_value, list):
            from .convolution import convolve_hrf_per_event
            keep_idx = np.where(keep)[0]
            result = convolve_hrf_per_event(
                grid=grid,
                onsets=self.onsets[keep_idx],
                durations=self.duration[keep_idx],
                amplitudes=self.amplitude[keep_idx],
                hrfs=[hrf_value[i] for i in keep_idx],
                span=span,
                precision=precision,
                summate=self.summate,
            )
        else:
            # Standard single-HRF convolution
            result = convolve_hrf(
                grid=grid,
                onsets=self.onsets[keep],
                durations=self.duration[keep],
                amplitudes=self.amplitude[keep],
                hrf=hrf_value,
                span=span,
                precision=precision,
                method=method,
                summate=self.summate
            )

        # Apply normalization if requested
        if normalize:
            if result.ndim == 2:
                for i in range(result.shape[1]):
                    peak = np.max(np.abs(result[:, i]))
                    if peak > 0:
                        result[:, i] /= peak
            else:
                peak = np.max(np.abs(result))
                if peak > 0:
                    result /= peak

        # Convert to sparse if requested
        if sparse:
            if result.ndim == 1:
                result = result[:, np.newaxis]
            return csr_matrix(result)

        return result
    
    def plot(
        self,
        grid: ArrayLike | None = None,
        precision: float = 0.33,
        show_onsets: bool = True,
        ax: Any = None,
        **kwargs: Any,
    ) -> object:
        """Plot this regressor.  See :func:`fmrimod.plotting.plot_regressor`."""
        from ..plotting import plot_regressor
        return plot_regressor(self, grid=grid, precision=precision, show_onsets=show_onsets, ax=ax, **kwargs)

    def shift(self, shift_amount: float) -> "Regressor":
        """Shift all event onsets by a temporal offset.

        Args:
            shift_amount: Amount to shift in seconds (positive = forward)

        Returns:
            New Regressor with shifted onsets
        """
        return Regressor(
            onsets=self.onsets + shift_amount,
            hrf=self.hrf,
            duration=self.duration,
            amplitude=self.amplitude,
            span=self.span,
            summate=self.summate
        )
    
    def __repr__(self) -> str:
        """String representation of Regressor."""
        n_events = len(self.onsets)
        hrf_value = self.hrf
        if isinstance(hrf_value, list):
            hrf_desc = f"list[{len(hrf_value)} HRFs]"
        else:
            hrf_desc = f"'{getattr(hrf_value, 'name', 'custom')}'"
        parts = [
            f"Regressor(n_events={n_events}",
            f"hrf={hrf_desc}",
            f"nbasis={self.nbasis}",
            f"span={self.span}",
        ]
        if n_events > 0:
            parts.append(f"onset_range=[{np.min(self.onsets):.1f}, {np.max(self.onsets):.1f}]")
        if self.hrf_is_list:
            parts.append("trial_varying=True")
        if not self.summate:
            parts.append("summate=False")
        return ", ".join(parts) + ")"


def regressor(
    onsets: ArrayLike,
    hrf: HRFLike = "spmg1",
    duration: Union[float, ArrayLike] = 0.0,
    amplitude: Union[float, ArrayLike] = 1.0,
    span: Optional[float] = None,
    summate: bool = True
) -> Regressor:
    """Create a Regressor object.

    This is the main public interface for creating regressor objects.

    Args:
        onsets: Event onset times in seconds.
        hrf: HRF object, string name, callable, or **list of HRFs** for
            trial-varying designs.  When a list is given its length must be
            1 (recycled to all events) or equal to ``len(onsets)``.
        duration: Event duration(s) in seconds.
        amplitude: Event amplitude(s).
        span: HRF temporal span in seconds.
        summate: Whether overlapping responses sum.

    Returns:
        Regressor object

    Examples:
        >>> reg = regressor(onsets=[10, 30, 50])
        >>> # Trial-varying HRFs
        >>> from fmrimod import boxcar_generator
        >>> h1 = boxcar_generator(width=4)
        >>> h2 = boxcar_generator(width=8)
        >>> reg_tv = regressor(onsets=[10, 30], hrf=[h1, h2])
    """
    return Regressor(
        onsets=np.asarray(onsets, dtype=np.float64),
        hrf=_coerce_hrf(hrf),
        duration=np.asarray(duration, dtype=np.float64),
        amplitude=np.asarray(amplitude, dtype=np.float64),
        span=span,
        summate=summate
    )


@dataclass
class RegressorSet:
    """A set of regressors for multiple conditions.
    
    Each condition shares the same HRF but has distinct onsets,
    durations, and amplitudes.
    
    Attributes:
        regressors: List of Regressor objects, one per condition
        levels: Condition names/labels
    """
    
    regressors: List[Regressor]
    levels: List[str]
    
    def evaluate(
        self,
        grid: ArrayLike,
        precision: float = 0.33,
        method: ConvolutionMethod = "conv",
        sparse: bool = False
    ) -> Union[NDArray[np.float64], scipy.sparse.csr_matrix]:
        """Evaluate all regressors to create design matrix.
        
        Args:
            grid: Time points at which to evaluate
            precision: Temporal precision
            method: Evaluation method
            sparse: Whether to return sparse matrix
            
        Returns:
            Design matrix with columns for each regressor
        """
        # Evaluate each regressor
        results = []
        for reg in self.regressors:
            result = reg.evaluate(grid, precision, method, sparse=False)
            if result.ndim == 1:
                result = result[:, np.newaxis]
            results.append(result)
        
        # Combine results
        design_matrix = np.hstack(results)
        
        if sparse:
            return csr_matrix(design_matrix)
        
        return design_matrix
    
    def __repr__(self) -> str:
        """String representation of RegressorSet."""
        n_conditions = len(self.levels)
        return f"RegressorSet(n_conditions={n_conditions}, levels={self.levels})"


def regressor_set(
    onsets: ArrayLike,
    fac: ArrayLike,
    hrf: HRFLike = "spmg1",
    duration: Union[float, ArrayLike] = 0.0,
    amplitude: Union[float, ArrayLike] = 1.0,
    span: Optional[float] = None,
    summate: bool = True
) -> RegressorSet:
    """Create a set of regressors for multiple conditions.
    
    Args:
        onsets: Event onset times
        fac: Factor indicating condition for each onset
        hrf: HRF to use for all conditions
        duration: Event duration(s)
        amplitude: Event amplitude(s)
        span: HRF temporal span
        summate: Whether overlapping responses sum
        
    Returns:
        RegressorSet object
        
    Examples:
        >>> # Create events for 3 conditions
        >>> onsets = [10, 20, 30, 40, 50, 60]
        >>> conditions = ['A', 'B', 'C', 'A', 'B', 'C']
        >>> 
        >>> # Create regressor set
        >>> rset = regressor_set(onsets, conditions)
        >>> 
        >>> # Evaluate to get design matrix
        >>> times = np.arange(0, 80, 0.1)
        >>> design_matrix = rset.evaluate(times)
    """
    # Convert inputs
    onsets = np.asarray(onsets, dtype=np.float64)
    fac = np.asarray(fac)
    hrf_typed = _coerce_hrf(hrf)

    # Match R as.factor() level ordering: sorted unique values.
    if np.issubdtype(fac.dtype, np.number):
        levels = np.unique(fac.astype(np.float64))
        if np.all(np.isclose(levels, np.round(levels))):
            levels_list = [str(int(round(lev))) for lev in levels]
        else:
            levels_list = [format(float(lev), "g") for lev in levels]
    else:
        levels = np.unique(fac.astype(str))
        levels_list = [str(lev) for lev in levels]
    
    # Recycle duration and amplitude
    duration = _recycle_or_error(duration, len(onsets), "duration")
    amplitude = _recycle_or_error(amplitude, len(onsets), "amplitude")
    
    # Create regressor for each level
    regressors = []
    for level in levels:
        idx = fac == level
        if np.any(idx):
            reg = Regressor(
                onsets=onsets[idx],
                hrf=hrf_typed,
                duration=duration[idx],
                amplitude=amplitude[idx],
                span=span,
                summate=summate
            )
        else:
            # Empty regressor for this level
            reg = Regressor(
                onsets=np.array([]),
                hrf=hrf_typed,
                duration=np.array([]),
                amplitude=np.array([]),
                span=span,
                summate=summate
            )
        regressors.append(reg)
    
    return RegressorSet(regressors=regressors, levels=levels_list)


def null_regressor(hrf: Optional[HRFLike] = None, span: float = 24.0) -> Regressor:
    """Create a null (empty) regressor with no events.

    Convenience function for creating a Regressor with zero events,
    used as a placeholder or baseline regressor.

    Args:
        hrf: HRF object, string name, or callable (default: SPM_CANONICAL)
        span: Temporal span (default: 24.0)

    Returns:
        Regressor with no onsets
    """
    if hrf is None:
        from ..hrf.library import SPM_CANONICAL
        hrf_typed: Union[HRF, List[HRF]] = SPM_CANONICAL
    else:
        hrf_typed = _coerce_hrf(hrf)
    return Regressor(
        onsets=np.array([], dtype=np.float64),
        hrf=hrf_typed,
        amplitude=np.array([0.0], dtype=np.float64),
        span=span,
    )
