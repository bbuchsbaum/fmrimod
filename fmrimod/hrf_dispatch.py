"""HRF dispatch adapters and wrapper classes.

The canonical HRF name registry lives in :mod:`fmrimod.hrf.registry`.
This module keeps compatibility imports plus generic conversion utilities
(``as_hrf``, ``ArrayHRF``, ``FunctionHRF``, ``DictHRF``) and generator
factories (``boxcar_hrf_gen``, ``duration_hrf_gen``, ``weighted_hrf_gen``).

For the full HRF basis library (SPM, Gamma, B-spline, FIR, etc.), see
:mod:`fmrimod.hrf`.
"""

from typing import Any, Callable, Dict, Literal, Optional, Union, cast

import numpy as np

from .types import HRFProtocol


class SimpleHRF(HRFProtocol):
    """Minimal gamma-shaped HRF for testing and prototyping.

    Implements ``h(t) = t * exp(-t/2)`` normalised to a peak of 1.
    This is *not* physiologically accurate and is intended only for
    quick testing when the full HRF library is not needed.

    Attributes
    ----------
    name : str
        Always ``"simple"``.
    nbasis : int
        Always 1 (single basis function).

    Examples
    --------
    >>> hrf = SimpleHRF()
    >>> t = np.arange(0, 20, 0.1)
    >>> response = hrf.evaluate(t)
    >>> response.shape
    (200,)
    """

    def __init__(self):
        self._name = "simple"
        self._nbasis = 1

    @property
    def name(self) -> str:
        """Name of the HRF."""
        return self._name

    @property
    def nbasis(self) -> int:
        """Number of basis functions."""
        return self._nbasis

    def evaluate(self, t: np.ndarray) -> np.ndarray:
        """Evaluate HRF at the given time points.

        Parameters
        ----------
        t : array-like
            Time points in seconds (negative values yield 0).

        Returns
        -------
        np.ndarray
            HRF values normalised so that the peak equals 1.
        """
        t = np.asarray(t)
        result = np.where(t > 0, t * np.exp(-t / 2), 0)
        if np.max(result) > 0:
            result = result / np.max(result)
        return result


class SPMCanonicalHRF(HRFProtocol):
    """Simplified SPM canonical (double-gamma) HRF.

    Implements a lightweight approximation of the SPM canonical HRF
    using two scaled gamma functions. For production use, prefer the
    full implementation in :mod:`fmrimod.hrf`.

    Attributes
    ----------
    name : str
        Always ``"spm_canonical"``.
    nbasis : int
        Always 1.

    Notes
    -----
    The HRF is defined as:

    .. math::

        h(t) = \\frac{(t/a_1)^{a_1 - 1} e^{-(t - a_1)}
              - c\\,(t/a_2)^{a_2 - 1} e^{-(t - a_2)}}{\\max|h|}

    with ``a1=6``, ``a2=16``, ``c=1/6``.
    """

    def __init__(self):
        self._name = "spm_canonical"
        self._nbasis = 1

    @property
    def name(self) -> str:
        """Name of the HRF."""
        return self._name

    @property
    def nbasis(self) -> int:
        """Number of basis functions."""
        return self._nbasis

    def evaluate(self, t: np.ndarray) -> np.ndarray:
        """Evaluate the SPM canonical HRF at given time points."""
        t = np.asarray(t)
        a1, a2 = 6.0, 16.0
        b1, b2 = 1.0, 1.0
        c = 1.0 / 6.0

        with np.errstate(divide='ignore', invalid='ignore'):
            gamma1 = np.where(t > 0, (t / a1) ** (a1 * b1 - 1) * np.exp(-(t - a1) * b1), 0)
            gamma2 = np.where(t > 0, (t / a2) ** (a2 * b2 - 1) * np.exp(-(t - a2) * b2), 0)

        hrf = gamma1 - c * gamma2
        max_val = np.max(np.abs(hrf))
        if max_val > 0:
            hrf = hrf / max_val
        return hrf


def get_hrf(name: str, **kwargs: object) -> HRFProtocol:
    """Get an HRF by name through the canonical registry.

    Parameters
    ----------
    name : str
        HRF name
    **kwargs
        Additional parameters passed to HRF constructor

    Returns
    -------
    HRFProtocol
        HRF function instance
    """
    from .hrf.registry import get_hrf as _get_hrf

    return cast(HRFProtocol, _get_hrf(name, **kwargs))


def register_hrf(name: str, hrf_class_or_factory: Union[type, callable]) -> None:
    """Register an HRF through the canonical registry.

    Parameters
    ----------
    name : str
        Name to register HRF under
    hrf_class_or_factory : type or callable
        HRF class (must implement HRFProtocol) or factory function
        that returns an HRF instance
    """
    from .hrf.registry import register_hrf as _register_hrf

    _register_hrf(name, cast(Any, hrf_class_or_factory), force=True)


def as_hrf(x: Any, **kwargs: Any) -> HRFProtocol:
    """Convert various objects to HRF instances.

    This is a runtime-dispatch entry point; ``x`` is intentionally
    ``Any`` because the function inspects the runtime type and routes
    to one of several typed factories (registry lookup for ``str``,
    :class:`ArrayHRF` for array-likes, :class:`DictHRF` for dicts
    carrying ``evaluate``, :class:`FunctionHRF` for callables). The
    HRFProtocol return type is the typed exit; callers should bind
    the result, not the input.

    Parameters
    ----------
    x : str, HRFProtocol, array-like, dict, or callable
        Object to convert to HRF
    **kwargs
        Additional arguments passed to HRF constructor

    Returns
    -------
    HRFProtocol
        HRF instance
    """
    if hasattr(x, 'evaluate') and hasattr(x, 'name') and hasattr(x, 'nbasis'):
        return x

    if isinstance(x, str):
        return get_hrf(x, **kwargs)

    if hasattr(x, '__array__') or isinstance(x, (list, tuple)):
        array = np.asarray(x)
        return ArrayHRF(array, **kwargs)

    if isinstance(x, dict) and 'evaluate' in x:
        return DictHRF(x, **kwargs)

    if isinstance(x, dict):
        raise ValueError(
            f"Dict HRF must contain an 'evaluate' key. "
            f"Got keys: {list(x.keys())}"
        )

    if callable(x):
        return FunctionHRF(x, **kwargs)

    raise TypeError(
        f"Cannot convert {type(x)} to HRF. "
        "Expected str, HRF, array, dict with 'evaluate', or callable."
    )


class ArrayHRF(HRFProtocol):
    """HRF constructed from a pre-sampled array of values."""

    def __init__(self, array: np.ndarray, sampling_rate: float = 1.0,
                 name: Optional[str] = None, **kwargs: object):
        self.array = np.asarray(array)
        self.sampling_rate = sampling_rate
        self._name = name or "array_hrf"
        self._nbasis = 1

    @property
    def name(self) -> str:
        return self._name

    @property
    def nbasis(self) -> int:
        return self._nbasis

    def evaluate(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t)
        t_array = np.arange(len(self.array)) / self.sampling_rate
        return np.interp(t, t_array, self.array, left=0, right=0)


class FunctionHRF(HRFProtocol):
    """HRF wrapping an arbitrary callable ``f(t) -> array``."""

    def __init__(self, func: Callable, name: Optional[str] = None, **kwargs: object):
        self.func = func
        self._name = name or getattr(func, '__name__', 'function_hrf')
        self._nbasis = kwargs.get('nbasis', 1)

    @property
    def name(self) -> str:
        return self._name

    @property
    def nbasis(self) -> int:
        return self._nbasis

    def evaluate(self, t: np.ndarray) -> np.ndarray:
        return self.func(t)

    def __call__(self, t: np.ndarray) -> np.ndarray:
        return self.evaluate(t)


class DictHRF(HRFProtocol):
    """HRF constructed from a dictionary specification."""

    def __init__(self, spec: Dict, **kwargs: object):
        if 'evaluate' not in spec:
            raise ValueError("Dictionary must contain 'evaluate' key")

        self._evaluate = spec['evaluate']
        self._name = spec.get('name', 'dict_hrf')
        self._nbasis = spec.get('nbasis', 1)

        for key, value in spec.items():
            if key not in ['evaluate', 'name', 'nbasis']:
                setattr(self, key, value)

    @property
    def name(self) -> str:
        return self._name

    @property
    def nbasis(self) -> int:
        return self._nbasis

    def evaluate(self, t: np.ndarray) -> np.ndarray:
        return self._evaluate(t)


# HRF Generator Factory Functions

def boxcar_hrf_gen(
    normalize: bool = True,
    min_duration: float = 0.1
) -> Callable:
    """Factory for duration-based boxcar HRF generation."""
    from .hrf.generators import boxcar_generator

    def generator(event_data):
        if hasattr(event_data, 'iloc'):
            durations = event_data['duration'].values
        elif isinstance(event_data, dict):
            durations = event_data['duration']
        else:
            raise TypeError(
                f"event_data must be DataFrame or dict, got {type(event_data)}"
            )

        # ``normalize=True`` on boxcar_generator was retired
        # (bd-01KRGCZ6QJME1JD8FD5D4PGC04); set amplitude=1/width explicitly,
        # which is what the retired flag did internally.
        hrfs = []
        for dur in durations:
            w = max(float(dur), min_duration)
            amplitude = 1.0 / w if normalize else 1.0
            hrfs.append(boxcar_generator(width=w, amplitude=amplitude))
        return hrfs

    return generator


def duration_hrf_gen(
    base: Optional[HRFProtocol] = None,
    min_duration: float = 0.0
) -> Callable:
    """Factory for duration-modulated HRF generation."""
    from .hrf.core import HRF
    from .hrf.decorators import block_hrf
    from .hrf.library import SPM_CANONICAL

    if base is None:
        base = SPM_CANONICAL
    base_hrf = cast(HRF, base)

    def generator(event_data):
        if hasattr(event_data, 'iloc'):
            durations = event_data['duration'].values
        elif isinstance(event_data, dict):
            durations = event_data['duration']
        else:
            raise TypeError(
                f"event_data must be DataFrame or dict, got {type(event_data)}"
            )

        # ``block_hrf(normalize=True)`` was retired
        # (bd-01KRGCZ6QJME1JD8FD5D4PGC04); compose the typed wrappers
        # explicitly: ``normalize(block_hrf(...), 'unit_peak_per_basis')``.
        from .hrf.normalization import normalize as _normalize

        hrfs = []
        for dur in durations:
            dur = max(float(dur), min_duration)
            if dur == 0:
                hrfs.append(base_hrf)
            else:
                hrfs.append(
                    _normalize(block_hrf(base_hrf, width=dur), "unit_peak_per_basis")
                )

        return hrfs

    return generator


def weighted_hrf_gen(
    times_col: str = "sub_times",
    weights_col: str = "sub_weights",
    relative: bool = False,
    method: str = "constant",
    normalize: bool = False
) -> Callable:
    """Factory for weighted HRF generation from list columns."""
    from .hrf.generators import weighted_generator
    if method not in ("constant", "linear"):
        raise ValueError("method must be 'constant' or 'linear'")
    method_lit = cast(Literal["constant", "linear"], method)

    def generator(event_data):
        if hasattr(event_data, 'iloc'):
            times_list = event_data[times_col].values
            weights_list = event_data[weights_col].values
            onsets = event_data['onset'].values
        elif isinstance(event_data, dict):
            times_list = event_data[times_col]
            weights_list = event_data[weights_col]
            onsets = event_data['onset']
        else:
            raise TypeError(
                f"event_data must be DataFrame or dict, got {type(event_data)}"
            )

        # ``weighted_generator(normalize=True)`` was retired
        # (bd-01KRGCZ6QJME1JD8FD5D4PGC04); scale weights explicitly here.
        hrfs = []
        for times, weights, onset in zip(times_list, weights_list, onsets):
            if relative:
                rel_times = np.asarray(times, dtype=np.float64)
            else:
                rel_times = np.array(times) - float(onset)

            w = np.asarray(weights, dtype=np.float64)
            if normalize:
                if method == "constant":
                    denom = float(np.sum(w[:-1]))
                    if abs(denom) > 1e-10:
                        w = w / denom
                else:  # linear
                    intervals = np.diff(rel_times)
                    avg_w = (w[:-1] + w[1:]) / 2.0
                    integral = float(np.sum(intervals * avg_w))
                    if abs(integral) > 1e-10:
                        w = w / integral

            hrfs.append(
                weighted_generator(
                    times=rel_times.tolist(),
                    weights=w.tolist(),
                    method=method_lit,
                )
            )

        return hrfs

    return generator
