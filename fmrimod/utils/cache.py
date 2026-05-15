"""Caching utilities for HRF evaluation."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any, Tuple

import numpy as np
from numpy.typing import NDArray

# Simple LRU cache for HRF evaluations
# TODO: Consider implementing bounded cache using cachetools or custom LRU cache
# to prevent unbounded memory growth in long sessions
_hrf_cache: dict[Any, NDArray[np.float64]] = {}
_cache_size = 0
_max_cache_size = 128


def _to_cache_value(value: object) -> Any:
    """Convert arbitrary values into hashable, cache-safe representations."""
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        array_bytes = array.tobytes()
        digest = hashlib.sha256(array_bytes).hexdigest()
        return ("ndarray", str(array.shape), str(array.dtype), digest)

    if isinstance(value, Mapping):
        items = []
        for key in sorted(value.keys(), key=str):
            key_repr = str(key)
            items.append((key_repr, _to_cache_value(value[key])))
        return ("mapping", tuple(items))

    if isinstance(value, (tuple, list)):
        return ("sequence", tuple(_to_cache_value(item) for item in value))

    if isinstance(value, np.number):
        return ("np_scalar", str(value))

    if callable(value):
        return ("callable", repr(value))

    if isinstance(value, (str, int, float, bool, type(None))):
        return ("scalar", value)

    return ("object", repr(value))


def _extract_hrf_cache_signature(hrf: Any) -> Tuple[Any, ...]:
    """Build a cache key fragment that works across HRF protocol implementations."""
    if isinstance(getattr(hrf, "params", None), Mapping):
        return (
            "params",
            tuple(
                (
                    str(key),
                    _to_cache_value(value),
                )
                for key, value in sorted(
                    getattr(hrf, "params").items(),
                    key=lambda item: str(item[0]),
                )
            ),
        )

    fragments = []
    for attr in ("sampling_rate", "nbasis", "name"):
        if hasattr(hrf, attr):
            fragments.append((attr, _to_cache_value(getattr(hrf, attr))))

    if hasattr(hrf, "array"):
        fragments.append(("array", _to_cache_value(getattr(hrf, "array"))))

    if hasattr(hrf, "func"):
        fragments.append(("func", _to_cache_value(getattr(hrf, "func"))))

    if not fragments:
        fragments = [("repr", repr(hrf))]

    return ("protocol", tuple(fragments))


def cached_hrf_eval(hrf: Any, span: float, dt: float) -> NDArray[np.float64]:
    """Evaluate HRF with caching to avoid recomputation.
    
    Args:
        hrf: HRF object to evaluate
        span: Temporal span in seconds
        dt: Time step for evaluation
        
    Returns:
        Array of HRF values evaluated from 0 to span with step dt
    """
    try:
        span = float(span)
        dt = float(dt)
    except (TypeError, ValueError) as exc:
        raise ValueError("span and dt must be scalar numeric values") from exc

    if not np.isfinite(span) or span < 0:
        raise ValueError("span must be a finite non-negative number")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be a finite positive number")

    # Create cache key from HRF parameters and evaluation settings
    # We use HRF name and params as they define the HRF uniquely
    cache_key = (hrf.name, _extract_hrf_cache_signature(hrf), span, dt)
    
    # Check cache
    if cache_key in _hrf_cache:
        return _hrf_cache[cache_key].copy()
    
    # Evaluate HRF
    # Match R seq(0, span, by=dt): include points <= span only.
    n_times = int(np.floor(span / dt)) + 1
    times = np.arange(n_times, dtype=np.float64) * dt
    if callable(hrf):
        values = hrf(times)
    else:
        values = hrf.evaluate(times)

    values = np.asarray(values, dtype=np.float64)

    # Ensure 2D
    if values.ndim == 0:
        values = np.full((times.size, 1), float(values), dtype=np.float64)
    elif values.ndim == 1:
        if values.shape[0] != times.size:
            raise ValueError(
                "HRF evaluation must return one value per sampled time point"
            )
        values = values[:, np.newaxis]
    elif values.ndim == 2:
        if values.shape[0] != times.size and values.shape[1] == times.size:
            values = values.T
        if values.shape[0] != times.size:
            raise ValueError(
                "HRF evaluation must return shape (n_times,) or (n_times, nbasis)"
            )
    else:
        raise ValueError(
            "HRF evaluation must return scalar, 1D, or 2D array-like output"
        )
    
    # Add to cache with simple size management
    global _cache_size
    if _cache_size >= _max_cache_size:
        # Simple FIFO eviction - remove first added item
        first_key = next(iter(_hrf_cache))
        del _hrf_cache[first_key]
        _cache_size -= 1
    
    _hrf_cache[cache_key] = values.copy()
    _cache_size += 1
    
    return values


def clear_hrf_cache() -> None:
    """Clear the HRF evaluation cache."""
    global _cache_size
    _hrf_cache.clear()
    _cache_size = 0
