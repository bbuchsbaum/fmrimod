"""Caching utilities for HRF evaluation."""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple
import numpy as np
from numpy.typing import NDArray


# Simple LRU cache for HRF evaluations
# TODO: Consider implementing bounded cache using cachetools or custom LRU cache
# to prevent unbounded memory growth in long sessions
_hrf_cache = {}
_cache_size = 0
_max_cache_size = 128


def cached_hrf_eval(hrf, span: float, dt: float) -> NDArray[np.float64]:
    """Evaluate HRF with caching to avoid recomputation.
    
    Args:
        hrf: HRF object to evaluate
        span: Temporal span in seconds
        dt: Time step for evaluation
        
    Returns:
        Array of HRF values evaluated from 0 to span with step dt
    """
    # Create cache key from HRF parameters and evaluation settings
    # We use HRF name and params as they define the HRF uniquely
    cache_key = (hrf.name, tuple(sorted(hrf.params.items())), span, dt)
    
    # Check cache
    if cache_key in _hrf_cache:
        return _hrf_cache[cache_key].copy()
    
    # Evaluate HRF
    # Match R seq(0, span, by=dt): include points <= span only.
    n_times = int(np.floor(span / dt)) + 1
    times = np.arange(n_times, dtype=np.float64) * dt
    values = hrf(times)
    
    # Ensure 2D
    if values.ndim == 1:
        values = values[:, np.newaxis]
    
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


def clear_hrf_cache():
    """Clear the HRF evaluation cache."""
    global _cache_size
    _hrf_cache.clear()
    _cache_size = 0
