"""HRF decorator functions for modifying HRF behavior.

The decorator factories (``lag_hrf`` and ``block_hrf``) return *typed*
HRF subclasses rather than opaque ``FunctionHRF`` wrappers, so
downstream code can introspect the decoration chain via the ``base``
attribute and pattern-match on subclass identity. HRF normalization is
centralized in :func:`fmrimod.hrf.normalization.normalize`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .core import HRF, FunctionHRF, as_hrf, bind_basis


def _block_offsets_weights(width: float, precision: float):
    """Compute trapezoidal quadrature offsets and weights for block evaluation.

    Matches the R helper ``.block_offsets_weights(width, precision)``.

    Args:
        width: Block duration in seconds.
        precision: Sampling step size in seconds.

    Returns:
        Tuple of (offsets, weights) as 1-D numpy arrays.
    """
    offsets = np.arange(0, width, precision)
    # Ensure the endpoint is included
    if len(offsets) == 0 or offsets[-1] < width:
        offsets = np.append(offsets, width)

    if len(offsets) == 1:
        return offsets, np.array([1.0])

    deltas = np.diff(offsets)
    weights = np.zeros(len(offsets))
    weights[0] = deltas[0] / 2.0
    weights[-1] = deltas[-1] / 2.0
    if len(offsets) > 2:
        weights[1:-1] = (deltas[:-1] + deltas[1:]) / 2.0

    return offsets, weights


@dataclass
class LaggedHRF(HRF):
    """HRF shifted in time by a fixed ``lag``.

    ``base`` is preserved so callers can introspect the chain. The
    inherited ``params`` dict is mirrored with ``_lag`` for cross-testing
    readers during the transition window; bead
    ``bd-01KRGCZJ6JAA4BKRTNQ91P2PE5`` retires the mirror.
    """

    name: str = ""
    nbasis: int = 1
    span: float = 24.0
    base: Optional[HRF] = None
    lag: float = 0.0

    def __post_init__(self) -> None:
        if self.base is None:
            raise ValueError("LaggedHRF requires `base`")
        if not np.isfinite(self.lag):
            raise ValueError("lag must be finite")
        self.name = f"{self.base.name}_lag({self.lag})"
        self.nbasis = self.base.nbasis
        self.span = self.base.span + max(0.0, self.lag)
        new_params = dict(self.base.params)
        new_params["_lag"] = self.lag
        self.params = new_params
        self.param_names = self.base.param_names

    def __call__(self, t: ArrayLike) -> NDArray[np.float64]:
        assert self.base is not None  # for type checkers; __post_init__ guarantees
        return self.base(np.asarray(t) - self.lag)


@dataclass
class BlockedHRF(HRF):
    """HRF convolved with a boxcar of duration ``width`` via trapezoidal quadrature.

    Optionally applies an exponential half-life decay across the block.
    Normalization is handled by ``fmrimod.hrf.normalization.normalize``.
    """

    name: str = ""
    nbasis: int = 1
    span: float = 24.0
    base: Optional[HRF] = None
    width: float = 0.0
    precision: float = 0.1
    half_life: float = float("inf")
    summate: bool = True

    def __post_init__(self) -> None:
        if self.base is None:
            raise ValueError("BlockedHRF requires `base`")
        if not np.isfinite(self.width):
            raise ValueError("width must be finite")
        if not np.isfinite(self.precision) or self.precision <= 0:
            raise ValueError("precision must be finite and positive")
        if self.half_life <= 0:
            raise ValueError("half_life must be positive")
        self.name = f"{self.base.name}_block(w={self.width})"
        self.nbasis = self.base.nbasis
        self.span = self.base.span + self.width
        new_params = dict(self.base.params)
        new_params.update({
            "_width": self.width,
            "_precision": self.precision,
            "_half_life": self.half_life,
            "_summate": self.summate,
        })
        self.params = new_params
        self.param_names = self.base.param_names

    def __call__(self, t: ArrayLike) -> NDArray[np.float64]:
        assert self.base is not None
        t = np.asarray(t)
        quad_offsets, quad_weights = _block_offsets_weights(self.width, self.precision)

        # Evaluate the base HRF at each quadrature offset, applying per-offset
        # decay when a finite half-life is configured.
        if not np.isfinite(self.half_life):
            decay = np.ones_like(quad_offsets)
        else:
            decay = np.exp(-np.log(2) * quad_offsets / self.half_life)

        hmat_list = [self.base(t - offset) * d for offset, d in zip(quad_offsets, decay)]

        if self.base.nbasis == 1:
            hmat = np.column_stack(hmat_list)  # (len(t), n_offsets)
            result = hmat @ quad_weights
            if not self.summate:
                weight_sum = float(np.sum(quad_weights))
                if weight_sum > 0:
                    result = result / weight_sum
        else:
            weighted = [vals * wt for vals, wt in zip(hmat_list, quad_weights)]
            result = weighted[0].copy()
            for w in weighted[1:]:
                result += w
            if not self.summate:
                weight_sum = float(np.sum(quad_weights))
                if weight_sum > 0:
                    result = result / weight_sum

        return result


def lag_hrf(hrf: HRF, lag: float) -> HRF:
    """Apply temporal lag to an HRF.

    Returns a :class:`LaggedHRF` whose ``base`` is the original HRF.

    Args:
        hrf: The HRF to lag
        lag: Time lag in seconds (must be finite)

    Returns:
        Typed lagged HRF.

    Raises:
        ValueError: If ``lag`` is not finite.
    """
    return LaggedHRF(base=hrf, lag=lag)


def block_hrf(
    hrf: HRF,
    width: float,
    precision: float = 0.1,
    half_life: float = float("inf"),
    summate: bool = True,
    normalize: bool = False,
) -> HRF:
    """Create a blocked/sustained version of an HRF.

    Returns a :class:`BlockedHRF` whose ``base`` is the original. For
    ``width <= precision`` the original HRF is returned unchanged.

    Args:
        hrf: The HRF to block.
        width: Duration of the block in seconds (must be finite).
        precision: Temporal precision for convolution (must be finite and positive).
        half_life: Half-life for exponential decay (default: no decay).
        summate: If True, responses accumulate; if False, averaged.
        normalize: If True, normalize result to unit peak per basis.

    Returns:
        Typed blocked HRF (or the original ``hrf`` when ``width <= precision``).
    """
    if normalize:
        raise ValueError(
            "block_hrf(normalize=True) is retired; "
            "use normalize(block_hrf(...), 'unit_peak_per_basis')."
        )
    if width <= precision:
        return hrf
    return BlockedHRF(
        base=hrf,
        width=width,
        precision=precision,
        half_life=half_life,
        summate=summate,
    )


def gen_hrf_lagged(
    hrf: Union[HRF, Callable],
    lags: ArrayLike,
    name: Optional[str] = None,
) -> HRF:
    """Generate a set of lagged HRFs.

    Creates a multi-basis HRF with each basis function being the original
    HRF shifted by different lag amounts.

    Args:
        hrf: Base HRF or function to lag.
        lags: Array of lag values in seconds.
        name: Optional name for the combined HRF.

    Returns:
        Multi-basis HRF with lagged versions (a :class:`BoundBasisHRF`
        when there are multiple lags, or a :class:`LaggedHRF` for one).
    """
    if not isinstance(hrf, HRF):
        hrf = as_hrf(hrf)

    lags = np.atleast_1d(np.asarray(lags, dtype=np.float64))

    lagged_hrfs = [lag_hrf(hrf, lag) for lag in lags]
    combined = bind_basis(*lagged_hrfs)

    if name is not None:
        # bind_basis returns the single HRF unchanged when len(lags) == 1, so
        # name overrides flow through unchanged regardless of count.
        combined = FunctionHRF(
            func=combined,
            name=name,
            nbasis=combined.nbasis,
            span=combined.span,
            params=combined.params,
        )

    return combined


def hrf_lagged(
    hrf: Union[HRF, Callable],
    lag: ArrayLike = 2.0,
    normalize: bool = False,
    name: Optional[str] = None,
) -> HRF:
    """R-compatible alias for creating lagged HRF bases."""
    lagged = gen_hrf_lagged(hrf, lag, name=name)
    if normalize:
        from .normalization import normalize as _normalize

        lagged = _normalize(lagged, "unit_peak_per_basis")
    return lagged


def gen_hrf_blocked(
    hrf: Union[HRF, Callable],
    widths: ArrayLike,
    precision: float = 0.1,
    half_life: float = float("inf"),
    summate: bool = True,
    normalize: bool = False,
    name: Optional[str] = None,
) -> HRF:
    """Generate a set of blocked HRFs with different widths.

    Args:
        hrf: Base HRF or function to block.
        widths: Array of block widths in seconds.
        precision: Temporal precision for convolution.
        half_life: Half-life for exponential decay.
        summate: If True, responses accumulate; if False, averaged.
        normalize: If True, normalize each basis.
        name: Optional name for the combined HRF.

    Returns:
        Multi-basis HRF with blocked versions.
    """
    if not isinstance(hrf, HRF):
        hrf = as_hrf(hrf)

    widths = np.atleast_1d(np.asarray(widths, dtype=np.float64))

    blocked_hrfs = []
    for width in widths:
        blocked = block_hrf(
            hrf,
            float(width),
            precision=precision,
            half_life=half_life,
            summate=summate,
        )
        if normalize:
            from .normalization import normalize as _normalize

            blocked = _normalize(blocked, "unit_peak_per_basis")
        blocked_hrfs.append(blocked)

    combined = bind_basis(*blocked_hrfs)

    if name is not None:
        combined = FunctionHRF(
            func=combined,
            name=name,
            nbasis=combined.nbasis,
            span=combined.span,
            params=combined.params,
        )

    return combined


def hrf_blocked(
    hrf: Optional[Union[HRF, Callable]] = None,
    width: ArrayLike = 5.0,
    precision: float = 0.1,
    half_life: float = float("inf"),
    summate: bool = True,
    normalize: bool = False,
    name: Optional[str] = None,
) -> HRF:
    """R-compatible alias for creating blocked HRF bases."""
    if hrf is None:
        from .library import GAUSSIAN_HRF
        hrf = GAUSSIAN_HRF
    return gen_hrf_blocked(
        hrf,
        width,
        precision=precision,
        half_life=half_life,
        summate=summate,
        normalize=normalize,
        name=name,
    )
