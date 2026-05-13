"""Pool per-run GLM fits into a single combined result.

The canonical user-facing entry point is :func:`combine_runs`::

    fits = [fm.fmri_lm(spec, ds_run) for ds_run in runs]
    combined = fm.combine_runs(fits)
    result   = combined.contrast(c)

For combining already-computed :class:`ContrastResult` objects, use
:func:`combine_contrasts`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import special as sp_special

from .contrasts import ContrastResult
from .fmri_lm import FmriLm

_METHODS = ("fixed", "ivw")


@dataclass
class CombinedFmriLm:
    """Combined view over per-run :class:`FmriLm` fits.

    Use ``.contrast(spec)`` to evaluate a contrast on every per-run fit
    and pool the results by the configured method.

    Attributes
    ----------
    fits : tuple of FmriLm
        Per-run fits being combined.
    method : str
        Pooling method (currently only ``"fixed"``).
    """

    fits: tuple[FmriLm, ...]
    method: str = "fixed"
    contrasts: dict[str, ContrastResult] = field(default_factory=dict)

    @property
    def n_runs(self) -> int:
        return len(self.fits)

    @property
    def n_voxels(self) -> int:
        return int(self.fits[0].n_voxels)

    def contrast(
        self,
        spec: NDArray[np.float64] | str | Mapping[str, Any],
        name: str | None = None,
    ) -> ContrastResult:
        """Compute a contrast across all runs and pool by ``self.method``."""
        per_run = [f.contrast(spec, name=name) for f in self.fits]
        pooled = combine_contrasts(per_run, method=self.method, name=name)
        # Propagate spatial context (combine_runs enforced matching n_voxels,
        # so per-run masks coincide).
        if pooled.spatial is None:
            for r in per_run:
                if r.spatial is not None:
                    pooled.spatial = r.spatial
                    break
        self.contrasts[pooled.name] = pooled
        return pooled

    def compute_contrasts(
        self,
        specs: dict[str, NDArray[np.float64]],
    ) -> dict[str, ContrastResult]:
        """Evaluate and pool a batch of named contrasts."""
        return {name: self.contrast(w, name=name) for name, w in specs.items()}

    def __repr__(self) -> str:
        return (
            f"CombinedFmriLm(n_runs={self.n_runs}, n_voxels={self.n_voxels}, "
            f"method={self.method!r}, contrasts={list(self.contrasts)})"
        )


def combine_runs(
    fits: Sequence[FmriLm],
    *,
    method: str = "fixed",
) -> CombinedFmriLm:
    """Pool per-run :class:`FmriLm` fits into a combined result.

    Parameters
    ----------
    fits : sequence of FmriLm
        Per-run fitted models. Must share the same voxel count.
    method : {"fixed", "ivw"}
        Pooling method, applied at the contrast level. ``"fixed"`` (default)
        matches Nilearn's ``compute_fixed_effect_contrast``: equal-weight
        averaging of per-run effects with variance ``mean(var) / n_runs``.
        ``"ivw"`` does inverse-variance-weighted pooling - the maximum-
        likelihood estimator under independent per-run noise. The two
        coincide when per-run residual variances are equal across runs.

    Returns
    -------
    CombinedFmriLm
        A combined fit exposing ``.contrast(spec)`` for pooled contrasts.

    Examples
    --------
    >>> fits = [fm.fmri_lm(spec, ds) for ds in run_datasets]  # doctest: +SKIP
    >>> combined = fm.combine_runs(fits)                       # doctest: +SKIP
    >>> result = combined.contrast(np.array([0.5, 0.5, -0.5, -0.5]))  # doctest: +SKIP
    """
    fits = tuple(fits)
    if not fits:
        raise ValueError("combine_runs: at least one fit required")

    if method not in _METHODS:
        raise ValueError(
            f"combine_runs: unknown method {method!r}. "
            f"Supported: {sorted(_METHODS)!r}"
        )

    n_voxels = fits[0].n_voxels
    for i, f in enumerate(fits[1:], start=1):
        if f.n_voxels != n_voxels:
            raise ValueError(
                f"combine_runs: fits[0] has {n_voxels} voxels but "
                f"fits[{i}] has {f.n_voxels} voxels"
            )

    return CombinedFmriLm(fits=fits, method=method)


def combine_contrasts(
    results: Sequence[ContrastResult],
    *,
    method: str = "fixed",
    name: str | None = None,
) -> ContrastResult:
    """Pool per-run :class:`ContrastResult` objects into one combined result.

    For ``method="fixed"`` (default, Nilearn-compatible)::

        eff_pool = mean(eff_i)
        var_pool = mean(var_i) / n_runs
        t_pool   = eff_pool / sqrt(var_pool)
        df_pool  = sum(df_i)

    For ``method="ivw"`` (inverse-variance-weighted, MLE under independent
    Gaussian per-run noise)::

        eff_pool = sum(eff_i / var_i) / sum(1 / var_i)
        var_pool = 1 / sum(1 / var_i)

    The two coincide when per-run variances are equal.

    Parameters
    ----------
    results : sequence of ContrastResult
        Per-run contrast outputs to pool. All must be t-statistics and have
        matching voxel counts.
    method : {"fixed", "ivw"}
        Pooling method.
    name : str, optional
        Output contrast name; defaults to the first result's name.

    Returns
    -------
    ContrastResult
        Pooled contrast with combined estimate, SE, t, p, and df.
    """
    if not results:
        raise ValueError("combine_contrasts: at least one result required")
    if method not in _METHODS:
        raise ValueError(
            f"combine_contrasts: unknown method {method!r}. "
            f"Supported: {sorted(_METHODS)!r}"
        )

    if any(r.stat_type != "t" for r in results):
        raise NotImplementedError(
            "combine_contrasts currently supports t-contrasts only"
        )

    se_arrays = [r.se for r in results]
    if any(se is None for se in se_arrays):
        raise ValueError(
            "combine_contrasts: every input must have a populated `se` "
            "(t-contrast results from FmriLm.contrast satisfy this)"
        )

    if any(r.estimate.shape != results[0].estimate.shape for r in results):
        raise ValueError("combine_contrasts: per-run estimate shapes must match")

    estimate_stack = np.vstack([np.asarray(r.estimate, dtype=np.float64) for r in results])
    se_stack = np.vstack([np.asarray(se, dtype=np.float64) for se in se_arrays])
    var_stack = se_stack * se_stack
    n_runs = estimate_stack.shape[0]

    if method == "fixed":
        # Nilearn-compatible: equal-weight average of effects, with the
        # variance of that average. Equivalent to summing Nilearn Contrast
        # objects via ``__add__`` and then multiplying by ``1 / n_runs``.
        effect_pool = estimate_stack.mean(axis=0)
        var_pool = var_stack.sum(axis=0) / (n_runs * n_runs)
    else:  # method == "ivw"
        var_floor = np.maximum(var_stack, np.finfo(np.float64).tiny)
        inv_var = 1.0 / var_floor
        sum_inv_var = inv_var.sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            effect_pool = np.where(
                sum_inv_var > 0,
                (estimate_stack * inv_var).sum(axis=0) / sum_inv_var,
                0.0,
            )
            var_pool = np.where(sum_inv_var > 0, 1.0 / sum_inv_var, np.inf)

    se_pool = np.sqrt(np.maximum(var_pool, 0.0))

    df_pool = float(sum(float(r.df) for r in results))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_pool = np.where(se_pool > 1e-15, effect_pool / se_pool, 0.0)
    p_pool = 2.0 * sp_special.stdtr(df_pool, -np.abs(t_pool))

    return ContrastResult(
        name=name or results[0].name,
        estimate=effect_pool,
        stat=t_pool,
        se=se_pool,
        p_value=p_pool,
        df=df_pool,
        stat_type="t",
    )
