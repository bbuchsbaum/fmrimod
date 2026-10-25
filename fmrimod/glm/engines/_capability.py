"""Capability checks shared by the plain-OLS fitting engines.

An engine that cannot honour a requested configuration must refuse it in
``preflight`` instead of fitting silently without it or crashing later in a
post-fit stage (fmrireg caab78d / 60f8430 / 505a32c).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fmrimod.glm.errors import UnsupportedEngineConfiguration
from fmrimod.model.config import FmriLmConfig

_REPAIR = 'use the default runwise engine (engine="runwise")'


def _has_censored_frames(model: Any) -> bool:
    dataset = getattr(model, "dataset", None)
    get_censor = getattr(dataset, "get_censor", None)
    if not callable(get_censor):
        return False
    for run in range(int(getattr(model, "n_runs", 1))):
        censor = get_censor(run)
        if censor is not None and np.any(np.asarray(censor).astype(bool)):
            return True
    return False


def require_unweighted_ols(
    engine: str,
    model: Any,
    config: FmriLmConfig,
    *,
    current_capability: tuple[str, ...],
) -> None:
    """Raise :class:`UnsupportedEngineConfiguration` for anything but plain OLS.

    Checks every :class:`FmriLmConfig` section that changes the estimator
    (AR, robust, volume weights, soft-subspace projection) and frames the
    dataset censors.
    """
    unsupported = (
        (config.ar.enabled, "AR modeling", 'set ar.struct="iid"'),
        (config.robust.enabled, "robust fitting", "set robust.type=False"),
        (
            config.volume_weights.enabled,
            "volume weights",
            "disable volume_weights",
        ),
        (
            config.soft_subspace.enabled,
            "soft-subspace projection",
            "disable soft_subspace",
        ),
    )
    for enabled, feature, local_repair in unsupported:
        if enabled:
            raise UnsupportedEngineConfiguration(
                engine=engine,
                feature=feature,
                repair=f"{_REPAIR} or {local_repair}",
                current_capability=current_capability,
            )
    if _has_censored_frames(model):
        raise UnsupportedEngineConfiguration(
            engine=engine,
            feature="frame censoring (the dataset censors frames)",
            repair=f"{_REPAIR} or drop the censored frames before fitting",
            current_capability=current_capability,
        )
