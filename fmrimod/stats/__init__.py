"""Statistical inference utilities."""

from __future__ import annotations

import sys
import types
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fmrimod.accessors import EstimateOrContrastMap

from .backends import available_second_level_backends, fmrigds_backend_available
from .inference import (
    fdr_correction,
    group_fit,
    p_to_z,
    r_to_z,
    t_to_d,
    z_to_p,
    z_to_r,
)
from .interfaces import GroupFitRequest, GroupFitResult
from .meta import FmriMetaResult, fmri_meta
from .meta_compat import (
    fmri_meta_fit,
    fmri_meta_fit_contrasts,
    fmri_meta_fit_cov,
    fmri_meta_fit_extended,
    meta_effective_n,
)
from .spatial_fdr import SpatialFdrResult, spatial_fdr
from .ttest import FmriTTestResult, fmri_ttest

__all__ = [
    "fdr_correction",
    "p_to_z",
    "z_to_p",
    "t_to_d",
    "r_to_z",
    "z_to_r",
    "group_fit",
    "GroupFitRequest",
    "GroupFitResult",
    "available_second_level_backends",
    "fmrigds_backend_available",
    "fmri_meta",
    "FmriMetaResult",
    "fmri_meta_fit",
    "fmri_meta_fit_contrasts",
    "fmri_meta_fit_cov",
    "fmri_meta_fit_extended",
    "meta_effective_n",
    "fmri_ttest",
    "FmriTTestResult",
    "SpatialFdrResult",
    "spatial_fdr",
]


class _CallableStatsModule(types.ModuleType):
    def __call__(
        self,
        x: object,
        type: str = "estimates",
        **kwargs: object,
    ) -> EstimateOrContrastMap:
        from fmrimod.accessors import stats as _accessor_stats

        return _accessor_stats(x, type=type, **kwargs)


sys.modules[__name__].__class__ = _CallableStatsModule
