"""Concrete FmriDataset implementation.

Wraps adapter objects together with an event table and optional
metadata to provide a complete dataset for GLM analysis.
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..sampling import SamplingFrame
from .protocols import DatasetProtocol


class FmriDataset:
    """fMRI dataset combining time-series data with experimental events.

    This is the primary concrete implementation of :class:`DatasetProtocol`.
    It wraps any adapter that satisfies the protocol together with an event
    table describing the experimental design.

    Parameters
    ----------
    data_source : DatasetProtocol
        Object providing ``get_data(run)``, ``get_mask()``, etc.
        Typically a :class:`~fmrimod.dataset.adapters.NumpyAdapter`.
    event_table : pd.DataFrame, optional
        Table of experimental events with at least ``onset`` and ``block``
        (or ``run``) columns.  Additional columns are event variables.
    censor : NDArray[np.bool\_] or list of NDArray, optional
        Boolean censoring vector(s).  ``True`` marks volumes to exclude.
        Can be a single array covering all runs concatenated, or a list
        of per-run arrays.
    """

    def __init__(
        self,
        data_source: DatasetProtocol,
        event_table: Optional[pd.DataFrame] = None,
        censor: Optional[Union[NDArray[np.bool_], List[NDArray[np.bool_]]]] = None,
    ):
        self._source = data_source
        self._event_table = event_table
        self._censor = self._validate_censor(censor)

    # -- DatasetProtocol delegation --

    def get_data(self, run: int) -> NDArray[np.float64]:
        """Return ``(time, voxels)`` data matrix for *run*."""
        return self._source.get_data(run)

    def get_mask(self) -> NDArray[np.bool_]:
        """Return the spatial mask."""
        return self._source.get_mask()

    def get_sampling_frame(self) -> SamplingFrame:
        """Return the :class:`SamplingFrame`."""
        return self._source.get_sampling_frame()

    @property
    def sampling_frame(self) -> SamplingFrame:
        """Convenience alias for ``get_sampling_frame()``."""
        return self._source.get_sampling_frame()

    @property
    def n_runs(self) -> int:
        return self._source.n_runs

    @property
    def n_timepoints(self) -> List[int]:
        return self._source.n_timepoints

    @property
    def n_voxels(self) -> int:
        return self._source.n_voxels

    # -- Event table --

    @property
    def event_table(self) -> Optional[pd.DataFrame]:
        """The event table, or *None* if not set."""
        return self._event_table

    # -- Censoring --

    @property
    def censor(self) -> Optional[List[NDArray[np.bool_]]]:
        """Per-run boolean censor vectors, or *None*."""
        return self._censor

    def get_censor(self, run: int) -> Optional[NDArray[np.bool_]]:
        """Return the censor vector for a single run, or *None*."""
        if self._censor is None:
            return None
        return self._censor[run]

    # -- Convenience --

    def get_all_data(self) -> NDArray[np.float64]:
        """Return all runs vertically concatenated as ``(total_time, voxels)``."""
        return np.vstack([self.get_data(r) for r in range(self.n_runs)])

    def __repr__(self) -> str:
        return (
            f"FmriDataset(n_runs={self.n_runs}, "
            f"n_timepoints={self.n_timepoints}, "
            f"n_voxels={self.n_voxels})"
        )

    # -- Internal --

    def _validate_censor(
        self, censor: Optional[Union[NDArray[np.bool_], List[NDArray[np.bool_]]]]
    ) -> Optional[List[NDArray[np.bool_]]]:
        if censor is None:
            return None

        n_tp = self.n_timepoints

        if isinstance(censor, np.ndarray) and censor.ndim == 1:
            # Split a single concatenated vector into per-run vectors
            total = sum(n_tp)
            if len(censor) != total:
                raise ValueError(
                    f"Censor vector length ({len(censor)}) != total timepoints ({total})"
                )
            splits = np.cumsum(n_tp[:-1])
            return [c.astype(bool) for c in np.split(censor, splits)]

        # Already a list
        censor_list = list(censor)
        if len(censor_list) != self.n_runs:
            raise ValueError(
                f"Number of censor arrays ({len(censor_list)}) != n_runs ({self.n_runs})"
            )
        for i, (c, nt) in enumerate(zip(censor_list, n_tp)):
            c = np.asarray(c, dtype=bool)
            if len(c) != nt:
                raise ValueError(
                    f"Run {i}: censor length ({len(c)}) != n_timepoints ({nt})"
                )
            censor_list[i] = c
        return censor_list
