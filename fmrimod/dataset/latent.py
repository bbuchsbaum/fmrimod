"""Latent-component dataset typed primitive.

Houses :class:`LatentDataset` and the :func:`latent_dataset` constructor.
Promoted from ``fmrimod/dataset/compat.py`` per the compat retirement
inventory follow-up — these are typed Python values, not R-name shims,
and the compat housing was a misnomer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from ..sampling import SamplingFrame


@dataclass
class LatentDataset:
    """Small Python representation of a latent-component dataset."""

    scores: NDArray[np.float64]
    loadings: Optional[NDArray[np.float64]]
    sampling_frame: SamplingFrame
    event_table: Optional[pd.DataFrame] = None

    def __post_init__(self) -> None:
        self.scores = np.asarray(self.scores, dtype=np.float64)
        if self.scores.ndim != 2:
            raise ValueError("scores must be a 2-D matrix")
        if self.loadings is not None:
            self.loadings = np.asarray(self.loadings, dtype=np.float64)
            if self.loadings.ndim != 2:
                raise ValueError("loadings must be a 2-D matrix")
            if self.loadings.shape[0] != self.scores.shape[1]:
                raise ValueError("loadings rows must match score columns")
        if sum(self.sampling_frame.blocklens) != self.scores.shape[0]:
            raise ValueError("sampling frame length must match scores rows")

    @property
    def n_runs(self) -> int:
        return len(self.sampling_frame.blocklens)

    @property
    def n_timepoints(self) -> list[int]:
        return [int(v) for v in self.sampling_frame.blocklens]

    @property
    def n_voxels(self) -> int:
        if self.loadings is None:
            return int(self.scores.shape[1])
        return int(self.loadings.shape[1])

    def get_scores(self, run: int = 0) -> NDArray[np.float64]:
        start = int(sum(self.sampling_frame.blocklens[:run]))
        end = start + int(self.sampling_frame.blocklens[run])
        return self.scores[start:end]

    def get_data(self, run: int = 0) -> NDArray[np.float64]:
        scores = self.get_scores(run)
        if self.loadings is None:
            return scores
        return scores @ self.loadings

    def get_mask(self) -> NDArray[np.bool_]:
        return np.ones((self.n_voxels, 1, 1), dtype=bool)

    def get_sampling_frame(self) -> SamplingFrame:
        return self.sampling_frame

    def get_censor(self, run: int = 0) -> None:
        return None


def latent_dataset(
    scores: ArrayLike,
    loadings: Optional[ArrayLike] = None,
    tr: float = 1.0,
    run_length: Optional[Union[int, Sequence[int]]] = None,
    *,
    event_table: Optional[pd.DataFrame] = None,
) -> LatentDataset:
    """Construct a latent-component dataset from scores and optional loadings."""
    scores_arr = np.asarray(scores, dtype=np.float64)
    if scores_arr.ndim != 2:
        raise ValueError("scores must be a 2-D matrix")
    if run_length is None:
        blocklens = [scores_arr.shape[0]]
    elif isinstance(run_length, int):
        if scores_arr.shape[0] % run_length:
            raise ValueError("run_length must divide score rows")
        blocklens = [int(run_length)] * (scores_arr.shape[0] // int(run_length))
    else:
        blocklens = [int(v) for v in run_length]
        if sum(blocklens) != scores_arr.shape[0]:
            raise ValueError("sum(run_length) must equal score rows")
    sf = SamplingFrame(blocklens=blocklens, tr=tr)
    return LatentDataset(
        scores_arr,
        None if loadings is None else np.asarray(loadings),
        sf,
        event_table,
    )


__all__ = ["LatentDataset", "latent_dataset"]
