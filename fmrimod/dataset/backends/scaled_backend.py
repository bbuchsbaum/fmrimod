"""Read-time signal scaling over another storage backend.

:class:`ScaledBackend` rescales every read to percent signal change of each
voxel's run mean, ``100 * (y / mean_run - 1)``. This is Nilearn's
``FirstLevelModel(signal_scaling=0)``: betas then read as percent signal
change, comparable across subjects and scanners. Voxels whose run mean is 0
are left divided by 1 (as Nilearn does).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..backend_protocol import BackendDims, StorageBackend
from ..space import VolumeSpace


class ScaledBackend(StorageBackend):
    """Percent-signal-change view of ``base``; run boundaries from ``run_starts``."""

    def __init__(self, base: StorageBackend, run_starts: NDArray[np.intp]) -> None:
        self._base = base
        self._run_starts = np.asarray(run_starts, dtype=np.intp)
        self._means: NDArray[np.float64] | None = None  # (n_runs, n_voxels)

    # -- delegation --

    def open(self) -> None:
        self._base.open()

    def close(self) -> None:
        self._base.close()

    def get_dims(self) -> BackendDims:
        return self._base.get_dims()

    def get_mask(self) -> NDArray[np.bool_]:
        return self._base.get_mask()

    def get_metadata(self) -> dict[str, object]:
        meta = dict(self._base.get_metadata())
        meta["signal_scaling"] = "psc"
        return meta

    @property
    def space(self) -> VolumeSpace:
        return self._base.space

    @property
    def source_run_lengths(self) -> tuple[int, ...] | None:
        return self._base.source_run_lengths

    # -- scaling --

    def _run_means(self) -> NDArray[np.float64]:
        if self._means is None:
            means = []
            for start, stop in zip(
                self._run_starts[:-1], self._run_starts[1:], strict=True
            ):
                rows = np.arange(start, stop, dtype=np.intp)
                run = np.asarray(self._base.get_data(rows=rows), dtype=np.float64)
                mean = run.mean(axis=0)
                means.append(np.where(mean == 0.0, 1.0, mean))
            self._means = np.vstack(means)
        return self._means

    def get_data(
        self,
        rows: NDArray[np.intp] | None = None,
        cols: NDArray[np.intp] | None = None,
    ) -> NDArray[np.float64]:
        data = np.asarray(self._base.get_data(rows=rows, cols=cols), dtype=np.float64)
        n_total = int(self._run_starts[-1])
        row_idx = np.arange(n_total) if rows is None else np.asarray(rows, dtype=np.intp)
        run_of_row = np.searchsorted(self._run_starts, row_idx, side="right") - 1
        means = self._run_means()
        if cols is not None:
            means = means[:, np.asarray(cols, dtype=np.intp)]
        scaled = data / means[run_of_row]
        scaled -= 1.0
        scaled *= 100.0
        return scaled
