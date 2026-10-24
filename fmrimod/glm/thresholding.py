"""Height and cluster-extent thresholding of statistic maps.

:meth:`ContrastResult.threshold` and :meth:`GroupLmResult.threshold` return a
:class:`ThresholdedMap`: the z map with sub-threshold voxels set to zero, the
z cutoff that was applied, and helpers to write it (:meth:`~ThresholdedMap.to_nifti`)
or tabulate its clusters (:meth:`~ThresholdedMap.clusters`).

The rules follow Nilearn's ``threshold_stats_img`` so results can be compared
directly:

- ``method="fdr"``: Benjamini-Hochberg on the one-sided p-values of ``|z|``
  at ``alpha / 2`` when two-sided (equivalently, BH on two-sided p at
  ``alpha``); the cutoff is the smallest surviving ``|z|``.
- ``method="bonferroni"``: ``|z| > Phi^-1(1 - alpha / (2 n))`` when two-sided.
- ``method="none"``: ``|z| > height`` for a given z ``height``.
- ``cluster_size=k`` then removes clusters smaller than ``k`` voxels;
  clusters are face-connected (6-neighbour) and positive and negative
  clusters are labelled separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import ndimage
from scipy import special as sp_special

if TYPE_CHECKING:
    from .spatial import SpatialContext

ThresholdMethod = Literal["fdr", "bonferroni", "none"]


@dataclass(frozen=True)
class ThresholdedMap:
    """A z map after height (and optionally cluster-extent) thresholding.

    Attributes
    ----------
    z
        ``(n_voxels,)`` z values with sub-threshold voxels set to ``0``.
    threshold
        The ``|z|`` cutoff applied (``inf`` when nothing survives FDR).
    method, alpha, cluster_size, two_sided
        The settings that produced it.
    name
        Label of the source map (contrast or predictor).
    spatial
        Voxel geometry, when the source carried one (needed for clusters and
        NIfTI output).
    """

    z: NDArray[np.float64]
    threshold: float
    method: ThresholdMethod
    alpha: float
    cluster_size: int
    two_sided: bool
    name: str = "map"
    spatial: SpatialContext | None = None

    @property
    def n_suprathreshold(self) -> int:
        """Number of voxels that survive."""
        return int(np.count_nonzero(self.z))

    def to_img(self, *, fill: float = 0.0) -> Any:
        """The thresholded z map as an in-memory ``nibabel.Nifti1Image``."""
        return self._require_spatial().to_img(self.z, fill=fill)

    def __repr__(self) -> str:
        return (
            f"ThresholdedMap(name={self.name!r}, method={self.method!r}, "
            f"alpha={self.alpha}, threshold={self.threshold:.3f}, "
            f"cluster_size={self.cluster_size}, "
            f"n_suprathreshold={self.n_suprathreshold}/{self.z.size})"
        )

    def to_nifti(self, path: str | Path, *, fill: float = 0.0) -> Path:
        """Write the thresholded z map to NIfTI."""
        return self._require_spatial().write_nifti(
            self.z, path, label=f"{self.name}.z_thresholded", fill=fill
        )

    def clusters(self) -> pd.DataFrame:
        """One row per surviving cluster, largest ``|peak z|`` first.

        Columns: ``cluster`` (1-based id), ``sign`` (+1/-1), ``size``
        (voxels), ``volume_mm3``, ``peak_z``, ``x``, ``y``, ``z`` (peak
        position in world mm), and ``i``, ``j``, ``k`` (peak voxel indices).
        Unlike Nilearn's ``get_clusters_table`` there are no sub-peak rows.

        Examples
        --------
        ::

            hits = fit.contrast("face - house").threshold(alpha=0.05,
                                                          cluster_size=10)
            hits.clusters().head()
        """
        spatial = self._require_spatial()
        volume = spatial.reconstruct(self.z, fill=0.0)
        affine = spatial.affine if spatial.affine is not None else np.eye(4)
        voxel_mm3 = float(abs(np.linalg.det(affine[:3, :3])))
        rows: list[dict[str, float | int]] = []
        for sign in (1, -1) if self.two_sided else (1,):
            labels, n_labels = ndimage.label(sign * volume > 0)
            if n_labels == 0:
                continue
            index = np.arange(1, n_labels + 1)
            sizes = ndimage.sum_labels(np.ones_like(volume), labels, index)
            peaks = ndimage.maximum_position(sign * volume, labels, index)
            for size, peak in zip(sizes, peaks, strict=True):
                ijk = tuple(int(v) for v in peak)
                xyz = affine @ np.array([*ijk, 1.0])
                rows.append(
                    {
                        "sign": sign,
                        "size": int(size),
                        "volume_mm3": float(size) * voxel_mm3,
                        "peak_z": float(volume[ijk]),
                        "x": float(xyz[0]),
                        "y": float(xyz[1]),
                        "z": float(xyz[2]),
                        "i": ijk[0],
                        "j": ijk[1],
                        "k": ijk[2],
                    }
                )
        columns = [
            "cluster",
            "sign",
            "size",
            "volume_mm3",
            "peak_z",
            "x",
            "y",
            "z",
            "i",
            "j",
            "k",
        ]
        if not rows:
            return pd.DataFrame(columns=columns)
        table = pd.DataFrame(rows)
        table = table.reindex(table["peak_z"].abs().sort_values(ascending=False).index)
        table.insert(0, "cluster", np.arange(1, len(table) + 1))
        return table.reset_index(drop=True)[columns]

    def _require_spatial(self) -> SpatialContext:
        if self.spatial is None:
            raise ValueError(
                "this map has no voxel geometry; fit from images (or attach a "
                "SpatialContext) to cluster or write NIfTI"
            )
        return self.spatial


def height_threshold(
    z: NDArray[np.float64],
    *,
    alpha: float,
    method: ThresholdMethod,
    height: float | None,
    two_sided: bool,
) -> float:
    """The ``|z|`` (or ``z``) cutoff for ``method``; see the module docstring."""
    values = np.asarray(z, dtype=np.float64)
    values = values[np.isfinite(values)]
    stats = np.abs(values) if two_sided else values
    if method == "none":
        if height is None:
            raise ValueError("method='none' needs height= (a z cutoff)")
        return float(height)
    if height is not None:
        raise ValueError("height= applies only to method='none'")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1); got {alpha!r}")
    level = alpha / 2 if two_sided else alpha
    n = stats.size
    if n == 0:
        return float("inf")
    if method == "bonferroni":
        return float(-sp_special.ndtri(level / n))
    if method == "fdr":
        ordered = -np.sort(-stats)
        p = sp_special.ndtr(-ordered)
        passing = p < level * np.arange(1, n + 1) / n
        return float(ordered[passing][-1] - 1e-12) if passing.any() else float("inf")
    raise ValueError(f"method must be 'fdr', 'bonferroni' or 'none'; got {method!r}")


def threshold_map(
    z: NDArray[np.float64],
    spatial: SpatialContext | None,
    *,
    alpha: float = 0.05,
    method: ThresholdMethod = "fdr",
    height: float | None = None,
    cluster_size: int = 0,
    two_sided: bool = True,
    name: str = "map",
) -> ThresholdedMap:
    """Threshold a z map by height, then by cluster extent."""
    values = np.asarray(z, dtype=np.float64).ravel()
    cutoff = height_threshold(
        values, alpha=alpha, method=method, height=height, two_sided=two_sided
    )
    stat = np.abs(values) if two_sided else values
    kept = np.where(np.isfinite(values) & (stat > cutoff), values, 0.0)
    if cluster_size > 1:
        if spatial is None:
            raise ValueError("cluster_size needs voxel geometry (fit from images)")
        volume = spatial.reconstruct(kept, fill=0.0)
        for sign in (1, -1) if two_sided else (1,):
            labels, n_labels = ndimage.label(sign * volume > 0)
            if n_labels == 0:
                continue
            sizes = np.bincount(labels.ravel())
            small = sizes < cluster_size
            small[0] = False
            volume[small[labels]] = 0.0
        kept = volume[spatial.mask]
    return ThresholdedMap(
        z=kept,
        threshold=cutoff,
        method=method,
        alpha=alpha,
        cluster_size=int(cluster_size),
        two_sided=two_sided,
        name=name,
        spatial=spatial,
    )
