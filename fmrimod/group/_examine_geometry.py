"""Deterministic residual-geometry projection.

The implicit projection is a function of global feature IDs, not of the
scan-block partition. This is the fmrigds ``.geometry_projection`` contract
(``test-examination-geometry.R``).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

_MOD = 2147483629


def _utf8_to_int(text: str) -> list[int]:
    return [ord(ch) for ch in text]


def _code_hash(code: Sequence[int]) -> int:
    acc = 0.0
    for i, value in enumerate(code, start=1):
        acc += float(value) * float(i)
    return int(acc % _MOD)


def geometry_projection(
    feature_ids: Sequence[str],
    dimension: int,
    seed: str = "digest",
    *,
    nonzero: int = 3,
) -> NDArray[np.float64]:
    """Sparse signed projection with ID-keyed columns.

    Rows for the same feature ID are identical whether the IDs are projected
    in one shot or in contiguous blocks.
    """
    n_feature = len(feature_ids)
    dim = int(dimension)
    if n_feature == 0 or dim < 1:
        return np.zeros((n_feature, 0), dtype=np.float64)
    k = min(int(nonzero), dim)
    omega: NDArray[np.float64] = np.zeros((n_feature, dim), dtype=np.float64)
    seed_hash = _code_hash(_utf8_to_int(seed))
    scale = 1.0 / np.sqrt(k)
    for i, feature_id in enumerate(feature_ids):
        code = _utf8_to_int(str(feature_id))
        base = (_code_hash(code) + seed_hash) % _MOD
        used: list[int] = []
        step = 1
        while len(used) < k:
            mixed = (base * 48271 + step * 69621 + step * step * 1237) % _MOD
            candidate = int(mixed % dim)
            if candidate not in used:
                used.append(candidate)
            step += 1
        for j, col in enumerate(used, start=1):
            sign = 1.0 if ((base + j * 130363) % 2) == 0 else -1.0
            omega[i, col] = sign * scale
    return omega
