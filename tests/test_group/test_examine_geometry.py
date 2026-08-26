"""Geometry projection depends on global feature IDs, not block partition."""

from __future__ import annotations

import numpy as np

from fmrimod.group import geometry_projection


def test_implicit_projection_is_block_invariant() -> None:
    ids = [f"contrast|feature-{i}" for i in range(1, 32)]
    full = geometry_projection(ids, dimension=12, seed="digest")
    blocked = np.vstack(
        [
            geometry_projection(ids[0:7], 12, "digest"),
            geometry_projection(ids[7:19], 12, "digest"),
            geometry_projection(ids[19:31], 12, "digest"),
        ]
    )
    np.testing.assert_array_equal(full, blocked)
    assert np.all(np.sum(full != 0, axis=1) == 3)
    assert int(np.sum(np.sum(np.abs(full), axis=0) > 0)) > 8
