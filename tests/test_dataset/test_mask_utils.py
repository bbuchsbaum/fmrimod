"""Tests for canonical dataset mask helpers."""

from __future__ import annotations

import numpy as np
import pytest

from fmrimod.dataset import mask_to_logical, mask_to_volume


def test_mask_to_logical_flattens_to_bool() -> None:
    mask = np.array([[1, 0], [2, 0]])

    np.testing.assert_array_equal(
        mask_to_logical(mask),
        np.array([True, False, True, False]),
    )


def test_mask_to_volume_validates_size() -> None:
    mask = np.array([True, False, True, False])

    np.testing.assert_array_equal(
        mask_to_volume(mask, (2, 2, 1)),
        np.array([[[True], [False]], [[True], [False]]]),
    )
    with pytest.raises(ValueError, match="Mask length"):
        mask_to_volume(mask, (2, 2, 2))
