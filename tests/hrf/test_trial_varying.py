"""Tests for trial-varying (list-of-HRFs) regressor support."""

import numpy as np
import pytest

from fmrimod.hrf.generators import boxcar_generator
from fmrimod.hrf.library import SPM_CANONICAL
from fmrimod.regressor.core import regressor


class TestTrialVaryingRegressor:
    def test_list_accepted(self):
        h1 = boxcar_generator(width=4)
        h2 = boxcar_generator(width=8)
        reg = regressor(onsets=[10, 30], hrf=[h1, h2])
        assert reg.hrf_is_list
        assert len(reg.hrf) == 2

    def test_length1_recycling(self):
        h = boxcar_generator(width=5)
        reg = regressor(onsets=[10, 20, 30], hrf=[h])
        assert reg.hrf_is_list
        assert len(reg.hrf) == 3

    def test_invalid_list_length(self):
        h1 = boxcar_generator(width=4)
        h2 = boxcar_generator(width=8)
        with pytest.raises(ValueError, match="length 1 or 3"):
            regressor(onsets=[10, 20, 30], hrf=[h1, h2])

    def test_evaluation(self):
        h1 = boxcar_generator(width=4)
        h2 = boxcar_generator(width=8)
        reg = regressor(onsets=[10, 30], hrf=[h1, h2])
        grid = np.arange(0, 50, 0.5)
        result = reg.evaluate(grid, precision=0.1)
        # Should produce non-zero values
        assert np.sum(result != 0) > 0
        # Result shape should be 1D for single-basis
        assert result.ndim == 1

    def test_per_event_convolution(self):
        """Each event should use its own HRF."""
        h_short = boxcar_generator(width=2)
        h_long = boxcar_generator(width=10)
        reg = regressor(onsets=[0, 50], hrf=[h_short, h_long])
        grid = np.arange(0, 70, 0.5)
        result = reg.evaluate(grid, precision=0.1)

        # The response near onset=0 should be shorter than near onset=50
        response_event1 = result[(grid >= 0) & (grid < 20)]
        response_event2 = result[(grid >= 50) & (grid < 70)]
        nonzero1 = np.sum(response_event1 != 0)
        nonzero2 = np.sum(response_event2 != 0)
        assert nonzero2 > nonzero1  # long boxcar has more non-zero points

    def test_amplitude_filtering_preserves_alignment(self):
        h1 = boxcar_generator(width=2)
        h2 = boxcar_generator(width=4)
        h3 = boxcar_generator(width=6)
        reg = regressor(
            onsets=[10, 20, 30],
            hrf=[h1, h2, h3],
            amplitude=[1.0, 0.0, 1.0],  # middle event filtered
        )
        assert len(reg.onsets) == 2
        assert len(reg.hrf) == 2
        assert reg.hrf[0].span == 2.0  # h1
        assert reg.hrf[1].span == 6.0  # h3 (not h2)

    def test_single_hrf_unchanged(self):
        """Passing a single HRF (not a list) still works."""
        reg = regressor(onsets=[10, 30], hrf=SPM_CANONICAL)
        assert not reg.hrf_is_list
        grid = np.arange(0, 60, 0.5)
        result = reg.evaluate(grid, precision=0.1)
        assert result.shape == (len(grid),)

    def test_nbasis_property(self):
        h1 = boxcar_generator(width=4)
        reg = regressor(onsets=[10, 20], hrf=[h1, h1])
        assert reg.nbasis == 1

    def test_inconsistent_nbasis_rejected(self):
        from fmrimod.hrf.generators import tent_generator
        h1 = boxcar_generator(width=4)  # nbasis=1
        h2 = tent_generator(nbasis=3, span=10)  # nbasis=3
        with pytest.raises(ValueError, match="same nbasis"):
            regressor(onsets=[10, 20], hrf=[h1, h2])

    def test_span_uses_max(self):
        h1 = boxcar_generator(width=4)
        h2 = boxcar_generator(width=10)
        reg = regressor(onsets=[0, 10], hrf=[h1, h2])
        assert reg.span == 10.0

    def test_repr_shows_trial_varying(self):
        h1 = boxcar_generator(width=4)
        h2 = boxcar_generator(width=8)
        reg = regressor(onsets=[10, 30], hrf=[h1, h2])
        r = repr(reg)
        assert "trial_varying=True" in r
        assert "list[2 HRFs]" in r

    def test_empty_onsets_with_list_hrf(self):
        """Empty onsets should still work with list HRF."""
        h1 = boxcar_generator(width=4)
        reg = regressor(onsets=[], hrf=[h1])
        assert reg.filtered_all
        grid = np.arange(0, 20, 0.5)
        result = reg.evaluate(grid)
        assert np.all(result == 0)
