"""Smoke tests for plotting functions."""

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fmrimod.plotting import plot_hrf, plot_hrfs, plot_regressor, plot_regressors
from fmrimod.hrf.library import SPM_CANONICAL, SPM_WITH_DERIVATIVE, GAMMA_HRF
from fmrimod.hrf.generators import boxcar_generator
from fmrimod.regressor.core import regressor


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


class TestPlotHRF:
    def test_single_basis(self):
        ax = plot_hrf(SPM_CANONICAL)
        assert ax is not None
        assert len(ax.lines) >= 1

    def test_multi_basis(self):
        ax = plot_hrf(SPM_WITH_DERIVATIVE)
        assert len(ax.lines) >= 2

    def test_custom_time(self):
        t = np.linspace(0, 30, 200)
        ax = plot_hrf(SPM_CANONICAL, time=t)
        assert ax is not None

    def test_normalize(self):
        ax = plot_hrf(SPM_CANONICAL, normalize=True)
        # Peak should be at 1.0
        ydata = ax.lines[0].get_ydata()
        assert np.max(ydata) == pytest.approx(1.0, abs=0.01)

    def test_no_peak_annotation(self):
        ax = plot_hrf(SPM_CANONICAL, show_peak=False)
        assert len(ax.texts) == 0

    def test_existing_axes(self):
        fig, ax = plt.subplots()
        returned_ax = plot_hrf(SPM_CANONICAL, ax=ax)
        assert returned_ax is ax

    def test_convenience_method(self):
        ax = SPM_CANONICAL.plot()
        assert ax is not None


class TestPlotHRFs:
    def test_overlay(self):
        ax = plot_hrfs(SPM_CANONICAL, GAMMA_HRF)
        # 2 HRF lines + 1 axhline baseline = 3
        assert len(ax.lines) >= 2

    def test_custom_labels(self):
        ax = plot_hrfs(SPM_CANONICAL, GAMMA_HRF, labels=["A", "B"])
        legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert legend_texts == ["A", "B"]

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            plot_hrfs()

    def test_normalize(self):
        ax = plot_hrfs(SPM_CANONICAL, GAMMA_HRF, normalize=True)
        # Check only the HRF data lines (first 2), not the axhline
        for line in ax.lines[:2]:
            assert np.max(line.get_ydata()) == pytest.approx(1.0, abs=0.02)


class TestPlotRegressor:
    def test_basic(self):
        reg = regressor(onsets=[10, 30, 50])
        ax = plot_regressor(reg)
        assert ax is not None

    def test_with_grid(self):
        reg = regressor(onsets=[10, 30])
        grid = np.arange(0, 60, 0.5)
        ax = plot_regressor(reg, grid=grid)
        assert ax is not None

    def test_no_onsets_markers(self):
        reg = regressor(onsets=[10, 30])
        ax = plot_regressor(reg, show_onsets=False)
        # Should have 2 lines: data + baseline, but no extra vlines
        # (axhline adds 1, plot adds 1 per basis)
        assert ax is not None

    def test_trial_varying(self):
        h1 = boxcar_generator(width=4)
        h2 = boxcar_generator(width=8)
        reg = regressor(onsets=[10, 30], hrf=[h1, h2])
        ax = plot_regressor(reg)
        assert ax is not None

    def test_convenience_method(self):
        reg = regressor(onsets=[10, 30])
        ax = reg.plot()
        assert ax is not None


class TestPlotRegressors:
    def test_overlay(self):
        reg1 = regressor(onsets=[10, 30, 50])
        reg2 = regressor(onsets=[20, 40])
        ax = plot_regressors(reg1, reg2)
        assert len(ax.lines) >= 2

    def test_custom_labels(self):
        reg1 = regressor(onsets=[10])
        reg2 = regressor(onsets=[20])
        ax = plot_regressors(reg1, reg2, labels=["A", "B"])
        legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert legend_texts == ["A", "B"]

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            plot_regressors()
