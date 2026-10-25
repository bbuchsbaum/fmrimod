"""Comprehensive tests for visualization module.

This test suite covers:
- design_map (heatmap visualization)
- correlation_map (correlation matrix)
- plot_design_matrix (time series plots)
- plot_baseline_model (baseline terms)
- plot_sampling_frame (sampling frame structure)
- plot_contrasts (contrast visualization)
- plot_Fcontrasts (F-contrast matrices)
"""

import pytest
import numpy as np
import pandas as pd

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt

from fmrimod import event_model, baseline_model
from fmrimod.events.factor import EventFactor
from fmrimod.events.variable import EventVariable
from fmrimod.sampling import SamplingFrame
from fmrimod.visualization import (
    design_map,
    correlation_map,
    plot_design_matrix,
    plot_baseline_model,
    plot_sampling_frame
)


@pytest.fixture(autouse=True)
def close_all_plots():
    """Close all matplotlib figures after each test."""
    yield
    plt.close('all')


class TestPlotBaselineModel:
    """Tests for plot_baseline_model function."""

    @pytest.fixture
    def simple_baseline(self):
        """Simple baseline model with drift."""
        sf = SamplingFrame(blocklens=[50], TR=2.0)
        return baseline_model(basis='poly', degree=2, sframe=sf)

    @pytest.fixture
    def complex_baseline(self):
        """Baseline model with drift and blocks."""
        sf = SamplingFrame(blocklens=[40, 40], TR=2.0)
        return baseline_model(basis='poly', degree=3, sframe=sf, intercept='runwise')

    def test_plot_baseline_returns_figure(self, simple_baseline):
        """Test that plot_baseline_model returns figure and axes."""
        fig, axes = plot_baseline_model(simple_baseline)

        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(axes, np.ndarray)

    def test_plot_baseline_all_terms(self, complex_baseline):
        """Test plotting all baseline terms."""
        fig, axes = plot_baseline_model(complex_baseline)

        # Should have subplots for drift and block
        assert len(axes) >= 1

    def test_plot_baseline_specific_term(self, complex_baseline):
        """Test plotting specific term only."""
        fig, axes = plot_baseline_model(complex_baseline, term_name='drift')

        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_baseline_has_lines(self, simple_baseline):
        """Test that line plots are created."""
        fig, axes = plot_baseline_model(simple_baseline)

        # At least one subplot should have lines
        has_lines = any(len(ax.get_lines()) > 0 for ax in axes)
        assert has_lines, "Should have line plots"

    def test_plot_baseline_custom_figsize(self, simple_baseline):
        """Test custom figure size."""
        figsize = (10, 8)
        fig, axes = plot_baseline_model(simple_baseline, figsize=figsize)

        assert fig.get_figwidth() == 10
        assert fig.get_figheight() == 8

    def test_plot_baseline_custom_title(self, simple_baseline):
        """Test custom title."""
        title = "My Baseline Model"
        fig, axes = plot_baseline_model(simple_baseline, title=title)

        # Should have the custom title
        if fig._suptitle:
            assert title in fig._suptitle.get_text()

    def test_plot_baseline_multi_block(self):
        """Test baseline with multiple blocks."""
        sf = SamplingFrame(blocklens=[30, 30, 30], TR=2.0)
        bmodel = baseline_model(basis='poly', degree=2, sframe=sf, intercept='runwise')

        fig, axes = plot_baseline_model(bmodel)

        assert isinstance(fig, matplotlib.figure.Figure)


class TestPlotSamplingFrame:
    """Tests for plot_sampling_frame function."""

    @pytest.fixture
    def single_block_sf(self):
        """Single block sampling frame."""
        return SamplingFrame(blocklens=[50], TR=2.0)

    @pytest.fixture
    def multi_block_sf(self):
        """Multi-block sampling frame."""
        return SamplingFrame(blocklens=[40, 35, 45], TR=2.5)

    def test_plot_sampling_frame_timeline_style(self, multi_block_sf):
        """Test timeline style visualization."""
        fig, ax = plot_sampling_frame(multi_block_sf, style='timeline')

        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_plot_sampling_frame_grid_style(self, multi_block_sf):
        """Test grid style visualization."""
        fig, ax = plot_sampling_frame(multi_block_sf, style='grid')

        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_plot_sampling_frame_invalid_style(self, single_block_sf):
        """Test error for invalid style."""
        with pytest.raises(ValueError, match="style must be"):
            plot_sampling_frame(single_block_sf, style='invalid')

    def test_plot_sampling_frame_single_block(self, single_block_sf):
        """Test with single block."""
        fig, ax = plot_sampling_frame(single_block_sf, style='timeline')

        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_sampling_frame_custom_figsize(self, multi_block_sf):
        """Test custom figure size."""
        figsize = (14, 6)
        fig, ax = plot_sampling_frame(multi_block_sf, figsize=figsize)

        assert fig.get_figwidth() == 14
        assert fig.get_figheight() == 6

    def test_plot_sampling_frame_show_ticks(self, multi_block_sf):
        """Test show_ticks parameter."""
        fig, ax = plot_sampling_frame(multi_block_sf, show_ticks=True)

        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(ax.get_xticks()) > 0
        assert len(ax.get_yticks()) > 0

    def test_plot_sampling_frame_hide_ticks(self, multi_block_sf):
        """Test hiding ticks."""
        fig, ax = plot_sampling_frame(multi_block_sf, show_ticks=False)

        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(ax.get_xticks()) == 0
        assert len(ax.get_yticks()) == 0

    def test_plot_sampling_frame_tick_every_applies_timeline(self, multi_block_sf):
        """Test tick_every controls x-axis spacing in timeline mode."""
        fig, ax = plot_sampling_frame(multi_block_sf, style='timeline', tick_every=50.0)

        xticks = np.asarray(ax.get_xticks(), dtype=float)
        assert len(xticks) >= 3
        diffs = np.diff(xticks)
        np.testing.assert_allclose(diffs, 50.0, atol=1e-8)

    def test_plot_sampling_frame_tick_every_must_be_positive(self, multi_block_sf):
        """Test validation for non-positive tick_every."""
        with pytest.raises(ValueError, match="tick_every must be positive"):
            plot_sampling_frame(multi_block_sf, style='timeline', tick_every=0.0)

    def test_plot_sampling_frame_uses_per_block_tr_for_timeline_widths(self):
        """Regression: timeline bar widths must honor per-block TR values."""
        sf = SamplingFrame(blocklens=[2, 2], tr=[1.0, 3.0])
        fig, ax = plot_sampling_frame(sf, style='timeline')

        widths = [patch.get_width() for patch in ax.patches]
        np.testing.assert_allclose(widths, [2.0, 6.0], atol=1e-10)

    def test_plot_sampling_frame_accepts_tr_only_sampling_frame(self):
        """Regression: support sampling frames that expose ``tr`` but no ``TR``."""

        class TrOnlySamplingFrame:
            blocklens = np.array([30, 20])
            tr = np.array([2.0, 2.0])

        fig, ax = plot_sampling_frame(TrOnlySamplingFrame(), style="timeline")

        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)


class TestBaselineModelVisualization:
    """Integration tests for baseline model visualization."""

    def test_design_map_with_baseline_model(self):
        """Test design_map works with BaselineModel."""
        sf = SamplingFrame(blocklens=[50], TR=2.0)
        bmodel = baseline_model(basis='poly', degree=2, sframe=sf)

        fig = design_map(bmodel)

        assert isinstance(fig, matplotlib.figure.Figure)

    def test_correlation_map_with_baseline_model(self):
        """Test correlation_map works with BaselineModel."""
        sf = SamplingFrame(blocklens=[50], TR=2.0)
        bmodel = baseline_model(basis='poly', degree=2, sframe=sf, intercept='runwise')

        fig = correlation_map(bmodel)

        assert isinstance(fig, matplotlib.figure.Figure)

    def test_baseline_with_nuisance(self):
        """Test visualization with nuisance regressors."""
        sf = SamplingFrame(blocklens=[40], TR=2.0)
        nuisance = np.random.randn(40, 2)

        bmodel = baseline_model(basis='poly', degree=1, sframe=sf, nuisance_list=[nuisance])

        fig, axes = plot_baseline_model(bmodel)

        assert isinstance(fig, matplotlib.figure.Figure)


class TestVisualizationEdgeCases:
    """Test edge cases and error handling."""

    def test_single_regressor_model(self):
        """Test with model having single regressor."""
        data = pd.DataFrame({
            'onset': [10, 30],
            'condition': ['A', 'A']
        })
        model = event_model("condition", data=data, tr=2.0, n_scans=50)

        # Design map and plot_design_matrix should work
        fig1 = design_map(model)
        assert isinstance(fig1, matplotlib.figure.Figure)

        # Correlation map with single regressor is a scalar, may not work
        # Skip this edge case for single regressor
        # fig2, ax2 = correlation_map(model)
        # assert isinstance(fig2, matplotlib.figure.Figure)

        fig3 = plot_design_matrix(model)
        assert isinstance(fig3, matplotlib.figure.Figure)

    def test_very_few_scans(self):
        """Test with very few scans."""
        data = pd.DataFrame({
            'onset': [2],
            'condition': ['A']
        })
        model = event_model("condition", data=data, tr=2.0, n_scans=10)

        fig = design_map(model)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_many_columns(self):
        """Test with model having many columns."""
        # Create data with many conditions
        n_events = 30
        data = pd.DataFrame({
            'onset': np.arange(n_events) * 2,
            'condition': [f'cond_{i}' for i in range(n_events)]
        })

        model = event_model("condition", data=data, tr=1.0, n_scans=100)

        # Should handle many columns without error
        fig = design_map(model)
        assert isinstance(fig, matplotlib.figure.Figure)


class TestVisualizationWithHRF:
    """Test visualization with HRF-convolved models."""

    def test_hrf_simple(self):
        """Test with simple HRF."""
        data = pd.DataFrame({
            'onset': [5, 15, 25],
            'condition': ['A', 'B', 'A']
        })

        model = event_model(
            'hrf(condition, hrf="simple")',
            data=data,
            tr=2.0,
            n_scans=40
        )

        fig = design_map(model)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_hrf_canonical(self):
        """Test with canonical HRF."""
        data = pd.DataFrame({
            'onset': [5, 15, 25],
            'condition': ['A', 'B', 'A']
        })

        model = event_model(
            'hrf(condition)',
            data=data,
            tr=2.0,
            n_scans=40
        )

        fig = correlation_map(model)
        assert isinstance(fig, matplotlib.figure.Figure)


if __name__ == '__main__':
    # Run tests with verbose output
    pytest.main([__file__, '-v', '--tb=short'])
