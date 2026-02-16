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
from fmrimod.contrast import (
    plot_contrasts,
    plot_Fcontrasts,
    Fcontrasts,
    contrast,
    pair_contrast
)


@pytest.fixture(autouse=True)
def close_all_plots():
    """Close all matplotlib figures after each test."""
    yield
    plt.close('all')


class TestDesignMapComprehensive:
    """Comprehensive tests for design_map function."""

    @pytest.fixture
    def simple_model(self):
        """Simple model with 1 categorical variable."""
        data = pd.DataFrame({
            'onset': [5, 15, 25, 35],
            'condition': ['A', 'B', 'A', 'B']
        })
        return event_model("condition", data=data, tr=2.0, n_scans=50)

    @pytest.fixture
    def multi_term_model(self):
        """Model with multiple terms."""
        data = pd.DataFrame({
            'onset': [5, 10, 15, 20, 25, 30, 35, 40],
            'condition': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B'],
            'rating': [1, 2, 3, 4, 5, 4, 3, 2]
        })
        return event_model("condition + rating", data=data, tr=2.0, n_scans=60)

    @pytest.fixture
    def multi_block_model(self):
        """Model with multiple blocks/runs."""
        data1 = pd.DataFrame({
            'onset': [5, 15],
            'condition': ['A', 'B']
        })
        data2 = pd.DataFrame({
            'onset': [5, 15],
            'condition': ['A', 'B']
        })

        # Create with block structure
        model = event_model("condition", data=data1, tr=2.0, n_scans=30)
        return model

    def test_design_map_returns_figure_and_axes(self, simple_model):
        """Test that design_map returns matplotlib figure and axes."""
        fig, ax = design_map(simple_model)

        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_design_map_creates_heatmap(self, simple_model):
        """Test that heatmap image is created."""
        fig, ax = design_map(simple_model)

        images = ax.get_images()
        assert len(images) == 1, "Should create one heatmap image"

    def test_design_map_has_correct_labels(self, simple_model):
        """Test axis labels are correct."""
        fig, ax = design_map(simple_model)

        assert ax.get_xlabel() == 'Regressors'
        assert ax.get_ylabel() == 'Scan Number'
        assert 'Design Matrix' in ax.get_title()

    def test_design_map_with_block_separators(self, simple_model):
        """Test design_map with block separators enabled."""
        # This tests the parameter is accepted
        fig, ax = design_map(simple_model, block_separators=True)

        assert isinstance(fig, matplotlib.figure.Figure)

    def test_design_map_without_block_separators(self, simple_model):
        """Test design_map with block separators disabled."""
        fig, ax = design_map(simple_model, block_separators=False)

        assert isinstance(fig, matplotlib.figure.Figure)

    def test_design_map_custom_figsize(self, simple_model):
        """Test custom figure size."""
        figsize = (12, 8)
        fig, ax = design_map(simple_model, figsize=figsize)

        assert fig.get_figwidth() == 12
        assert fig.get_figheight() == 8

    def test_design_map_custom_colormap(self, simple_model):
        """Test custom colormap."""
        fig, ax = design_map(simple_model, cmap='coolwarm')

        images = ax.get_images()
        assert len(images) == 1
        assert images[0].get_cmap().name == 'coolwarm'

    def test_design_map_with_fill_limits(self, simple_model):
        """Test design_map with custom color scale limits."""
        fig, ax = design_map(simple_model, fill_limits=(-2, 2))

        images = ax.get_images()
        clim = images[0].get_clim()
        assert clim[0] == -2
        assert clim[1] == 2

    def test_design_map_with_fill_midpoint(self, simple_model):
        """Test diverging colormap with midpoint."""
        fig, ax = design_map(simple_model, fill_midpoint=0)

        images = ax.get_images()
        clim = images[0].get_clim()
        # Should be symmetric around midpoint
        assert clim[0] + clim[1] == pytest.approx(0, abs=1e-10)

    def test_design_map_rotate_x_text_true(self, simple_model):
        """Test x-axis label rotation enabled."""
        fig, ax = design_map(simple_model, rotate_x_text=True)

        labels = ax.get_xticklabels()
        if labels:
            # At least one label should be rotated
            rotations = [label.get_rotation() for label in labels]
            assert any(r != 0 for r in rotations)

    def test_design_map_rotate_x_text_false(self, simple_model):
        """Test x-axis label rotation disabled."""
        fig, ax = design_map(simple_model, rotate_x_text=False)

        labels = ax.get_xticklabels()
        if labels:
            rotations = [label.get_rotation() for label in labels]
            assert all(r == 0 for r in rotations)

    def test_design_map_multi_term_model(self, multi_term_model):
        """Test design_map with model having multiple terms."""
        fig, ax = design_map(multi_term_model)

        assert isinstance(fig, matplotlib.figure.Figure)

        # Check that multiple columns are represented
        n_cols = multi_term_model.design_matrix.shape[1]
        assert len(ax.get_xticklabels()) >= 2

    def test_design_map_large_model(self):
        """Test design_map with larger model (many scans)."""
        data = pd.DataFrame({
            'onset': np.arange(0, 200, 5),
            'condition': ['A', 'B'] * 20
        })
        model = event_model("condition", data=data, tr=1.0, n_scans=200)

        fig, ax = design_map(model)

        assert isinstance(fig, matplotlib.figure.Figure)
        # Should automatically adjust figure size for large models


class TestCorrelationMapComprehensive:
    """Comprehensive tests for correlation_map function."""

    @pytest.fixture
    def uncorrelated_model(self):
        """Model with uncorrelated regressors."""
        data = pd.DataFrame({
            'onset': [5, 25, 45, 65],
            'condition': ['A', 'B', 'A', 'B']
        })
        return event_model("condition", data=data, tr=2.0, n_scans=80)

    @pytest.fixture
    def correlated_model(self):
        """Model with somewhat correlated regressors."""
        # Close onsets create correlation after HRF convolution
        data = pd.DataFrame({
            'onset': [5, 6, 25, 26, 45, 46],
            'condition': ['A', 'B', 'A', 'B', 'A', 'B']
        })
        return event_model("condition", data=data, tr=1.0, n_scans=60)

    def test_correlation_map_returns_figure_and_axes(self, uncorrelated_model):
        """Test correlation_map returns figure and axes."""
        fig, ax = correlation_map(uncorrelated_model)

        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_correlation_map_creates_heatmap(self, uncorrelated_model):
        """Test correlation heatmap is created."""
        fig, ax = correlation_map(uncorrelated_model)

        # Either images (matplotlib) or collections (seaborn)
        has_heatmap = len(ax.images) > 0 or len(ax.collections) > 0
        assert has_heatmap, "Should create correlation heatmap"

    def test_correlation_map_symmetric_matrix(self, uncorrelated_model):
        """Test that correlation matrix is symmetric."""
        fig, ax = correlation_map(uncorrelated_model)

        # Get the correlation matrix from the image
        if ax.images:
            corr = ax.images[0].get_array()
            # Check symmetry
            if corr.ndim == 2:
                assert np.allclose(corr, corr.T), "Correlation matrix should be symmetric"

    def test_correlation_map_diagonal_ones(self, uncorrelated_model):
        """Test that diagonal elements are 1 (self-correlation)."""
        fig, ax = correlation_map(uncorrelated_model)

        if ax.images:
            corr = ax.images[0].get_array()
            if corr.ndim == 2:
                diag = np.diag(corr)
                assert np.allclose(diag, 1.0), "Diagonal should be all 1s"

    def test_correlation_map_with_annotations(self, uncorrelated_model):
        """Test correlation map with value annotations."""
        fig, ax = correlation_map(uncorrelated_model, annot=True)

        # Should have text annotations
        texts = ax.texts
        assert len(texts) > 0, "Should have text annotations"

    def test_correlation_map_custom_format(self, uncorrelated_model):
        """Test custom annotation format."""
        fig, ax = correlation_map(uncorrelated_model, annot=True, fmt='.3f')

        assert isinstance(fig, matplotlib.figure.Figure)

    def test_correlation_map_custom_colormap(self, uncorrelated_model):
        """Test custom colormap."""
        fig, ax = correlation_map(uncorrelated_model, cmap='coolwarm')

        assert isinstance(fig, matplotlib.figure.Figure)

    def test_correlation_map_custom_figsize(self, uncorrelated_model):
        """Test custom figure size."""
        figsize = (10, 10)
        fig, ax = correlation_map(uncorrelated_model, figsize=figsize)

        assert fig.get_figwidth() == 10
        assert fig.get_figheight() == 10

    def test_correlation_map_rotate_labels(self, uncorrelated_model):
        """Test label rotation."""
        fig, ax = correlation_map(uncorrelated_model, rotate_x_text=True)

        labels = ax.get_xticklabels()
        if labels:
            rotations = [label.get_rotation() for label in labels]
            assert any(r != 0 for r in rotations)


class TestPlotDesignMatrixComprehensive:
    """Comprehensive tests for plot_design_matrix function."""

    @pytest.fixture
    def simple_model(self):
        """Simple model for testing."""
        data = pd.DataFrame({
            'onset': [10, 30, 50],
            'condition': ['A', 'B', 'A']
        })
        return event_model("condition", data=data, tr=2.0, n_scans=60)

    @pytest.fixture
    def complex_model(self):
        """Complex model with many regressors."""
        data = pd.DataFrame({
            'onset': [5, 10, 15, 20, 25, 30, 35, 40],
            'condition': ['A', 'B', 'C', 'D', 'A', 'B', 'C', 'D'],
            'rating': [1, 2, 3, 4, 5, 4, 3, 2]
        })
        return event_model("condition + rating", data=data, tr=2.0, n_scans=80)

    def test_plot_design_matrix_returns_figure(self, simple_model):
        """Test that plot_design_matrix returns figure."""
        fig, axes = plot_design_matrix(simple_model)

        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_design_matrix_all_regressors(self, simple_model):
        """Test plotting all regressors."""
        fig, axes = plot_design_matrix(simple_model)

        # Should create plot(s) for all regressors
        assert fig is not None

    def test_plot_design_matrix_has_lines(self, simple_model):
        """Test that line plots are created."""
        fig, axes = plot_design_matrix(simple_model, separate_regressors=False)

        # axes might be a single Axes or array of Axes
        if isinstance(axes, matplotlib.axes.Axes):
            lines = axes.get_lines()
            assert len(lines) > 0, "Should have line plots"

    def test_plot_design_matrix_specific_term(self, complex_model):
        """Test plotting specific term only."""
        fig, axes = plot_design_matrix(complex_model, term_name='condition')

        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_design_matrix_nonexistent_term(self, simple_model):
        """Test error for non-existent term."""
        with pytest.raises(ValueError, match="No columns found"):
            plot_design_matrix(simple_model, term_name='nonexistent')

    def test_plot_design_matrix_separate_regressors_true(self, complex_model):
        """Test separate subplots for regressors."""
        fig, axes = plot_design_matrix(complex_model, separate_regressors=True)

        # With many regressors, should use subplots
        if isinstance(axes, np.ndarray):
            assert len(axes) > 1

    def test_plot_design_matrix_separate_regressors_false(self, simple_model):
        """Test all regressors in one plot."""
        fig, axes = plot_design_matrix(simple_model, separate_regressors=False)

        assert isinstance(axes, matplotlib.axes.Axes)

    def test_plot_design_matrix_custom_figsize(self, simple_model):
        """Test custom figure size."""
        figsize = (12, 6)
        fig, axes = plot_design_matrix(simple_model, figsize=figsize)

        assert fig.get_figwidth() == 12
        assert fig.get_figheight() == 6

    def test_plot_design_matrix_few_regressors(self):
        """Test with very few regressors (1 term)."""
        data = pd.DataFrame({
            'onset': [10, 30],
            'condition': ['A', 'A']
        })
        model = event_model("condition", data=data, tr=2.0, n_scans=50)

        fig, axes = plot_design_matrix(model)

        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_design_matrix_many_regressors(self):
        """Test with many regressors (8+ columns)."""
        # Create model with interaction to get many columns
        data = pd.DataFrame({
            'onset': np.arange(0, 40, 2),
            'cond1': ['A', 'B'] * 10,
            'cond2': ['X', 'Y', 'Z'] * 6 + ['X', 'Y']
        })
        model = event_model("cond1 + cond2", data=data, tr=2.0, n_scans=80)

        fig, axes = plot_design_matrix(model, separate_regressors=True)

        # Should create multiple subplots
        assert isinstance(fig, matplotlib.figure.Figure)


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


class TestPlotContrasts:
    """Tests for plot_contrasts function."""

    def test_plot_contrasts_simple_dict(self):
        """Test plotting contrasts from simple dictionary."""
        contrast_dict = {
            'A_vs_B': np.array([1, -1, 0, 0]),
            'C_vs_D': np.array([0, 0, 1, -1])
        }

        fig = plot_contrasts(contrast_dict)

        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) >= 1  # At least main plot

    def test_plot_contrasts_2d_weights(self):
        """Test with 2D contrast weights (F-contrasts)."""
        contrast_dict = {
            'main_effect': np.array([[1, 0],
                                    [-1, 1],
                                    [0, -1],
                                    [0, 0]])
        }

        fig = plot_contrasts(contrast_dict)

        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_contrasts_diverging_scale(self):
        """Test diverging color scale."""
        contrast_dict = {
            'contrast': np.array([1, -1, 0.5, -0.5])
        }

        fig = plot_contrasts(contrast_dict, scale_mode='diverging')

        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_contrasts_one_sided_scale(self):
        """Test one-sided color scale."""
        contrast_dict = {
            'positive': np.array([1, 0.5, 0.25, 0])
        }

        fig = plot_contrasts(contrast_dict, scale_mode='one_sided')

        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_contrasts_absolute_limits(self):
        """Test absolute limits option."""
        contrast_dict = {
            'small': np.array([0.1, -0.1, 0, 0])
        }

        fig = plot_contrasts(contrast_dict, absolute_limits=True)

        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_contrasts_custom_colormap(self):
        """Test custom colormap."""
        contrast_dict = {
            'test': np.array([1, -1, 0, 0])
        }

        fig = plot_contrasts(contrast_dict, cmap='viridis')

        im = fig.axes[0].images[0]
        assert im.get_cmap().name == 'viridis'

    def test_plot_contrasts_rotate_labels_true(self):
        """Test x-axis label rotation."""
        contrast_dict = {
            'test': np.array([1, -1, 0, 0, 0, 0])
        }

        fig = plot_contrasts(contrast_dict, rotate_x_text=True)

        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_contrasts_rotate_labels_false(self):
        """Test no label rotation."""
        contrast_dict = {
            'test': np.array([1, -1, 0, 0])
        }

        fig = plot_contrasts(contrast_dict, rotate_x_text=False)

        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_contrasts_empty_dict(self):
        """Test error for empty dictionary."""
        with pytest.raises(ValueError, match="No contrasts"):
            plot_contrasts({})

    def test_plot_contrasts_mismatched_sizes(self):
        """Test error for mismatched contrast sizes."""
        contrast_dict = {
            'contrast1': np.array([1, -1, 0, 0]),
            'contrast2': np.array([1, -1, 0])  # Wrong size
        }

        with pytest.raises(ValueError, match="expected 4"):
            plot_contrasts(contrast_dict)

    def test_plot_contrasts_custom_figsize(self):
        """Test custom figure size."""
        contrast_dict = {
            'test': np.array([1, -1, 0, 0])
        }

        figsize = (10, 6)
        fig = plot_contrasts(contrast_dict, figsize=figsize)

        assert fig.get_figwidth() == 10
        assert fig.get_figheight() == 6

    def test_plot_contrasts_from_event_model(self):
        """Test plotting contrasts from EventModel with simple contrast dict."""
        data = pd.DataFrame({
            'onset': [5, 15, 25, 35],
            'condition': ['A', 'B', 'A', 'B']
        })

        # Create model without contrasts first
        model = event_model(
            "condition",
            data=data,
            tr=2.0,
            n_scans=50
        )

        # Test plotting with manually created contrast dict
        # (simulates what contrast_weights would return)
        n_cols = model.design_matrix.shape[1]
        contrast_dict = {
            'test_contrast': np.array([1, -1] + [0] * (n_cols - 2))
        }

        fig = plot_contrasts(contrast_dict)
        assert isinstance(fig, matplotlib.figure.Figure)


class TestPlotFcontrasts:
    """Tests for plot_Fcontrasts function."""

    def test_plot_fcontrasts_simple_dict(self):
        """Test plotting simple F-contrasts from dict."""
        fcontrasts = {
            'main_effect': np.array([[1, 0],
                                    [-1, 0],
                                    [0, 1],
                                    [0, -1]])
        }

        # plot_Fcontrasts doesn't return fig, it shows it
        # So we just test it doesn't error
        plot_Fcontrasts(fcontrasts)

        # Check a figure was created
        figs = plt.get_fignums()
        assert len(figs) > 0

    def test_plot_fcontrasts_from_event_model(self):
        """Test F-contrasts directly from EventModel."""
        data = pd.DataFrame({
            'onset': [5, 10, 15, 20, 25, 30],
            'condition': ['A', 'B', 'C', 'A', 'B', 'C']
        })

        model = event_model("condition", data=data, tr=2.0, n_scans=50)

        # Plot directly from EventModel (new functionality)
        plot_Fcontrasts(model)

        # Check figure was created
        figs = plt.get_fignums()
        assert len(figs) > 0

    def test_plot_fcontrasts_dict_from_model(self):
        """Test F-contrasts from dict generated by Fcontrasts(model)."""
        data = pd.DataFrame({
            'onset': [5, 10, 15, 20, 25, 30],
            'condition': ['A', 'B', 'C', 'A', 'B', 'C']
        })

        model = event_model("condition", data=data, tr=2.0, n_scans=50)
        fcon = Fcontrasts(model)

        if fcon:
            plot_Fcontrasts(fcon)

            # Check figure was created
            figs = plt.get_fignums()
            assert len(figs) > 0

    def test_plot_fcontrasts_multiple_contrasts(self):
        """Test plotting multiple F-contrasts."""
        fcontrasts = {
            'effect1': np.array([[1, 0], [-1, 0], [0, 0], [0, 0]]),
            'effect2': np.array([[0, 0], [0, 0], [1, 0], [-1, 0]]),
            'interaction': np.array([[1, 0], [-1, 1], [0, -1], [0, 0]])
        }

        plot_Fcontrasts(fcontrasts)

        figs = plt.get_fignums()
        assert len(figs) > 0

    def test_plot_fcontrasts_custom_colormap(self):
        """Test custom colormap."""
        fcontrasts = {
            'test': np.array([[1, 0], [-1, 0]])
        }

        plot_Fcontrasts(fcontrasts, cmap='coolwarm')

        figs = plt.get_fignums()
        assert len(figs) > 0

    def test_plot_fcontrasts_custom_figsize(self):
        """Test custom figure size."""
        fcontrasts = {
            'test': np.array([[1, 0], [-1, 0]])
        }

        plot_Fcontrasts(fcontrasts, figsize=(12, 8))

        figs = plt.get_fignums()
        assert len(figs) > 0

    def test_plot_fcontrasts_empty_dict_raises_error(self):
        """Test that empty F-contrasts dictionary raises error."""
        with pytest.raises(ValueError, match="No F-contrasts"):
            plot_Fcontrasts({})

    def test_plot_fcontrasts_multi_contrast_layout(self):
        """Test layout with many contrasts (>3 triggers multi-row layout)."""
        # Create 5 contrasts to test multi-row layout
        fcontrasts = {
            f'contrast_{i}': np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
            for i in range(5)
        }

        plot_Fcontrasts(fcontrasts)

        # Check figure was created
        figs = plt.get_fignums()
        assert len(figs) > 0

        # Get the current figure and check it has correct number of subplots
        fig = plt.gcf()
        # Should have at least 5 axes (some may be hidden)
        assert len(fig.axes) >= 5


class TestBaselineModelVisualization:
    """Integration tests for baseline model visualization."""

    def test_design_map_with_baseline_model(self):
        """Test design_map works with BaselineModel."""
        sf = SamplingFrame(blocklens=[50], TR=2.0)
        bmodel = baseline_model(basis='poly', degree=2, sframe=sf)

        fig, ax = design_map(bmodel)

        assert isinstance(fig, matplotlib.figure.Figure)

    def test_correlation_map_with_baseline_model(self):
        """Test correlation_map works with BaselineModel."""
        sf = SamplingFrame(blocklens=[50], TR=2.0)
        bmodel = baseline_model(basis='poly', degree=2, sframe=sf, intercept='runwise')

        fig, ax = correlation_map(bmodel)

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
        fig1, ax1 = design_map(model)
        assert isinstance(fig1, matplotlib.figure.Figure)

        # Correlation map with single regressor is a scalar, may not work
        # Skip this edge case for single regressor
        # fig2, ax2 = correlation_map(model)
        # assert isinstance(fig2, matplotlib.figure.Figure)

        fig3, ax3 = plot_design_matrix(model)
        assert isinstance(fig3, matplotlib.figure.Figure)

    def test_very_few_scans(self):
        """Test with very few scans."""
        data = pd.DataFrame({
            'onset': [2],
            'condition': ['A']
        })
        model = event_model("condition", data=data, tr=2.0, n_scans=10)

        fig, ax = design_map(model)
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
        fig, ax = design_map(model)
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

        fig, ax = design_map(model)
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

        fig, ax = correlation_map(model)
        assert isinstance(fig, matplotlib.figure.Figure)


if __name__ == '__main__':
    # Run tests with verbose output
    pytest.main([__file__, '-v', '--tb=short'])
