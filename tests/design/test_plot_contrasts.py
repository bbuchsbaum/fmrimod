"""Tests for plot_contrasts functionality."""

import pytest
import numpy as np

# Force non-interactive backend for isolated test execution.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from fmrimod.contrast.plot_contrasts import plot_contrasts


class TestPlotContrasts:
    """Test contrast plotting functionality."""

    def test_plot_contrasts_uses_headless_backend(self):
        """Regression: avoid GUI backend crashes in isolated runs."""
        assert matplotlib.get_backend().lower().startswith("agg")
    
    def test_plot_contrasts_dict_simple(self):
        """Test plotting contrasts from a simple dictionary."""
        # Create simple contrast weights
        contrast_dict = {
            'contrast1': np.array([1, -1, 0, 0]),
            'contrast2': np.array([0, 0, 1, -1]),
        }
        
        fig = plot_contrasts(contrast_dict)
        
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 2  # Main plot + colorbar
        
        # Check that the plot has correct dimensions
        ax = fig.axes[0]
        assert ax.get_xlim()[1] > 3  # 4 regressors
        # For imshow, y-axis can be inverted
        ylim = ax.get_ylim()
        assert abs(ylim[1] - ylim[0]) > 1  # 2 contrasts
        
        plt.close(fig)
        
    def test_plot_contrasts_f_contrasts(self):
        """Test plotting F-contrasts with multiple components."""
        # Create F-contrast with multiple columns
        contrast_dict = {
            'main_effect': np.array([[1, 0],
                                   [-1, 1],
                                   [0, -1],
                                   [0, 0]]),
            'simple': np.array([1, 1, -1, -1]),
        }
        
        fig = plot_contrasts(contrast_dict)
        
        # Should have 3 rows: 2 for main_effect components + 1 for simple
        ax = fig.axes[0]
        ylabels = [t.get_text() for t in ax.get_yticklabels()]
        assert 'main_effect_component1' in ylabels
        assert 'main_effect_component2' in ylabels
        assert 'simple' in ylabels
        
        plt.close(fig)
        
    def test_plot_contrasts_diverging_scale(self):
        """Test diverging color scale for positive and negative weights."""
        contrast_dict = {
            'positive': np.array([1, 0.5, 0, 0]),
            'mixed': np.array([1, -1, 0.5, -0.5]),
        }
        
        # Auto mode should detect negative values and use diverging
        fig = plot_contrasts(contrast_dict, scale_mode='auto')
        
        # Check colorbar has negative and positive limits
        cbar = fig.axes[1]  # Colorbar axis
        clim = cbar.get_ylim()
        assert clim[0] < 0
        assert clim[1] > 0
        
        plt.close(fig)
        
    def test_plot_contrasts_one_sided_scale(self):
        """Test one-sided color scale for all positive weights."""
        contrast_dict = {
            'pos1': np.array([1, 0.5, 0, 0]),
            'pos2': np.array([0, 0, 0.5, 1]),
        }
        
        fig = plot_contrasts(contrast_dict, scale_mode='one_sided')
        
        # Check colorbar has only positive limits
        cbar = fig.axes[1]
        clim = cbar.get_ylim()
        assert clim[0] >= 0
        
        plt.close(fig)
        
    def test_plot_contrasts_absolute_limits(self):
        """Test absolute limits option."""
        contrast_dict = {
            'small': np.array([0.1, -0.1, 0, 0]),
        }
        
        # Without absolute limits
        fig1 = plot_contrasts(contrast_dict, absolute_limits=False)
        cbar1 = fig1.axes[1]
        clim1 = cbar1.get_ylim()
        
        # With absolute limits
        fig2 = plot_contrasts(contrast_dict, absolute_limits=True)
        cbar2 = fig2.axes[1]
        clim2 = cbar2.get_ylim()
        
        # Absolute limits should be wider
        assert abs(clim2[1] - clim2[0]) > abs(clim1[1] - clim1[0])
        
        plt.close(fig1)
        plt.close(fig2)
        
    def test_plot_contrasts_custom_colormap(self):
        """Test custom colormap option."""
        contrast_dict = {
            'test': np.array([1, -1, 0, 0]),
        }
        
        fig = plot_contrasts(contrast_dict, cmap='viridis')
        
        # Check that the colormap was applied
        im = fig.axes[0].images[0]
        assert im.cmap.name == 'viridis'
        
        plt.close(fig)
        
    def test_plot_contrasts_rotate_labels(self):
        """Test x-axis label rotation."""
        contrast_dict = {
            'test': np.array([1, -1, 0, 0, 0, 0]),  # 6 regressors
        }
        
        # With rotation
        fig1 = plot_contrasts(contrast_dict, rotate_x_text=True)
        ax1 = fig1.axes[0]
        rotation1 = ax1.get_xticklabels()[0].get_rotation()
        
        # Without rotation
        fig2 = plot_contrasts(contrast_dict, rotate_x_text=False)
        ax2 = fig2.axes[0]
        rotation2 = ax2.get_xticklabels()[0].get_rotation()
        
        assert rotation1 != rotation2
        
        plt.close(fig1)
        plt.close(fig2)
        
    def test_plot_contrasts_empty_dict(self):
        """Test error handling for empty contrast dictionary."""
        with pytest.raises(ValueError, match="No contrasts provided"):
            plot_contrasts({})
            
    def test_plot_contrasts_mismatched_sizes(self):
        """Test error handling for mismatched contrast sizes."""
        contrast_dict = {
            'contrast1': np.array([1, -1, 0, 0]),
            'contrast2': np.array([1, -1, 0]),  # Wrong size
        }
        
        with pytest.raises(ValueError, match="expected 4"):
            plot_contrasts(contrast_dict)
            
    def test_plot_contrasts_custom_figsize(self):
        """Test custom figure size."""
        contrast_dict = {
            'test': np.array([1, -1, 0, 0]),
        }
        
        fig = plot_contrasts(contrast_dict, figsize=(10, 8))
        
        assert fig.get_figwidth() == 10
        assert fig.get_figheight() == 8
        
        plt.close(fig)
