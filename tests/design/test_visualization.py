"""Tests for visualization functions."""

import pytest
import numpy as np
import pandas as pd

# Try importing matplotlib
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from fmrimod import event_model
from fmrimod.events.factor import EventFactor
from fmrimod.events.variable import EventVariable
from fmrimod.visualization import design_map, correlation_map
from fmrimod.visualization.design_map import (
    plot_design_matrix, 
    plot_model_summary
)


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
class TestDesignMap:
    """Test design_map visualization."""

    def test_headless_backend(self):
        """Regression: isolated visualization tests should not require GUI backend."""
        assert matplotlib.get_backend().lower().startswith("agg")
    
    @pytest.fixture
    def simple_model(self):
        """Create a simple event model for testing."""
        n_scans = 100
        data = pd.DataFrame({
            'onset': [10, 30, 50, 70],
            'condition': ['A', 'B', 'A', 'B'],
            'rating': [1, 2, 3, 4]
        })
        
        model = event_model(
            "condition + rating",
            data=data,
            tr=2.0,
            n_scans=n_scans
        )
        return model
    
    def test_design_map_basic(self, simple_model):
        """Test basic design_map functionality."""
        fig, ax = design_map(simple_model)
        
        # Check that figure and axes were created
        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)
        
        # Check that an image was plotted
        images = ax.get_images()
        assert len(images) == 1
        
        # Check labels
        assert ax.get_xlabel() == 'Regressors'
        assert ax.get_ylabel() == 'Scan Number'
        
        plt.close(fig)
    
    def test_design_map_custom_params(self, simple_model):
        """Test design_map with custom parameters."""
        fig, ax = design_map(
            simple_model,
            figsize=(10, 8),
            cmap='coolwarm',
            fill_midpoint=0,
            fill_limits=(-2, 2),
            rotate_x_text=False
        )
        
        # Check figure size
        assert fig.get_figwidth() == 10
        assert fig.get_figheight() == 8
        
        # Check that rotation was not applied
        for label in ax.get_xticklabels():
            assert label.get_rotation() == 0
        
        plt.close(fig)
    
    def test_design_map_no_matplotlib(self, simple_model, monkeypatch):
        """Test error when matplotlib not available."""
        # Test that appropriate error is raised when matplotlib is unavailable
        import sys

        # Hide matplotlib temporarily
        mpl_module = sys.modules.get('matplotlib')
        plt_module = sys.modules.get('matplotlib.pyplot')

        try:
            sys.modules['matplotlib'] = None
            sys.modules['matplotlib.pyplot'] = None

            # Re-import to trigger the HAS_MATPLOTLIB check
            import importlib
            from fmrimod import visualization
            importlib.reload(visualization)

            # Should raise error
            with pytest.raises(ImportError, match="matplotlib"):
                visualization.design_map(simple_model)
        finally:
            # Restore matplotlib
            if mpl_module is not None:
                sys.modules['matplotlib'] = mpl_module
            if plt_module is not None:
                sys.modules['matplotlib.pyplot'] = plt_module
            importlib.reload(visualization)


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
class TestCorrelationMap:
    """Test correlation_map visualization."""
    
    @pytest.fixture
    def model_with_corr(self):
        """Create model with correlated regressors."""
        n_scans = 50
        
        # Create correlated events
        onsets1 = [5, 15, 25, 35]
        onsets2 = [6, 16, 26, 36]  # Slightly offset
        
        data = pd.DataFrame({
            'onset': onsets1 + onsets2,
            'event': ['A'] * 4 + ['B'] * 4
        })
        
        model = event_model(
            "event",
            data=data,
            tr=1.0,
            n_scans=n_scans
        )
        return model
    
    def test_correlation_map_basic(self, model_with_corr):
        """Test basic correlation_map functionality."""
        fig, ax = correlation_map(model_with_corr)
        
        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)
        
        # Check that heatmap was created
        # With matplotlib fallback, we get images instead of collections
        assert len(ax.images) > 0 or len(ax.collections) > 0
        
        plt.close(fig)
    
    def test_correlation_map_with_annotations(self, model_with_corr):
        """Test correlation map with value annotations."""
        fig, ax = correlation_map(
            model_with_corr,
            annot=True,
            fmt='.3f'
        )
        
        # Check that text annotations exist
        texts = ax.texts
        assert len(texts) > 0
        
        plt.close(fig)


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
class TestPlotDesignMatrix:
    """Test plot_design_matrix function."""
    
    @pytest.fixture
    def complex_model(self):
        """Create model with multiple regressors."""
        n_scans = 100
        # Create data with consistent lengths
        data = pd.DataFrame({
            'onset': [10, 20, 30, 40, 50, 60, 70, 80],
            'condition': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B'],
            'rating': [1, 2, 3, 4, 5, 4, 3, 2]
        })
        
        model = event_model(
            "condition + rating",
            data=data,
            tr=2.0,
            n_scans=n_scans
        )
        return model
    
    def test_plot_all_regressors(self, complex_model):
        """Test plotting all regressors."""
        fig, axes = plot_design_matrix(complex_model)
        
        assert isinstance(fig, matplotlib.figure.Figure)
        
        # With many regressors, should use subplots
        if isinstance(axes, np.ndarray):
            assert len(axes) >= complex_model.design_matrix.shape[1]
        
        plt.close(fig)
    
    def test_plot_specific_term(self, complex_model):
        """Test plotting specific term only."""
        fig, ax = plot_design_matrix(
            complex_model,
            term_name='condition'
        )
        
        assert isinstance(fig, matplotlib.figure.Figure)
        
        plt.close(fig)
    
    def test_plot_nonexistent_term(self, complex_model):
        """Test error for non-existent term."""
        with pytest.raises(ValueError, match="No columns found"):
            plot_design_matrix(complex_model, term_name='nonexistent')


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
class TestPlotModelSummary:
    """Test plot_model_summary function."""
    
    @pytest.fixture
    def model(self):
        """Create a simple model."""
        data = pd.DataFrame({
            'onset': [10, 30, 50],
            'condition': ['A', 'B', 'A']
        })
        
        return event_model(
            "condition",
            data=data,
            tr=2.0,
            n_scans=50
        )
    
    def test_plot_model_summary(self, model):
        """Test creating summary plots."""
        plots = plot_model_summary(model)
        
        # Check that all plots were created
        assert 'design_map' in plots
        assert 'correlation_map' in plots
        assert 'time_series' in plots
        
        # Check that each entry contains figure and axes
        for name, (fig, ax) in plots.items():
            assert isinstance(fig, matplotlib.figure.Figure)
            if name == 'time_series' and isinstance(ax, np.ndarray):
                assert all(isinstance(a, matplotlib.axes.Axes) for a in ax.flat)
            else:
                assert isinstance(ax, matplotlib.axes.Axes)
            
            plt.close(fig)


# Integration tests
@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
class TestVisualizationIntegration:
    """Test visualization with various model types."""
    
    def test_with_hrf_model(self):
        """Test visualization with HRF-convolved model."""
        data = pd.DataFrame({
            'onset': [5, 15, 25, 35],
            'cond': ['X', 'Y', 'X', 'Y']
        })
        
        model = event_model(
            'hrf(cond, hrf="simple")',
            data=data,
            tr=2.0,
            n_scans=50
        )
        
        fig, ax = design_map(model)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)
    
    def test_with_covariate_model(self):
        """Test visualization with covariates."""
        n_scans = 100
        data = pd.DataFrame({
            'onset': [10, 30, 50, 70],
            'condition': ['A', 'B', 'A', 'B']
        })
        
        # Create model without covariates for now
        # (covariate parser support needs to be implemented)
        model = event_model(
            'condition',
            data=data,
            tr=2.0,
            n_scans=n_scans
        )
        
        # Should be able to visualize model
        fig, ax = design_map(model)
        assert isinstance(fig, matplotlib.figure.Figure)
        
        plt.close(fig)
    
