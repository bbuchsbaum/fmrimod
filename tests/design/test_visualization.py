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
from fmrimod.visualization import correlation_map


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
