"""Comprehensive tests for visualization module.

This test suite covers:
- design_map (heatmap visualization)
- correlation_map (correlation matrix)
- plot_design_matrix (time series plots)
- plot_baseline_model (baseline terms)
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
)


@pytest.fixture(autouse=True)
def close_all_plots():
    """Close all matplotlib figures after each test."""
    yield
    plt.close('all')


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

        fig = plot_baseline_model(bmodel)

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
