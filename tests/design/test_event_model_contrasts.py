"""Test EventModel contrast_weights handler."""

import pytest
import numpy as np
import pandas as pd

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fmrimod import event_model
from fmrimod.contrast import (
    contrast_weights,
    plot_contrasts,
    pair_contrast,
    unit_contrast,
    contrast_set,
)
from fmrimod.contrast.contrast_spec import Formula
from fmrimod.formula.base import Term


@pytest.fixture(autouse=True)
def close_all_plots():
    """Close all matplotlib figures after each test."""
    yield
    plt.close('all')


@pytest.fixture
def simple_data():
    """Create simple test data with 3-level categorical variable."""
    return pd.DataFrame({
        'onset': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        'condition': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C'],
    })


@pytest.fixture
def model_with_contrasts(simple_data):
    """Create EventModel with contrast_specs on a term."""
    # Create a Term with contrast specifications
    term = Term('condition')
    term.contrast_specs = contrast_set(
        pair_contrast(Formula("A"), Formula("B"), name='A_vs_B'),
        pair_contrast(Formula("A"), Formula("C"), name='A_vs_C'),
        unit_contrast(Formula("A"), name='just_A'),
    )

    return event_model([term], data=simple_data, tr=2.0, n_scans=20)


class TestEventModelContrastWeights:
    """Test contrast_weights dispatch for EventModel."""
    
    def test_contrast_weights_returns_nested_dict(self, model_with_contrasts):
        """Test that contrast_weights returns nested dict structure."""
        result = contrast_weights(model_with_contrasts)
        
        assert isinstance(result, dict)
        assert 'condition' in result
        assert isinstance(result['condition'], dict)
    
    def test_contrast_weights_extracts_all_contrasts(self, model_with_contrasts):
        """Test that all contrast specs are extracted."""
        result = contrast_weights(model_with_contrasts)

        condition_contrasts = result['condition']
        assert 'A_vs_B' in condition_contrasts
        assert 'A_vs_C' in condition_contrasts
        assert 'just_A' in condition_contrasts
    
    def test_contrast_weights_have_full_design_matrix_size(self, model_with_contrasts):
        """Test that contrast weights are in full design matrix space."""
        result = contrast_weights(model_with_contrasts)
        
        ncond = model_with_contrasts.design_matrix.shape[1]
        
        for term_name, term_contrasts in result.items():
            for contrast_name, contrast_obj in term_contrasts.items():
                assert contrast_obj.weights.shape[0] == ncond
                assert hasattr(contrast_obj, 'offset_weights')
                assert contrast_obj.offset_weights.shape[0] == ncond
    
    def test_contrast_weights_have_correct_nonzero_entries(self, model_with_contrasts):
        """Test that weights are placed at correct indices."""
        result = contrast_weights(model_with_contrasts)
        
        # Get the A_vs_B contrast
        a_vs_b = result['condition']['A_vs_B']
        
        # Should have nonzero weights only in condition term columns
        # The condition term should have 3 columns (A, B, C)
        nonzero_indices = np.where(np.abs(a_vs_b.offset_weights).sum(axis=1) > 1e-10)[0]
        
        # Should have exactly 2 nonzero entries (for A and B)
        assert len(nonzero_indices) == 2
    
    def test_contrast_object_structure(self, model_with_contrasts):
        """Test that contrast objects have expected structure."""
        result = contrast_weights(model_with_contrasts)

        a_vs_b = result['condition']['A_vs_B']

        # Should have weights and name
        assert hasattr(a_vs_b, 'weights')
        assert hasattr(a_vs_b, 'name')
        assert a_vs_b.name == 'A_vs_B'


class TestEventModelPlotContrasts:
    """Test plot_contrasts dispatch for EventModel."""
    
    def test_plot_contrasts_with_event_model(self, model_with_contrasts):
        """Test that plot_contrasts works with EventModel."""
        fig = plot_contrasts(model_with_contrasts)
        
        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) > 0
    
    def test_plot_contrasts_shows_all_contrasts(self, model_with_contrasts):
        """Test that all contrasts are shown in the plot."""
        fig = plot_contrasts(model_with_contrasts)
        
        ax = fig.axes[0]
        y_labels = [label.get_text() for label in ax.get_yticklabels()]
        
        # Should have labels for all three contrasts
        assert any('A_vs_B' in label for label in y_labels)
        assert any('A_vs_C' in label for label in y_labels)
        assert any('just_A' in label for label in y_labels)


class TestEventModelWithoutContrasts:
    """Test EventModel without contrast_specs."""
    
    def test_contrast_weights_returns_empty_dict(self, simple_data):
        """Test that model without contrast_specs returns empty dict."""
        model = event_model("condition", data=simple_data, tr=2.0, n_scans=20)
        
        result = contrast_weights(model)
        
        assert isinstance(result, dict)
        assert len(result) == 0
