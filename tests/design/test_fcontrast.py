"""Tests for F-contrast functionality."""

import pytest
import numpy as np
import pandas as pd

from fmrimod.events.factor import EventFactor
from fmrimod.events.variable import EventVariable
from fmrimod.contrast import Fcontrasts
from fmrimod.events.term import EventTerm, create_interaction
from fmrimod.design.event_model import EventModel


class TestFcontrasts:
    """Test F-contrast generation."""
    
    def test_fcontrasts_single_factor(self):
        """Test F-contrasts for single categorical event."""
        onsets = np.array([1, 2, 3, 4, 5, 6])
        values = ['A', 'B', 'C', 'A', 'B', 'C']
        
        event = EventFactor('condition', onsets, values)
        fcon = Fcontrasts(event)
        
        # Should have one contrast matrix
        assert len(fcon) == 1
        assert 'condition' in fcon
        
        # Check dimensions - 3 levels -> 2 contrast columns
        con_mat = fcon['condition']
        assert con_mat.shape == (3, 2)
        
        # Check sum-to-zero property
        assert np.allclose(con_mat.sum(axis=0), 0)
        
    def test_fcontrasts_continuous_event(self):
        """Test that continuous events return empty F-contrasts."""
        onsets = np.array([1, 2, 3, 4])
        values = np.array([1.5, 2.3, 3.1, 1.8])
        
        event = EventVariable('rating', onsets, values)
        fcon = Fcontrasts(event)
        
        # Should be empty for continuous
        assert len(fcon) == 0
        
    def test_fcontrasts_interaction_term(self):
        """Test F-contrasts for interaction term."""
        onsets = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        
        # Two factors
        cond_values = ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
        group_values = ['X', 'X', 'Y', 'Y', 'X', 'X', 'Y', 'Y']
        
        cond_event = EventFactor('condition', onsets, cond_values)
        group_event = EventFactor('group', onsets, group_values)
        
        # Create interaction term
        term = create_interaction(cond_event, group_event)
        fcon = Fcontrasts(term)
        
        # Should have main effects and interaction
        assert len(fcon) == 3
        assert 'condition' in fcon
        assert 'group' in fcon
        assert 'condition:group' in fcon
        
        # Check dimensions
        assert fcon['condition'].shape == (4, 1)  # 2x2 = 4 cells, 1 df
        assert fcon['group'].shape == (4, 1)      # 2x2 = 4 cells, 1 df
        assert fcon['condition:group'].shape == (4, 1)  # interaction has 1 df
        
    def test_fcontrasts_three_way(self):
        """Test F-contrasts with three categorical factors."""
        onsets = np.arange(12)
        
        # Three factors (2x2x3)
        factor1 = ['A', 'B'] * 6
        factor2 = ['X', 'X', 'Y', 'Y'] * 3
        factor3 = ['1', '1', '1', '1', '2', '2', '2', '2', '3', '3', '3', '3']
        
        ev1 = EventFactor('f1', onsets, factor1)
        ev2 = EventFactor('f2', onsets, factor2)
        ev3 = EventFactor('f3', onsets, factor3)
        
        # Create term with all three
        term = EventTerm([ev1, ev2, ev3], name='f1:f2:f3')
        fcon = Fcontrasts(term, max_inter=3)
        
        # Should have all main effects, 2-way, and 3-way interactions
        expected = ['f1', 'f2', 'f3', 'f1:f2', 'f1:f3', 'f2:f3', 'f1:f2:f3']
        assert sorted(fcon.keys()) == sorted(expected)
        
        # Check dimensions (2x2x3 = 12 cells)
        assert fcon['f1'].shape == (12, 1)      # 2 levels - 1
        assert fcon['f2'].shape == (12, 1)      # 2 levels - 1
        assert fcon['f3'].shape == (12, 2)      # 3 levels - 1
        assert fcon['f1:f2:f3'].shape == (12, 2)  # (2-1)*(2-1)*(3-1) = 2
        
    def test_fcontrasts_max_interaction_limit(self):
        """Test max_inter parameter limits interaction order."""
        onsets = np.arange(8)
        
        # Three factors
        f1 = ['A', 'B'] * 4
        f2 = ['X', 'Y'] * 4
        f3 = ['1', '1', '2', '2'] * 2
        
        ev1 = EventFactor('f1', onsets, f1)
        ev2 = EventFactor('f2', onsets, f2)
        ev3 = EventFactor('f3', onsets, f3)
        
        term = EventTerm([ev1, ev2, ev3])
        
        # Limit to 2-way interactions
        fcon = Fcontrasts(term, max_inter=2)
        
        # Should not have 3-way interaction
        assert 'f1:f2:f3' not in fcon
        assert 'f1:f2' in fcon
        assert 'f1:f3' in fcon
        assert 'f2:f3' in fcon
        
    def test_fcontrasts_event_model(self):
        """Test F-contrasts for full event model."""
        from fmrimod.sampling import SamplingFrame
        from fmrimod.formula.base import Term

        events = {
            'condition': EventFactor(
                name='condition',
                onsets=[1, 2, 3, 4, 5, 6],
                values=['A', 'B', 'A', 'B', 'A', 'B'],
                durations=1
            )
        }

        sf = SamplingFrame(tr=2.0, n_scans=10)
        terms = [Term('condition')]

        model = EventModel(
            terms=terms,
            events=events,
            sampling_info=sf
        )

        fcon = Fcontrasts(model)

        # Should have contrast for the condition factor
        # EventModel returns keys in format "term#contrast"
        assert 'condition#condition' in fcon
        assert fcon['condition#condition'].shape[1] == 1  # 2 levels -> 1 contrast
            
    def test_fcontrasts_kronecker_product(self):
        """Test that Kronecker product is computed correctly."""
        onsets = np.array([1, 2, 3, 4])
        
        # 2x2 factorial design
        factor1 = ['A', 'B', 'A', 'B'] 
        factor2 = ['X', 'X', 'Y', 'Y']
        
        ev1 = EventFactor('f1', onsets, factor1)
        ev2 = EventFactor('f2', onsets, factor2)
        
        term = create_interaction(ev1, ev2)
        fcon = Fcontrasts(term)
        
        # Manually compute what main effect for f1 should be
        # C matrix for f2 (all ones), D matrix for f1 (contrast)
        C2 = np.ones((2, 1))
        D1 = np.array([[1], [-1]])  # Sum-to-zero for 2 levels
        
        expected_f1 = np.kron(D1, C2)  # Kronecker product
        assert np.allclose(fcon['f1'], expected_f1)
        
    def test_fcontrasts_single_level_factor(self):
        """Test that single-level factors are handled gracefully."""
        onsets = np.array([1, 2, 3, 4])
        values = ['A', 'A', 'A', 'A']  # Only one level
        
        event = EventFactor('constant', onsets, values)
        fcon = Fcontrasts(event)
        
        # Should be empty - can't create contrasts for single level
        assert len(fcon) == 0
        
    def test_fcontrasts_mixed_term(self):
        """Test F-contrasts for term with both categorical and continuous."""
        onsets = np.arange(8)
        
        # Categorical and continuous
        conditions = ['A', 'B'] * 4
        ratings = np.random.randn(8)
        
        cat_event = EventFactor('condition', onsets, conditions)
        cont_event = EventVariable('rating', onsets, ratings)
        
        # Mixed term
        term = EventTerm([cat_event, cont_event])
        fcon = Fcontrasts(term)
        
        # Should only have contrast for categorical part
        assert len(fcon) == 1
        assert 'condition' in fcon
        assert 'rating' not in fcon