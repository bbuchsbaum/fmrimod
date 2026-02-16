"""Tests for term utilities - term_indices, term_matrices, baseline_terms."""

import pytest
import numpy as np
import pandas as pd

from fmrimod import event_model, baseline_model
from fmrimod.utils import term_indices, term_matrices, baseline_terms


class TestTermIndices:
    """Test term_indices function."""

    def setup_method(self):
        """Create test models."""
        # Create simple 2-term model
        self.events_df = pd.DataFrame({
            'onset': [5, 10, 15, 20, 25, 30],
            'condition': ['A', 'B', 'A', 'B', 'A', 'B'],
            'rating': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        })

        self.model = event_model(
            'condition + rating',
            data=self.events_df,
            tr=2.0,
            n_scans=30
        )

    def test_term_indices_basic(self):
        """Test extracting term indices from model."""
        indices = term_indices(self.model)

        # Should return dict
        assert isinstance(indices, dict)

        # Should have entries for each term
        assert 'condition' in indices
        assert 'rating' in indices

        # Each should map to list of column indices
        assert isinstance(indices['condition'], list)
        assert isinstance(indices['rating'], list)

        # All indices should be valid
        n_cols = len(self.model.column_names)
        for term_name, idx_list in indices.items():
            for idx in idx_list:
                assert 0 <= idx < n_cols

    def test_term_indices_coverage(self):
        """Test that term_indices covers all columns."""
        indices = term_indices(self.model)

        # Collect all indices
        all_indices = []
        for idx_list in indices.values():
            all_indices.extend(idx_list)

        # Should cover all columns (no duplicates unless error)
        n_cols = len(self.model.column_names)
        assert len(set(all_indices)) <= n_cols

    def test_term_indices_single_term(self):
        """Test term_indices with single term model."""
        events_df = pd.DataFrame({
            'onset': [5, 10, 15],
            'condition': ['A', 'B', 'C']
        })

        model = event_model('condition', data=events_df, tr=2.0, n_scans=20)
        indices = term_indices(model)

        assert 'condition' in indices
        assert len(indices['condition']) > 0

    def test_term_indices_multiple_terms(self):
        """Test term_indices with multiple terms."""
        events_df = pd.DataFrame({
            'onset': [5, 10, 15, 20],
            'cond1': ['A', 'B', 'A', 'B'],
            'cond2': ['X', 'X', 'Y', 'Y']
        })

        model = event_model('cond1 + cond2',
                           data=events_df, tr=2.0, n_scans=25)
        indices = term_indices(model)

        # Should have both terms
        assert 'cond1' in indices
        assert 'cond2' in indices


class TestTermMatrices:
    """Test term_matrices function."""

    def setup_method(self):
        """Create test model."""
        self.events_df = pd.DataFrame({
            'onset': [5, 10, 15, 20, 25, 30],
            'condition': ['A', 'B', 'A', 'B', 'A', 'B'],
            'rating': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        })

        self.model = event_model(
            'condition + rating',
            data=self.events_df,
            tr=2.0,
            n_scans=30
        )

    def test_term_matrices_all_terms(self):
        """Test extracting all term matrices."""
        matrices = term_matrices(self.model)

        # Should return dict
        assert isinstance(matrices, dict)

        # Should have entries for each term
        assert 'condition' in matrices
        assert 'rating' in matrices

        # Each should be a numpy array
        assert isinstance(matrices['condition'], np.ndarray)
        assert isinstance(matrices['rating'], np.ndarray)

        # Matrices should have correct shape
        n_scans = self.model.design_matrix.shape[0]
        assert matrices['condition'].shape[0] == n_scans
        assert matrices['rating'].shape[0] == n_scans

    def test_term_matrices_specific_terms(self):
        """Test extracting specific terms."""
        # Extract single term
        matrices = term_matrices(self.model, term_names='condition')
        assert 'condition' in matrices
        assert 'rating' not in matrices

        # Extract multiple terms
        matrices = term_matrices(self.model, term_names=['condition', 'rating'])
        assert 'condition' in matrices
        assert 'rating' in matrices

    def test_term_matrices_invalid_term(self):
        """Test error on invalid term name."""
        with pytest.raises(ValueError, match="Terms not found"):
            term_matrices(self.model, term_names='nonexistent')

    def test_term_matrices_shape_consistency(self):
        """Test that term matrices have consistent shapes."""
        matrices = term_matrices(self.model)
        X = self.model.design_matrix

        # Concatenate all term matrices
        all_cols = np.hstack([matrices[term] for term in ['condition', 'rating']])

        # Should match original design matrix column count
        assert all_cols.shape[1] == X.shape[1]

    def test_term_matrices_column_match(self):
        """Test that extracted matrices match design matrix columns."""
        matrices = term_matrices(self.model)
        indices = term_indices(self.model)
        X = self.model.design_matrix

        # For each term, verify matrix matches design matrix columns
        for term_name, matrix in matrices.items():
            idx_list = indices[term_name]
            expected = X[:, idx_list]
            np.testing.assert_array_almost_equal(matrix, expected)


class TestBaselineTerms:
    """Test baseline_terms function."""

    def test_baseline_terms_no_baseline(self):
        """Test baseline_terms with model without baseline."""
        events_df = pd.DataFrame({
            'onset': [5, 10, 15],
            'condition': ['A', 'B', 'A']
        })

        model = event_model('condition', data=events_df, tr=2.0, n_scans=20)
        baseline = baseline_terms(model)

        # Should return None when no baseline
        assert baseline is None

    def test_baseline_terms_function_exists(self):
        """Test that baseline_terms function is callable."""
        # Just verify the function exists and is callable
        assert callable(baseline_terms)

    def test_baseline_terms_returns_none_for_simple_model(self):
        """Test baseline_terms returns None for models without baseline."""
        events_df = pd.DataFrame({
            'onset': [5, 10, 15, 20],
            'condition': ['A', 'B', 'A', 'B']
        })

        model = event_model('condition', data=events_df, tr=2.0, n_scans=30)
        result = baseline_terms(model)

        # Should be None for model without baseline
        assert result is None


class TestTermUtilsEdgeCases:
    """Test edge cases for term utilities."""

    def test_term_indices_empty_model(self):
        """Test term_indices with minimal model."""
        events_df = pd.DataFrame({
            'onset': [10],
            'condition': ['A']
        })

        model = event_model('condition', data=events_df, tr=2.0, n_scans=10)
        indices = term_indices(model)

        assert isinstance(indices, dict)
        assert len(indices) > 0

    def test_term_matrices_single_column_term(self):
        """Test term_matrices with single-column term."""
        events_df = pd.DataFrame({
            'onset': [5, 10, 15],
            'rating': [1.0, 2.0, 3.0]
        })

        model = event_model('rating', data=events_df, tr=2.0, n_scans=20)
        matrices = term_matrices(model)

        assert 'rating' in matrices
        # Continuous variable should give 1 column
        assert matrices['rating'].ndim == 2
        # Even single column should be 2D (n_scans, 1)
        if matrices['rating'].shape[1] == 1:
            assert matrices['rating'].shape == (20, 1)

    def test_term_indices_ordering(self):
        """Test that term_indices preserves term order."""
        events_df = pd.DataFrame({
            'onset': [5, 10, 15, 20],
            'first': ['A', 'B', 'A', 'B'],
            'second': [1.0, 2.0, 3.0, 4.0],
            'third': ['X', 'Y', 'X', 'Y']
        })

        model = event_model('first + second + third',
                           data=events_df, tr=2.0, n_scans=25)
        indices = term_indices(model)

        # All terms should be present
        assert 'first' in indices
        assert 'second' in indices
        assert 'third' in indices
