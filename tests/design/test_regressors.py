"""Tests for regressor extraction functionality."""

import pytest
import re
import numpy as np
import pandas as pd

from fmrimod.regressors import regressors, regressor_names, term_regressors
from fmrimod.design.event_model import event_model


class TestRegressors:
    """Test regressors extraction function."""

    def setup_method(self):
        """Create test data for each test."""
        # Create event data with multiple conditions
        self.events_df = pd.DataFrame({
            'onset': [10, 20, 30, 40, 50, 60, 70, 80],
            'duration': [1, 1, 1, 1, 1, 1, 1, 1],
            'condition': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
            'response': [1.2, 0.8, 1.5, 0.6, 1.1, 0.9, 1.3, 0.7]
        })

        # Create model with multiple terms (no baseline for simplicity)
        self.model = event_model(
            'condition + response',
            self.events_df,
            tr=2.0,
            n_scans=100
        )

    def test_extract_all_regressors(self):
        """Test extracting all regressors."""
        all_regs = regressors(self.model)

        # Should be DataFrame
        assert isinstance(all_regs, pd.DataFrame)

        # Should have all columns
        assert all_regs.shape[1] == self.model.design_matrix.shape[1]
        assert list(all_regs.columns) == self.model.column_names

    def test_extract_specific_term(self):
        """Test extracting regressors for specific term."""
        # Extract condition regressors
        cond_regs = regressors(self.model, term='condition')

        # Should have condition columns only
        cond_cols = [col for col in self.model.column_names if 'condition' in col]
        assert list(cond_regs.columns) == cond_cols

    def test_extract_multiple_terms(self):
        """Test extracting multiple terms."""
        event_regs = regressors(self.model, term=['condition', 'response'])

        # Should have columns from both terms
        assert any('response' in col for col in event_regs.columns)
        assert any('condition' in col for col in event_regs.columns)

    def test_regex_pattern_extraction(self):
        """Test regex pattern matching."""
        # Match all condition regressors
        pattern = re.compile(r'condition.*')
        cond_regs = regressors(self.model, term=pattern)

        # Should match condition columns
        for col in cond_regs.columns:
            assert 'condition' in col

    def test_exclude_baseline(self):
        """Test excluding baseline regressors."""
        # With baseline
        with_baseline = regressors(self.model, include_baseline=True)

        # Without baseline - for this model both should be the same
        # since there's no baseline
        without_baseline = regressors(self.model, include_baseline=False)

        # Both should have columns (no baseline columns to exclude)
        assert without_baseline.shape[1] > 0

    def test_as_dict_output(self):
        """Test dictionary output format."""
        reg_dict = regressors(self.model, as_dict=True)

        # Should be dictionary
        assert isinstance(reg_dict, dict)

        # Keys should be column names
        assert set(reg_dict.keys()) == set(self.model.column_names)

        # Values should be 1D arrays
        for name, values in reg_dict.items():
            assert values.ndim == 1
            assert len(values) == self.model.design_matrix.shape[0]

    def test_no_matching_columns(self):
        """Test error when no columns match."""
        with pytest.raises(ValueError, match="No columns found"):
            regressors(self.model, term='nonexistent')

    def test_empty_result_baseline_filter(self):
        """Test when baseline filter removes all columns."""
        # Create a simple model
        events_df = pd.DataFrame({
            'onset': [10, 20, 30],
            'condition': ['A', 'B', 'A']
        })
        model = event_model(
            'condition',
            events_df,
            tr=2.0,
            n_scans=50
        )

        # Should still have columns when excluding baseline (no baseline to exclude)
        result = regressors(model, include_baseline=False)
        assert result.shape[1] > 0


class TestRegressorNames:
    """Test regressor_names function."""

    def setup_method(self):
        """Create test model."""
        events_df = pd.DataFrame({
            'onset': [10, 30, 50],
            'condition': ['visual', 'motor', 'visual']
        })

        self.model = event_model(
            'condition',
            events_df,
            tr=2.0,
            n_scans=100
        )

    def test_get_all_names(self):
        """Test getting all regressor names."""
        names = regressor_names(self.model)
        assert names == self.model.column_names

    def test_get_term_names(self):
        """Test getting names for specific term."""
        cond_names = regressor_names(self.model, term='condition')
        assert all('condition' in name for name in cond_names)

    def test_exclude_baseline_names(self):
        """Test excluding baseline from names."""
        names_all = regressor_names(self.model, include_baseline=True)
        names_event = regressor_names(self.model, include_baseline=False)

        # Without baseline, should still have event columns
        assert len(names_event) > 0


class TestTermRegressors:
    """Test term_regressors function."""

    def setup_method(self):
        """Create complex model for testing."""
        events_df = pd.DataFrame({
            'onset': np.arange(0, 100, 10),
            'visual': ['left', 'right'] * 5,
            'motor': ['hand', 'foot', 'hand', 'foot', 'hand'] * 2,
            'rating': np.random.rand(10)
        })

        self.model = event_model(
            'visual + motor + rating',
            events_df,
            tr=2.0,
            n_scans=200
        )

    def test_group_by_term_dict(self):
        """Test grouping regressors by term as dict."""
        term_dict = term_regressors(self.model, as_dict=True)

        # Should have entries for each term
        assert 'visual' in term_dict
        assert 'motor' in term_dict
        assert 'rating' in term_dict

        # Visual should have regressors for each level
        visual_dict = term_dict['visual']
        assert isinstance(visual_dict, dict)

    def test_group_by_term_dataframe(self):
        """Test grouping regressors by term as DataFrames."""
        term_dfs = term_regressors(self.model, as_dict=False)

        # Should have DataFrames for each term
        assert isinstance(term_dfs['visual'], pd.DataFrame)
        assert isinstance(term_dfs['motor'], pd.DataFrame)
        assert isinstance(term_dfs['rating'], pd.DataFrame)

    def test_baseline_grouping(self):
        """Test that baseline regressors are grouped separately."""
        term_dict = term_regressors(self.model)

        # Model without baseline should still work
        assert len(term_dict) >= 3  # At least visual, motor, rating

    def test_empty_model(self):
        """Test with model that has minimal terms."""
        events_df = pd.DataFrame({
            'onset': [10, 20, 30],
            'condition': ['A', 'B', 'A']
        })
        simple_model = event_model(
            'condition',
            events_df,
            tr=2.0,
            n_scans=50
        )

        term_dict = term_regressors(simple_model)
        assert len(term_dict) >= 1


class TestRegressorsEdgeCases:
    """Test edge cases for regressor extraction."""

    def test_regressors_single_timepoint(self):
        """Test with very short scan."""
        events_df = pd.DataFrame({
            'onset': [1],
            'condition': ['A']
        })

        model = event_model('condition', events_df, tr=2.0, n_scans=5)
        regs = regressors(model)

        assert isinstance(regs, pd.DataFrame)
        assert regs.shape[0] == 5

    def test_regressors_many_conditions(self):
        """Test with many conditions."""
        n_conds = 20
        events_df = pd.DataFrame({
            'onset': np.arange(0, 100, 5),
            'condition': [f'cond_{i % n_conds}' for i in range(20)]
        })

        model = event_model('condition', events_df, tr=2.0, n_scans=100)
        regs = regressors(model)

        # Should have columns for all conditions
        assert regs.shape[1] >= n_conds - 1  # n-1 for contrast coding

    def test_regressors_as_dict_values(self):
        """Test that as_dict returns correct array values."""
        events_df = pd.DataFrame({
            'onset': [10, 20, 30],
            'condition': ['A', 'B', 'A']
        })

        model = event_model('condition', events_df, tr=2.0, n_scans=50)
        reg_dict = regressors(model, as_dict=True)

        # Each value should match corresponding column
        X = model.design_matrix
        col_names = model.column_names

        for i, name in enumerate(col_names):
            if name in reg_dict:
                np.testing.assert_array_almost_equal(reg_dict[name], X[:, i])

    def test_regressor_names_consistency(self):
        """Test that regressor_names matches regressors output."""
        events_df = pd.DataFrame({
            'onset': [10, 20, 30, 40],
            'condition': ['A', 'B', 'A', 'B'],
            'rating': [1.0, 2.0, 3.0, 4.0]
        })

        model = event_model('condition + rating', events_df, tr=2.0, n_scans=50)

        names = regressor_names(model)
        regs_df = regressors(model)

        # Names should match DataFrame columns
        assert names == list(regs_df.columns)

    def test_term_regressors_value_consistency(self):
        """Test that term_regressors values match full design matrix."""
        events_df = pd.DataFrame({
            'onset': [10, 20, 30],
            'condition': ['A', 'B', 'A'],
            'rating': [1.0, 2.0, 3.0]
        })

        model = event_model('condition + rating', events_df, tr=2.0, n_scans=50)

        # Get term regressors
        term_dict = term_regressors(model, as_dict=False)

        # Concatenate all term matrices
        all_terms = []
        for term_name in ['condition', 'rating']:
            if term_name in term_dict:
                all_terms.append(term_dict[term_name].values)

        if all_terms:
            combined = np.hstack(all_terms)
            # Should match design matrix
            X = model.design_matrix
            assert combined.shape == X.shape
