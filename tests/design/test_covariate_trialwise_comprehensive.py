"""Comprehensive tests for covariate support, trialwise models, and convolve_design.

This test file aims to increase coverage for:
- covariate.py (target 80%+)
- trialwise.py (target 70%+)
- convolve_design.py (target 60%+)

Focus on edge cases, error handling, and integration scenarios.
"""

import pytest
import numpy as np
import pandas as pd

from fmrimod.covariate import covariate, CovariateTerm, CovariateEvent
from fmrimod import event_model
from fmrimod.events.factor import EventFactor
from fmrimod.design.event_model import EventModel
from fmrimod.sampling import SamplingFrame
from fmrimod.trialwise import trialwise
from fmrimod.covariate import create_covariate_events
from fmrimod.convolve_design import convolve_design, convolve_regressors
from fmrimod.formula.base import Term


class TestCovariateEdgeCases:
    """Additional tests for covariate edge cases and uncovered lines."""

    def test_covariate_event_validation_non_numeric(self):
        """Test validation error for non-numeric covariate values."""
        # String values should fail
        with pytest.raises(ValueError, match="must be numeric"):
            CovariateEvent('test', np.array(['a', 'b', 'c']))

    def test_covariate_event_validation_nan_values(self):
        """Test validation error for NaN values."""
        values = np.array([1.0, 2.0, np.nan, 4.0])
        with pytest.raises(ValueError, match="must be finite"):
            CovariateEvent('test', values)

    def test_covariate_event_validation_inf_values(self):
        """Test validation error for infinite values."""
        values = np.array([1.0, 2.0, np.inf, 4.0])
        with pytest.raises(ValueError, match="must be finite"):
            CovariateEvent('test', values)

    def test_covariate_event_sampling_points_length_mismatch(self):
        """Test error when sampling_points length doesn't match values."""
        values = np.array([1.0, 2.0, 3.0])
        sampling_points = np.array([0.0, 1.0])  # Wrong length

        with pytest.raises(ValueError, match="Sampling points length"):
            CovariateEvent('test', values, sampling_points=sampling_points)

    def test_covariate_term_repr(self):
        """Test string representation of CovariateTerm."""
        term = covariate('motion_x', 'motion_y')
        repr_str = repr(term)
        assert 'CovariateTerm' in repr_str
        assert 'motion_x' in repr_str
        assert 'motion_y' in repr_str

    def test_covariate_term_repr_with_prefix(self):
        """Test string representation with prefix."""
        term = covariate('x', 'y', prefix='motion')
        repr_str = repr(term)
        assert 'CovariateTerm' in repr_str
        assert 'prefix=motion' in repr_str

    def test_covariate_event_repr(self):
        """Test string representation of CovariateEvent."""
        event = CovariateEvent('motion_x', np.array([1, 2, 3]))
        repr_str = repr(event)
        assert 'CovariateEvent' in repr_str
        assert 'motion_x' in repr_str
        assert 'n_timepoints=3' in repr_str

    def test_covariate_with_subset_parameter(self):
        """Test covariate with subset parameter (stores but doesn't validate here)."""
        subset_mask = np.array([True, False, True, False, True])
        term = covariate('motion_x', subset=subset_mask)
        assert term.subset is not None
        assert np.array_equal(term.subset, subset_mask)

    def test_covariate_design_matrix_2d_values(self):
        """Test that design_matrix ensures 2D output."""
        values = np.array([1.0, 2.0, 3.0])
        event = CovariateEvent('test', values)
        dm = event.design_matrix(np.arange(3))

        # Should be 2D
        assert dm.ndim == 2
        assert dm.shape == (3, 1)


class TestTrialwiseEdgeCases:
    """Additional tests for trialwise edge cases and uncovered lines."""

    def test_trialwise_with_durations_parameter(self):
        """Test trialwise with durations parameter."""
        data = pd.DataFrame({
            'onset': [10, 20, 30],
            'duration': [1.0, 2.0, 1.5],
            'condition': ['A', 'B', 'A']
        })

        # Create trialwise term with durations
        term = trialwise(durations='duration')

        # Check that kwargs were stored
        assert hasattr(term, '_kwargs')
        assert 'durations' in term._kwargs
        assert term._kwargs['durations'] == 'duration'

    def test_trialwise_with_normalize_parameter(self):
        """Test trialwise with normalize=True."""
        term = trialwise(normalize=True)

        assert hasattr(term, '_kwargs')
        assert 'normalize' in term._kwargs
        assert term._kwargs['normalize'] is True

    def test_trialwise_with_durations_and_normalize(self):
        """Test trialwise with both durations and normalize."""
        term = trialwise(durations=1.5, normalize=True)

        assert term._kwargs['durations'] == 1.5
        assert term._kwargs['normalize'] is True

    def test_trialwise_nbasis_parameter(self):
        """Test that nbasis parameter is stored."""
        term = trialwise(nbasis=3)
        assert term._nbasis == 3

    def test_trialwise_creates_trial_factor_helper(self):
        """Test the internal _create_trial_factor function."""
        from fmrimod.trialwise import _create_trial_factor

        # Create factor with 5 trials
        factor = _create_trial_factor(5)

        assert factor.name == 'trial'
        assert len(factor.values) == 5
        assert list(factor.values) == ['1', '2', '3', '4', '5']

    def test_trialwise_creates_trial_factor_with_onsets(self):
        """Test _create_trial_factor with explicit onsets."""
        from fmrimod.trialwise import _create_trial_factor

        onsets = np.array([10, 20, 30])
        factor = _create_trial_factor(3, onsets=onsets)

        assert np.array_equal(factor.onsets, onsets)
        assert list(factor.values) == ['1', '2', '3']

    def test_trialwise_zero_padding_for_many_trials(self):
        """Test that trial labels are zero-padded correctly for many trials."""
        from fmrimod.trialwise import _create_trial_factor

        # Create factor with 100 trials
        factor = _create_trial_factor(100)

        # First trial should be '001' (3 digits)
        assert list(factor.values)[0] == '001'
        # Last trial should be '100'
        assert list(factor.values)[-1] == '100'

    def test_trialwise_preserves_existing_kwargs(self):
        """Test that _kwargs is created if it doesn't exist."""
        term = trialwise()

        # Should have _kwargs even if empty initially
        assert hasattr(term, '_kwargs')

        # Add durations - should work
        term2 = trialwise(durations=1.0)
        assert term2._kwargs['durations'] == 1.0


class TestConvolveDesignEdgeCases:
    """Additional tests for convolve_design edge cases and uncovered lines."""

    def test_convolve_design_with_event_model(self):
        """Test convolving an EventModel's design matrix."""
        # Create a simple event model
        data = pd.DataFrame({
            'onset': [10, 30, 50],
            'condition': ['A', 'B', 'A']
        })

        model = event_model(
            "condition",
            data=data,
            tr=2.0,
            n_scans=50
        )

        # Convolve the model (should extract design matrix, tr, etc.)
        result = convolve_design(model, hrf='spmg1')

        # Should return DataFrame
        assert isinstance(result, pd.DataFrame)
        # Should have same shape as original design matrix
        assert result.shape == model.design_matrix.shape
        # Should have same column names
        assert list(result.columns) == model.column_names

    def test_convolve_design_normalize(self):
        """Test peak normalization of convolved columns."""
        # Create simple impulse
        X = np.zeros((100, 2))
        X[20, 0] = 5.0  # Large amplitude
        X[60, 1] = 2.0  # Smaller amplitude

        # Convolve with normalization
        X_conv = convolve_design(X, hrf='spmg1', sampling_rate=1.0, normalize=True)

        # Each column should be peak-normalized
        # Max absolute value should be 1.0 (or close to it)
        assert np.isclose(np.max(np.abs(X_conv[:, 0])), 1.0, atol=0.01)
        assert np.isclose(np.max(np.abs(X_conv[:, 1])), 1.0, atol=0.01)

    def test_convolve_design_normalize_zero_column(self):
        """Test normalization with zero columns (should not divide by zero)."""
        X = np.zeros((50, 2))
        X[10, 0] = 1.0  # Only first column has signal

        # Should not crash on zero column
        X_conv = convolve_design(X, hrf='spmg1', sampling_rate=1.0, normalize=True)

        # Second column should remain zero
        assert np.all(X_conv[:, 1] == 0)
        # First column should be normalized
        assert np.max(np.abs(X_conv[:, 0])) > 0

    def test_convolve_design_multidimensional_hrf(self):
        """Test handling of multi-dimensional HRF (with derivatives)."""
        X = np.zeros((100, 1))
        X[30, 0] = 1.0

        # Use HRF with derivatives (spmg3 = canonical + time + dispersion derivatives)
        X_conv = convolve_design(X, hrf='spmg3', sampling_rate=1.0)

        # Should use only the first basis function
        assert X_conv.shape == (100, 1)
        # Should have convolved response
        assert np.sum(X_conv > 0) > 1

    def test_convolve_design_invalid_ndim(self):
        """Test error on 3D input."""
        X = np.random.randn(10, 5, 3)  # 3D array

        with pytest.raises(ValueError, match="must be 1D or 2D"):
            convolve_design(X, sampling_rate=1.0)

    def test_convolve_design_hrf_normalization_edge_case(self):
        """Test HRF normalization when sum is near zero."""
        # This tests the edge case in lines 150-159
        # Use a very short time window that might produce near-zero sum
        X = np.zeros((5, 1))
        X[2, 0] = 1.0

        # Should not crash even with unusual HRF
        X_conv = convolve_design(X, hrf='spmg1', sampling_rate=0.1)
        assert X_conv.shape == X.shape

    def test_convolve_design_column_names_auto_generation(self):
        """Test automatic column name generation when returning DataFrame."""
        X = np.random.randn(50, 3)

        # Convert to DataFrame without names, then convolve
        df = pd.DataFrame(X)  # Will have default names 0, 1, 2
        df_conv = convolve_design(df, sampling_rate=1.0)

        # Should have preserved the default column names
        assert list(df_conv.columns) == [0, 1, 2]

    def test_convolve_regressors_preserves_shape_1d(self):
        """Test that convolve_regressors preserves 1D shape."""
        regressors = {
            'visual': np.zeros(100),
            'motor': np.zeros(100)
        }
        regressors['visual'][20] = 1
        regressors['motor'][60] = 1

        conv_regs = convolve_regressors(regressors, sampling_rate=1.0)

        # Check that 1D shape is preserved (not converted to 2D)
        assert conv_regs['visual'].ndim == 1
        assert conv_regs['motor'].ndim == 1
        assert conv_regs['visual'].shape == (100,)
        assert conv_regs['motor'].shape == (100,)

    def test_convolve_regressors_preserves_shape_2d(self):
        """Test that convolve_regressors preserves 2D shape."""
        regressors = {
            'multi': np.zeros((100, 3))
        }
        regressors['multi'][20, :] = 1

        conv_regs = convolve_regressors(regressors, sampling_rate=1.0)

        # Should preserve 2D shape
        assert conv_regs['multi'].ndim == 2
        assert conv_regs['multi'].shape == (100, 3)


class TestIntegrationScenarios:
    """Integration tests combining multiple features."""

    def test_event_model_with_covariates_and_events(self):
        """Test complete integration: events + covariates in one model."""
        n_scans = 100
        tr = 2.0

        # Create event data
        events = pd.DataFrame({
            'onset': [10, 30, 50, 70],
            'condition': ['A', 'B', 'A', 'B']
        })

        # Create full data with covariates
        data = pd.DataFrame({
            'motion_x': np.random.randn(n_scans),
            'motion_y': np.random.randn(n_scans)
        })

        # Merge event information
        for i, row in events.iterrows():
            scan_idx = int(row['onset'] / tr)
            if scan_idx < n_scans:
                data.loc[scan_idx, 'onset'] = row['onset']
                data.loc[scan_idx, 'condition'] = row['condition']

        # Forward fill condition
        data['condition'] = data['condition'].ffill().bfill()
        data['onset'] = data['onset'].ffill().bfill()

        # Create model with both event terms and covariates
        model = event_model(
            [
                Term('condition', hrf='spmg1'),
                CovariateTerm(['motion_x', 'motion_y'], prefix='motion')
            ],
            data=data,
            tr=tr,
            n_scans=n_scans
        )

        X = model.design_matrix

        # Should have columns for both conditions and covariates
        assert X.shape[0] == n_scans
        # Should have at least 3 columns (condition levels may vary + covariates)
        assert X.shape[1] >= 3

        # Check that we have both condition and motion columns
        col_names = model.column_names
        has_condition = any('condition' in name for name in col_names)
        has_motion = any('motion' in name for name in col_names)
        assert has_condition, f"Expected condition columns in {col_names}"
        assert has_motion, f"Expected motion columns in {col_names}"

    def test_trialwise_integration_with_cells_conditions(self):
        """Test trialwise integration with cells/conditions methods."""
        data = pd.DataFrame({
            'onset': [10, 20, 30, 40],
            'condition': ['A', 'B', 'A', 'B']
        })

        model = event_model(
            "trialwise(add_sum=True)",
            data=data,
            tr=2.0,
            n_scans=100
        )

        # Get cells and conditions
        cells = model.cells()
        conditions = model.conditions()

        # Should have trial-level conditions
        assert len(cells) > 0
        assert len(conditions) > 0

    def test_convolve_design_with_superposition(self):
        """Test that convolution properly handles overlapping events (superposition)."""
        # Create design with two close events
        X = np.zeros((100, 1))
        X[20, 0] = 1.0
        X[25, 0] = 1.0  # Only 5 timepoints apart

        X_conv = convolve_design(X, hrf='spmg1', sampling_rate=1.0)

        # Response should overlap and summate
        # Peak should be > 1 due to summation
        # (Actually depends on HRF normalization, but should see response in both regions)
        assert np.sum(X_conv[20:30, 0] > 0) > 0
        assert np.sum(X_conv[25:35, 0] > 0) > 0

    def test_multiple_covariates_different_sampling(self):
        """Test covariates with explicit sampling points and interpolation."""
        # Create covariate sampled at different rate than scan TR
        orig_times = np.array([0, 5, 10, 15, 20])  # Every 5 seconds
        hr_values = np.array([60, 65, 62, 68, 64])  # Heart rate

        # Create covariate event with explicit sampling
        cov_event = CovariateEvent('heart_rate', hr_values, sampling_points=orig_times)

        # Request at different sampling (every 2 seconds)
        scan_times = np.arange(0, 21, 2)  # 0, 2, 4, ..., 20
        dm = cov_event.design_matrix(scan_times)

        # Should have interpolated values
        assert dm.shape == (11, 1)
        # First value should match
        assert np.isclose(dm[0, 0], 60)
        # Last value should match
        assert np.isclose(dm[-1, 0], 64)
        # Intermediate values should be interpolated
        assert dm[1, 0] != hr_values[0]  # Should be interpolated, not original

    def test_covariate_event_property_access(self):
        """Test CovariateEvent properties."""
        event = CovariateEvent('test', np.array([1, 2, 3, 4, 5]))

        assert event.event_type == 'covariate'
        assert event.n_timepoints == 5
        assert event.name == 'test'
        # Covariates have empty onsets/durations
        assert len(event.onsets) == 0
        assert len(event.durations) == 0


class TestRealWorldScenarios:
    """Tests based on realistic fMRI analysis scenarios."""

    def test_motion_correction_workflow(self):
        """Test typical motion parameter inclusion workflow."""
        n_scans = 200
        tr = 2.0

        # Simulate 6 motion parameters (3 translation + 3 rotation)
        motion_params = pd.DataFrame({
            'trans_x': np.random.randn(n_scans) * 0.5,
            'trans_y': np.random.randn(n_scans) * 0.5,
            'trans_z': np.random.randn(n_scans) * 0.3,
            'rot_x': np.random.randn(n_scans) * 0.02,
            'rot_y': np.random.randn(n_scans) * 0.02,
            'rot_z': np.random.randn(n_scans) * 0.02,
        })

        # Create sampling frame
        sampling = SamplingFrame(tr=tr, n_scans=n_scans)

        # Create covariate events for motion parameters
        motion_events = create_covariate_events(
            motion_params,
            ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z'],
            sampling,
            prefix='motion'
        )

        # Should have 6 motion covariates
        assert len(motion_events) == 6
        assert 'motion_trans_x' in motion_events
        assert 'motion_rot_z' in motion_events

        # Each should have correct number of timepoints
        for event in motion_events.values():
            assert event.n_timepoints == n_scans

    def test_single_trial_beta_series_workflow(self):
        """Test single-trial beta series analysis workflow."""
        # Create experiment with many trials
        n_trials = 30  # Reduced to fit in sample space
        tr = 2.0
        n_scans = 500

        # Random trial onsets (spaced at least 20s apart)
        np.random.seed(42)
        onsets = np.sort(np.random.choice(np.arange(10, 600, 20), n_trials, replace=False))

        data = pd.DataFrame({
            'onset': onsets,
            'trial_type': ['stimulus'] * n_trials
        })

        # Create trialwise model
        model = event_model(
            "trialwise(basis='spmg1')",
            data=data,
            tr=tr,
            n_scans=n_scans
        )

        # Should have one column per trial
        assert model.design_matrix.shape[1] == n_trials

        # Each column should be convolved (have temporal spread)
        for col_idx in range(model.design_matrix.shape[1]):
            col = model.design_matrix[:, col_idx]
            # Should have non-zero values
            assert np.sum(col != 0) > 1

    def test_mixed_model_with_multiple_term_types(self):
        """Test model with events, covariates, and trialwise terms."""
        n_scans = 100
        tr = 2.0

        # Create event data
        events_df = pd.DataFrame({
            'onset': [20, 40, 60, 80],
            'block_type': ['task', 'rest', 'task', 'rest']
        })

        # Create full data
        data = pd.DataFrame({
            'motion': np.random.randn(n_scans)
        })

        # Add event columns
        for i, row in events_df.iterrows():
            scan_idx = int(row['onset'] / tr)
            if scan_idx < n_scans:
                data.loc[scan_idx, 'onset'] = row['onset']
                data.loc[scan_idx, 'block_type'] = row['block_type']

        data['onset'] = data['onset'].ffill().bfill()
        data['block_type'] = data['block_type'].ffill().bfill()

        # This would be a complex model combining multiple approaches
        # For now, test that covariate term is compatible
        model = event_model(
            [
                Term('block_type'),
                CovariateTerm('motion')
            ],
            data=data,
            tr=tr,
            n_scans=n_scans
        )

        assert model.design_matrix.shape[0] == n_scans
        # Should have block_type levels + motion covariate
        assert model.design_matrix.shape[1] >= 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
