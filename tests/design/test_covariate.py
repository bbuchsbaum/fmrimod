"""Tests for covariate functionality."""

import pytest
import numpy as np
import pandas as pd

from fmrimod.covariate import covariate, CovariateTerm, CovariateEvent
from fmrimod import event_model
from fmrimod.events.factor import EventFactor
from fmrimod.covariate import create_covariate_events
from fmrimod.sampling import SamplingFrame
from fmrimod.formula.base import Term


class TestCovariateTerm:
    """Test CovariateTerm creation and properties."""
    
    def test_single_covariate(self):
        """Test single covariate term."""
        term = covariate('motion_x')
        
        assert isinstance(term, CovariateTerm)
        assert term.covariates == ['motion_x']
        assert term.events == ['motion_x']  # Should mirror covariates
        assert term.hrf is None  # Never convolved
        assert term.name == 'motion_x'
        assert term.is_covariate is True
    
    def test_multiple_covariates(self):
        """Test multiple covariates."""
        term = covariate('motion_x', 'motion_y', 'motion_z')
        
        assert term.covariates == ['motion_x', 'motion_y', 'motion_z']
        # Default Term behavior uses : for multiple events
        assert term.name == 'motion_x:motion_y:motion_z'
    
    def test_covariate_with_prefix(self):
        """Test covariate with prefix."""
        term = covariate('x', 'y', 'z', prefix='motion')
        
        assert term.prefix == 'motion'
        # With prefix, underscores are used
        assert term.name == 'motion_x_y_z'
    
    def test_covariate_with_custom_name(self):
        """Test covariate with custom name."""
        term = covariate('heart_rate', 'respiration', name='physio')
        
        assert term.name == 'physio'
        assert term.covariates == ['heart_rate', 'respiration']
    
    def test_covariate_validation(self):
        """Test covariate validation with data."""
        data = pd.DataFrame({
            'motion_x': np.random.randn(100),
            'motion_y': np.random.randn(100)
        })
        
        # Should work with valid columns
        term = covariate('motion_x', 'motion_y', data=data)
        assert term is not None
        
        # Should fail with missing columns
        with pytest.raises(ValueError, match="Variables not found"):
            covariate('motion_x', 'missing_col', data=data)
    
    def test_empty_covariate(self):
        """Test error on empty covariate list."""
        with pytest.raises(ValueError, match="At least one variable"):
            covariate()


class TestCovariateEvent:
    """Test CovariateEvent functionality."""
    
    def test_basic_covariate_event(self):
        """Test basic covariate event creation."""
        values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        event = CovariateEvent('motion_x', values)
        
        assert event.name == 'motion_x'
        assert event.event_type == 'covariate'
        assert event.n_timepoints == 5
        assert np.array_equal(event.values, values)
    
    def test_design_matrix_matching_points(self):
        """Test design matrix when sampling points match."""
        values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        sampling_points = np.arange(5)
        
        event = CovariateEvent('motion_x', values)
        dm = event.design_matrix(sampling_points)
        
        assert dm.shape == (5, 1)
        assert np.array_equal(dm[:, 0], values)
    
    def test_design_matrix_interpolation(self):
        """Test design matrix with interpolation."""
        # Original data at times 0, 2, 4
        orig_times = np.array([0, 2, 4])
        values = np.array([0.0, 1.0, 2.0])
        
        event = CovariateEvent('motion_x', values, sampling_points=orig_times)
        
        # Request at times 0, 1, 2, 3, 4
        new_times = np.arange(5)
        dm = event.design_matrix(new_times)
        
        assert dm.shape == (5, 1)
        # Check interpolated values
        assert np.isclose(dm[0, 0], 0.0)  # t=0
        assert np.isclose(dm[1, 0], 0.5)  # t=1 (interpolated)
        assert np.isclose(dm[2, 0], 1.0)  # t=2
        assert np.isclose(dm[3, 0], 1.5)  # t=3 (interpolated)
        assert np.isclose(dm[4, 0], 2.0)  # t=4
    
    def test_design_matrix_mismatch_error(self):
        """Test error when timepoints don't match without sampling_points."""
        values = np.array([0.1, 0.2, 0.3])
        event = CovariateEvent('motion_x', values)
        
        # Request different number of points
        with pytest.raises(ValueError, match="has 3 values but 5 sampling points"):
            event.design_matrix(np.arange(5))


class TestCreateCovariateEvents:
    """Test create_covariate_events function."""
    
    def test_create_single_covariate(self):
        """Test creating single covariate event."""
        n_timepoints = 10
        data = pd.DataFrame({
            'motion_x': np.random.randn(n_timepoints)
        })
        
        sampling_info = SamplingFrame(tr=2.0, n_scans=n_timepoints)
        
        events = create_covariate_events(
            data, ['motion_x'], sampling_info
        )
        
        assert 'motion_x' in events
        assert isinstance(events['motion_x'], CovariateEvent)
        assert events['motion_x'].n_timepoints == n_timepoints
    
    def test_create_with_prefix(self):
        """Test creating covariates with prefix."""
        n_timepoints = 10
        data = pd.DataFrame({
            'x': np.random.randn(n_timepoints),
            'y': np.random.randn(n_timepoints)
        })
        
        sampling_info = SamplingFrame(tr=2.0, n_scans=n_timepoints)
        
        events = create_covariate_events(
            data, ['x', 'y'], sampling_info, prefix='motion'
        )
        
        assert 'motion_x' in events
        assert 'motion_y' in events
        assert events['motion_x'].name == 'motion_x'
    
    def test_create_length_mismatch(self):
        """Test error on length mismatch."""
        data = pd.DataFrame({
            'motion_x': np.random.randn(5)  # Wrong length
        })
        
        sampling_info = SamplingFrame(tr=2.0, n_scans=10)
        
        with pytest.raises(ValueError, match="has 5 values but sampling frame expects 10"):
            create_covariate_events(data, ['motion_x'], sampling_info)
    
    def test_create_missing_column(self):
        """Test error on missing column."""
        data = pd.DataFrame({
            'motion_x': np.random.randn(10)
        })
        
        sampling_info = SamplingFrame(tr=2.0, n_scans=10)
        
        with pytest.raises(ValueError, match="Covariate 'motion_y' not found"):
            create_covariate_events(data, ['motion_x', 'motion_y'], sampling_info)


class TestCovariateIntegration:
    """Test integration with EventModel."""
    
    def test_event_model_with_covariates(self):
        """Test event model with both events and covariates."""
        n_scans = 100
        tr = 2.0
        
        # Create data with events and covariates
        event_data = pd.DataFrame({
            'onset': [10, 30, 50, 70],
            'condition': ['A', 'B', 'A', 'B']
        })
        
        # Create full-length data for covariates
        full_data = pd.DataFrame({
            'motion_x': np.random.randn(n_scans),
            'motion_y': np.random.randn(n_scans),
            'heart_rate': np.random.randn(n_scans)
        })
        
        # Merge event data into full data (repeat event values)
        for col in event_data.columns:
            full_data[col] = pd.NA
            for i, (idx, row) in enumerate(event_data.iterrows()):
                scan_idx = int(row['onset'] / tr)
                if scan_idx < n_scans:
                    full_data.loc[scan_idx, col] = row[col]
        
        # Forward fill event data
        full_data['condition'] = full_data['condition'].ffill().fillna('A')
        # Create onset array properly sized
        onset_list = event_data['onset'].tolist()
        full_onset = []
        for i in range(n_scans):
            full_onset.append(onset_list[i % len(onset_list)])
        full_data['onset'] = full_onset
        
        # Create model with mixed terms
        model = event_model(
            [
                Term('condition'),  # Regular event
                CovariateTerm(['motion_x', 'motion_y'], prefix='motion'),  # Covariates
                CovariateTerm('heart_rate')  # Single covariate
            ],
            data=full_data,
            tr=tr,
            n_scans=n_scans
        )
        
        # Check design matrix
        X = model.design_matrix
        assert X.shape[0] == n_scans
        
        # Check column names
        col_names = model.column_names
        assert any('condition' in name for name in col_names)
        assert any('motion_x' in name for name in col_names)
        assert any('motion_y' in name for name in col_names)
        assert any('heart_rate' in name for name in col_names)
    
    def test_covariate_no_hrf_convolution(self):
        """Test that covariates are not convolved with HRF."""
        n_scans = 50
        tr = 2.0
        
        # Simple data
        data = pd.DataFrame({
            'covariate': np.ones(n_scans)  # Constant covariate
        })
        
        # Create model with covariate term
        model = event_model(
            [CovariateTerm('covariate')],
            data=data,
            tr=tr,
            n_scans=n_scans
        )
        
        # Get design matrix
        X = model.design_matrix
        
        # Should be constant (no HRF convolution)
        assert X.shape == (n_scans, 1)
        assert np.allclose(X[:, 0], 1.0)
    
    def test_cells_conditions_with_covariates(self):
        """Test cells and conditions methods with covariates."""
        n_scans = 50
        
        # Create event data
        event_data = pd.DataFrame({
            'onset': [10, 20, 30, 40],
            'condition': ['A', 'B', 'A', 'B']
        })
        
        # Create full data with covariates
        data = pd.DataFrame({
            'motion_x': np.random.randn(n_scans)
        })
        
        # Add event columns - just repeat the condition for simplicity
        data['onset'] = [event_data['onset'].iloc[i % len(event_data)] for i in range(n_scans)]
        data['condition'] = [event_data['condition'].iloc[i % len(event_data)] for i in range(n_scans)]
        
        model = event_model(
            [
                Term('condition'),
                CovariateTerm('motion_x')
            ],
            data=data,
            tr=2.0,
            n_scans=n_scans
        )
        
        # Get cells - should handle both event and covariate terms
        cells = model.cells()
        assert len(cells) == 2
        
        # Get conditions
        conditions = model.conditions()
        assert len(conditions) == 2
        # First term should have categorical conditions
        assert 'condition.A' in conditions[0]
        # Second term should have covariate name
        assert 'motion_x' in conditions[1]