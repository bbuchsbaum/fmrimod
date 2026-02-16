"""Tests for design matrix convolution functionality."""

import pytest
import numpy as np
import pandas as pd

from fmrimod.convolve_design import convolve_design, convolve_regressors
from fmrimod.design.event_model import EventModel, event_model


class TestConvolveDesign:
    """Test convolve_design function."""
    
    def test_convolve_array_basic(self):
        """Test basic array convolution."""
        # Create simple design matrix
        n_timepoints = 100
        X = np.zeros((n_timepoints, 2))
        X[10, 0] = 1  # Impulse in first column
        X[50, 1] = 1  # Impulse in second column
        
        # Convolve
        X_conv = convolve_design(X, hrf='spmg1', sampling_rate=1.0)
        
        # Check shape preserved
        assert X_conv.shape == X.shape
        
        # Check that impulses were spread out by convolution
        assert np.sum(X_conv[:, 0] > 0) > 1
        assert np.sum(X_conv[:, 1] > 0) > 1
        
        # Check peaks are after the impulses (HRF delay)
        assert np.argmax(X_conv[:, 0]) > 10
        assert np.argmax(X_conv[:, 1]) > 50
    
    def test_convolve_dataframe(self):
        """Test DataFrame convolution."""
        # Create DataFrame
        n_timepoints = 100
        df = pd.DataFrame({
            'visual': np.zeros(n_timepoints),
            'motor': np.zeros(n_timepoints)
        })
        df.loc[20, 'visual'] = 1
        df.loc[60, 'motor'] = 1
        
        # Convolve
        df_conv = convolve_design(df, hrf='gamma', sampling_rate=2.0)
        
        # Check type and columns preserved
        assert isinstance(df_conv, pd.DataFrame)
        assert list(df_conv.columns) == ['visual', 'motor']
        
        # Check convolution occurred
        assert df_conv['visual'].sum() > df['visual'].sum()
        assert df_conv['motor'].sum() > df['motor'].sum()
    
    def test_convolve_event_model(self):
        """Test EventModel convolution."""
        from fmrimod.design.event_model import EventModel
        from fmrimod.events.factor import EventFactor
        from fmrimod.sampling import SamplingFrame
        from fmrimod.formula.base import Term
        import pandas as pd

        events = {
            'stimulus': EventFactor(
                name='stimulus',
                onsets=[1, 5, 9],  # Align with TR=2.0
                values=['A', 'A', 'A'],
                durations=1.5
            )
        }

        sf = SamplingFrame(tr=2.0, n_scans=10)
        terms = [Term('stimulus', hrf='spmg1')]

        model = EventModel(
            terms=terms,
            events=events,
            sampling_info=sf
        )

        # Get original design matrix
        X_orig = model.design_matrix

        # Convolve the model (should apply additional convolution)
        convolved = convolve_design(model, hrf='spmg1')

        # Should return DataFrame with convolved design
        assert isinstance(convolved, pd.DataFrame)
        assert convolved.shape[0] == 10  # same number of scans
        assert convolved.shape[1] == X_orig.shape[1]  # same number of columns
    
    def test_missing_parameters(self):
        """Test error handling for missing parameters."""
        X = np.random.randn(100, 3)
        
        # Array without sampling_rate should fail
        with pytest.raises(ValueError, match="sampling_rate or time_points"):
            convolve_design(X)
        
        # DataFrame without sampling_rate should fail
        df = pd.DataFrame(X)
        with pytest.raises(ValueError, match="sampling_rate or time_points"):
            convolve_design(df)
    
    def test_time_points_specification(self):
        """Test using explicit time points."""
        X = np.ones((50, 1))
        time_points = np.linspace(0, 100, 50)  # Non-uniform spacing
        
        X_conv = convolve_design(X, time_points=time_points)
        
        assert X_conv.shape == X.shape
        
        # Wrong length time_points should fail
        with pytest.raises(ValueError, match="time_points length"):
            convolve_design(X, time_points=np.arange(30))
    
    def test_custom_hrf_parameters(self):
        """Test passing custom HRF parameters."""
        X = np.eye(20)
        
        # Custom gamma parameters
        X_conv = convolve_design(
            X,
            hrf='gamma',
            sampling_rate=1.0,
            shape=8,
            scale=1.5
        )
        
        assert X_conv.shape == X.shape
    
    def test_column_names_preservation(self):
        """Test that column names are preserved or generated."""
        X = np.random.randn(50, 3)
        
        # With column names
        names = ['reg1', 'reg2', 'reg3']
        X_conv = convolve_design(X, sampling_rate=1.0, column_names=names)
        
        # When returning as DataFrame
        df = pd.DataFrame(X, columns=names)
        df_conv = convolve_design(df, sampling_rate=1.0)
        assert list(df_conv.columns) == names
    
    def test_1d_input(self):
        """Test handling of 1D input."""
        x = np.zeros(100)
        x[50] = 1
        
        x_conv = convolve_design(x, sampling_rate=1.0)
        
        # Should be 2D output
        assert x_conv.ndim == 2
        assert x_conv.shape == (100, 1)


class TestConvolveRegressors:
    """Test convolve_regressors function."""
    
    def test_basic_dictionary_convolution(self):
        """Test convolving dictionary of regressors."""
        regressors = {
            'visual': np.zeros(100),
            'motor': np.zeros(100),
            'auditory': np.zeros(100)
        }
        
        # Add impulses
        regressors['visual'][20] = 1
        regressors['motor'][40] = 1
        regressors['auditory'][60] = 1
        
        # Convolve
        conv_regs = convolve_regressors(regressors, sampling_rate=2.0)
        
        # Check all keys preserved
        assert set(conv_regs.keys()) == set(regressors.keys())
        
        # Check convolution occurred
        for name in regressors:
            # Since HRF is normalized, sum should be approximately preserved
            assert np.abs(conv_regs[name].sum() - regressors[name].sum()) < 0.1
            assert conv_regs[name].shape == regressors[name].shape
            # Check that impulse was spread out
            assert np.sum(conv_regs[name] > 0.01) > 1
    
    def test_mixed_dimensions(self):
        """Test handling mixed 1D and 2D regressors."""
        regressors = {
            'single': np.zeros(50),
            'multi': np.zeros((50, 3))
        }
        
        regressors['single'][10] = 1
        regressors['multi'][20, :] = 1
        
        conv_regs = convolve_regressors(regressors)
        
        # Check shapes preserved
        assert conv_regs['single'].shape == regressors['single'].shape
        assert conv_regs['multi'].shape == regressors['multi'].shape
    
    def test_empty_dictionary(self):
        """Test handling empty dictionary."""
        conv_regs = convolve_regressors({})
        assert conv_regs == {}