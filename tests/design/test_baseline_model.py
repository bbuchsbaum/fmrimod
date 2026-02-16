"""Tests for baseline model functionality."""

import pytest
import numpy as np
import pandas as pd

from fmrimod import baseline_model, design_matrix
from fmrimod.baseline import BaselineModel, baseline
from fmrimod.baseline.baseline_model import BaselineSpec
from fmrimod.sampling import SamplingFrame


class TestBaselineModel:
    """Test baseline model creation and functionality."""
    
    def test_constant_baseline(self):
        """Test constant (intercept only) baseline."""
        # Single block
        sframe = SamplingFrame(tr=2.0, n_scans=50)
        
        bmodel = baseline_model(
            basis='constant',
            sframe=sframe,
            intercept='runwise'
        )
        
        assert isinstance(bmodel, BaselineModel)
        assert 'drift' in bmodel.terms
        
        # Get design matrix
        X = design_matrix(bmodel)
        assert X.shape == (50, 1)
        assert np.all(X == 1)
        assert bmodel.column_names == ["base_constant1_block_1"]
    
    def test_polynomial_baseline(self):
        """Test polynomial drift baseline."""
        sframe = SamplingFrame(tr=2.0, n_scans=100)
        
        bmodel = baseline_model(
            basis='poly',
            degree=3,
            sframe=sframe,
            intercept='runwise'
        )
        
        X = design_matrix(bmodel)
        # poly degree 3 + 1 intercept = 4 columns
        assert X.shape == (100, 4)
        assert bmodel.column_names == [
            "base_poly1_block_1",
            "base_poly2_block_1",
            "base_poly3_block_1",
            "constant_global",
        ]
        
        # Check orthogonality of polynomial columns
        poly_cols = X[:, :3]  # First 3 columns are polynomial
        gram = poly_cols.T @ poly_cols
        # Should be approximately diagonal
        assert np.allclose(gram, np.diag(np.diag(gram)), atol=1e-10)
    
    def test_bspline_baseline(self):
        """Test B-spline baseline."""
        sframe = SamplingFrame(tr=2.0, n_scans=100)
        
        bmodel = baseline_model(
            basis='bs',
            degree=4,
            sframe=sframe,
            intercept='none'
        )
        
        X = design_matrix(bmodel)
        # B-spline with degree 4
        assert X.shape[0] == 100
        assert X.shape[1] >= 4  # At least degree+1 basis functions
    
    def test_multiblock_baseline(self):
        """Test baseline with multiple blocks."""
        # Create multi-block sampling frame
        sframe = SamplingFrame(blocklens=[50, 60], TR=2.0)
        
        bmodel = baseline_model(
            basis='poly',
            degree=2,
            sframe=sframe,
            intercept='runwise'
        )
        
        X = design_matrix(bmodel)
        # 2 blocks * (2 poly + 1 intercept) = 6 columns
        assert X.shape == (110, 6)
        
        # Check block structure
        # With poly basis and runwise intercept, structure is:
        # [drift_block1 | drift_block2 | intercept_block1 | intercept_block2]
        # Column order: 2 poly block1, 2 poly block2, intercept block1, intercept block2
        
        # First block poly columns should be zero in second block rows
        assert np.all(X[50:, :2] == 0)
        # Second block poly columns should be zero in first block rows  
        assert np.all(X[:50, 2:4] == 0)
        
        # Check intercept columns
        # Block 1 intercept (column 4) should be 1 for first 50 rows, 0 for rest
        assert np.all(X[:50, 4] == 1)
        assert np.all(X[50:, 4] == 0)
        # Block 2 intercept (column 5) should be 0 for first 50 rows, 1 for rest
        assert np.all(X[:50, 5] == 0) 
        assert np.all(X[50:, 5] == 1)
    
    def test_global_intercept(self):
        """Test global intercept option."""
        sframe = SamplingFrame(blocklens=[50, 50], TR=2.0)
        
        bmodel = baseline_model(
            basis='poly',
            degree=2,
            sframe=sframe,
            intercept='global'
        )
        
        X = design_matrix(bmodel)
        # 2 blocks * 2 poly + 1 global intercept = 5 columns
        assert X.shape == (100, 5)
        
        # Last column should be all ones (global intercept)
        assert np.all(X[:, -1] == 1)
    
    def test_no_intercept(self):
        """Test no intercept option."""
        sframe = SamplingFrame(tr=2.0, n_scans=50)
        
        bmodel = baseline_model(
            basis='poly',
            degree=3,
            sframe=sframe,
            intercept='none'
        )
        
        X = design_matrix(bmodel)
        # Only polynomial terms, no intercept
        assert X.shape == (50, 3)
    
    def test_nuisance_regressors(self):
        """Test adding nuisance regressors."""
        sframe = SamplingFrame(blocklens=[50, 60], TR=2.0)
        
        # Create nuisance regressors (e.g., motion parameters)
        nuisance1 = np.random.randn(50, 6)
        nuisance2 = np.random.randn(60, 6)
        
        bmodel = baseline_model(
            basis='poly',
            degree=2,
            sframe=sframe,
            intercept='runwise',
            nuisance_list=[nuisance1, nuisance2]
        )
        
        X = design_matrix(bmodel)
        # 2 blocks * (2 poly + 1 intercept) + 2 blocks * 6 nuisance = 18 columns
        assert X.shape == (110, 18)
        
        # Check that nuisance values are included
        # Last 12 columns should be nuisance
        X_nuisance = X[:, -12:]
        # First block nuisance (rows 0-49, cols 0-5)
        assert np.allclose(X_nuisance[:50, :6], nuisance1)
        # Second block nuisance (rows 50-109, cols 6-11)
        assert np.allclose(X_nuisance[50:, 6:], nuisance2)
    
    def test_block_subsetting(self):
        """Test extracting specific blocks."""
        sframe = SamplingFrame(blocklens=[40, 50, 60], TR=2.0)
        
        bmodel = baseline_model(
            basis='poly',
            degree=2,
            sframe=sframe
        )
        
        # Get design matrix for block 2 only (1-indexed)
        X_block2 = design_matrix(bmodel, blockid=2)
        assert X_block2.shape == (50, 3)  # 50 timepoints, 3 columns
        
        # Get design matrix for blocks 1 and 3
        X_blocks13 = design_matrix(bmodel, blockid=[1, 3])
        assert X_blocks13.shape == (100, 6)  # 40+60 timepoints, 2*3 columns
    
    def test_baseline_spec(self):
        """Test baseline specification creation."""
        spec = baseline(degree=3, basis='poly', name='my_drift')
        
        assert isinstance(spec, BaselineSpec)
        assert spec.degree == 3
        assert spec.basis == 'poly'
        assert spec.name == 'my_drift'
        assert spec.intercept == 'runwise'
    
    def test_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        sframe = SamplingFrame(tr=2.0, n_scans=100)
        
        # Invalid basis
        with pytest.raises(ValueError, match="Invalid basis"):
            baseline_model(basis='invalid', sframe=sframe)
        
        # Invalid degree for splines
        with pytest.raises(ValueError, match="degree >= 3"):
            baseline_model(basis='bs', degree=2, sframe=sframe)
        
        # Missing sampling frame
        with pytest.raises(ValueError, match="sframe"):
            baseline_model(basis='poly')
        
        # Wrong nuisance dimensions
        with pytest.raises(ValueError, match="Length of nuisance_list"):
            baseline_model(
                sframe=sframe,
                nuisance_list=[np.random.randn(50, 3), np.random.randn(50, 3)]
            )


class TestBaselineIntegration:
    """Integration tests for baseline models."""
    
    def test_combined_with_event_model(self):
        """Test combining baseline with event model."""
        from fmrimod import event_model
        from fmrimod.design.event_model import EventModel
        
        # Create event model
        df = pd.DataFrame({
            'onset': [10, 30, 50, 70],
            'condition': ['A', 'B', 'A', 'B'],
            'duration': 2
        })
        
        sframe = SamplingFrame(tr=2.0, n_scans=50)
        
        emodel = event_model('condition', data=df, sampling_info=sframe)
        bmodel = baseline_model(basis='poly', degree=3, sframe=sframe)
        
        # Get design matrices
        X_event = design_matrix(emodel)
        X_baseline = design_matrix(bmodel)
        
        # Combine
        X_full = np.hstack([X_event, X_baseline])
        
        assert X_full.shape == (50, 6)  # 2 conditions + 3 poly + 1 intercept
    
    def test_natural_splines(self):
        """Test natural spline basis."""
        sframe = SamplingFrame(tr=2.0, n_scans=200)
        
        bmodel = baseline_model(
            basis='ns',
            degree=5,
            sframe=sframe,
            intercept='none'
        )
        
        X = design_matrix(bmodel)
        assert X.shape[0] == 200
        assert X.shape[1] == 5  # df = degree for natural splines
