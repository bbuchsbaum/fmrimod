"""Tests for HRF integration with pyfmrihrf."""

import pytest
import numpy as np

from fmrimod.dispatch import get_hrf
from fmrimod.hrf_integration import (
    PyFMRIHRF,
    get_fmrimod,
    list_hrfs,
    create_hrf_basis,
)
from fmrimod.formula.parser import parse_formula
from fmrimod import event_model
from fmrimod.formula.base import Term
from fmrimod.events.factor import EventFactor


class TestPyFMRIHRF:
    """Test PyFMRIHRF wrapper class."""
    
    def test_create_from_string(self):
        """Test creating HRF from string name."""
        hrf = PyFMRIHRF("spm")
        assert hrf.name == "spm"
        assert hrf.nbasis >= 1
        
        # Test evaluation
        t = np.linspace(0, 20, 100)
        y = hrf.evaluate(t)
        assert y.shape == (100,) or y.shape == (100, hrf.nbasis)
    
    def test_create_with_alias(self):
        """Test HRF aliases."""
        # These should all map to SPMG1
        for alias in ["spm", "spm_canonical", "canonical"]:
            hrf = PyFMRIHRF(alias)
            assert hrf.name == alias
    
    def test_create_from_dict(self):
        """Test creating HRF from dictionary."""
        hrf = PyFMRIHRF({"name": "gamma", "shape": 6})
        assert hrf.name == "gamma"
    
    def test_evaluate_scalar(self):
        """Test evaluating HRF at single time point."""
        hrf = PyFMRIHRF("gamma")
        
        # Single time point
        y = hrf.evaluate(5.0)
        assert isinstance(y, (float, np.ndarray))
        if isinstance(y, np.ndarray):
            assert y.ndim in (0, 1)  # Scalar or 1D array
    
    def test_evaluate_vector(self):
        """Test evaluating HRF at multiple time points."""
        hrf = PyFMRIHRF("gamma")
        
        t = np.array([0, 5, 10, 15, 20])
        y = hrf.evaluate(t)
        assert y.shape == (5,) or y.shape[0] == 5
    
    def test_unknown_hrf(self):
        """Test error on unknown HRF."""
        with pytest.raises(ValueError, match="Unknown HRF"):
            PyFMRIHRF("nonexistent_hrf")


class TestHRFIntegrationFunctions:
    """Test HRF integration utility functions."""
    
    def test_get_pyfmrihrf(self):
        """Test get_fmrimod convenience function (renamed from get_pyfmrihrf)."""
        hrf = get_fmrimod("spm")
        assert isinstance(hrf, PyFMRIHRF)
        assert hrf.name == "spm"
    
    def test_list_hrfs(self):
        """Test listing available HRFs."""
        hrfs = list_hrfs()
        assert isinstance(hrfs, list)
        assert len(hrfs) > 0
        # Check some expected HRFs
        assert any("gamma" in h.lower() for h in hrfs)
        assert any("spm" in h.lower() for h in hrfs)
    
    def test_create_hrf_basis(self):
        """Test creating HRF basis sets."""
        # FIR basis
        fir = create_hrf_basis("fir", n_basis=10, length=20)
        assert fir.nbasis == 10
        
        # Evaluate
        t = np.linspace(0, 20, 100)
        y = fir.evaluate(t)
        assert y.shape == (100, 10)  # Should have 10 basis functions


class TestDispatchIntegration:
    """Test HRF dispatch integration."""
    
    def test_get_hrf_string(self):
        """Test getting HRF by string through dispatch."""
        # Built-in HRF
        hrf1 = get_hrf("simple")
        assert hrf1.name == "simple"
        
        # pyfmrihrf HRF
        hrf2 = get_hrf("gamma")
        assert hasattr(hrf2, "evaluate")
    
    def test_get_hrf_dict(self):
        """Test getting HRF from dictionary specification."""
        hrf = get_hrf({"name": "gamma", "shape": 6})
        assert hasattr(hrf, "evaluate")
    
    def test_get_hrf_with_params(self):
        """Test getting HRF with parameters."""
        hrf = get_hrf("gamma", shape=6, scale=1)
        assert hasattr(hrf, "evaluate")


class TestFormulaIntegration:
    """Test HRF integration with formula parsing."""
    
    def test_parse_hrf_formula(self):
        """Test parsing formula with HRF specification."""
        # Parse formula without context (for event model)
        terms = parse_formula('hrf(condition, hrf="gamma")')
        assert len(terms) == 1
        assert terms[0].events == ["condition"]
        assert terms[0].hrf == "gamma"
    
    def test_parse_multiple_hrfs(self):
        """Test parsing formula with multiple HRF terms."""
        terms = parse_formula('hrf(cond1, hrf="spm") + hrf(cond2, hrf="gamma")')
        assert len(terms) == 2
        assert terms[0].hrf == "spm"
        assert terms[1].hrf == "gamma"
    
    def test_parse_default_hrf(self):
        """Test default HRF when not specified."""
        terms = parse_formula('hrf(condition)')
        assert len(terms) == 1
        # Should default to 'simple' based on parser logic
        assert terms[0].hrf == "simple"


class TestEventModelIntegration:
    """Test HRF integration with event models."""
    
    def test_event_model_with_hrf(self):
        """Test event model with HRF specification."""
        import pandas as pd
        
        # Create test data
        n_scans = 100
        data = pd.DataFrame({
            'onset': [10, 30, 50, 70],
            'condition': ['A', 'B', 'A', 'B']
        })
        
        # Create model with HRF
        model = event_model(
            [Term('condition', hrf='gamma')],
            data=data,
            tr=2.0,
            n_scans=n_scans
        )
        
        # Check design matrix
        X = model.design_matrix
        assert X.shape[0] == n_scans
        assert X.shape[1] > 0
    
    def test_event_model_with_basis_hrf(self):
        """Test event model with basis HRF."""
        import pandas as pd
        
        # Create test data
        n_scans = 100
        data = pd.DataFrame({
            'onset': [10, 30, 50, 70],
            'condition': ['A', 'B', 'A', 'B']
        })
        
        # Create model with basis HRF
        model = event_model(
            [Term('condition', hrf='SPMG2')],  # SPM with time derivative
            data=data,
            tr=2.0,
            n_scans=n_scans
        )
        
        # Check design matrix - should have more columns due to basis
        X = model.design_matrix
        assert X.shape[0] == n_scans
        # SPMG2 has 2 basis functions, and we have 2 conditions
        assert X.shape[1] == 4  # 2 conditions × 2 basis functions
    
    def test_formula_string_with_hrf(self):
        """Test event model from formula string with HRF."""
        import pandas as pd
        
        # Create test data
        n_scans = 50
        data = pd.DataFrame({
            'onset': [5, 15, 25, 35],
            'cond': ['X', 'Y', 'X', 'Y'],
            'rating': [1, 2, 3, 4]
        })
        
        # Create model from formula
        model = event_model(
            'hrf(cond, hrf="spm") + hrf(rating, hrf="gamma")',
            data=data,
            tr=2.0,
            n_scans=n_scans
        )
        
        # Check that we have the right terms
        assert len(model.terms) == 2
        assert model.terms[0].hrf == "spm"
        assert model.terms[1].hrf == "gamma"
        
        # Check design matrix
        X = model.design_matrix
        assert X.shape[0] == n_scans
        assert X.shape[1] >= 3  # At least 2 for cond + 1 for rating


@pytest.mark.skipif(
    not hasattr(pytest, 'importorskip'),
    reason="pytest.importorskip not available"
)
class TestPyFMRIHRFAvailability:
    """Test that pyfmrihrf HRFs are available."""
    
    def test_pyfmrihrf_import(self):
        """Test that pyfmrihrf can be imported."""
        pytest.importorskip("pyfmrihrf")
    
    def test_registered_hrfs(self):
        """Test that pyfmrihrf HRFs are registered."""
        from fmrimod.hrf import _HRF_REGISTRY
        
        # Should have more than just the built-in HRFs
        assert len(_HRF_REGISTRY) > 3
        
        # Check some expected HRFs are registered
        expected_hrfs = ['gamma', 'spmg1', 'spmg2', 'spmg3', 'fir']
        for hrf_name in expected_hrfs:
            assert hrf_name in _HRF_REGISTRY