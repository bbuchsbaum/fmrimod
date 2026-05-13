"""Tests for baseline model specification functions."""

from dataclasses import FrozenInstanceError

import pytest
import numpy as np
import pandas as pd

from fmrimod.baseline import nuisance, block
from fmrimod.baseline import NuisanceSpec, BlockSpec


class TestNuisance:
    """Test nuisance specification functionality."""
    
    def test_nuisance_with_array(self):
        """Test nuisance with numpy array."""
        data = np.random.randn(100, 6)
        spec = nuisance(data)
        
        assert isinstance(spec, NuisanceSpec)
        assert spec.name == 'nuisance'
        assert spec.data is data
        
    def test_nuisance_with_string(self):
        """Test nuisance with string reference."""
        spec = nuisance('motion_params')
        
        assert isinstance(spec, NuisanceSpec)
        assert spec.name == 'motion_params'
        assert spec.data == 'motion_params'
        
    def test_nuisance_with_dataframe(self):
        """Test nuisance with pandas DataFrame."""
        df = pd.DataFrame(np.random.randn(100, 3), 
                         columns=['x', 'y', 'z'])
        spec = nuisance(df)
        
        assert isinstance(spec, NuisanceSpec)
        assert spec.data is df
        
    def test_nuisance_repr(self):
        """Test string representation of NuisanceSpec."""
        spec = nuisance('my_nuisance')
        assert repr(spec) == "NuisanceSpec(name='my_nuisance')"
        
    def test_nuisance_with_matrix(self):
        """Test nuisance with 2D array input."""
        mat = np.random.randn(50, 4)
        spec = nuisance(mat)
        
        assert isinstance(spec, NuisanceSpec)
        assert spec.data is mat

    def test_nuisance_spec_is_frozen_payload_reference(self):
        """NuisanceSpec preserves the payload but freezes the wrapper."""
        data = np.random.randn(10, 2)
        spec = nuisance(data)

        with pytest.raises(FrozenInstanceError):
            spec.name = 'changed'


class TestBlock:
    """Test block specification functionality."""
    
    def test_block_with_string(self):
        """Test block with string name."""
        spec = block('run')
        
        assert isinstance(spec, BlockSpec)
        assert spec.name == 'run'
        assert spec.label == 'run'
        
    def test_block_with_array(self):
        """Test block with array of indicators."""
        run_ids = [1, 1, 1, 2, 2, 2, 3, 3, 3]
        spec = block(run_ids)
        
        assert isinstance(spec, BlockSpec)
        assert spec.name == 'block'  # Default name
        
    def test_block_repr(self):
        """Test string representation of BlockSpec."""
        spec = block('session')
        assert repr(spec) == "BlockSpec(name='session')"
        
    def test_block_with_custom_label(self):
        """Test block preserves label."""
        spec = block('my_block_var')
        assert spec.name == 'my_block_var'
        assert spec.label == 'my_block_var'

    def test_block_spec_is_frozen(self):
        """BlockSpec is an immutable declarative wrapper."""
        spec = block('run')

        with pytest.raises(FrozenInstanceError):
            spec.label = 'changed'


class TestIntegration:
    """Test integration of nuisance and block specs."""
    
    def test_multiple_nuisance_specs(self):
        """Test creating multiple nuisance specs."""
        motion = nuisance(np.random.randn(100, 6))
        physio = nuisance('physio_regressors')
        
        assert isinstance(motion, NuisanceSpec)
        assert isinstance(physio, NuisanceSpec)
        assert motion.name != physio.name
        
    def test_block_and_nuisance_together(self):
        """Test using block and nuisance together."""
        run_spec = block('run')
        motion_spec = nuisance(np.random.randn(100, 6))
        
        # These should be different types
        assert type(run_spec) != type(motion_spec)
        assert isinstance(run_spec, BlockSpec)
        assert isinstance(motion_spec, NuisanceSpec)
