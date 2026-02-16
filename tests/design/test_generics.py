"""Tests for generic functions - mirroring and extending R tests."""

import pytest
import numpy as np
import pandas as pd

from fmrimod.events.factor import EventFactor
from fmrimod.events.variable import EventVariable
from fmrimod.events.matrix import EventMatrix
from fmrimod.events.basis import EventBasis
from fmrimod.events.term import EventTerm
from fmrimod import event_model, baseline_model
from fmrimod.basis import Poly, BSpline
from fmrimod.utils.generics import event_terms, construct, columns, labels, levels, nbasis
from fmrimod.baseline import baseline, nuisance, block
from fmrimod.utils import (
    onsets, durations, elements, is_categorical, is_continuous,
    cells, conditions
)
from fmrimod.baseline.specs import NuisanceSpec, BlockSpec
from fmrimod import SamplingFrame


class TestEventTerms:
    """Test event_terms generic function."""
    
    def test_event_terms_from_model(self):
        """Test extracting event terms from EventModel."""
        data = pd.DataFrame({
            'onset': [1, 5, 10, 15],
            'condition': ['A', 'B', 'A', 'B'],
            'response': [0.5, 0.7, 0.6, 0.8]
        })

        # Model with single categorical term
        model1 = event_model('condition', data=data, tr=2.0, n_scans=20)
        terms1 = event_terms(model1)
        assert len(terms1) == 1
        assert isinstance(terms1[0], EventTerm)

        # Model with multiple terms
        model2 = event_model('condition + response', data=data, tr=2.0, n_scans=20)
        terms2 = event_terms(model2)
        assert len(terms2) == 2

    def test_event_terms_empty_model(self):
        """Test event_terms returns list for simple model."""
        data = pd.DataFrame({
            'onset': [1, 5, 10],
            'condition': ['A', 'B', 'A']
        })
        model = event_model('condition', data=data, tr=2.0, n_scans=20)
        terms = event_terms(model)
        assert isinstance(terms, list)
        assert len(terms) >= 1


class TestConstruct:
    """Test construct generic function."""
    
    def test_construct_baseline_spec(self):
        """Test constructing baseline from BaselineSpec."""
        from fmrimod.baseline.baseline_model import BaselineSpec

        spec = BaselineSpec(degree=3, basis='poly')
        sframe = SamplingFrame(blocklens=[100], TR=2.0)

        result = construct(spec, sframe)
        assert hasattr(result, 'design_matrix')

        # Check that polynomial baseline was created
        dm = result.design_matrix
        assert dm.shape[0] == 100

    def test_construct_nuisance_spec(self):
        """Test constructing nuisance from NuisanceSpec."""
        # NuisanceSpec wraps nuisance data
        motion_params = np.random.randn(100, 6)
        spec = NuisanceSpec(name='motion', data=motion_params)

        # NuisanceSpec stores data directly
        assert spec.name == 'motion'
        assert_array_equal(spec.data, motion_params)

    def test_construct_block_spec(self):
        """Test constructing block variable from BlockSpec."""
        spec = BlockSpec(name='run', label='run')

        assert spec.name == 'run'


class TestColumns:
    """Test columns generic function."""
    
    def test_columns_event_factor(self):
        """Test extracting columns from EventFactor."""
        event = EventFactor(
            onsets=[1, 5, 10],
            durations=[1, 1, 1],
            values=['A', 'B', 'A'],
            name='condition'
        )
        
        cols = columns(event)
        assert cols == ['condition.A', 'condition.B']
    
    def test_columns_event_variable(self):
        """Test extracting columns from EventVariable."""
        event = EventVariable(
            onsets=[1, 5, 10],
            durations=[1, 1, 1],
            values=[0.5, 0.7, 0.6],
            name='response_time'
        )
        
        cols = columns(event)
        assert cols == ['response_time']
    
    def test_columns_event_matrix(self):
        """Test extracting columns from EventMatrix."""
        values = np.array([[1, 0], [0, 1], [1, 0]])
        event = EventMatrix(
            name='stimulus',
            onsets=[1, 5, 10],
            durations=[1, 1, 1],
            values=values,
            column_names=['left', 'right']
        )
        
        cols = columns(event)
        assert cols == ['left', 'right']
    
    def test_columns_event_term(self):
        """Test extracting columns from EventTerm."""
        factor = EventFactor(
            onsets=[1, 5], durations=[1, 1], values=['A', 'B'], name='cond'
        )
        variable = EventVariable(
            onsets=[1, 5], durations=[1, 1], values=[0.5, 0.7], name='rt'
        )
        
        # Single event term
        term1 = EventTerm([factor])
        cols1 = columns(term1)
        # EventTerm.get_column_names returns raw level names
        assert 'A' in cols1 and 'B' in cols1
        
        # Interaction term
        term2 = EventTerm([factor, variable])
        cols2 = columns(term2)
        # Should have interaction columns
        assert len(cols2) > 0
    
    def test_columns_numpy_array(self):
        """Test extracting columns from numpy array."""
        # 1D array
        arr1 = np.array([1, 2, 3, 4])
        cols1 = columns(arr1)
        assert cols1 == ['V1']
        
        # 2D array
        arr2 = np.random.randn(10, 5)
        cols2 = columns(arr2)
        assert cols2 == ['V1', 'V2', 'V3', 'V4', 'V5']
    
    def test_columns_dataframe(self):
        """Test extracting columns from DataFrame."""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6],
            'C': [7, 8, 9]
        })
        
        cols = columns(df)
        assert cols == ['A', 'B', 'C']
    
    def test_columns_event_model(self):
        """Test extracting columns from EventModel."""
        data = pd.DataFrame({
            'onset': [1, 5, 10],
            'condition': ['A', 'B', 'A']
        })
        
        model = event_model('condition', data=data, tr=2.0, n_scans=20)
        cols = columns(model)
        
        # Should have columns for each condition
        assert any('A' in col for col in cols)
        assert any('B' in col for col in cols)


class TestLabelsAndLevels:
    """Test labels and levels generic functions."""
    
    def test_labels_levels_event_factor(self):
        """Test labels/levels for EventFactor."""
        event = EventFactor(
            onsets=[1, 5, 10, 15],
            durations=[1, 1, 1, 1],
            values=['A', 'B', 'A', 'C'],
            name='condition'
        )
        
        # Labels should be the unique levels
        labs = labels(event)
        assert set(labs) == {'A', 'B', 'C'}
        
        # Levels should be the same
        levs = levels(event)
        assert set(levs) == {'A', 'B', 'C'}
    
    def test_labels_levels_event_variable(self):
        """Test labels/levels for EventVariable."""
        event = EventVariable(
            onsets=[1, 5, 10],
            durations=[1, 1, 1],
            values=[0.5, 0.7, 0.6],
            name='response_time'
        )
        
        # Labels should be the variable name
        labs = labels(event)
        assert labs == ['response_time']
        
        # Levels should be None for continuous
        levs = levels(event)
        assert levs is None
    
    def test_labels_levels_event_matrix(self):
        """Test labels/levels for EventMatrix."""
        values = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        event = EventMatrix(
            name='matrix_event',
            onsets=[1, 5, 10],
            durations=[1, 1, 1],
            values=values,
            column_names=['col1', 'col2', 'col3']
        )
        
        # Labels should be column names
        labs = labels(event)
        assert labs == ['col1', 'col2', 'col3']
        
        # Levels should be None for matrix
        levs = levels(event)
        assert levs is None
    
    def test_labels_levels_event_basis(self):
        """Test labels/levels for EventBasis."""
        # Create a simple basis event
        poly_basis = Poly(degree=2)
        event = EventBasis(
            name='poly_event',
            onsets=[1, 5, 10],
            values=[0.5, 0.7, 0.6],
            basis=poly_basis,
            durations=[1, 1, 1],
        )

        # Labels should be basis names
        labs = labels(event)
        assert len(labs) > 0

        # Levels should be None for basis
        levs = levels(event)
        assert levs is None
    
    def test_labels_levels_event_term(self):
        """Test labels/levels for EventTerm."""
        # Categorical term
        factor1 = EventFactor(
            onsets=[1, 5], durations=[1, 1], values=['A', 'B'], name='cond1'
        )
        factor2 = EventFactor(
            onsets=[1, 5], durations=[1, 1], values=['X', 'Y'], name='cond2'
        )
        
        # Single factor term
        term1 = EventTerm([factor1])
        labs1 = labels(term1)
        assert 'A' in labs1 and 'B' in labs1
        levs1 = levels(term1)
        assert set(levs1) == {'A', 'B'}
        
        # Interaction term
        term2 = EventTerm([factor1, factor2])
        levs2 = levels(term2)
        # Should have interaction levels
        assert 'A:X' in levs2
        assert 'B:Y' in levs2
        
        # Mixed term (categorical + continuous)
        variable = EventVariable(
            onsets=[1, 5], durations=[1, 1], values=[0.5, 0.7], name='rt'
        )
        term3 = EventTerm([factor1, variable])
        levs3 = levels(term3)
        assert levs3 is None  # Mixed terms have no levels


class TestNbasis:
    """Test nbasis generic function."""
    
    def test_nbasis_event_factor(self):
        """Test nbasis for EventFactor."""
        event = EventFactor(
            onsets=[1, 5, 10],
            durations=[1, 1, 1],
            values=['A', 'B', 'C'],
            name='condition'
        )
        
        # Number of basis = levels - 1 (for contrast coding)
        n = nbasis(event)
        assert n == 2  # 3 levels - 1
    
    def test_nbasis_event_variable(self):
        """Test nbasis for EventVariable."""
        event = EventVariable(
            onsets=[1, 5, 10],
            durations=[1, 1, 1],
            values=[0.5, 0.7, 0.6],
            name='rt'
        )
        
        # Continuous variables have 1 basis function
        n = nbasis(event)
        assert n == 1
    
    def test_nbasis_event_matrix(self):
        """Test nbasis for EventMatrix."""
        values = np.random.randn(5, 4)
        event = EventMatrix(
            name='matrix_event',
            onsets=[1, 5, 10, 15, 20],
            durations=[1, 1, 1, 1, 1],
            values=values,
            column_names=[f'col{i}' for i in range(4)]
        )
        
        # Number of columns
        n = nbasis(event)
        assert n == 4
    
    def test_nbasis_event_basis(self):
        """Test nbasis for EventBasis."""
        poly_basis = Poly(degree=3)
        event = EventBasis(
            name='poly_event',
            onsets=[1, 5, 10],
            values=[0.5, 0.7, 0.6],
            basis=poly_basis,
            durations=[1, 1, 1],
        )

        n = nbasis(event)
        assert n == event.n_basis
    
    def test_nbasis_basis_objects(self):
        """Test nbasis for basis objects."""
        # Polynomial basis
        poly = Poly(degree=4)
        assert nbasis(poly) == 5  # degree + 1
        
        # BSpline basis
        spline = BSpline(df=7)
        assert nbasis(spline) == 7
    
    def test_nbasis_event_term(self):
        """Test nbasis for EventTerm."""
        # Single factor
        factor = EventFactor(
            onsets=[1, 5, 10], durations=[1, 1, 1], values=['A', 'B', 'A'], name='cond'
        )
        term1 = EventTerm([factor])
        assert nbasis(term1) == 1  # 2 levels - 1

        # Two factors (interaction)
        factor2 = EventFactor(
            onsets=[1, 5, 10], durations=[1, 1, 1], values=['X', 'Y', 'Z'], name='cond2'
        )
        term2 = EventTerm([factor, factor2])
        # Interaction: (2-1) * (3-1) = 1 * 2 = 2
        assert nbasis(term2) == 2


class TestGenericEdgeCases:
    """Test edge cases for generic functions."""
    
    def test_empty_events(self):
        """Test generic functions with minimal events."""
        # Factor with single event but explicit levels
        factor = EventFactor(
            name='minimal',
            onsets=[1], durations=[1], values=['A'],
            levels=['A', 'B']  # Explicit levels even though only A appears
        )

        assert columns(factor) == ['minimal.A', 'minimal.B']
        assert labels(factor) == ['A', 'B']
        assert levels(factor) == ['A', 'B']
        assert nbasis(factor) == 1
    
    def test_single_level_factor(self):
        """Test factor with only one level."""
        single_factor = EventFactor(
            onsets=[1, 5, 10],
            durations=[1, 1, 1],
            values=['A', 'A', 'A'],
            name='single'
        )
        
        assert len(labels(single_factor)) == 1
        assert nbasis(single_factor) == 0  # No contrast possible
    
    def test_none_inputs(self):
        """Test handling of None inputs."""
        # Most functions should raise for None
        with pytest.raises((TypeError, AttributeError)):
            columns(None)
        
        with pytest.raises((TypeError, AttributeError)):
            labels(None)
    
    def test_3d_array_columns(self):
        """Test columns with 3D array (should raise)."""
        arr3d = np.random.randn(5, 4, 3)
        
        with pytest.raises(ValueError, match="3D array"):
            columns(arr3d)


def assert_array_equal(a, b):
    """Helper for array comparison."""
    np.testing.assert_array_equal(a, b)


class TestOnsetsDurations:
    """Test onsets and durations generic functions."""

    def test_onsets_event_factor(self):
        """Test extracting onsets from EventFactor."""
        event = EventFactor(
            name='condition',
            onsets=[5.0, 10.0, 15.0, 20.0],
            values=['A', 'B', 'A', 'B'],
            durations=1.0
        )

        onset_array = onsets(event)
        assert isinstance(onset_array, np.ndarray)
        np.testing.assert_array_equal(onset_array, [5.0, 10.0, 15.0, 20.0])

    def test_onsets_event_variable(self):
        """Test extracting onsets from EventVariable."""
        event = EventVariable(
            name='rating',
            onsets=[5.0, 10.0, 15.0],
            values=[1.0, 2.0, 3.0],
            durations=1.0
        )

        onset_array = onsets(event)
        np.testing.assert_array_equal(onset_array, [5.0, 10.0, 15.0])

    def test_onsets_event_term(self):
        """Test extracting onsets from EventTerm."""
        factor = EventFactor(
            name='condition',
            onsets=[5.0, 10.0, 15.0],
            values=['A', 'B', 'A'],
            durations=1.0
        )
        term = EventTerm([factor])

        onset_array = onsets(term)
        np.testing.assert_array_equal(onset_array, [5.0, 10.0, 15.0])

    def test_durations_event_factor(self):
        """Test extracting durations from EventFactor."""
        event = EventFactor(
            name='condition',
            onsets=[5.0, 10.0, 15.0],
            values=['A', 'B', 'A'],
            durations=[1.0, 2.0, 1.5]
        )

        dur_array = durations(event)
        np.testing.assert_array_equal(dur_array, [1.0, 2.0, 1.5])

    def test_durations_event_variable(self):
        """Test extracting durations from EventVariable."""
        event = EventVariable(
            name='rating',
            onsets=[5.0, 10.0],
            values=[1.0, 2.0],
            durations=[1.0, 1.0]
        )

        dur_array = durations(event)
        np.testing.assert_array_equal(dur_array, [1.0, 1.0])

    def test_durations_event_term(self):
        """Test extracting durations from EventTerm."""
        variable = EventVariable(
            name='rating',
            onsets=[5.0, 10.0],
            values=[1.0, 2.0],
            durations=[0.5, 1.5]
        )
        term = EventTerm([variable])

        dur_array = durations(term)
        np.testing.assert_array_equal(dur_array, [0.5, 1.5])


class TestElementsAndTypes:
    """Test elements, is_categorical, is_continuous functions."""

    def test_elements_event_factor_values(self):
        """Test extracting values from EventFactor."""
        event = EventFactor(
            name='condition',
            onsets=[5, 10, 15],
            values=['A', 'B', 'A'],
            durations=1.0
        )

        vals = elements(event, what='values')
        assert list(vals) == ['A', 'B', 'A']

    def test_elements_event_factor_labels(self):
        """Test extracting labels from EventFactor."""
        event = EventFactor(
            name='condition',
            onsets=[5, 10, 15],
            values=['A', 'B', 'A'],
            durations=1.0
        )

        labs = elements(event, what='labels')
        assert set(labs) == {'A', 'B'}

    def test_elements_event_variable_values(self):
        """Test extracting values from EventVariable."""
        event = EventVariable(
            name='rating',
            onsets=[5, 10, 15],
            values=[1.0, 2.0, 3.0],
            durations=1.0,
            center=False  # Don't center for this test
        )

        vals = elements(event, what='values')
        np.testing.assert_array_equal(vals, [1.0, 2.0, 3.0])

    def test_elements_event_variable_labels(self):
        """Test extracting labels from EventVariable."""
        event = EventVariable(
            name='rating',
            onsets=[5, 10],
            values=[1.0, 2.0],
            durations=1.0
        )

        labs = elements(event, what='labels')
        assert labs == ['rating']

    def test_is_categorical_event_factor(self):
        """Test is_categorical for EventFactor."""
        event = EventFactor(
            name='condition',
            onsets=[5, 10],
            values=['A', 'B'],
            durations=1.0
        )

        assert is_categorical(event) is True
        assert is_continuous(event) is False

    def test_is_continuous_event_variable(self):
        """Test is_continuous for EventVariable."""
        event = EventVariable(
            name='rating',
            onsets=[5, 10],
            values=[1.0, 2.0],
            durations=1.0
        )

        assert is_continuous(event) is True
        assert is_categorical(event) is False

    def test_is_categorical_event_term_categorical(self):
        """Test is_categorical for categorical EventTerm."""
        factor = EventFactor(
            name='condition',
            onsets=[5, 10],
            values=['A', 'B'],
            durations=1.0
        )
        term = EventTerm([factor])

        assert is_categorical(term) is True
        assert is_continuous(term) is False

    def test_is_continuous_event_term_continuous(self):
        """Test is_continuous for continuous EventTerm."""
        variable = EventVariable(
            name='rating',
            onsets=[5, 10],
            values=[1.0, 2.0],
            durations=1.0
        )
        term = EventTerm([variable])

        assert is_continuous(term) is True
        assert is_categorical(term) is False


class TestCellsAndConditions:
    """Test cells and conditions generic functions."""

    def test_cells_event_factor(self):
        """Test extracting cells from EventFactor."""
        event = EventFactor(
            name='condition',
            onsets=[5, 10, 15, 20],
            values=['A', 'B', 'A', 'B'],
            durations=1.0
        )

        cell_df = cells(event)
        assert isinstance(cell_df, pd.DataFrame)
        # Should have counts for each level
        assert len(cell_df) > 0

    def test_cells_event_variable(self):
        """Test extracting cells from EventVariable."""
        event = EventVariable(
            name='rating',
            onsets=[5, 10, 15],
            values=[1.0, 2.0, 3.0],
            durations=1.0
        )

        cell_df = cells(event)
        assert isinstance(cell_df, pd.DataFrame)
        # Continuous events should have single cell
        assert len(cell_df) >= 1

    def test_conditions_event_factor(self):
        """Test extracting conditions from EventFactor."""
        event = EventFactor(
            name='condition',
            onsets=[5, 10, 15],
            values=['A', 'B', 'A'],
            durations=1.0
        )

        cond_list = conditions(event)
        assert isinstance(cond_list, list)
        assert set(cond_list) == {'condition.A', 'condition.B'}

    def test_conditions_event_variable(self):
        """Test extracting conditions from EventVariable."""
        event = EventVariable(
            name='rating',
            onsets=[5, 10],
            values=[1.0, 2.0],
            durations=1.0
        )

        cond_list = conditions(event)
        assert isinstance(cond_list, list)
        assert 'rating' in cond_list

    def test_conditions_event_term(self):
        """Test extracting conditions from EventTerm."""
        factor = EventFactor(
            name='condition',
            onsets=[5, 10, 15],
            values=['A', 'B', 'A'],
            durations=1.0
        )
        term = EventTerm([factor])

        cond_list = conditions(term)
        assert isinstance(cond_list, list)
        assert len(cond_list) > 0


class TestEventModel:
    """Test generic functions on EventModel."""

    def setup_method(self):
        """Create test EventModel."""
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

    def test_term_names_event_model(self):
        """Test extracting term names from EventModel."""
        from fmrimod.utils import term_names

        names = term_names(self.model)
        assert isinstance(names, list)
        assert 'condition' in names
        assert 'rating' in names

    def test_longnames_event_model(self):
        """Test extracting long names from EventModel."""
        from fmrimod.utils import longnames

        long_names = longnames(self.model)
        assert isinstance(long_names, list)
        assert len(long_names) == len(self.model.column_names)

    def test_shortnames_event_model(self):
        """Test extracting short names from EventModel."""
        from fmrimod.utils import shortnames

        short_names = shortnames(self.model)
        assert isinstance(short_names, list)
        assert len(short_names) == len(self.model.column_names)

    def test_columns_event_model(self):
        """Test extracting columns from EventModel."""
        cols = columns(self.model)
        assert cols == self.model.column_names

    def test_event_terms_event_model(self):
        """Test extracting event terms from EventModel."""
        terms = event_terms(self.model)
        assert isinstance(terms, list)
        assert len(terms) == 2  # condition + rating

        # Each should be EventTerm
        for term in terms:
            assert isinstance(term, EventTerm)


class TestEventConditionsAndEvents:
    """Test event_conditions and events generic functions."""

    def test_event_conditions_event_term(self):
        """Test extracting event conditions from EventTerm."""
        from fmrimod.utils import event_conditions

        factor = EventFactor(
            name='condition',
            onsets=[5, 10, 15, 20],
            values=['A', 'B', 'A', 'B'],
            durations=1.0
        )
        term = EventTerm([factor])

        cond_array = event_conditions(term)
        assert isinstance(cond_array, np.ndarray)
        assert len(cond_array) == 4

    def test_events_event_term(self):
        """Test extracting events table from EventTerm."""
        from fmrimod.utils import events

        factor = EventFactor(
            name='condition',
            onsets=[5.0, 10.0, 15.0],
            values=['A', 'B', 'A'],
            durations=[1.0, 1.0, 1.0]
        )
        term = EventTerm([factor])

        events_df = events(term)
        assert isinstance(events_df, pd.DataFrame)
        assert 'onset' in events_df.columns
        assert 'duration' in events_df.columns
        assert 'condition' in events_df.columns
        assert len(events_df) == 3

    def test_events_event_model(self):
        """Test extracting events from EventModel."""
        from fmrimod.utils import events

        events_df = pd.DataFrame({
            'onset': [5, 10, 15],
            'condition': ['A', 'B', 'A']
        })

        model = event_model(
            'condition',
            data=events_df,
            tr=2.0,
            n_scans=20
        )

        evt_table = events(model)
        assert isinstance(evt_table, pd.DataFrame)
        assert 'onset' in evt_table.columns
        assert 'condition' in evt_table.columns


class TestBlockidsAndBlocklens:
    """Test blockids and blocklens generic functions."""

    def test_blockids_list(self):
        """Test blockids with list input."""
        from fmrimod.utils import blockids

        block_list = [1, 1, 1, 2, 2, 2, 3, 3, 3]
        result = blockids(block_list)
        assert result == block_list

    def test_blockids_array(self):
        """Test blockids with array input."""
        from fmrimod.utils import blockids

        block_array = np.array([1, 1, 2, 2, 3, 3])
        result = blockids(block_array)
        np.testing.assert_array_equal(result, block_array)

    def test_blocklens_sampling_frame(self):
        """Test blocklens with SamplingFrame."""
        from fmrimod.utils import blocklens

        sframe = SamplingFrame(blocklens=[100, 100, 100], TR=2.0)
        lens = blocklens(sframe)
        np.testing.assert_array_equal(lens, [100, 100, 100])

    def test_blockids_sampling_frame(self):
        """Test blockids with SamplingFrame."""
        from fmrimod.utils import blockids

        sframe = SamplingFrame(blocklens=[3, 2, 2], TR=2.0)
        ids = blockids(sframe)
        # Should generate [1, 1, 1, 2, 2, 3, 3]
        expected = np.array([1, 1, 1, 2, 2, 3, 3])
        np.testing.assert_array_equal(ids, expected)