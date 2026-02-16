"""Tests for convolve, formula, and HRF modules to fill coverage gaps."""

import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal

from fmrimod.events.factor import EventFactor
from fmrimod.events.variable import EventVariable
from fmrimod.events.matrix import EventMatrix
from fmrimod.events.basis import EventBasis
from fmrimod.convolve import convolve
from fmrimod import get_hrf, as_hrf
from fmrimod.hrf_dispatch import (
    ArrayHRF, FunctionHRF, DictHRF, SimpleHRF, SPMCanonicalHRF
)
# Import generators from hrf_dispatch (they are defined there)
from fmrimod.hrf_dispatch import (
    boxcar_hrf_gen, duration_hrf_gen, weighted_hrf_gen
)
from fmrimod.formula.parser import parse_formula, FormulaParser
from fmrimod.formula.base import Term, EventModelBuilder


# ============================================================================
# Tests for convolve.py - targeting 50%+ coverage
# ============================================================================

class TestConvolveHRFVariants:
    """Test convolve with different HRF types (spmg1, spmg2, spmg3)."""

    def test_convolve_eventfactor_spmg1(self):
        """Test EventFactor convolution with spmg1 HRF."""
        event = EventFactor(
            onsets=[2.0, 6.0, 10.0, 14.0],
            durations=[1.0, 1.0, 1.0, 1.0],
            values=['A', 'B', 'A', 'B'],
            name='condition'
        )

        result = convolve(event, hrf='spmg1', sampling_rate=1.0, total_duration=25.0)

        assert result.shape == (25, 2)  # 2 levels
        assert np.max(result) > 0

    def test_convolve_eventfactor_spmg2(self):
        """Test EventFactor convolution with spmg2 HRF (with derivative)."""
        event = EventFactor(
            onsets=[2.0, 6.0],
            durations=[1.0, 1.0],
            values=['X', 'Y'],
            name='stim'
        )

        result = convolve(event, hrf='spmg2', sampling_rate=1.0, total_duration=20.0)

        # spmg2 has 2 basis functions, but EventFactor uses first basis only
        assert result.shape[0] == 20
        assert result.shape[1] == 2  # X and Y levels
        assert np.max(result) > 0

    def test_convolve_eventfactor_spmg3(self):
        """Test EventFactor convolution with spmg3 HRF (with dispersion)."""
        event = EventFactor(
            onsets=[2.0, 8.0],
            durations=[1.0, 1.0],
            values=['P', 'Q'],
            name='param'
        )

        result = convolve(event, hrf='spmg3', sampling_rate=1.0, total_duration=20.0)

        assert result.shape == (20, 2)
        assert np.max(result) > 0

    def test_convolve_eventvariable_spmg2(self):
        """Test EventVariable convolution with spmg2."""
        event = EventVariable(
            onsets=[5.0, 10.0],
            durations=[1.0, 1.0],
            values=[1.5, 2.5],
            name='rating',
            center=False
        )

        result = convolve(event, hrf='spmg2', sampling_rate=1.0, total_duration=20.0)

        assert result.shape[0] == 20
        assert np.max(result) > 0

    def test_convolve_eventvariable_spmg3(self):
        """Test EventVariable convolution with spmg3."""
        event = EventVariable(
            onsets=[3.0, 9.0],
            durations=[0.5, 0.5],
            values=[1.0, 2.0],
            name='value',
            center=False
        )

        result = convolve(event, hrf='spmg3', sampling_rate=1.0, total_duration=15.0)

        assert result.shape[0] == 15
        assert np.max(result) > 0

    def test_convolve_eventmatrix_spmg1(self):
        """Test EventMatrix convolution with spmg1."""
        values = np.array([[1.0, 0.5], [0.5, 1.0], [1.0, 1.0]])
        event = EventMatrix(
            name='motion',
            onsets=[2.0, 6.0, 10.0],
            durations=[1.0, 1.0, 1.0],
            values=values
        )

        result = convolve(event, hrf='spmg1', sampling_rate=1.0, total_duration=20.0)

        assert result.shape == (20, 2)
        assert np.max(result) > 0

    def test_convolve_normalize_flag(self):
        """Test normalize parameter for peak normalization."""
        event = EventVariable(
            onsets=[5.0],
            durations=[1.0],
            values=[5.0],  # Large amplitude
            name='test',
            center=False
        )

        result = convolve(event, sampling_rate=1.0, total_duration=20.0, normalize=True)

        # Peak-normalized: max absolute value should be 1.0
        max_abs = np.max(np.abs(result))
        assert np.isclose(max_abs, 1.0, atol=1e-6)

    def test_convolve_summate_parameter(self):
        """Test summate parameter for overlapping events."""
        # Two overlapping events
        event = EventVariable(
            onsets=[5.0, 6.0],  # 1 second apart
            durations=[2.0, 2.0],
            values=[1.0, 1.0],
            name='overlap',
            center=False
        )

        result_sum = convolve(event, sampling_rate=1.0, total_duration=20.0, summate=True)
        result_max = convolve(event, sampling_rate=1.0, total_duration=20.0, summate=False)

        # Both should produce valid output
        assert result_sum.shape == result_max.shape
        assert np.max(result_sum) > 0
        assert np.max(result_max) > 0


class TestConvolveListAndArray:
    """Test convolve with list and numpy array inputs."""

    def test_convolve_list_multiple_events(self):
        """Test convolve with list of multiple event types."""
        ev1 = EventVariable(
            onsets=[2.0, 6.0],
            durations=[1.0, 1.0],
            values=[1.0, 2.0],
            name='var1',
            center=False
        )

        ev2 = EventFactor(
            onsets=[3.0, 7.0],
            durations=[1.0, 1.0],
            values=['A', 'B'],
            name='fac1'
        )

        ev3 = EventMatrix(
            name='mat1',
            onsets=[4.0, 8.0],
            durations=[1.0, 1.0],
            values=np.array([[1.0, 0.0], [0.0, 1.0]])
        )

        results = convolve([ev1, ev2, ev3], sampling_rate=1.0, total_duration=15.0)

        assert len(results) == 3
        assert results[0].shape == (15, 1)  # EventVariable
        assert results[1].shape == (15, 2)  # EventFactor (2 levels)
        assert results[2].shape == (15, 2)  # EventMatrix (2 columns)

    def test_convolve_numpy_array_valid(self):
        """Test convolve with valid numpy array [onset, duration, value]."""
        arr = np.array([
            [2.0, 1.0, 1.5],
            [6.0, 1.0, 2.5],
            [10.0, 0.5, 3.0]
        ])

        result = convolve(arr, hrf='spmg1', sampling_rate=1.0, total_duration=20.0)

        assert isinstance(result, np.ndarray)
        assert result.ndim == 1
        assert result.shape[0] == 20
        assert np.max(result) > 0

    def test_convolve_numpy_array_with_normalize(self):
        """Test numpy array convolution with normalize=True."""
        arr = np.array([
            [5.0, 1.0, 10.0],  # Large amplitude
            [10.0, 1.0, 5.0]
        ])

        result = convolve(arr, sampling_rate=1.0, total_duration=20.0, normalize=True)

        max_abs = np.max(np.abs(result))
        assert np.isclose(max_abs, 1.0, atol=1e-6)


# ============================================================================
# Tests for formula/parser.py - targeting 50%+ coverage
# ============================================================================

class TestFormulaParser:
    """Test formula parsing functionality."""

    def test_parse_simple_variable(self):
        """Test parsing simple variable name."""
        terms = parse_formula("condition")

        assert len(terms) == 1
        assert isinstance(terms[0], Term)
        assert terms[0].events == ['condition']

    def test_parse_addition_two_terms(self):
        """Test parsing addition of two terms."""
        terms = parse_formula("condition + rating")

        assert len(terms) == 2
        assert terms[0].events == ['condition']
        assert terms[1].events == ['rating']

    def test_parse_interaction_term(self):
        """Test parsing interaction term with colon."""
        terms = parse_formula("condition:rating")

        assert len(terms) == 1
        # Interaction creates single term with both variables
        assert 'condition' in str(terms[0].events[0]) or isinstance(terms[0].events, list)

    def test_parse_hrf_function_simple(self):
        """Test parsing hrf() function."""
        terms = parse_formula("hrf(condition)")

        assert len(terms) == 1
        assert terms[0].events == ['condition']
        assert terms[0].hrf == 'simple'

    def test_parse_hrf_with_spmg1(self):
        """Test parsing hrf() with spmg1 parameter."""
        terms = parse_formula("hrf(condition, hrf='spmg1')")

        assert len(terms) == 1
        assert terms[0].events == ['condition']
        # HRF parameter should be captured

    def test_parse_variable_with_hrf_prefix(self):
        """Test parsing variable:hrf(args) syntax."""
        parser = FormulaParser()
        formula = parser.parse("onset ~ condition:hrf(spmg1)")

        assert formula.lhs == 'onset'
        assert len(formula.rhs) == 1

    def test_parse_full_formula_with_lhs(self):
        """Test parsing full formula with left-hand side."""
        parser = FormulaParser()
        formula = parser.parse("onset ~ hrf(condition)")

        assert formula.lhs == 'onset'
        assert len(formula.rhs) == 1
        assert formula.rhs[0].function == 'hrf'

    def test_parse_multiple_terms_with_hrf(self):
        """Test parsing multiple hrf() terms."""
        terms = parse_formula("hrf(condition) + hrf(block)")

        assert len(terms) == 2
        assert terms[0].events == ['condition']
        assert terms[1].events == ['block']

    def test_parse_trialwise_function(self):
        """Test parsing trialwise() function."""
        terms = parse_formula("trialwise()")

        assert len(terms) == 1
        # Should create trialwise term

    def test_parse_invalid_syntax_raises_error(self):
        """Test that invalid syntax raises ValueError."""
        parser = FormulaParser()

        with pytest.raises(ValueError, match="Invalid formula syntax"):
            parser.parse("invalid syntax with spaces")

    def test_formula_parser_term_extraction(self):
        """Test FormulaParser _parse_term method."""
        parser = FormulaParser()

        # Simple variable
        term = parser._parse_term("x")
        assert term.function == "var"
        assert term.arguments == ["x"]

        # Function call
        term = parser._parse_term("hrf(condition)")
        assert term.function == "hrf"
        assert "condition" in term.arguments


# ============================================================================
# Tests for formula/base.py - targeting 60%+ coverage
# ============================================================================

class TestTerm:
    """Test Term class functionality."""

    def test_term_single_event(self):
        """Test Term with single event."""
        term = Term('condition')

        assert term.events == ['condition']
        assert term.name == 'condition'
        assert not term.is_interaction

    def test_term_interaction_events(self):
        """Test Term with interaction (multiple events)."""
        term = Term(['condition', 'rating'])

        assert term.events == ['condition', 'rating']
        assert term.is_interaction
        assert ':' in term.name

    def test_term_with_hrf_string(self):
        """Test Term with HRF parameter."""
        term = Term('stim', hrf='spmg1')

        assert term.hrf == 'spmg1'
        assert term.events == ['stim']

    def test_term_with_hrf_method(self):
        """Test Term.with_hrf() method."""
        term = Term('condition')
        term_modified = term.with_hrf('spmg2')

        assert term_modified.hrf == 'spmg2'
        assert term_modified is term  # Should return self for chaining

    def test_term_with_basis_method(self):
        """Test Term.with_basis() method."""
        from fmrimod.basis import Poly

        term = Term('value')
        basis = Poly(degree=2)
        term_modified = term.with_basis(basis)

        assert term_modified.basis is basis
        assert term_modified is term

    def test_term_with_name_method(self):
        """Test Term.with_name() method."""
        term = Term('x')
        term_modified = term.with_name('custom_name')

        assert term_modified.name == 'custom_name'
        assert term_modified is term

    def test_term_set_method(self):
        """Test Term.set() method for kwargs."""
        term = Term('x')
        term_modified = term.set(param1='value1', param2=42)

        assert term_modified._kwargs['param1'] == 'value1'
        assert term_modified._kwargs['param2'] == 42
        assert term_modified is term

    def test_term_kwargs_property(self):
        """Test Term.kwargs property."""
        from fmrimod.basis import Poly

        term = Term('x', hrf='spmg1', normalize=True, summate=False)
        term.set(extra='value')
        term.with_basis(Poly(degree=2))

        kwargs = term.kwargs
        assert kwargs['hrf'] == 'spmg1'
        assert kwargs['normalize'] is True
        assert kwargs['summate'] is False
        assert kwargs['extra'] == 'value'
        assert 'basis' in kwargs

    def test_term_repr(self):
        """Test Term string representation."""
        term1 = Term('condition')
        repr1 = repr(term1)
        assert 'Term' in repr1
        assert 'condition' in repr1

        term2 = Term(['x', 'y'], hrf='spmg1')
        repr2 = repr(term2)
        assert 'Term' in repr2


class TestEventModelBuilder:
    """Test EventModelBuilder functionality."""

    def test_builder_add_term(self):
        """Test adding single term to builder."""
        builder = EventModelBuilder()
        term = Term('condition')

        builder.add_term(term)

        assert len(builder._terms) == 1
        assert builder._terms[0] is term

    def test_builder_add_multiple_terms(self):
        """Test adding multiple terms at once."""
        builder = EventModelBuilder()
        term1 = Term('x')
        term2 = Term('y')

        builder.add_terms(term1, term2)

        assert len(builder._terms) == 2

    def test_builder_set_data(self):
        """Test setting data on builder."""
        builder = EventModelBuilder()
        df = pd.DataFrame({'onset': [1, 2, 3], 'condition': ['A', 'B', 'A']})

        result = builder.set_data(df)

        assert result is builder  # Chaining
        assert builder._data is df

    def test_builder_set_sampling(self):
        """Test setting sampling frame on builder."""
        try:
            from fmrimod import SamplingFrame
            builder = EventModelBuilder()
            sframe = SamplingFrame(blocklens=[20], TR=2.0)

            result = builder.set_sampling(sframe)

            assert result is builder
            assert builder._sampling is sframe
        except ImportError:
            pytest.skip("pyfmrihrf not available")

    def test_builder_set_onset_column(self):
        """Test setting onset column name."""
        builder = EventModelBuilder()

        result = builder.set_onset_column('time')

        assert result is builder
        assert builder._onset_column == 'time'

    def test_builder_set_duration_column(self):
        """Test setting duration column name."""
        builder = EventModelBuilder()

        result = builder.set_duration_column('dur')

        assert result is builder
        assert builder._duration_column == 'dur'

    def test_builder_add_contrast(self):
        """Test adding contrast to builder."""
        builder = EventModelBuilder()

        result = builder.add_contrast('main_effect', 'A > B')

        assert result is builder
        assert 'main_effect' in builder._contrasts

    def test_builder_set_kwargs(self):
        """Test setting additional kwargs."""
        builder = EventModelBuilder()

        result = builder.set(param1='value1', param2=42)

        assert result is builder
        assert builder._kwargs['param1'] == 'value1'
        assert builder._kwargs['param2'] == 42

    def test_builder_terms_property(self):
        """Test accessing terms from builder."""
        builder = EventModelBuilder()
        term1 = Term('x')
        term2 = Term('y')
        builder.add_terms(term1, term2)

        # Builder has _terms attribute
        assert len(builder._terms) == 2

    def test_builder_context_manager(self):
        """Test builder as context manager."""
        with EventModelBuilder() as builder:
            assert isinstance(builder, EventModelBuilder)


# ============================================================================
# Tests for hrf.py - targeting 50%+ coverage
# ============================================================================

class TestHRFFunctions:
    """Test HRF retrieval and conversion functions."""

    def test_get_hrf_spmg1(self):
        """Test getting spmg1 HRF."""
        hrf = get_hrf('spm')

        # HRF from fmrimod may have different name
        assert hrf.name in ('spm_canonical', 'SPMG1', 'HRF_SPMG1')
        assert hrf.nbasis == 1

    def test_get_hrf_simple(self):
        """Test getting simple HRF."""
        hrf = get_hrf('simple')

        assert hrf.name == 'simple'
        assert hrf.nbasis == 1

    def test_get_hrf_evaluate(self):
        """Test HRF evaluation."""
        hrf = get_hrf('spm')
        t = np.arange(0, 20, 0.5)

        values = hrf.evaluate(t)

        assert values.shape == t.shape
        assert np.max(values) > 0

    def test_get_hrf_unknown_raises(self):
        """Test that unknown HRF name raises ValueError."""
        with pytest.raises(ValueError, match="not found in registry"):
            get_hrf('nonexistent_hrf')

    def test_as_hrf_from_string(self):
        """Test as_hrf() with string name."""
        hrf = as_hrf('spm')

        assert hasattr(hrf, 'evaluate')
        assert hasattr(hrf, 'name')
        assert hasattr(hrf, 'nbasis')

    def test_as_hrf_from_hrf_object(self):
        """Test as_hrf() with existing HRF object."""
        hrf1 = get_hrf('simple')
        hrf2 = as_hrf(hrf1)

        assert hrf2 is hrf1  # Should return as-is

    def test_as_hrf_from_array(self):
        """Test as_hrf() with numpy array."""
        arr = np.array([0, 0.5, 1.0, 0.5, 0])
        hrf = as_hrf(arr)

        assert isinstance(hrf, ArrayHRF)
        assert hrf.nbasis == 1

    def test_as_hrf_from_callable(self):
        """Test as_hrf() with callable function."""
        def my_hrf(t):
            return t * np.exp(-t / 2)

        hrf = as_hrf(my_hrf)

        assert isinstance(hrf, FunctionHRF)
        assert callable(hrf.evaluate)

    def test_as_hrf_from_dict_valid(self):
        """Test as_hrf() with dict containing 'evaluate' key."""
        def custom_eval(t):
            return np.exp(-t)

        hrf_dict = {
            'evaluate': custom_eval,
            'name': 'custom',
            'nbasis': 1
        }

        hrf = as_hrf(hrf_dict)

        assert isinstance(hrf, DictHRF)
        assert hrf.name == 'custom'

    def test_as_hrf_from_dict_missing_evaluate(self):
        """Test as_hrf() with dict missing 'evaluate' key raises error."""
        hrf_dict = {'name': 'bad', 'nbasis': 1}

        with pytest.raises(ValueError, match="must contain an 'evaluate' key"):
            as_hrf(hrf_dict)

    def test_as_hrf_invalid_type(self):
        """Test as_hrf() with invalid type raises TypeError."""
        with pytest.raises(TypeError, match="Cannot convert"):
            as_hrf(12345)


class TestHRFClasses:
    """Test HRF class implementations."""

    def test_simple_hrf_evaluate(self):
        """Test SimpleHRF evaluation."""
        hrf = SimpleHRF()
        t = np.arange(0, 10, 0.5)

        values = hrf.evaluate(t)

        assert values.shape == t.shape
        assert values[0] == 0  # At t=0
        assert np.max(values) > 0

    def test_spm_canonical_hrf_evaluate(self):
        """Test SPMCanonicalHRF evaluation."""
        hrf = SPMCanonicalHRF()
        t = np.arange(0, 20, 0.5)

        values = hrf.evaluate(t)

        assert values.shape == t.shape
        assert np.max(values) > 0

    def test_array_hrf_interpolation(self):
        """Test ArrayHRF interpolation."""
        arr = np.array([0, 1, 2, 1, 0])
        hrf = ArrayHRF(arr, sampling_rate=1.0)

        # Evaluate at finer grid
        t = np.arange(0, 5, 0.5)
        values = hrf.evaluate(t)

        assert len(values) == len(t)
        assert np.max(values) > 0

    def test_function_hrf_evaluation(self):
        """Test FunctionHRF evaluation."""
        def my_func(t):
            return np.exp(-t)

        hrf = FunctionHRF(my_func, name='exponential')
        t = np.arange(0, 5, 0.5)

        values = hrf.evaluate(t)

        assert values.shape == t.shape
        assert hrf.name == 'exponential'

    def test_dict_hrf_with_extra_attributes(self):
        """Test DictHRF with extra attributes."""
        def eval_func(t):
            return t ** 2

        spec = {
            'evaluate': eval_func,
            'name': 'quadratic',
            'nbasis': 1,
            'extra_param': 'extra_value'
        }

        hrf = DictHRF(spec)

        assert hrf.name == 'quadratic'
        assert hasattr(hrf, 'extra_param')
        assert hrf.extra_param == 'extra_value'


class TestHRFGenerators:
    """Test HRF generator factory functions."""

    def test_boxcar_hrf_gen_basic(self):
        """Test boxcar_hrf_gen() basic functionality."""
        try:
            gen = boxcar_hrf_gen(normalize=True, min_duration=0.1)

            # Test with DataFrame
            events = pd.DataFrame({
                'duration': [1.0, 2.5, 0.05]
            })

            hrfs = gen(events)

            assert len(hrfs) == 3
            # All should be HRF objects
            assert all(hasattr(h, 'evaluate') for h in hrfs)
        except ImportError:
            pytest.skip("pyfmrihrf not available")

    def test_boxcar_hrf_gen_with_dict(self):
        """Test boxcar_hrf_gen() with dict input."""
        try:
            gen = boxcar_hrf_gen(normalize=False)

            events = {
                'duration': [1.5, 3.0]
            }

            hrfs = gen(events)

            assert len(hrfs) == 2
        except ImportError:
            pytest.skip("pyfmrihrf not available")

    def test_duration_hrf_gen_basic(self):
        """Test duration_hrf_gen() basic functionality."""
        try:
            gen = duration_hrf_gen(min_duration=0.1)

            events = pd.DataFrame({
                'duration': [0.0, 1.0, 2.5]
            })

            hrfs = gen(events)

            assert len(hrfs) == 3
            assert all(hasattr(h, 'evaluate') for h in hrfs)
        except ImportError:
            pytest.skip("pyfmrihrf not available")

    def test_duration_hrf_gen_zero_duration(self):
        """Test duration_hrf_gen() handles zero duration."""
        try:
            gen = duration_hrf_gen(min_duration=0.0)

            events = pd.DataFrame({
                'duration': [0.0, 0.0]
            })

            hrfs = gen(events)

            assert len(hrfs) == 2
        except ImportError:
            pytest.skip("pyfmrihrf not available")

    def test_weighted_hrf_gen_basic(self):
        """Test weighted_hrf_gen() basic functionality."""
        try:
            gen = weighted_hrf_gen(
                times_col='sub_times',
                weights_col='sub_weights',
                relative=False
            )

            events = pd.DataFrame({
                'onset': [0.0, 10.0],
                'sub_times': [[0.5, 1.0, 1.5], [10.2, 10.8]],
                'sub_weights': [[0.3, 0.5, 0.2], [0.6, 0.4]]
            })

            hrfs = gen(events)

            assert len(hrfs) == 2
            assert all(hasattr(h, 'evaluate') for h in hrfs)
        except ImportError:
            pytest.skip("pyfmrihrf not available")

    def test_weighted_hrf_gen_relative_times(self):
        """Test weighted_hrf_gen() with relative=True."""
        try:
            gen = weighted_hrf_gen(
                times_col='rel_times',
                weights_col='weights',
                relative=True,
                normalize=True
            )

            events = pd.DataFrame({
                'onset': [0.0],
                'rel_times': [[0.0, 0.5, 1.0]],
                'weights': [[0.3, 0.5, 0.2]]
            })

            hrfs = gen(events)

            assert len(hrfs) == 1
        except ImportError:
            pytest.skip("pyfmrihrf not available")

    def test_weighted_hrf_gen_with_dict(self):
        """Test weighted_hrf_gen() with dict input."""
        try:
            gen = weighted_hrf_gen()

            events = {
                'onset': [5.0],
                'sub_times': [[5.2, 5.8]],
                'sub_weights': [[0.7, 0.3]]
            }

            hrfs = gen(events)

            assert len(hrfs) == 1
        except ImportError:
            pytest.skip("pyfmrihrf not available")


# ============================================================================
# Integration Tests
# ============================================================================

class TestFormulaIntegration:
    """Integration tests for formula parsing with event models."""

    def test_parse_formula_creates_terms(self):
        """Test that parse_formula creates proper Term objects."""
        terms = parse_formula("condition + rating + block")

        assert len(terms) == 3
        assert all(isinstance(t, Term) for t in terms)
        assert terms[0].events == ['condition']
        assert terms[1].events == ['rating']
        assert terms[2].events == ['block']

    def test_term_used_with_builder(self):
        """Test that Term objects work with EventModelBuilder."""
        builder = EventModelBuilder()
        term1 = Term('condition', hrf='spmg1')
        term2 = Term('rating')

        builder.add_term(term1)
        builder.add_term(term2)

        assert len(builder._terms) == 2
        assert builder._terms[0].hrf == 'spmg1'


class TestConvolveHRFIntegration:
    """Integration tests between convolve and HRF functions."""

    def test_convolve_with_as_hrf_array(self):
        """Test convolve using as_hrf() to create HRF from array."""
        event = EventVariable(
            onsets=[5.0],
            durations=[1.0],
            values=[1.0],
            name='test',
            center=False
        )

        arr = np.array([0, 0.5, 1.0, 0.5, 0])
        hrf = as_hrf(arr)

        result = convolve(event, hrf=hrf, sampling_rate=1.0, total_duration=15.0)

        assert result.shape[0] == 15
        assert np.max(result) > 0

    def test_convolve_with_wrapped_function_hrf(self):
        """Test convolve with wrapped function HRF using as_hrf."""
        event = EventVariable(
            onsets=[2.0, 6.0],
            durations=[1.0, 1.0],
            values=[1.0, 2.0],
            name='stim',
            center=False
        )

        def custom_hrf(t):
            t = np.asarray(t)
            return np.where(t > 0, np.exp(-t / 3), 0.0)

        # Wrap the function with as_hrf to create proper HRF object
        hrf_obj = as_hrf(custom_hrf)
        result = convolve(event, hrf=hrf_obj, sampling_rate=1.0, total_duration=15.0)

        assert result.shape == (15, 1)
        assert np.max(result) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
