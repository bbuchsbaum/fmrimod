"""Tests for formula interfaces."""

import pytest
import numpy as np
import pandas as pd

from fmrimod.formula import (
    # Core
    Term,
    EventModelBuilder,
    # DSL
    event,
    dsl_hrf,
    dsl_basis,
    # Functional
    term,
    hrf,
    poly,
    # Parser
    parse_formula,
    FormulaParser,
    FormulaContext,
)


class TestTerm:
    """Test Term class."""
    
    def test_single_event_term(self):
        """Test creating a single event term."""
        t = Term('condition')
        assert t.events == ['condition']
        assert t.name == 'condition'
        assert not t.is_interaction
    
    def test_interaction_term(self):
        """Test creating an interaction term."""
        t = Term(['condition', 'block'])
        assert t.events == ['condition', 'block']
        assert t.name == 'condition:block'
        assert t.is_interaction
    
    def test_term_with_hrf(self):
        """Test adding HRF to term."""
        t = Term('condition').with_hrf('spm_canonical')
        assert t.hrf == 'spm_canonical'
        assert 'hrf' in t.kwargs
    
    def test_term_chaining(self):
        """Test method chaining."""
        t = (Term('condition')
             .with_hrf('gamma')
             .with_name('main_effect')
             .set(lag=2.0))
        
        assert t.hrf == 'gamma'
        assert t.name == 'main_effect'
        assert t.kwargs['lag'] == 2.0


class TestEventModelBuilder:
    """Test EventModelBuilder."""
    
    def test_builder_basic(self):
        """Test basic builder usage."""
        builder = EventModelBuilder()
        
        # Check method chaining
        result = (builder
                  .set_onset_column('time')
                  .add_term(Term('condition')))
        
        assert result is builder
        assert builder._onset_column == 'time'
        assert len(builder._terms) == 1
    
    def test_builder_validation(self):
        """Test builder validation."""
        builder = EventModelBuilder()
        
        # Should fail without terms
        with pytest.raises(ValueError, match="No terms"):
            builder.build()
        
        # Should fail without data
        builder.add_term(Term('condition'))
        with pytest.raises(ValueError, match="No data"):
            builder.build()
    
    def test_builder_context_manager(self):
        """Test builder as context manager."""
        with EventModelBuilder() as builder:
            builder.add_term(Term('condition'))
        
        # Should work fine
        assert len(builder._terms) == 1


class TestDSL:
    """Test DSL with operators."""
    
    def test_event_var_creation(self):
        """Test creating event variables."""
        condition = event('condition')
        assert condition.name == 'condition'
        assert isinstance(condition._term, Term)
    
    def test_event_interaction(self):
        """Test event interaction with * operator."""
        condition = event('condition')
        block = event('block')
        
        interaction = condition * block
        assert interaction.events == ['condition', 'block']
    
    def test_hrf_application(self):
        """Test applying HRF with @ operator."""
        condition = event('condition')
        term = condition @ dsl_hrf.spm_canonical
        
        assert isinstance(term, Term)
        assert term.hrf == 'spm_canonical'
    
    def test_complex_dsl_expression(self):
        """Test complex DSL expression."""
        condition = event('condition')
        block = event('block')
        parametric = event('parametric')
        
        # Interaction with HRF
        term1 = (condition * block) @ dsl_hrf.gamma
        assert term1.events == ['condition', 'block']
        assert term1.hrf == 'gamma'
        
        # Chained transformations
        term2 = parametric @ dsl_basis.poly(3) @ dsl_hrf.spm_canonical
        assert term2.events == ['parametric']
        assert term2.basis is not None
        assert term2.hrf == 'spm_canonical'


class TestFunctional:
    """Test functional interface with pipes."""
    
    def test_simple_pipe(self):
        """Test simple pipe operation."""
        t = term('condition') | hrf('spm_canonical')
        
        assert isinstance(t, Term)
        assert t.events == ['condition']
        assert t.hrf == 'spm_canonical'
    
    def test_interaction_pipe(self):
        """Test interaction term with pipe."""
        t = term('condition', 'block') | hrf('gamma')
        
        assert t.events == ['condition', 'block']
        assert t.hrf == 'gamma'
    
    def test_chained_pipes(self):
        """Test chaining multiple pipe operations."""
        t = term('parametric') | poly(3) | hrf('gamma')
        
        assert t.events == ['parametric']
        assert t.basis is not None
        assert t.basis.degree == 3
        assert t.hrf == 'gamma'
    
    def test_pipe_with_name(self):
        """Test setting name with pipe."""
        from fmrimod.formula import name
        
        t = term('condition') | name('main_effect') | hrf('spm_canonical')
        
        assert t.name == 'main_effect'
        assert t.hrf == 'spm_canonical'


class TestFormulaParser:
    """Test formula string parsing."""
    
    def test_simple_formula(self):
        """Test parsing simple formula."""
        parser = FormulaParser()
        formula = parser.parse("onset ~ hrf(condition)")
        
        assert formula.lhs == "onset"
        assert len(formula.rhs) == 1
        assert formula.rhs[0].function == "hrf"
        assert formula.rhs[0].arguments == ["condition"]
    
    def test_formula_with_kwargs(self):
        """Test parsing formula with keyword arguments."""
        parser = FormulaParser()
        formula = parser.parse('onset ~ hrf(condition, basis="spm")')
        
        term = formula.rhs[0]
        assert term.function == "hrf"
        assert term.arguments == ["condition"]
        assert term.kwargs["basis"] == "spm"
    
    def test_multiple_terms(self):
        """Test parsing formula with multiple terms."""
        parser = FormulaParser()
        formula = parser.parse("onset ~ hrf(condition) + hrf(block)")
        
        assert len(formula.rhs) == 2
        assert formula.rhs[0].arguments == ["condition"]
        assert formula.rhs[1].arguments == ["block"]
    
    def test_interaction_formula(self):
        """Test parsing interaction formula."""
        parser = FormulaParser()
        formula = parser.parse("onset ~ hrf(condition:block)")
        
        term = formula.rhs[0]
        assert term.function == "hrf"
        assert term.arguments == ["condition:block"]
    
    def test_invalid_formula(self):
        """Test invalid formula raises error."""
        parser = FormulaParser()
        
        with pytest.raises(ValueError, match="Invalid formula syntax"):
            parser.parse("invalid formula")
    
    def test_formula_evaluation(self):
        """Test formula evaluation with context."""
        # Create test data
        df = pd.DataFrame({
            'onset': [1.0, 2.0, 3.0],
            'condition': ['A', 'B', 'A']
        })
        
        context = FormulaContext(data=df)
        result = parse_formula("onset ~ hrf(condition)", context)
        
        assert result['onset_var'] == 'onset'
        assert np.array_equal(result['onsets'], df['onset'].values)
        assert len(result['events']) == 1
        assert result['events'][0]['event']['name'] == 'condition'

    def test_event_model_mode_interaction_term_conversion(self):
        """Interaction inside hrf() should map to a multi-event Term."""
        terms = parse_formula(
            "onset ~ hrf(condition:block, basis='spmg1')",
            for_event_model=True,
        )

        assert len(terms) == 1
        assert terms[0].events == ["condition", "block"]
        assert terms[0].is_interaction
