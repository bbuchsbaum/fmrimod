"""Test EventModel contrast_weights handler."""

import pytest
import numpy as np
import pandas as pd

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fmrimod import event_model
from fmrimod.contrast import (
    contrast_weights,
    plot_contrasts,
    pair_contrast,
    unit_contrast,
    contrast_set,
)
from fmrimod.contrast.contrast_spec import Formula
from fmrimod.formula.base import Term


@pytest.fixture(autouse=True)
def close_all_plots():
    """Close all matplotlib figures after each test."""
    yield
    plt.close('all')


@pytest.fixture
def simple_data():
    """Create simple test data with 3-level categorical variable."""
    return pd.DataFrame({
        'onset': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        'condition': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C'],
    })


@pytest.fixture
def model_with_contrasts(simple_data):
    """Create EventModel with contrast_specs on a term."""
    # Create a Term with contrast specifications
    term = Term('condition')
    term.contrast_specs = contrast_set(
        pair_contrast(Formula("A"), Formula("B"), name='A_vs_B'),
        pair_contrast(Formula("A"), Formula("C"), name='A_vs_C'),
        unit_contrast(Formula("A"), name='just_A'),
    )

    return event_model([term], data=simple_data, tr=2.0, n_scans=20)


class TestEventModelContrastWeights:
    """Test contrast_weights dispatch for EventModel."""
    
    def test_contrast_weights_returns_nested_dict(self, model_with_contrasts):
        """Test that contrast_weights returns nested dict structure."""
        result = contrast_weights(model_with_contrasts)
        
        assert isinstance(result, dict)
        assert 'condition' in result
        assert isinstance(result['condition'], dict)
    
    def test_contrast_weights_extracts_all_contrasts(self, model_with_contrasts):
        """Test that all contrast specs are extracted."""
        result = contrast_weights(model_with_contrasts)

        condition_contrasts = result['condition']
        assert 'A_vs_B' in condition_contrasts
        assert 'A_vs_C' in condition_contrasts
        assert 'just_A' in condition_contrasts
    
    def test_contrast_weights_have_full_design_matrix_size(self, model_with_contrasts):
        """Test that contrast weights are in full design matrix space."""
        result = contrast_weights(model_with_contrasts)
        
        ncond = model_with_contrasts.design_matrix.shape[1]
        
        for term_name, term_contrasts in result.items():
            for contrast_name, contrast_obj in term_contrasts.items():
                assert contrast_obj.weights.shape[0] == ncond
                assert hasattr(contrast_obj, 'offset_weights')
                assert contrast_obj.offset_weights.shape[0] == ncond
    
    def test_contrast_weights_have_correct_nonzero_entries(self, model_with_contrasts):
        """Test that weights are placed at correct indices."""
        result = contrast_weights(model_with_contrasts)
        
        # Get the A_vs_B contrast
        a_vs_b = result['condition']['A_vs_B']
        
        # Should have nonzero weights only in condition term columns
        # The condition term should have 3 columns (A, B, C)
        nonzero_indices = np.where(np.abs(a_vs_b.offset_weights).sum(axis=1) > 1e-10)[0]
        
        # Should have exactly 2 nonzero entries (for A and B)
        assert len(nonzero_indices) == 2
    
    def test_contrast_object_structure(self, model_with_contrasts):
        """Test that contrast objects have expected structure."""
        result = contrast_weights(model_with_contrasts)

        a_vs_b = result['condition']['A_vs_B']

        # Should have weights and name
        assert hasattr(a_vs_b, 'weights')
        assert hasattr(a_vs_b, 'name')
        assert a_vs_b.name == 'A_vs_B'


class TestEventModelPlotContrasts:
    """Test plot_contrasts dispatch for EventModel."""
    
    def test_plot_contrasts_with_event_model(self, model_with_contrasts):
        """Test that plot_contrasts works with EventModel."""
        fig = plot_contrasts(model_with_contrasts)
        
        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) > 0
    
    def test_plot_contrasts_shows_all_contrasts(self, model_with_contrasts):
        """Test that all contrasts are shown in the plot."""
        fig = plot_contrasts(model_with_contrasts)
        
        ax = fig.axes[0]
        y_labels = [label.get_text() for label in ax.get_yticklabels()]
        
        # Should have labels for all three contrasts
        assert any('A_vs_B' in label for label in y_labels)
        assert any('A_vs_C' in label for label in y_labels)
        assert any('just_A' in label for label in y_labels)


def _nonzero_columns(model, weights):
    """``{column_name: weight}`` for the non-zero entries of a full-design vector."""
    w = np.asarray(weights).ravel()
    return {
        model.column_names[i]: round(float(w[i]), 12)
        for i in np.flatnonzero(np.abs(w) > 1e-12)
    }


class TestAttachedContrastPlacement:
    """Term-attached contrasts land on the term's columns by name.

    DES-01: weights were embedded by position, so on multi-basis terms they
    hit the wrong basis columns, a contrast on one of two same-named terms
    landed on the other, and spec-layer ``hrf(..., contrasts=)`` was dropped
    (R ``.embed_term_weights``, fmridesign 4415504 / a0081c7;
    test-multibasis-column-order.R).
    """

    @pytest.fixture
    def ab_data(self):
        return pd.DataFrame({
            'onset': [2.0, 20.0, 40.0, 60.0],
            'cond': ['A', 'B', 'A', 'B'],
        })

    def test_multibasis_contrasts_follow_condition_and_basis(self, ab_data):
        term = Term('cond', hrf='spmg3')
        term.contrast_specs = contrast_set(
            pair_contrast(Formula("cond == 'A'"), Formula("cond == 'B'"), name='A_vs_B'),
            unit_contrast(Formula("cond == 'B'"), name='just_B'),
        )
        em = event_model([term], data=ab_data, tr=2.0, n_scans=50)
        res = contrast_weights(em)['cond']
        facts = em.column_facts

        just_b = res['just_B'].offset_weights[:, 0]
        for fact, w in zip(facts, just_b, strict=True):
            assert (w != 0) == (fact['condition'] == 'cond.B')
        assert just_b[em.column_names.index('cond_cond.A_b02')] == 0.0
        # Replicated on every basis column of B, like R's basis expansion.
        assert _nonzero_columns(em, just_b) == {
            'cond_cond.B_b01': 1.0, 'cond_cond.B_b02': 1.0, 'cond_cond.B_b03': 1.0,
        }

        a_vs_b = res['A_vs_B'].offset_weights[:, 0]
        expected = {
            f'cond_cond.{c}_b0{b}': (1.0 if c == 'A' else -1.0)
            for c in 'AB' for b in (1, 2, 3)
        }
        assert _nonzero_columns(em, a_vs_b) == expected

    def test_same_named_terms_resolve_by_term_identity(self, ab_data):
        t1 = Term('cond', hrf='spmg1')
        t2 = Term('cond', hrf='spmg2')
        t1.contrast_specs = contrast_set(
            pair_contrast(Formula("cond == 'A'"), Formula("cond == 'B'"), name='first'))
        t2.contrast_specs = contrast_set(
            pair_contrast(Formula("cond == 'A'"), Formula("cond == 'B'"), name='second'))
        em = event_model([t1, t2], data=ab_data, tr=2.0, n_scans=60)
        assert em.column_names == [
            'cond_cond.A', 'cond_cond.B',
            'cond#1_cond.A_b01', 'cond#1_cond.A_b02', 'cond#1_cond.B_b01', 'cond#1_cond.B_b02',
        ]
        # The repeated name no longer overwrites the first term's indices.
        assert em.column_indices == {'cond': [0, 1], 'cond#1': [2, 3, 4, 5]}
        res = contrast_weights(em)
        assert set(res) == {'cond', 'cond#1'}
        assert _nonzero_columns(em, res['cond']['first'].offset_weights) == {
            'cond_cond.A': 1.0, 'cond_cond.B': -1.0,
        }
        assert _nonzero_columns(em, res['cond#1']['second'].offset_weights) == {
            'cond#1_cond.A_b01': 1.0, 'cond#1_cond.A_b02': 1.0,
            'cond#1_cond.B_b01': -1.0, 'cond#1_cond.B_b02': -1.0,
        }

    def test_modulated_term_contrast_lands_on_its_cells(self):
        data = pd.DataFrame({
            'onset': [2.0, 20.0, 40.0, 60.0, 80.0, 100.0],
            'cond': ['A', 'B', 'C'] * 2,
            'rt': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        })
        term = Term(['cond', 'rt'])
        term.contrast_specs = contrast_set(
            pair_contrast(Formula("cond == 'A'"), Formula("cond == 'B'"), name='ab'))
        em = event_model([term], data=data, tr=2.0, n_scans=70)
        w = contrast_weights(em)['cond:rt']['ab'].offset_weights[:, 0]
        by_level = {fact['level']: w[fact['index']] for fact in em.column_facts}
        assert by_level == {'A': 1.0, 'B': -1.0, 'C': 0.0}

    def test_spec_layer_contrasts_are_not_dropped(self):
        from fmrimod.sampling import SamplingFrame
        from fmrimod.spec import as_spec, compile_events
        from fmrimod.spec import hrf as spec_hrf

        events = pd.DataFrame({
            'onset': [2.0, 20.0, 40.0, 60.0], 'cond': ['A', 'B', 'A', 'B'], 'run': 1,
        })
        term = spec_hrf(
            'cond', basis='spmg2',
            contrasts=[pair_contrast({'cond': 'A'}, {'cond': 'B'}, name='AvB')],
        )
        em = compile_events(as_spec(term), events, SamplingFrame(blocklens=[50], tr=2.0),
                            block='run', durations=None, precision=None)
        res = contrast_weights(em)
        assert list(res) == ['cond']
        assert _nonzero_columns(em, res['cond']['AvB'].offset_weights) == {
            'cond_cond.A_b01': 1.0, 'cond_cond.A_b02': 1.0,
            'cond_cond.B_b01': -1.0, 'cond_cond.B_b02': -1.0,
        }

    def test_spec_layer_contrast_reaches_the_fit(self):
        import fmrimod as fm
        from fmrimod.spec import hrf as spec_hrf

        events = pd.DataFrame({
            'onset': [4.0, 24.0, 44.0, 64.0], 'duration': 0.0,
            'cond': ['A', 'B', 'A', 'B'], 'run': 1,
        })
        rng = np.random.default_rng(0)
        ds = fm.fmri_dataset(rng.normal(size=(50, 3)), tr=2.0, events=events)
        fit = fm.fmri_lm(
            spec_hrf('cond', contrasts=[pair_contrast({'cond': 'A'}, {'cond': 'B'}, name='AvB')]),
            ds,
        )
        named = fit.contrast('cond#AvB')
        direct = fit.contrast(
            np.asarray([1.0, -1.0] + [0.0] * (fit.betas.shape[0] - 2)), name='direct'
        )
        np.testing.assert_allclose(named.estimate, direct.estimate)


class TestEventModelWithoutContrasts:
    """Test EventModel without contrast_specs."""
    
    def test_contrast_weights_returns_empty_dict(self, simple_data):
        """Test that model without contrast_specs returns empty dict."""
        model = event_model("condition", data=simple_data, tr=2.0, n_scans=20)
        
        result = contrast_weights(model)
        
        assert isinstance(result, dict)
        assert len(result) == 0
