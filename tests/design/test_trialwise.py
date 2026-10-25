"""Tests for trialwise functionality."""

import pytest
import numpy as np
import pandas as pd

from fmrimod import event_model
from fmrimod.trialwise import trialwise
from fmrimod.formula import Term


class TestTrialwise:
    """Test trialwise single-trial regressor functionality."""
    
    @pytest.fixture
    def trial_data(self):
        """Create data with multiple trials."""
        return pd.DataFrame({
            'onset': [10, 20, 30, 40, 50],
            'condition': ['A', 'B', 'A', 'B', 'A']
        })
    
    def test_trialwise_basic(self, trial_data):
        """Test basic trialwise functionality."""
        model = event_model(
            "trialwise()",
            data=trial_data,
            tr=2.0,
            n_scans=100
        )
        
        # Check that we have one column per trial
        assert model.design_matrix.shape[1] == 5
        
        # Check column names
        assert model.column_names == ['trial_1', 'trial_2', 'trial_3', 'trial_4', 'trial_5']
    
    def test_trialwise_with_sum(self, trial_data):
        """Test trialwise with add_sum=True."""
        model = event_model(
            "trialwise(add_sum=True)",
            data=trial_data,
            tr=2.0,
            n_scans=100
        )
        
        # Check that we have one column per trial plus sum
        assert model.design_matrix.shape[1] == 6
        
        # Check column names
        expected_names = ['trial_1', 'trial_2', 'trial_3', 'trial_4', 'trial_5', 'trial_mean']
        assert model.column_names == expected_names
        
        # Check that last column is mean of others
        mean_col = model.design_matrix[:, -1]
        calc_mean = model.design_matrix[:, :-1].mean(axis=1)
        np.testing.assert_allclose(mean_col, calc_mean)
    
    def test_trialwise_custom_label(self, trial_data):
        """Test trialwise with custom label."""
        model = event_model(
            'trialwise(label="event")',
            data=trial_data,
            tr=2.0,
            n_scans=100
        )
        
        # Check column names use custom label
        assert all(name.startswith('event_') for name in model.column_names)
    
    def test_trialwise_with_hrf(self, trial_data):
        """Test trialwise with different HRF basis."""
        model = event_model(
            'trialwise(basis="spmg1")',
            data=trial_data,
            tr=2.0,
            n_scans=100
        )
        
        # Should still have one column per trial
        assert model.design_matrix.shape[1] == 5
        
        # Check that design matrix has been convolved (non-zero beyond event times)
        # Find non-zero entries
        nonzero_rows = np.any(model.design_matrix != 0, axis=1)
        nonzero_indices = np.where(nonzero_rows)[0]
        
        # Should have response beyond just the event times
        assert len(nonzero_indices) > 5
    
    def test_trialwise_combined_with_regular_terms(self, trial_data):
        """Test trialwise combined with regular event terms."""
        model = event_model(
            "condition + trialwise(add_sum=True)",
            data=trial_data,
            tr=2.0,
            n_scans=100
        )
        
        # Should have 2 condition columns + 5 trial columns + 1 mean column
        assert model.design_matrix.shape[1] == 8
        
        # Check that first two columns are for conditions
        assert model.column_names[0] == 'condition_condition.A'
        assert model.column_names[1] == 'condition_condition.B'
        
        # Check trial columns
        assert model.column_names[2:7] == ['trial_1', 'trial_2', 'trial_3', 'trial_4', 'trial_5']
        assert model.column_names[7] == 'trial_mean'
    
    def test_trialwise_function_directly(self):
        """Test trialwise function returns proper Term."""
        term = trialwise(
            basis="gamma",
            lag=2.0,
            add_sum=True,
            label="mytrial"
        )
        
        assert isinstance(term, Term)
        assert hasattr(term, '_is_trialwise')
        assert term._is_trialwise is True
        assert term._add_sum is True
        assert term._trialwise_label == "mytrial"
        assert term.hrf == "gamma"
        assert term._lag == 2.0
    
    def test_trialwise_no_data(self):
        """Test error when no trial data available."""
        empty_data = pd.DataFrame({
            'onset': [],
            'condition': []
        })
        
        # Create model - this should work
        model = event_model(
            "trialwise()",
            data=empty_data,
            tr=2.0,
            n_scans=100
        )
        
        # Error should happen when accessing design_matrix
        with pytest.raises(ValueError, match="no trials found"):
            _ = model.design_matrix
    
    def test_trialwise_large_dataset(self):
        """Test trialwise with many trials."""
        # Create data with 100 trials
        n_trials = 100
        data = pd.DataFrame({
            'onset': np.arange(n_trials) * 10,
            'condition': ['A', 'B'] * (n_trials // 2)
        })
        
        model = event_model(
            "trialwise()",
            data=data,
            tr=2.0,
            n_scans=1200
        )
        
        # Should have one column per trial
        assert model.design_matrix.shape[1] == n_trials
        
        # Check zero-padded names
        assert model.column_names[0] == 'trial_001'
        assert model.column_names[-1] == 'trial_100'


class TestTrialwiseHrfOptions:
    """``trialwise()`` forwards ``nbasis``/``lag`` to the HRF (R e799253).

    DES-07: the legacy and string-formula ``trialwise()`` stored these as
    private attributes that nothing read.
    """

    @pytest.fixture
    def trial_data(self):
        return pd.DataFrame({'onset': [10.0, 40.0, 70.0, 100.0]})

    def test_string_trialwise_nbasis_sizes_fir(self, trial_data):
        model = event_model(
            "onset ~ trialwise(basis='fir', nbasis=6)",
            data=trial_data, tr=2.0, n_scans=80,
        )
        # 6 FIR bins per trial (the FIR default is 12, so 6 is distinguishable)
        assert model.design_matrix.shape[1] == 6 * len(trial_data)
        assert model.column_names[:6] == [f"trial_1_b{b:02d}" for b in range(1, 7)]

    def test_legacy_trialwise_nbasis_sizes_fir(self, trial_data):
        model = event_model(
            [trialwise(basis='fir', nbasis=6)], data=trial_data, tr=2.0, n_scans=80,
        )
        assert model.design_matrix.shape[1] == 6 * len(trial_data)

    @pytest.mark.parametrize(
        "lagged",
        ["onset ~ trialwise(lag=4)", [trialwise(lag=4.0)]],
        ids=["string", "legacy"],
    )
    def test_trialwise_lag_shifts_regressors(self, trial_data, lagged):
        base = event_model("onset ~ trialwise()", data=trial_data, tr=2.0, n_scans=80)
        shifted = event_model(lagged, data=trial_data, tr=2.0, n_scans=80)
        X0 = base.design_matrix
        X4 = shifted.design_matrix
        # Same as an ordinary hrf(lag=4) term with one level per trial.
        per_trial = trial_data.assign(tid=[f"t{i}" for i in range(len(trial_data))])
        ref = event_model("onset ~ hrf(tid, lag=4)", data=per_trial, tr=2.0, n_scans=80)
        np.testing.assert_allclose(X4, ref.design_matrix, atol=1e-12)
        # A 4 s lag at TR=2 moves each regressor down by exactly 2 scans.
        # Compared over the first 20 s after each onset, i.e. inside the
        # unlagged HRF span (24 s) minus the lag.
        times = np.arange(80) * 2.0
        for j, onset in enumerate(trial_data['onset']):
            rows = np.flatnonzero((times >= onset) & (times < onset + 20.0))
            np.testing.assert_allclose(X4[rows + 2, j], X0[rows, j], atol=1e-12)
            assert np.max(np.abs(X0[rows, j])) > 0.1

    def test_string_trialwise_matches_legacy_term_options(self):
        from fmrimod.formula import parse_formula

        (parsed,) = parse_formula(
            "onset ~ trialwise(basis='fir', nbasis=6, lag=2, durations='dur', normalize=True)",
            for_event_model=True,
        )
        direct = trialwise(basis='fir', nbasis=6, lag=2.0, durations='dur', normalize=True)
        assert parsed._kwargs == direct._kwargs
        assert parsed._kwargs['nbasis'] == 6
        assert parsed._kwargs['lag'] == 2.0
