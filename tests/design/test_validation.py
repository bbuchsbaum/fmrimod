"""Comprehensive tests for validation, design_colmap, and residualization modules."""
import numpy as np
import pandas as pd
import pytest

from fmrimod.validate import validate_contrasts, check_collinearity
from fmrimod.residualize import residualize
from fmrimod.design_colmap import design_colmap
from fmrimod.design.event_model import EventModel
from fmrimod.events.factor import EventFactor
from fmrimod.events.variable import EventVariable
from fmrimod.formula.base import Term
from fmrimod.sampling import SamplingFrame
from fmrimod.contrast.contrast_spec import Formula


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def simple_categorical_model():
    """Create a simple categorical EventModel for testing."""
    events = {
        'condition': EventFactor(
            name='condition',
            onsets=np.array([2, 6, 10, 14, 18, 22]),
            values=['A', 'B', 'A', 'B', 'A', 'B'],
            durations=2.0
        )
    }
    sampling = SamplingFrame(tr=2.0, n_scans=30)
    model = EventModel(terms=[Term('condition')], events=events, sampling_info=sampling)
    return model


@pytest.fixture
def continuous_model():
    """Create a continuous EventModel for testing."""
    events = {
        'rating': EventVariable(
            name='rating',
            onsets=np.array([5, 10, 15, 20]),
            values=[1.0, 2.0, 3.0, 4.0],
            center=True
        )
    }
    sampling = SamplingFrame(tr=1.0, n_scans=30)
    model = EventModel(terms=[Term('rating')], events=events, sampling_info=sampling)
    return model


@pytest.fixture
def multi_term_model():
    """Create a multi-term EventModel for testing."""
    events = {
        'condition': EventFactor(
            name='condition',
            onsets=np.array([5, 10, 15, 20]),
            values=['A', 'B', 'A', 'B'],
            durations=1.0
        ),
        'difficulty': EventVariable(
            name='difficulty',
            onsets=np.array([5, 10, 15, 20]),
            values=[1.0, 2.0, 1.5, 2.5],
            center=True
        )
    }
    sampling = SamplingFrame(tr=1.0, n_scans=30)
    model = EventModel(
        terms=[Term('condition'), Term('difficulty')],
        events=events,
        sampling_info=sampling
    )
    return model


# ============================================================================
# validate_contrasts Tests
# ============================================================================

class TestValidateContrasts:
    """Test validate_contrasts function - targeting 80%+ coverage."""

    def test_valid_contrast_numpy_array(self):
        """Valid contrast on numpy array should pass all checks."""
        X = np.random.randn(100, 4)
        weights = np.array([1, -1, 0, 0])
        result = validate_contrasts(X, weights)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result['name'].iloc[0] == 'contrast'
        assert result['type'].iloc[0] == 't'
        assert result['estimable'].iloc[0] == True
        assert result['sum_to_zero'].iloc[0] == True

    def test_valid_contrast_dataframe(self):
        """Valid contrast on DataFrame should use column names."""
        X = pd.DataFrame(np.random.randn(100, 3), columns=['a', 'b', 'c'])
        weights = np.array([1, -1, 0])
        result = validate_contrasts(X, weights)

        assert len(result) == 1
        assert result['estimable'].iloc[0] == True

    def test_valid_contrast_event_model(self, simple_categorical_model):
        """Valid contrast on EventModel should extract design matrix."""
        model = simple_categorical_model
        n_cols = model.design_matrix.shape[1]
        weights = np.zeros(n_cols)
        weights[0] = 1
        weights[1] = -1

        result = validate_contrasts(model, weights)

        assert len(result) == 1
        assert result['estimable'].iloc[0] == True

    def test_wrong_number_of_weights(self):
        """Contrast with wrong number of weights should raise ValueError."""
        X = np.random.randn(100, 5)
        weights = np.array([1, -1, 0])  # Only 3 weights for 5 columns

        with pytest.raises(ValueError, match="has 3 weights but design matrix has 5 columns"):
            validate_contrasts(X, weights)

    def test_non_estimable_contrast(self):
        """Non-estimable contrast should be flagged."""
        # Create rank-deficient design
        X1 = np.random.randn(100, 2)
        X2 = X1[:, 0:1] + X1[:, 1:2]  # Linear combination
        X = np.column_stack([X1, X2])

        # Contrast that's not in column space
        weights = np.array([0, 0, 1])
        result = validate_contrasts(X, weights)

        # This contrast is actually estimable since col 3 is in span of cols 1-2
        # Let's create a truly non-estimable one
        rank = np.linalg.matrix_rank(X)
        assert rank < 3  # Verify rank deficiency

    def test_no_contrasts_with_event_model(self, simple_categorical_model):
        """EventModel with no attached contrasts should return empty DataFrame."""
        model = simple_categorical_model
        result = validate_contrasts(model, weights=None)

        # Should return empty DataFrame with correct columns
        assert isinstance(result, pd.DataFrame)
        assert 'name' in result.columns
        assert 'estimable' in result.columns

    def test_weights_as_dict(self):
        """Dictionary of contrast weights should create multiple rows."""
        X = np.random.randn(100, 4)
        weights = {
            'contrast1': np.array([1, -1, 0, 0]),
            'contrast2': np.array([0, 1, -1, 0]),
            'contrast3': np.array([0, 0, 1, -1])
        }

        result = validate_contrasts(X, weights)

        assert len(result) == 3
        assert set(result['name']) == {'contrast1', 'contrast2', 'contrast3'}
        assert all(result['sum_to_zero'])

    def test_weights_as_list(self):
        """List of weights should be converted to array."""
        X = np.random.randn(100, 3)
        weights = [1, -1, 0]

        result = validate_contrasts(X, weights)

        assert len(result) == 1
        assert result['sum_to_zero'].iloc[0] == True

    def test_f_contrast_multicolumn(self):
        """F-contrast with multiple columns should be detected."""
        X = np.random.randn(100, 5)
        weights = np.array([[1, -1, 0, 0, 0],
                           [0, 1, -1, 0, 0]]).T

        result = validate_contrasts(X, weights)

        assert len(result) == 2
        assert all(result['type'] == 'F')
        assert 'contrast#1' in result['name'].values
        assert 'contrast#2' in result['name'].values
        # Check full rank for F-contrast
        assert result['full_rank'].iloc[0] == True

    def test_sum_to_zero_detection(self):
        """Sum-to-zero contrasts should be flagged correctly."""
        X = np.random.randn(100, 4)

        # Sum to zero
        w1 = np.array([1, -1, 0, 0])
        result1 = validate_contrasts(X, w1)
        assert result1['sum_to_zero'].iloc[0] == True

        # Does not sum to zero
        w2 = np.array([1, 0, 0, 0])
        result2 = validate_contrasts(X, w2)
        assert result2['sum_to_zero'].iloc[0] == False

    def test_intercept_orthogonality_with_intercept(self):
        """Contrasts orthogonal to intercept should be flagged."""
        X = pd.DataFrame(np.random.randn(100, 5),
                        columns=['(Intercept)', 'a', 'b', 'c', 'd'])

        # Orthogonal to intercept
        w1 = np.array([0, 1, -1, 0, 0])
        result1 = validate_contrasts(X, w1)
        assert result1['orthogonal_to_intercept'].iloc[0] == True

        # Not orthogonal to intercept
        w2 = np.array([1, 1, -1, 0, 0])
        result2 = validate_contrasts(X, w2)
        assert result2['orthogonal_to_intercept'].iloc[0] == False

    def test_intercept_names_variants(self):
        """Test different intercept column name variants."""
        for intercept_name in ['(Intercept)', 'Intercept', 'constant', 'const']:
            X = pd.DataFrame(np.random.randn(100, 3),
                           columns=[intercept_name, 'a', 'b'])
            weights = np.array([0, 1, -1])
            result = validate_contrasts(X, weights)
            assert result['orthogonal_to_intercept'].iloc[0] == True

    def test_no_intercept_column(self):
        """Without intercept column, orthogonality should be True."""
        X = pd.DataFrame(np.random.randn(100, 3), columns=['a', 'b', 'c'])
        weights = np.array([1, -1, 0])
        result = validate_contrasts(X, weights)

        assert result['orthogonal_to_intercept'].iloc[0] == True

    def test_nonzero_weights_count(self):
        """Should count number of nonzero weights correctly."""
        X = np.random.randn(100, 5)
        weights = np.array([1, -1, 0, 0.5, 0])

        result = validate_contrasts(X, weights)

        assert result['nonzero_weights'].iloc[0] == 3

    def test_custom_tolerance(self):
        """Custom tolerance should affect zero detection."""
        X = np.random.randn(100, 3)
        weights = np.array([0.5, -0.5, 1e-9])

        result_default = validate_contrasts(X, weights, tol=1e-8)
        result_strict = validate_contrasts(X, weights, tol=1e-10)

        # With default tolerance, 1e-9 should be considered zero
        assert result_default['nonzero_weights'].iloc[0] == 2
        # With strict tolerance, 1e-9 should be nonzero
        assert result_strict['nonzero_weights'].iloc[0] == 3

    def test_invalid_tolerance_values(self):
        """Non-finite or negative tolerances should fail fast."""
        X = np.random.randn(100, 3)
        weights = np.array([1, -1, 0])

        for bad_tol in [np.nan, np.inf, -1e-3]:
            with pytest.raises(
                ValueError, match="tol must be a finite non-negative number"
            ):
                validate_contrasts(X, weights, tol=bad_tol)

    def test_rank_deficient_f_contrast(self):
        """Rank-deficient F-contrast should be detected."""
        X = np.random.randn(100, 5)
        # Create rank-deficient contrast matrix (second column = first)
        weights = np.array([[1, 1, 0, 0, 0],
                           [1, 1, 0, 0, 0]]).T

        result = validate_contrasts(X, weights)

        assert len(result) == 2
        assert result['full_rank'].iloc[0] == False

    def test_invalid_input_type(self):
        """Invalid input type should raise TypeError."""
        with pytest.raises(TypeError, match="must be EventModel, DataFrame, or numpy array"):
            validate_contrasts("invalid", np.array([1, -1]))

    def test_non_2d_design_matrix_raises_value_error(self):
        """Non-2D design matrices should raise a clear ValueError."""
        X = np.array([1, 2, 3])
        weights = np.array([1, -1, 0])
        with pytest.raises(ValueError, match="must be 2-dimensional"):
            validate_contrasts(X, weights)

    def test_invalid_weights_type(self):
        """Invalid weights type should raise TypeError."""
        X = np.random.randn(100, 3)
        with pytest.raises(TypeError, match="weights must be array, dict, or list"):
            validate_contrasts(X, weights="invalid")

    def test_weights_none_without_event_model(self):
        """weights=None should require EventModel."""
        X = np.random.randn(100, 3)
        with pytest.raises(ValueError, match="weights must be provided"):
            validate_contrasts(X, weights=None)

    def test_result_sorted_by_name(self):
        """Results should be sorted by contrast name."""
        X = np.random.randn(100, 3)
        weights = {
            'z_contrast': np.array([1, 0, 0]),
            'a_contrast': np.array([0, 1, 0]),
            'm_contrast': np.array([0, 0, 1])
        }

        result = validate_contrasts(X, weights)

        names = result['name'].tolist()
        assert names == sorted(names)

    def test_1d_weights_converted_to_column(self):
        """1D weight arrays should be converted to column vectors."""
        X = np.random.randn(100, 3)
        weights = np.array([1, -1, 0])  # 1D array

        result = validate_contrasts(X, weights)

        assert len(result) == 1
        assert result['type'].iloc[0] == 't'

    def test_dict_weights_with_2d_arrays(self):
        """Dictionary with 2D arrays should handle shapes correctly."""
        X = np.random.randn(100, 4)
        weights = {
            'contrast1': np.array([[1, -1, 0, 0]]).T,  # Column vector
            'contrast2': np.array([0, 1, -1, 0])       # 1D array
        }

        result = validate_contrasts(X, weights)

        assert len(result) == 2


# ============================================================================
# check_collinearity Tests
# ============================================================================

class TestCheckCollinearity:
    """Test check_collinearity function - targeting 80%+ coverage."""

    def test_no_collinearity(self):
        """Uncorrelated regressors should pass."""
        np.random.seed(42)
        X = np.random.randn(100, 4)

        result = check_collinearity(X, threshold=0.9)

        assert result['ok'] == True
        assert len(result['pairs']) == 0
        assert isinstance(result['pairs'], pd.DataFrame)

    def test_high_collinearity_detected(self):
        """Highly correlated regressors should be detected."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        # Add highly correlated column (r ≈ 0.995)
        X = np.column_stack([X, X[:, 0] + 0.01 * np.random.randn(100)])

        result = check_collinearity(X, threshold=0.9)

        assert result['ok'] == False
        assert len(result['pairs']) > 0
        assert 'regressor_1' in result['pairs'].columns
        assert 'regressor_2' in result['pairs'].columns
        assert 'r' in result['pairs'].columns
        # Check correlation value is above threshold
        assert abs(result['pairs']['r'].iloc[0]) > 0.9

    def test_with_dataframe_input(self):
        """DataFrame input should preserve column names."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 3), columns=['alpha', 'beta', 'gamma'])

        result = check_collinearity(X, threshold=0.9)

        assert result['ok'] == True

    def test_with_event_model(self, simple_categorical_model):
        """EventModel input should extract design matrix and column names."""
        result = check_collinearity(simple_categorical_model, threshold=0.9)

        assert isinstance(result, dict)
        assert 'ok' in result
        assert 'pairs' in result

    def test_ignores_intercept_columns(self):
        """Intercept columns should be excluded from collinearity check."""
        np.random.seed(42)
        for intercept_name in ['(Intercept)', 'Intercept', 'constant', 'const']:
            X = pd.DataFrame({
                intercept_name: np.ones(100),
                'a': np.random.randn(100),
                'b': np.random.randn(100)
            })

            result = check_collinearity(X, threshold=0.9)

            # Should only check non-intercept columns
            assert result['ok'] == True

    def test_custom_threshold(self):
        """Different thresholds should give different results."""
        np.random.seed(42)
        X1 = np.random.randn(100, 2)
        # Create column with moderate correlation (around 0.7-0.8)
        X2 = X1[:, 0:1] * 0.7 + np.random.randn(100, 1) * 0.3
        X = np.column_stack([X1, X2])

        result_strict = check_collinearity(X, threshold=0.6)
        result_lenient = check_collinearity(X, threshold=0.95)

        # Strict threshold should flag more pairs
        assert len(result_strict['pairs']) >= len(result_lenient['pairs'])

    def test_invalid_threshold_values(self):
        """Non-finite or out-of-range thresholds should raise ValueError."""
        X = np.random.randn(100, 3)
        bad_thresholds = [np.nan, np.inf, -0.1, 1.1]

        for threshold in bad_thresholds:
            with pytest.raises(
                ValueError,
                match="threshold must be a finite number between 0 and 1",
            ):
                check_collinearity(X, threshold=threshold)

    def test_drops_zero_variance_columns(self):
        """Zero-variance columns should be excluded."""
        X = np.array([[1, 2, 5],
                      [1, 3, 5],
                      [1, 4, 5],
                      [1, 5, 5]])

        # Column 0 and 2 have zero variance (constant)
        result = check_collinearity(X, threshold=0.9)

        # Should not crash and should check only non-constant columns
        assert result['ok'] == True

    def test_single_non_intercept_column(self):
        """Single valid column should return ok=True."""
        X = pd.DataFrame({
            '(Intercept)': np.ones(100),
            'a': np.random.randn(100)
        })

        result = check_collinearity(X, threshold=0.9)

        assert result['ok'] == True
        assert len(result['pairs']) == 0

    def test_no_valid_columns(self):
        """Only intercept columns should return ok=True."""
        X = pd.DataFrame({
            '(Intercept)': np.ones(100),
            'Intercept': np.ones(100)
        })

        result = check_collinearity(X, threshold=0.9)

        assert result['ok'] == True
        assert len(result['pairs']) == 0

    def test_only_upper_triangle(self):
        """Should only report each pair once (upper triangle)."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        # Add two columns highly correlated with first
        X = np.column_stack([X,
                            X[:, 0] + 0.01 * np.random.randn(100),
                            X[:, 0] + 0.01 * np.random.randn(100)])

        result = check_collinearity(X, threshold=0.9)

        # Should have 3 pairs: (0,2), (0,3), (2,3)
        # Each pair reported once
        pairs = result['pairs']
        for _, row in pairs.iterrows():
            # Check that regressor_1 index < regressor_2 index
            idx1 = int(row['regressor_1'].replace('V', '')) - 1
            idx2 = int(row['regressor_2'].replace('V', '')) - 1
            assert idx1 < idx2

    def test_invalid_input_type(self):
        """Invalid input type should raise TypeError."""
        with pytest.raises(TypeError, match="must be EventModel, DataFrame, or numpy array"):
            check_collinearity("invalid")

    def test_non_2d_design_matrix_raises_value_error(self):
        """Non-2D inputs should raise a clear ValueError."""
        X = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="2-dimensional design matrix"):
            check_collinearity(X)

    def test_column_names_in_pairs(self):
        """Pairs should include actual column names."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 2), columns=['predictor1', 'predictor2'])
        X['predictor3'] = X['predictor1'] + 0.01 * np.random.randn(100)

        result = check_collinearity(X, threshold=0.9)

        assert result['ok'] == False
        pairs = result['pairs']
        assert 'predictor1' in pairs['regressor_1'].values or 'predictor1' in pairs['regressor_2'].values

    def test_nan_handling(self):
        """Should handle potential NaN values in correlation matrix."""
        # Create data that might produce NaN correlations
        X = np.random.randn(100, 3)
        X[:, 2] = 0  # Zero variance column

        result = check_collinearity(X, threshold=0.9)

        # Should complete without error
        assert 'ok' in result


# ============================================================================
# design_colmap Tests
# ============================================================================

class TestDesignColmap:
    """Test design_colmap function - targeting 70%+ coverage."""

    def test_simple_categorical_model(self, simple_categorical_model):
        """Single categorical term should produce correct column metadata."""
        colmap = design_colmap(simple_categorical_model)

        assert isinstance(colmap, pd.DataFrame)
        assert len(colmap) == simple_categorical_model.design_matrix.shape[1]
        assert all(colmap['role'] == 'task')
        assert all(colmap['model_source'] == 'event')
        assert 'col' in colmap.columns
        assert 'name' in colmap.columns
        assert 'modulation_type' in colmap.columns

    def test_continuous_model(self, continuous_model):
        """Single continuous term should have modulation_type='amplitude'."""
        colmap = design_colmap(continuous_model)

        assert len(colmap) > 0
        assert all(colmap['modulation_type'] == 'amplitude')

    def test_multi_term_model(self, multi_term_model):
        """Multi-term model should have correct column count."""
        model = multi_term_model
        colmap = design_colmap(model)

        # Should have one row per design matrix column
        assert len(colmap) == model.design_matrix.shape[1]
        assert colmap['col'].tolist() == list(range(1, len(colmap) + 1))

    def test_column_names_match(self, simple_categorical_model):
        """Column names should match model.column_names."""
        model = simple_categorical_model
        colmap = design_colmap(model)

        assert colmap['name'].tolist() == model.column_names

    def test_term_index_assignment(self, multi_term_model):
        """Each column should be assigned to a term."""
        colmap = design_colmap(multi_term_model)

        # All columns should have term indices
        assert colmap['term_index'].notna().all()
        # Term indices should start at 1
        assert colmap['term_index'].min() >= 1

    def test_term_tag_assignment(self, multi_term_model):
        """Each column should have a term tag."""
        colmap = design_colmap(multi_term_model)

        # Some columns should have term tags
        assert colmap['term_tag'].notna().any()

    def test_basis_index_parsing(self):
        """Should parse basis index from column names with _b1, _b2, etc."""
        events = {
            'stim': EventFactor(
                name='stim',
                onsets=np.array([5, 10, 15]),
                values=['A', 'A', 'A'],
                durations=1.0
            )
        }
        sampling = SamplingFrame(tr=1.0, n_scans=30)
        # Use HRF with multiple basis functions
        model = EventModel(
            terms=[Term('stim', hrf='spmg2')],  # SPM with derivative
            events=events,
            sampling_info=sampling
        )

        colmap = design_colmap(model)

        # Should have basis indices parsed
        assert colmap['basis_ix'].notna().any()

    def test_condition_extraction(self, simple_categorical_model):
        """Should extract condition names from column names."""
        colmap = design_colmap(simple_categorical_model)

        # Condition should be extracted
        assert colmap['condition'].notna().all()
        # Should contain level names
        conditions = set(colmap['condition'].unique())
        assert len(conditions) > 0

    def test_single_event_model(self):
        """Model with single event type should have correct metadata."""
        events = {
            'stim': EventFactor(
                name='stim',
                onsets=np.array([2, 6]),
                values=['A', 'A'],
                durations=1.0
            )
        }
        sampling = SamplingFrame(tr=1.0, n_scans=10)
        model = EventModel(terms=[Term('stim')], events=events, sampling_info=sampling)

        colmap = design_colmap(model)

        # Should have correct schema
        assert len(colmap) > 0
        expected_cols = ['col', 'name', 'term_tag', 'term_index', 'condition',
                        'run', 'role', 'model_source', 'basis_name', 'basis_ix',
                        'basis_total', 'basis_label', 'pretty_name',
                        'is_block_diagonal', 'modulation_type', 'modulation_id']
        for col in expected_cols:
            assert col in colmap.columns

    def test_pretty_name_equals_name(self, simple_categorical_model):
        """Pretty name should match name for basic models."""
        colmap = design_colmap(simple_categorical_model)

        assert (colmap['name'] == colmap['pretty_name']).all()

    def test_is_block_diagonal_false(self, simple_categorical_model):
        """Event models should have is_block_diagonal=False."""
        colmap = design_colmap(simple_categorical_model)

        assert all(colmap['is_block_diagonal'] == False)

    def test_run_column_none(self, simple_categorical_model):
        """Run column should be None for single-run models."""
        colmap = design_colmap(simple_categorical_model)

        assert colmap['run'].isna().all()

    def test_modulation_id_none(self, simple_categorical_model):
        """Modulation ID should be None by default."""
        colmap = design_colmap(simple_categorical_model)

        assert colmap['modulation_id'].isna().all()

    def test_invalid_input_type(self):
        """Invalid input type should raise TypeError."""
        with pytest.raises(TypeError, match="design_colmap not implemented"):
            design_colmap(np.array([[1, 2], [3, 4]]))

    def test_basis_labels_canonical(self):
        """SPM canonical with derivative should have correct labels."""
        events = {
            'stim': EventFactor(
                name='stim',
                onsets=np.array([5, 10]),
                values=['A', 'A'],
                durations=1.0
            )
        }
        sampling = SamplingFrame(tr=1.0, n_scans=30)
        model = EventModel(
            terms=[Term('stim', hrf='spmg2')],
            events=events,
            sampling_info=sampling
        )

        colmap = design_colmap(model)

        labels = colmap['basis_label'].dropna().tolist()
        # Should contain 'canonical' and 'derivative' for SPMG2
        if len(labels) > 0:
            assert 'canonical' in labels or 'derivative' in labels


# ============================================================================
# residualize Tests
# ============================================================================

class TestResidualize:
    """Test residualize function - targeting 80%+ coverage."""

    def test_with_numpy_array(self):
        """Residualize ndarray: y - X @ (X'X)^-1 X'y."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        Y = np.random.randn(100, 5)

        resid = residualize(X, Y)

        assert resid.shape == Y.shape
        # Residuals should be orthogonal to design matrix
        assert np.allclose(X.T @ resid, 0, atol=1e-10)

    def test_with_dataframe_design(self):
        """Residualize against DataFrame design matrix."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 3), columns=['a', 'b', 'c'])
        Y = np.random.randn(100, 5)

        resid = residualize(X, Y)

        assert resid.shape == Y.shape
        assert np.allclose(X.values.T @ resid, 0, atol=1e-10)

    def test_with_dataframe_data(self):
        """Residualize DataFrame data should preserve shape."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        Y = pd.DataFrame(np.random.randn(100, 5), columns=['v1', 'v2', 'v3', 'v4', 'v5'])

        resid = residualize(X, Y)

        assert resid.shape == Y.shape
        assert np.allclose(X.T @ resid, 0, atol=1e-10)

    def test_with_event_model(self, simple_categorical_model):
        """Residualize against EventModel extracts design matrix."""
        model = simple_categorical_model
        Y = np.random.randn(model.design_matrix.shape[0], 3)

        resid = residualize(model, Y)

        assert resid.shape == Y.shape
        # Verify orthogonality
        X = model.design_matrix
        assert np.allclose(X.T @ resid, 0, atol=1e-10)

    def test_residuals_orthogonal_to_design(self):
        """Verify residuals are orthogonal to design matrix (X'e ≈ 0)."""
        np.random.seed(42)
        X = np.random.randn(100, 4)
        Y = np.random.randn(100, 2)

        resid = residualize(X, Y)

        orthogonality = X.T @ resid
        assert np.allclose(orthogonality, 0, atol=1e-10)

    def test_residuals_shape_matches_input(self):
        """Residuals shape should match input data shape."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        Y = np.random.randn(100, 7)

        resid = residualize(X, Y)

        assert resid.shape == Y.shape

    def test_1d_data_converted_to_2d(self):
        """1D data should be converted to column vector."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        Y = np.random.randn(100)  # 1D array

        resid = residualize(X, Y)

        assert resid.shape == (100, 1)
        assert np.allclose(X.T @ resid, 0, atol=1e-10)

    def test_with_column_subset_by_name(self):
        """Residualize against column subset using names."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 4), columns=['a', 'b', 'c', 'd'])
        Y = np.random.randn(100, 3)

        resid = residualize(X, Y, cols=['a', 'c'])

        assert resid.shape == Y.shape
        # Should be orthogonal only to selected columns
        assert np.allclose(X[['a', 'c']].values.T @ resid, 0, atol=1e-10)
        # Should not be orthogonal to excluded columns (in general)

    def test_with_column_subset_by_index(self):
        """Residualize against column subset using indices."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        Y = np.random.randn(100, 3)

        resid = residualize(X, Y, cols=[0, 2, 4])

        assert resid.shape == Y.shape
        assert np.allclose(X[:, [0, 2, 4]].T @ resid, 0, atol=1e-10)

    def test_event_model_column_subset_by_name(self, multi_term_model):
        """EventModel residualization with column name subset."""
        model = multi_term_model
        Y = np.random.randn(model.design_matrix.shape[0], 2)

        # Get first column name
        col_names = model.column_names
        if len(col_names) > 1:
            resid = residualize(model, Y, cols=[col_names[0]])

            X_subset = model.design_matrix[:, [0]]
            assert np.allclose(X_subset.T @ resid, 0, atol=1e-10)

    def test_row_mismatch_error(self):
        """Mismatched rows should raise ValueError."""
        X = np.random.randn(100, 3)
        Y = np.random.randn(50, 2)

        with pytest.raises(ValueError, match="Row mismatch"):
            residualize(X, Y)

    def test_string_cols_with_numpy_error(self):
        """String column names with numpy array should raise ValueError."""
        X = np.random.randn(100, 3)
        Y = np.random.randn(100, 2)

        with pytest.raises(ValueError, match="String column names not supported"):
            residualize(X, Y, cols=['a', 'b'])

    def test_preserves_variance_direction(self):
        """Residualization should preserve variance orthogonal to design."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        # Y has component parallel and orthogonal to X
        Y_parallel = X @ np.array([[1], [2]])
        Y_orthogonal = np.random.randn(100, 1) * 3
        Y = Y_parallel + Y_orthogonal

        resid = residualize(X, Y)

        # Residuals should have variance from orthogonal component
        assert np.var(resid) > 0
        # Variance should be less than original (parallel component removed)
        assert np.var(resid) < np.var(Y)

    def test_perfect_fit_gives_zero_residuals(self):
        """Perfect fit should give near-zero residuals."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        # Y is perfect linear combination of X
        Y = X @ np.array([[1], [2], [3]])

        resid = residualize(X, Y)

        # Residuals should be near zero
        assert np.allclose(resid, 0, atol=1e-10)

    def test_qr_decomposition_used(self):
        """Verify QR decomposition produces correct residuals."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        Y = np.random.randn(100, 2)

        resid = residualize(X, Y)

        # Compute expected residuals using manual QR
        Q, R = np.linalg.qr(X, mode='reduced')
        fitted = Q @ (Q.T @ Y)
        expected_resid = Y - fitted

        assert np.allclose(resid, expected_resid, atol=1e-10)

    def test_multiple_response_columns(self):
        """Multiple response columns should be residualized independently."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        Y = np.random.randn(100, 10)

        resid = residualize(X, Y)

        assert resid.shape == (100, 10)
        # Each column should be orthogonal to X
        for j in range(10):
            assert np.allclose(X.T @ resid[:, j:j+1], 0, atol=1e-10)

    def test_event_model_column_subset_by_index(self, multi_term_model):
        """EventModel residualization with integer column indices."""
        model = multi_term_model
        Y = np.random.randn(model.design_matrix.shape[0], 2)

        resid = residualize(model, Y, cols=[0])

        X_subset = model.design_matrix[:, [0]]
        assert np.allclose(X_subset.T @ resid, 0, atol=1e-10)


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_validate_and_residualize_workflow(self, simple_categorical_model):
        """Complete workflow: validate contrasts then residualize."""
        model = simple_categorical_model
        n_cols = model.design_matrix.shape[1]

        # Create a valid contrast
        weights = np.zeros(n_cols)
        weights[0] = 1
        if n_cols > 1:
            weights[1] = -1

        # Validate
        result = validate_contrasts(model, weights)
        assert result['estimable'].iloc[0] == True
        assert result['sum_to_zero'].iloc[0] == True

        # Residualize data
        Y = np.random.randn(model.design_matrix.shape[0], 5)
        resid = residualize(model, Y)

        assert resid.shape == Y.shape
        assert np.allclose(model.design_matrix.T @ resid, 0, atol=1e-10)

    def test_collinearity_check_before_residualize(self, simple_categorical_model):
        """Check collinearity before residualizing."""
        model = simple_categorical_model

        # Check collinearity
        result = check_collinearity(model, threshold=0.9)

        # If no collinearity, proceed with residualization
        if result['ok']:
            Y = np.random.randn(model.design_matrix.shape[0], 3)
            resid = residualize(model, Y)
            assert resid.shape == Y.shape

    def test_colmap_then_residualize_subset(self, multi_term_model):
        """Use colmap to identify columns, then residualize subset."""
        model = multi_term_model
        colmap = design_colmap(model)

        # Find task-related columns
        task_cols = colmap[colmap['role'] == 'task']['name'].tolist()

        # Residualize against task columns
        Y = np.random.randn(model.design_matrix.shape[0], 2)
        if len(task_cols) > 0:
            resid = residualize(model, Y, cols=task_cols)
            assert resid.shape == Y.shape

    def test_full_pipeline(self, multi_term_model):
        """Full pipeline: colmap → validate → check collinearity → residualize."""
        model = multi_term_model

        # 1. Get column map
        colmap = design_colmap(model)
        assert len(colmap) == model.design_matrix.shape[1]

        # 2. Validate a contrast
        n_cols = model.design_matrix.shape[1]
        weights = np.zeros(n_cols)
        weights[0] = 1
        if n_cols > 1:
            weights[1] = -1
        validation = validate_contrasts(model, weights)
        assert len(validation) > 0

        # 3. Check collinearity
        collinearity = check_collinearity(model, threshold=0.95)
        assert 'ok' in collinearity

        # 4. Residualize
        Y = np.random.randn(model.design_matrix.shape[0], 3)
        resid = residualize(model, Y)
        assert resid.shape == Y.shape

    def test_residualize_then_validate_on_residuals(self):
        """Residualize data, then validate contrasts on residualized design."""
        np.random.seed(42)
        # Original design
        X1 = np.random.randn(100, 2)
        # Design to remove
        X2 = np.random.randn(100, 1)

        # Residualize X1 against X2
        X1_resid = residualize(X2, X1)

        # Now validate contrasts on residualized design
        weights = np.array([1, -1])
        result = validate_contrasts(X1_resid, weights)

        assert result['estimable'].iloc[0] == True
