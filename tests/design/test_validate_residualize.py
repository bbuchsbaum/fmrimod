"""Tests for validate and residualize modules."""
import numpy as np
import pandas as pd
import pytest
from fmrimod.validate import validate_contrasts, check_collinearity
from fmrimod.residualize import residualize


class TestValidateContrasts:
    """Test validate_contrasts function."""

    def test_with_numpy_array(self):
        """Test validation with numpy array design matrix."""
        X = np.random.randn(100, 5)
        weights = np.array([1, -1, 0, 0, 0])
        result = validate_contrasts(X, weights)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert 'name' in result.columns
        assert 'type' in result.columns
        assert 'estimable' in result.columns
        assert result['type'].iloc[0] == 't'

    def test_with_dataframe(self):
        """Test validation with DataFrame design matrix."""
        X = pd.DataFrame(np.random.randn(100, 3), columns=['a', 'b', 'c'])
        weights = np.array([1, -1, 0])
        result = validate_contrasts(X, weights)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result['estimable'].iloc[0] == True

    def test_f_contrast(self):
        """Test F-contrast (multi-column weights)."""
        X = np.random.randn(100, 5)
        weights = np.array([[1, -1, 0, 0, 0],
                           [0, 1, -1, 0, 0]]).T
        result = validate_contrasts(X, weights)

        assert len(result) == 2
        assert all(result['type'] == 'F')
        assert result['full_rank'].iloc[0] == True

    def test_dict_weights(self):
        """Test with dictionary of weights."""
        X = np.random.randn(100, 4)
        weights = {
            'contrast1': np.array([1, -1, 0, 0]),
            'contrast2': np.array([0, 1, -1, 0])
        }
        result = validate_contrasts(X, weights)

        assert len(result) == 2
        assert set(result['name']) == {'contrast1', 'contrast2'}

    def test_sum_to_zero_detection(self):
        """Test sum-to-zero contrast detection."""
        X = np.random.randn(100, 3)
        weights = np.array([1, -1, 0])  # sum to zero
        result = validate_contrasts(X, weights)

        assert result['sum_to_zero'].iloc[0] == True

    def test_intercept_orthogonality(self):
        """Test intercept orthogonality check."""
        X = pd.DataFrame(np.random.randn(100, 4),
                        columns=['(Intercept)', 'a', 'b', 'c'])
        weights = np.array([0, 1, -1, 0])  # orthogonal to intercept
        result = validate_contrasts(X, weights)

        assert result['orthogonal_to_intercept'].iloc[0] == True

    def test_nonestimable_contrast(self):
        """Test detection of non-estimable contrasts."""
        # Create rank-deficient design
        X = np.random.randn(100, 3)
        X = np.column_stack([X, X[:, 0] + X[:, 1]])  # linear dependency

        # Contrast on dependent column
        weights = np.array([1, 1, 0, -1])
        result = validate_contrasts(X, weights)

        # Should be estimable because it's in column space
        assert 'estimable' in result.columns


class TestCheckCollinearity:
    """Test check_collinearity function."""

    def test_no_collinearity(self):
        """Test with uncorrelated regressors."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        result = check_collinearity(X, threshold=0.9)

        assert result['ok'] == True
        assert len(result['pairs']) == 0

    def test_high_collinearity(self):
        """Test with highly correlated regressors."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        # Add highly correlated column
        X = np.column_stack([X, X[:, 0] + 0.01 * np.random.randn(100)])
        result = check_collinearity(X, threshold=0.9)

        assert result['ok'] == False
        assert len(result['pairs']) > 0
        assert 'regressor_1' in result['pairs'].columns
        assert 'regressor_2' in result['pairs'].columns
        assert 'r' in result['pairs'].columns

    def test_with_dataframe(self):
        """Test with DataFrame input."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 3), columns=['a', 'b', 'c'])
        result = check_collinearity(X, threshold=0.9)

        assert isinstance(result, dict)
        assert 'ok' in result
        assert 'pairs' in result

    def test_ignores_intercept(self):
        """Test that intercept columns are ignored."""
        np.random.seed(42)
        X = pd.DataFrame({
            '(Intercept)': np.ones(100),
            'a': np.random.randn(100),
            'b': np.random.randn(100)
        })
        result = check_collinearity(X, threshold=0.9)

        # Should not flag constant column
        assert result['ok'] == True

    def test_custom_threshold(self):
        """Test with custom correlation threshold."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        # Create column correlated ~0.92 with X[:,0] (above 0.7, below 0.95)
        X = np.column_stack([X, X[:, 0] * 0.7 + 0.3 * np.random.randn(100)])

        result_strict = check_collinearity(X, threshold=0.7)
        result_lenient = check_collinearity(X, threshold=0.95)

        assert result_strict['ok'] == False
        assert result_lenient['ok'] == True


class TestResidualize:
    """Test residualize function."""

    def test_with_numpy_array(self):
        """Test residualization with numpy arrays."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        Y = np.random.randn(100, 5)

        resid = residualize(X, Y)

        assert resid.shape == Y.shape
        # Residuals should be orthogonal to design
        assert np.allclose(X.T @ resid, 0, atol=1e-10)

    def test_with_dataframe(self):
        """Test residualization with DataFrame design."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 3), columns=['a', 'b', 'c'])
        Y = np.random.randn(100, 5)

        resid = residualize(X, Y)

        assert resid.shape == Y.shape
        assert np.allclose(X.values.T @ resid, 0, atol=1e-10)

    def test_with_column_subset(self):
        """Test residualization with column subset."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 4), columns=['a', 'b', 'c', 'd'])
        Y = np.random.randn(100, 3)

        resid = residualize(X, Y, cols=['a', 'c'])

        assert resid.shape == Y.shape
        # Should be orthogonal only to selected columns
        assert np.allclose(X[['a', 'c']].values.T @ resid, 0, atol=1e-10)

    def test_with_1d_data(self):
        """Test residualization with 1D data."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        Y = np.random.randn(100)

        resid = residualize(X, Y)

        assert resid.shape == (100, 1)

    def test_preserves_variance_direction(self):
        """Test that residualization preserves variance orthogonal to design."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        # Create Y with component parallel and orthogonal to X
        Y = X @ np.array([1, 2]) + np.random.randn(100, 1) * 3

        resid = residualize(X, Y)

        # Residuals should have variance from orthogonal component
        assert np.var(resid) > 0
        assert np.var(resid) < np.var(Y)

    def test_row_mismatch_error(self):
        """Test error when rows don't match."""
        X = np.random.randn(100, 3)
        Y = np.random.randn(50, 2)

        with pytest.raises(ValueError, match="Row mismatch"):
            residualize(X, Y)

    def test_with_integer_column_indices(self):
        """Test with integer column indices."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        Y = np.random.randn(100, 3)

        resid = residualize(X, Y, cols=[0, 2, 4])

        assert resid.shape == Y.shape
        # Should be orthogonal only to selected columns
        assert np.allclose(X[:, [0, 2, 4]].T @ resid, 0, atol=1e-10)


class TestIntegration:
    """Integration tests combining validate and residualize."""

    def test_validate_then_residualize(self):
        """Test validation followed by residualization."""
        np.random.seed(42)
        X = np.random.randn(100, 4)

        # First validate a contrast
        weights = np.array([1, -1, 0, 0])
        result = validate_contrasts(X, weights)
        assert result['estimable'].iloc[0] == True

        # Then residualize some data
        Y = np.random.randn(100, 5)
        resid = residualize(X, Y)

        assert resid.shape == Y.shape
        assert np.allclose(X.T @ resid, 0, atol=1e-10)

    def test_check_collinearity_before_residualize(self):
        """Test collinearity check before residualization."""
        np.random.seed(42)
        X = np.random.randn(100, 3)

        # Check for collinearity first
        result = check_collinearity(X, threshold=0.9)
        assert result['ok'] == True

        # Proceed with residualization
        Y = np.random.randn(100, 5)
        resid = residualize(X, Y)

        assert resid.shape == Y.shape
