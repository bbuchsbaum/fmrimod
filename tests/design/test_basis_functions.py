"""Tests for new basis functions: predict, sub_basis, Standardized."""

import pytest
import numpy as np

from fmrimod.basis import Poly, BSpline, Standardized, sub_basis
from fmrimod.dispatch import nbasis, construct


class TestPredict:
    """Test predict method for ParametricBasis."""
    
    def test_predict_without_coef(self):
        """Test predict returns basis matrix when no coefficients."""
        poly = Poly(degree=2)
        x = np.array([0, 1, 2, 3])
        
        # predict without coef should equal evaluate
        pred = poly.predict(x)
        eval_result = poly.evaluate(x)
        
        np.testing.assert_array_equal(pred, eval_result)
    
    def test_predict_with_coef(self):
        """Test predict with coefficients."""
        poly = Poly(degree=2, intercept=True, raw=True)
        x = np.array([0, 1, 2, 3])
        coef = np.array([1.0, 2.0, 0.5])  # intercept, linear, quadratic

        pred = poly.predict(x, coef)

        # Expected: 1 + 2*x + 0.5*x^2
        expected = 1 + 2*x + 0.5*x**2
        np.testing.assert_allclose(pred, expected)
    
    def test_predict_with_2d_coef(self):
        """Test predict with 2D coefficient matrix."""
        poly = Poly(degree=1, intercept=True, raw=True)
        x = np.array([0, 1, 2, 3])
        
        # Multiple sets of coefficients
        coef = np.array([[1.0, 0.0],   # First set
                        [0.5, 1.0]])   # Second set
        
        pred = poly.predict(x, coef)
        
        # Should have shape (4, 2)
        assert pred.shape == (4, 2)
        
        # Column 0: basis @ coef[:, 0] = basis @ [1.0, 0.5] = 1 + 0.5*x
        np.testing.assert_allclose(pred[:, 0], [1, 1.5, 2, 2.5])

        # Column 1: basis @ coef[:, 1] = basis @ [0.0, 1.0] = 0 + 1*x
        np.testing.assert_allclose(pred[:, 1], [0, 1, 2, 3])
    
    def test_predict_wrong_coef_size(self):
        """Test predict with wrong coefficient size."""
        poly = Poly(degree=2)
        x = np.array([0, 1, 2, 3])
        
        # Wrong number of coefficients
        with pytest.raises(ValueError, match="Number of coefficients"):
            poly.predict(x, [1, 2])  # Need 3, gave 2
    
    def test_predict_spline(self):
        """Test predict with spline basis."""
        spline = BSpline(df=4)
        x = np.linspace(0, 1, 50)
        
        # Get basis matrix
        basis_mat = spline.predict(x)
        assert basis_mat.shape == (50, 4)
        
        # With coefficients
        coef = np.ones(4)
        pred = spline.predict(x, coef)
        assert pred.shape == (50,)


class TestSubBasis:
    """Test sub_basis functionality."""
    
    def test_sub_basis_indices(self):
        """Test subsetting with list of indices."""
        poly = Poly(degree=5, raw=True)

        # Select 1st, 3rd, and 5th basis functions
        sub = sub_basis(poly, [0, 2, 4])

        assert sub.n_basis == 3
        assert len(sub.basis_names) == 3

        # Evaluate (need at least n_basis points for proper evaluation)
        x = np.array([0, 1, 2, 3, 4, 5, 6])
        full = poly.evaluate(x)
        subset = sub.evaluate(x)
        
        # Check we got the right columns
        assert subset.shape == (7, 3)
        np.testing.assert_array_equal(subset[:, 0], full[:, 0])
        np.testing.assert_array_equal(subset[:, 1], full[:, 2])
        np.testing.assert_array_equal(subset[:, 2], full[:, 4])
    
    def test_sub_basis_slice(self):
        """Test subsetting with slice."""
        spline = BSpline(df=10)
        
        # Select middle splines
        sub = sub_basis(spline, slice(3, 7))
        
        assert sub.n_basis == 4
        assert sub.indices == [3, 4, 5, 6]
    
    def test_sub_basis_negative_indices(self):
        """Test negative indexing."""
        poly = Poly(degree=5)
        
        # Select last three
        sub = sub_basis(poly, [-3, -2, -1])
        
        assert sub.n_basis == 3
        assert sub.indices == [3, 4, 5]
    
    def test_sub_basis_out_of_range(self):
        """Test error on out of range indices."""
        poly = Poly(degree=2)
        
        with pytest.raises(IndexError):
            sub_basis(poly, [0, 1, 10])  # Index 10 out of range
    
    def test_sub_basis_empty(self):
        """Test error on empty selection."""
        poly = Poly(degree=2)
        
        with pytest.raises(ValueError, match="at least one"):
            sub_basis(poly, [])
    
    def test_sub_basis_names(self):
        """Test that basis names are preserved."""
        poly = Poly(degree=3, intercept=True)
        sub = sub_basis(poly, [0, 2])  # intercept and quadratic
        
        parent_names = poly.basis_names
        sub_names = sub.basis_names
        
        assert sub_names[0] == parent_names[0]
        assert sub_names[1] == parent_names[2]


class TestStandardized:
    """Test Standardized transform."""
    
    def test_standardized_basic(self):
        """Test basic standardization."""
        std = Standardized()
        x = np.array([1, 2, 3, 4, 5])
        
        z = std.evaluate(x)
        
        # Check shape
        assert z.shape == (5, 1)
        
        # Check standardized
        np.testing.assert_allclose(np.mean(z), 0, atol=1e-10)
        np.testing.assert_allclose(np.std(z), 1, atol=1e-10)
    
    def test_standardized_center_only(self):
        """Test centering without scaling."""
        std = Standardized(center=True, scale=False)
        x = np.array([1, 2, 3, 4, 5])
        
        z = std.evaluate(x)
        
        # Check centered
        np.testing.assert_allclose(np.mean(z), 0, atol=1e-10)
        
        # But not scaled
        assert np.std(z) != 1
        np.testing.assert_allclose(np.std(z), np.std(x))
    
    def test_standardized_scale_only(self):
        """Test scaling without centering."""
        std = Standardized(center=False, scale=True)
        x = np.array([1, 2, 3, 4, 5])
        
        z = std.evaluate(x)
        
        # Not centered
        assert np.mean(z) != 0
        
        # But scaled
        np.testing.assert_allclose(np.std(z), 1, atol=1e-10)
    
    def test_standardized_fit_transform(self):
        """Test separate fit and transform."""
        std = Standardized()
        
        # Fit on training data
        x_train = np.array([1, 2, 3, 4, 5])
        std.fit(x_train)
        
        # Transform new data
        x_test = np.array([2, 3, 4])
        z_test = std.evaluate(x_test)
        
        # Should use training statistics
        expected = (x_test - 3) / np.std([1, 2, 3, 4, 5])
        np.testing.assert_allclose(z_test.ravel(), expected)
    
    def test_standardized_zero_variance(self):
        """Test handling of zero variance."""
        std = Standardized()
        x = np.array([5, 5, 5, 5])  # Constant
        
        z = std.evaluate(x)
        
        # Should handle gracefully
        assert z.shape == (4, 1)
        np.testing.assert_array_equal(z.ravel(), [0, 0, 0, 0])


class TestDispatchFunctions:
    """Test nbasis and construct dispatch functions."""
    
    def test_nbasis_dispatch(self):
        """Test nbasis generic function."""
        poly = Poly(degree=3)
        assert nbasis(poly) == 4
        
        spline = BSpline(df=5)
        assert nbasis(spline) == 5
        
        std = Standardized()
        assert nbasis(std) == 1
    
    def test_construct_dispatch(self):
        """Test construct generic function."""
        poly = Poly(degree=2)
        x = np.array([0, 1, 2, 3])
        
        # construct should call evaluate
        result = construct(poly, x)
        expected = poly.evaluate(x)
        
        np.testing.assert_array_equal(result, expected)