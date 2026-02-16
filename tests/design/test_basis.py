"""Tests for basis functions."""

import pytest
import numpy as np
from scipy import interpolate

from fmrimod.basis import (
    ParametricBasis,
    Poly,
    NaturalPoly,
    BSpline,
    NaturalSpline,
    Scale,
    ScaleWithin,
    RobustScale,
    Ident,
)


class TestPoly:
    """Test polynomial basis functions."""
    
    def test_linear_with_intercept(self):
        """Test linear polynomial with intercept."""
        basis = Poly(degree=1, intercept=True)
        x = np.array([0, 1, 2, 3])
        
        assert basis.n_basis == 2
        assert basis.basis_names == ["poly1_intercept", "poly1_1"]
        
        # Raw polynomial
        basis_raw = Poly(degree=1, intercept=True, raw=True)
        X = basis_raw.evaluate(x)
        
        assert X.shape == (4, 2)
        assert np.allclose(X[:, 0], 1)  # Intercept
        assert np.allclose(X[:, 1], x)  # Linear term
    
    def test_quadratic_without_intercept(self):
        """Test quadratic polynomial without intercept."""
        basis = Poly(degree=2, intercept=False, raw=True)
        x = np.array([1, 2, 3])
        
        assert basis.n_basis == 2
        
        X = basis.evaluate(x)
        assert X.shape == (3, 2)
        assert np.allclose(X[:, 0], x)      # Linear
        assert np.allclose(X[:, 1], x**2)   # Quadratic
    
    def test_orthogonal_polynomials(self):
        """Test orthogonal polynomial generation."""
        basis = Poly(degree=3, intercept=True, raw=False)
        x = np.linspace(-1, 1, 50)
        
        X = basis.evaluate(x)
        assert X.shape == (50, 4)
        
        # Check orthogonality
        gram = X.T @ X
        # Should be approximately diagonal
        off_diag = gram - np.diag(np.diag(gram))
        assert np.max(np.abs(off_diag)) < 1e-10
    
    def test_degree_validation(self):
        """Test degree validation."""
        with pytest.raises(ValueError, match="Degree must be at least 1"):
            Poly(degree=0)
    
    def test_call_method(self):
        """Test __call__ convenience method."""
        basis = Poly(degree=2)
        x = np.array([1, 2, 3])
        
        X1 = basis.evaluate(x)
        X2 = basis(x)
        
        assert np.array_equal(X1, X2)


class TestNaturalPoly:
    """Test natural polynomial basis."""
    
    def test_centering(self):
        """Test that natural polynomials center at mean."""
        basis = NaturalPoly(degree=2)
        x = np.array([1, 2, 3, 4, 5])
        
        X = basis.evaluate(x)
        
        # After evaluation, mean should be stored
        assert basis._x_mean == 3.0
        
        # First column should be all 1s (intercept)
        assert np.allclose(X[:, 0], 1)
        
        # Second column should be centered x
        expected_linear = x - 3.0
        assert np.allclose(X[:, 1], expected_linear)


class TestBSpline:
    """Test B-spline basis functions."""
    
    def test_basic_bspline(self):
        """Test basic B-spline construction."""
        basis = BSpline(df=4, degree=3)
        x = np.linspace(0, 1, 100)
        
        assert basis.n_basis == 4
        assert len(basis.basis_names) == 4
        
        X = basis.evaluate(x)
        assert X.shape == (100, 4)
        
        # B-splines should be non-negative
        assert np.all(X >= 0)
        
        # Each row should sum to approximately 1 (partition of unity)
        row_sums = np.sum(X, axis=1)
        assert np.allclose(row_sums[10:-10], 1.0, atol=1e-10)
    
    def test_knot_specification(self):
        """Test specifying knots directly."""
        knots = [0.25, 0.5, 0.75]
        basis = BSpline(knots=knots, degree=2)
        
        assert basis.knots is not None
        assert np.array_equal(basis.knots, knots)
        
        # df should be computed from knots
        # df = len(knots) + degree + 1 (with intercept)
        assert basis.df == len(knots) + 2 + 1
    
    def test_without_intercept(self):
        """Test B-spline without intercept."""
        basis = BSpline(df=4, intercept=False)
        x = np.linspace(0, 1, 50)
        
        X = basis.evaluate(x)
        # Should have df columns
        assert X.shape == (50, 4)
    
    def test_boundary_knots(self):
        """Test custom boundary knots."""
        basis = BSpline(df=4, boundary_knots=(0, 10))
        x = np.linspace(0, 10, 100)
        
        X = basis.evaluate(x)
        assert X.shape == (100, 4)
        
        # Values outside boundaries should be 0
        x_test = np.array([-1, 5, 11])
        X_test = basis.evaluate(x_test)
        assert X_test[0, :].sum() == 0  # x = -1
        assert X_test[2, :].sum() == 0  # x = 11
        assert X_test[1, :].sum() > 0   # x = 5
    
    def test_error_handling(self):
        """Test error handling."""
        # Neither df nor knots specified
        with pytest.raises(ValueError, match="Either df or knots"):
            BSpline()
        
        # Both df and knots specified
        with pytest.raises(ValueError, match="Cannot specify both"):
            BSpline(df=4, knots=[0.5])


class TestNaturalSpline:
    """Test natural spline basis."""
    
    def test_natural_spline_basic(self):
        """Test basic natural spline."""
        basis = NaturalSpline(df=4)
        x = np.linspace(0, 1, 50)
        
        X = basis.evaluate(x)
        assert X.shape == (50, 4)
        
        # Should be cubic (degree 3)
        assert basis.degree == 3


class TestScale:
    """Test scaling transformations."""
    
    def test_standard_scaling(self):
        """Test standard z-score scaling."""
        transform = Scale()
        x = np.array([1, 2, 3, 4, 5])
        
        z = transform.evaluate(x)
        assert z.shape == (5, 1)
        
        # Should have mean 0, std 1
        assert np.abs(np.mean(z)) < 1e-10
        assert np.abs(np.std(z) - 1.0) < 1e-10
    
    def test_center_only(self):
        """Test centering without scaling."""
        transform = Scale(center=True, scale=False)
        x = np.array([1, 2, 3, 4, 5])
        
        z = transform.evaluate(x)
        
        # Mean should be 0, std unchanged
        assert np.abs(np.mean(z)) < 1e-10
        assert np.abs(np.std(z) - np.std(x)) < 1e-10
    
    def test_scale_only(self):
        """Test scaling without centering."""
        transform = Scale(center=False, scale=True)
        x = np.array([2, 4, 6, 8, 10])
        
        z = transform.evaluate(x)
        
        # Std should be 1, mean should be scaled
        assert np.abs(np.std(z) - 1.0) < 1e-10
        assert np.mean(z) == pytest.approx(np.mean(x) / np.std(x))
    
    def test_constant_values(self):
        """Test handling of constant values."""
        transform = Scale()
        x = np.array([5, 5, 5, 5])
        
        z = transform.evaluate(x)
        
        # Should center but not scale (std = 0)
        assert np.all(z == 0)


class TestScaleWithin:
    """Test group-wise scaling."""
    
    def test_basic_groups(self):
        """Test scaling within groups."""
        groups = ['A', 'A', 'B', 'B', 'B']
        x = np.array([1, 2, 10, 20, 30])
        
        transform = ScaleWithin(groups)
        z = transform.evaluate(x)
        
        assert z.shape == (5, 1)
        
        # Check group A
        z_a = z[np.array(groups) == 'A', 0]
        assert np.abs(np.mean(z_a)) < 1e-10
        assert np.abs(np.std(z_a) - 1.0) < 1e-10
        
        # Check group B
        z_b = z[np.array(groups) == 'B', 0]
        assert np.abs(np.mean(z_b)) < 1e-10
        assert np.abs(np.std(z_b) - 1.0) < 1e-10
    
    def test_length_mismatch(self):
        """Test error on length mismatch."""
        groups = ['A', 'B']
        x = np.array([1, 2, 3])
        
        transform = ScaleWithin(groups)
        with pytest.raises(ValueError, match="Length mismatch"):
            transform.evaluate(x)


class TestRobustScale:
    """Test robust scaling."""
    
    def test_mad_scaling(self):
        """Test MAD scaling."""
        transform = RobustScale(scale='mad')
        x = np.array([1, 2, 3, 4, 100])  # 100 is outlier
        
        z = transform.evaluate(x)
        
        # Should be centered at median
        assert transform._median == 3
        
        # MAD should handle outlier better than standard scaling
        assert np.median(np.abs(z)) < 2
    
    def test_iqr_scaling(self):
        """Test IQR scaling."""
        transform = RobustScale(scale='iqr')
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])
        
        z = transform.evaluate(x)
        
        # Should be centered at median
        assert transform._median == 5.5
        
        # Scale factor should be IQR
        q1, q3 = np.percentile(x, [25, 75])
        assert transform._scale_factor == q3 - q1
    
    def test_invalid_scale(self):
        """Test error on invalid scale method."""
        with pytest.raises(ValueError, match="scale must be"):
            RobustScale(scale='invalid')


class TestIdent:
    """Test identity transformation."""
    
    def test_identity(self):
        """Test that identity returns unchanged values."""
        transform = Ident()
        x = np.array([1, 2, 3, 4, 5])
        
        y = transform.evaluate(x)
        
        assert y.shape == (5, 1)
        assert np.array_equal(y.ravel(), x)
        assert transform.n_basis == 1
        assert transform.basis_names == ["ident"]


class TestIntegration:
    """Integration tests with EventBasis."""
    
    def test_with_event_basis(self):
        """Test basis functions with EventBasis."""
        from fmrimod.events import EventBasis
        
        # Polynomial basis
        poly_basis = Poly(degree=2)
        event = EventBasis(
            name='rating',
            onsets=[1, 2, 3],
            values=[1, 2, 3],
            basis=poly_basis
        )
        
        assert event.n_basis == 3
        assert event.expanded_values.shape == (3, 3)
    
    def test_basis_in_pipeline(self):
        """Test combining multiple bases."""
        x = np.linspace(0, 1, 100)
        
        # Different basis types
        bases = [
            Poly(degree=3),
            BSpline(df=4),
            Scale(),
        ]
        
        for basis in bases:
            X = basis.evaluate(x)
            assert X.shape[0] == len(x)
            assert X.shape[1] == basis.n_basis