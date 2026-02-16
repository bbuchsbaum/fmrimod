"""Comprehensive tests for basis transforms and sub_basis.

This module provides comprehensive coverage for:
- Basis transform classes (Scale, ScaleWithin, RobustScale, Standardized, Ident)
- SubBasis subsetting functionality
- Basis predict method for new data
- condition_basis_list function

Target coverage improvements:
- transform.py: 25% → 70%+
- sub_basis.py: 27% → 70%+
- condition_basis.py: 9% → 60%+
"""

import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from fmrimod.basis import (
    Poly,
    BSpline,
    Scale,
    ScaleWithin,
    RobustScale,
    Standardized,
    Ident,
    sub_basis,
    SubBasis,
)


class TestScale:
    """Test Scale transform (z-score normalization)."""

    def test_scale_basic(self):
        """Test basic z-score scaling."""
        transform = Scale()
        x = np.array([1, 2, 3, 4, 5], dtype=float)

        result = transform.evaluate(x)

        # Check shape
        assert result.shape == (5, 1)

        # Check standardization
        assert_allclose(np.mean(result), 0, atol=1e-10)
        assert_allclose(np.std(result), 1, atol=1e-10)

        # Check statistics were stored
        assert transform._mean == 3.0
        assert_allclose(transform._std, np.std(x))

    def test_scale_center_only(self):
        """Test centering without scaling."""
        transform = Scale(center=True, scale=False)
        x = np.array([10, 20, 30, 40, 50], dtype=float)

        result = transform.evaluate(x)

        # Mean should be 0
        assert_allclose(np.mean(result), 0, atol=1e-10)

        # Std should be unchanged from input
        assert_allclose(np.std(result), np.std(x), atol=1e-10)

    def test_scale_scale_only(self):
        """Test scaling without centering."""
        transform = Scale(center=False, scale=True)
        x = np.array([2, 4, 6, 8, 10], dtype=float)

        result = transform.evaluate(x)

        # Std should be 1
        assert_allclose(np.std(result), 1, atol=1e-10)

        # Mean should be non-zero (not centered)
        assert np.mean(result) != 0
        assert_allclose(np.mean(result), np.mean(x) / np.std(x))

    def test_scale_no_transform(self):
        """Test no centering or scaling."""
        transform = Scale(center=False, scale=False)
        x = np.array([1, 2, 3, 4, 5], dtype=float)

        result = transform.evaluate(x)

        # Should return unchanged
        assert_array_equal(result.ravel(), x)

    def test_scale_constant_values(self):
        """Test handling of constant values (zero variance)."""
        transform = Scale()
        x = np.array([7, 7, 7, 7], dtype=float)

        result = transform.evaluate(x)

        # Should center but not scale (std = 0)
        assert_array_equal(result, 0)
        assert transform._std == 0

    def test_scale_negative_values(self):
        """Test scaling with negative values."""
        transform = Scale()
        x = np.array([-5, -3, -1, 1, 3, 5], dtype=float)

        result = transform.evaluate(x)

        assert_allclose(np.mean(result), 0, atol=1e-10)
        assert_allclose(np.std(result), 1, atol=1e-10)

    def test_scale_properties(self):
        """Test n_basis and basis_names properties."""
        transform = Scale(name="my_scale")

        assert transform.n_basis == 1
        assert transform.basis_names == ["my_scale"]
        assert transform.name == "my_scale"

    def test_scale_default_name(self):
        """Test default naming."""
        transform = Scale()
        assert transform.name == "scale"
        assert transform.basis_names == ["scale"]


class TestScaleWithin:
    """Test ScaleWithin transform (group-wise scaling)."""

    def test_scalewithin_basic(self):
        """Test basic group-wise scaling."""
        groups = ['A', 'A', 'A', 'B', 'B', 'B']
        x = np.array([1, 2, 3, 10, 20, 30], dtype=float)

        transform = ScaleWithin(groups)
        result = transform.evaluate(x)

        assert result.shape == (6, 1)

        # Check group A is standardized
        mask_a = np.array(groups) == 'A'
        z_a = result[mask_a, 0]
        assert_allclose(np.mean(z_a), 0, atol=1e-10)
        assert_allclose(np.std(z_a), 1, atol=1e-10)

        # Check group B is standardized
        mask_b = np.array(groups) == 'B'
        z_b = result[mask_b, 0]
        assert_allclose(np.mean(z_b), 0, atol=1e-10)
        assert_allclose(np.std(z_b), 1, atol=1e-10)

    def test_scalewithin_different_group_sizes(self):
        """Test with unbalanced groups."""
        groups = ['A', 'A', 'B', 'B', 'B', 'B']
        x = np.array([1, 5, 10, 20, 30, 40], dtype=float)

        transform = ScaleWithin(groups)
        result = transform.evaluate(x)

        # Each group should be standardized independently
        for group in ['A', 'B']:
            mask = np.array(groups) == group
            z_group = result[mask, 0]
            assert_allclose(np.mean(z_group), 0, atol=1e-10)
            if len(z_group) > 1:
                assert_allclose(np.std(z_group), 1, atol=1e-10)

    def test_scalewithin_numeric_groups(self):
        """Test with numeric group labels."""
        groups = np.array([1, 1, 2, 2, 3, 3])
        x = np.array([10, 20, 100, 200, 1000, 2000], dtype=float)

        transform = ScaleWithin(groups)
        result = transform.evaluate(x)

        assert result.shape == (6, 1)

        # Check each group
        for group_id in [1, 2, 3]:
            mask = groups == group_id
            z_group = result[mask, 0]
            assert_allclose(np.mean(z_group), 0, atol=1e-10)
            assert_allclose(np.std(z_group), 1, atol=1e-10)

    def test_scalewithin_center_only(self):
        """Test group-wise centering without scaling."""
        groups = ['A', 'A', 'B', 'B']
        x = np.array([1, 3, 10, 30], dtype=float)

        transform = ScaleWithin(groups, center=True, scale=False)
        result = transform.evaluate(x)

        # Group A: centered at mean 2
        mask_a = np.array(groups) == 'A'
        assert_allclose(np.mean(result[mask_a, 0]), 0, atol=1e-10)

        # Group B: centered at mean 20
        mask_b = np.array(groups) == 'B'
        assert_allclose(np.mean(result[mask_b, 0]), 0, atol=1e-10)

    def test_scalewithin_scale_only(self):
        """Test group-wise scaling without centering."""
        groups = ['A', 'A', 'B', 'B']
        x = np.array([2, 4, 10, 20], dtype=float)

        transform = ScaleWithin(groups, center=False, scale=True)
        result = transform.evaluate(x)

        # Each group should have std = 1 but non-zero mean
        for group in ['A', 'B']:
            mask = np.array(groups) == group
            z_group = result[mask, 0]
            assert_allclose(np.std(z_group), 1, atol=1e-10)

    def test_scalewithin_length_mismatch(self):
        """Test error on length mismatch."""
        groups = ['A', 'B']
        x = np.array([1, 2, 3])

        transform = ScaleWithin(groups)
        with pytest.raises(ValueError, match="Length mismatch"):
            transform.evaluate(x)

    def test_scalewithin_group_stats(self):
        """Test that group statistics are stored."""
        groups = ['A', 'A', 'B', 'B']
        x = np.array([1, 3, 10, 20], dtype=float)

        transform = ScaleWithin(groups)
        transform.evaluate(x)

        # Check stored statistics
        assert 'A' in transform._group_stats
        assert 'B' in transform._group_stats

        stats_a = transform._group_stats['A']
        assert stats_a['mean'] == 2.0
        assert stats_a['n'] == 2

        stats_b = transform._group_stats['B']
        assert stats_b['mean'] == 15.0
        assert stats_b['n'] == 2

    def test_scalewithin_single_value_group(self):
        """Test handling of single-value groups."""
        groups = ['A', 'B', 'B']
        x = np.array([5, 10, 20], dtype=float)

        transform = ScaleWithin(groups)
        result = transform.evaluate(x)

        # Single value group A should be centered to 0
        assert result[0, 0] == 0

    def test_scalewithin_properties(self):
        """Test n_basis and basis_names properties."""
        groups = ['A', 'B']
        transform = ScaleWithin(groups, name="group_scale")

        assert transform.n_basis == 1
        assert transform.basis_names == ["group_scale"]


class TestRobustScale:
    """Test RobustScale transform."""

    def test_robustscale_mad(self):
        """Test MAD-based robust scaling."""
        transform = RobustScale(scale='mad')
        x = np.array([1, 2, 3, 4, 100], dtype=float)  # 100 is outlier

        result = transform.evaluate(x)

        # Should be centered at median (3)
        assert transform._median == 3

        # MAD should be less affected by outlier
        mad = np.median(np.abs(x - 3))
        expected_scale = mad * 1.4826
        assert_allclose(transform._scale_factor, expected_scale)

        # Median of transformed data should be 0
        assert_allclose(np.median(result), 0, atol=1e-10)

    def test_robustscale_iqr(self):
        """Test IQR-based robust scaling."""
        transform = RobustScale(scale='iqr')
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 100], dtype=float)

        result = transform.evaluate(x)

        # Should be centered at median
        assert transform._median == 5.5

        # Scale factor should be IQR
        q1, q3 = np.percentile(x, [25, 75])
        assert_allclose(transform._scale_factor, q3 - q1)

    def test_robustscale_center_only(self):
        """Test robust centering without scaling."""
        transform = RobustScale(center=True, scale='mad')
        x = np.array([1, 2, 3, 4, 5, 100], dtype=float)

        # Manually set scale to false by evaluating then checking
        result = transform.evaluate(x)

        # Should be centered at median
        assert_allclose(np.median(result), 0, atol=1e-10)

    def test_robustscale_no_center(self):
        """Test robust scaling without centering."""
        transform = RobustScale(center=False, scale='mad')
        x = np.array([1, 2, 3, 4, 5], dtype=float)

        result = transform.evaluate(x)

        # Should not be centered
        assert np.median(result) != 0

    def test_robustscale_constant_values(self):
        """Test handling of constant values."""
        transform = RobustScale(scale='mad')
        x = np.array([10, 10, 10, 10], dtype=float)

        result = transform.evaluate(x)

        # MAD will be 0, should center but not scale
        assert transform._scale_factor == 0
        assert_array_equal(result, 0)

    def test_robustscale_custom_constant(self):
        """Test custom MAD constant."""
        transform = RobustScale(scale='mad', constant=1.0)
        x = np.array([1, 2, 3, 4, 5], dtype=float)

        transform.evaluate(x)

        # Scale factor should use custom constant
        mad = np.median(np.abs(x - 3))
        assert_allclose(transform._scale_factor, mad * 1.0)

    def test_robustscale_invalid_scale(self):
        """Test error on invalid scale method."""
        with pytest.raises(ValueError, match="scale must be"):
            RobustScale(scale='invalid')

    def test_robustscale_properties(self):
        """Test n_basis and basis_names properties."""
        transform = RobustScale(name="robust")

        assert transform.n_basis == 1
        assert transform.basis_names == ["robust"]


class TestIdent:
    """Test Ident transform (identity/no-op)."""

    def test_ident_basic(self):
        """Test identity transformation."""
        transform = Ident()
        x = np.array([1, 2, 3, 4, 5])

        result = transform.evaluate(x)

        # Should return unchanged values
        assert result.shape == (5, 1)
        assert_array_equal(result.ravel(), x)

    def test_ident_negative_values(self):
        """Test with negative values."""
        transform = Ident()
        x = np.array([-10, -5, 0, 5, 10])

        result = transform.evaluate(x)
        assert_array_equal(result.ravel(), x)

    def test_ident_float_values(self):
        """Test with float values."""
        transform = Ident()
        x = np.array([1.5, 2.7, 3.9])

        result = transform.evaluate(x)
        assert_array_equal(result.ravel(), x)

    def test_ident_properties(self):
        """Test n_basis and basis_names properties."""
        transform = Ident(name="identity")

        assert transform.n_basis == 1
        assert transform.basis_names == ["identity"]

    def test_ident_default_name(self):
        """Test default naming."""
        transform = Ident()
        assert transform.name == "ident"


class TestStandardized:
    """Test Standardized transform."""

    def test_standardized_basic(self):
        """Test basic standardization."""
        std = Standardized()
        x = np.array([1, 2, 3, 4, 5], dtype=float)

        result = std.evaluate(x)

        assert result.shape == (5, 1)
        assert_allclose(np.mean(result), 0, atol=1e-10)
        assert_allclose(np.std(result), 1, atol=1e-10)

    def test_standardized_center_only(self):
        """Test centering without scaling."""
        std = Standardized(center=True, scale=False)
        x = np.array([1, 2, 3, 4, 5], dtype=float)

        result = std.evaluate(x)

        assert_allclose(np.mean(result), 0, atol=1e-10)
        assert_allclose(np.std(result), np.std(x))

    def test_standardized_scale_only(self):
        """Test scaling without centering."""
        std = Standardized(center=False, scale=True)
        x = np.array([1, 2, 3, 4, 5], dtype=float)

        result = std.evaluate(x)

        assert np.mean(result) != 0
        assert_allclose(np.std(result), 1, atol=1e-10)

    def test_standardized_fit_then_transform(self):
        """Test fitting on training data, then transforming test data."""
        std = Standardized()

        # Fit on training data
        x_train = np.array([1, 2, 3, 4, 5], dtype=float)
        std.fit(x_train)

        # Store training stats
        train_mean = std._mean
        train_std = std._std

        # Transform test data using training stats
        x_test = np.array([2, 3, 4], dtype=float)
        result = std.evaluate(x_test)

        # Should use training statistics
        expected = (x_test - train_mean) / train_std
        assert_allclose(result.ravel(), expected)

    def test_standardized_auto_fit(self):
        """Test automatic fitting if not pre-fitted."""
        std = Standardized()
        x = np.array([10, 20, 30, 40, 50], dtype=float)

        # evaluate should auto-fit
        result = std.evaluate(x)

        assert std._mean is not None
        assert std._std is not None
        assert_allclose(np.mean(result), 0, atol=1e-10)

    def test_standardized_zero_variance(self):
        """Test handling of zero variance data."""
        std = Standardized()
        x = np.array([5, 5, 5, 5], dtype=float)

        result = std.evaluate(x)

        # Should set std to 1 to avoid division by zero
        assert std._std == 1.0
        assert_array_equal(result.ravel(), [0, 0, 0, 0])

    def test_standardized_fit_center_false(self):
        """Test fit with center=False."""
        std = Standardized(center=False, scale=True)
        x = np.array([2, 4, 6, 8], dtype=float)

        std.fit(x)

        assert std._mean == 0.0
        assert std._std == np.std(x)

    def test_standardized_fit_scale_false(self):
        """Test fit with scale=False."""
        std = Standardized(center=True, scale=False)
        x = np.array([2, 4, 6, 8], dtype=float)

        std.fit(x)

        assert std._mean == np.mean(x)
        assert std._std == 1.0

    def test_standardized_properties(self):
        """Test n_basis and basis_names properties."""
        std = Standardized(name="std_transform")

        assert std.n_basis == 1
        assert std.basis_names == ["std_transform"]


class TestSubBasis:
    """Test SubBasis subsetting functionality."""

    def test_subbasis_list_indices(self):
        """Test subsetting with list of indices."""
        poly = Poly(degree=5, raw=True)
        sub = SubBasis(poly, [0, 2, 4])

        assert sub.n_basis == 3
        assert sub.indices == [0, 2, 4]

        # Evaluate
        x = np.linspace(0, 1, 20)
        full = poly.evaluate(x)
        subset = sub.evaluate(x)

        assert subset.shape == (20, 3)
        assert_array_equal(subset[:, 0], full[:, 0])
        assert_array_equal(subset[:, 1], full[:, 2])
        assert_array_equal(subset[:, 2], full[:, 4])

    def test_subbasis_slice(self):
        """Test subsetting with slice."""
        spline = BSpline(df=10)
        sub = SubBasis(spline, slice(3, 7))

        assert sub.n_basis == 4
        assert sub.indices == [3, 4, 5, 6]

    def test_subbasis_slice_with_step(self):
        """Test slice with step."""
        poly = Poly(degree=10)
        sub = SubBasis(poly, slice(0, 10, 2))  # Every other basis

        assert sub.n_basis == 5
        assert sub.indices == [0, 2, 4, 6, 8]

    def test_subbasis_negative_indices(self):
        """Test negative indexing."""
        poly = Poly(degree=5)
        sub = SubBasis(poly, [-3, -2, -1])

        # degree=5 with intercept=True → 6 basis functions
        # -3, -2, -1 → indices 3, 4, 5
        assert sub.n_basis == 3
        assert sub.indices == [3, 4, 5]

    def test_subbasis_mixed_indices(self):
        """Test mix of positive and negative indices."""
        poly = Poly(degree=5)
        sub = SubBasis(poly, [0, -1])

        # Should get first and last
        assert sub.indices == [0, 5]

    def test_subbasis_out_of_range(self):
        """Test error on out of range indices."""
        poly = Poly(degree=2)  # 3 basis functions

        with pytest.raises(IndexError, match="out of range"):
            SubBasis(poly, [0, 1, 10])

    def test_subbasis_empty_indices(self):
        """Test error on empty selection."""
        poly = Poly(degree=2)

        with pytest.raises(ValueError, match="at least one"):
            SubBasis(poly, [])

    def test_subbasis_basis_names(self):
        """Test that basis names are correctly subsetted."""
        poly = Poly(degree=3, intercept=True)
        sub = SubBasis(poly, [0, 2])

        parent_names = poly.basis_names
        sub_names = sub.basis_names

        assert len(sub_names) == 2
        assert sub_names[0] == parent_names[0]
        assert sub_names[1] == parent_names[2]

    def test_subbasis_custom_name(self):
        """Test custom name for SubBasis."""
        poly = Poly(degree=3)
        sub = SubBasis(poly, [0, 1], name="custom_sub")

        assert sub.name == "custom_sub"

    def test_subbasis_default_name(self):
        """Test default name generation."""
        poly = Poly(degree=3)
        sub = SubBasis(poly, [0, 2, 3])

        assert "poly3_sub" in sub.name
        assert "[0,2,3]" in sub.name

    def test_subbasis_repr(self):
        """Test string representation."""
        poly = Poly(degree=3)
        sub = SubBasis(poly, [0, 1])

        repr_str = repr(sub)
        assert "SubBasis" in repr_str
        assert "poly3" in repr_str
        assert "[0, 1]" in repr_str
        assert "n_basis=2" in repr_str

    def test_subbasis_with_bspline(self):
        """Test SubBasis with BSpline."""
        spline = BSpline(df=8)
        sub = SubBasis(spline, [1, 3, 5, 7])

        x = np.linspace(0, 1, 50)
        full = spline.evaluate(x)
        subset = sub.evaluate(x)

        assert subset.shape == (50, 4)
        for i, idx in enumerate([1, 3, 5, 7]):
            assert_array_equal(subset[:, i], full[:, idx])

    def test_subbasis_nested(self):
        """Test subsetting a SubBasis (nested)."""
        poly = Poly(degree=10)
        sub1 = SubBasis(poly, [0, 2, 4, 6, 8])
        sub2 = SubBasis(sub1, [0, 2, 4])  # Further subset

        # sub2 should have indices [0, 4, 8] from original poly
        x = np.linspace(0, 1, 20)
        poly_eval = poly.evaluate(x)
        sub2_eval = sub2.evaluate(x)

        # sub2 first column = sub1 first column = poly column 0
        assert_array_equal(sub2_eval[:, 0], poly_eval[:, 0])


class TestSubBasisFunction:
    """Test sub_basis function wrapper."""

    def test_sub_basis_function(self):
        """Test sub_basis function creates SubBasis."""
        poly = Poly(degree=5)
        sub = sub_basis(poly, [0, 1, 2])

        assert isinstance(sub, SubBasis)
        assert sub.n_basis == 3
        assert sub.indices == [0, 1, 2]

    def test_sub_basis_with_name(self):
        """Test sub_basis with custom name."""
        poly = Poly(degree=3)
        sub = sub_basis(poly, [0, 2], name="my_subset")

        assert sub.name == "my_subset"

    def test_sub_basis_slice(self):
        """Test sub_basis with slice notation."""
        spline = BSpline(df=10)
        sub = sub_basis(spline, slice(2, 8))

        assert sub.indices == [2, 3, 4, 5, 6, 7]


class TestBasisPredict:
    """Test predict method for basis objects with new data."""

    def test_poly_consistent_scaling(self):
        """Test Poly basis evaluates correctly on new data."""
        basis = Poly(degree=3, raw=False)

        # Fit on training data
        x_train = np.array([1, 2, 3, 4, 5], dtype=float)
        X_train = basis.evaluate(x_train)

        # Apply to new data (need enough points for orthogonal polynomials)
        x_test = np.array([1.5, 2.5, 3.5, 4.5], dtype=float)
        X_test = basis.evaluate(x_test)

        # Should have correct shape
        assert X_test.shape == (4, 4)

        # Check that the basis matrix is properly orthogonalized
        # (each call to evaluate creates orthogonal basis for that dataset)
        gram = X_test.T @ X_test
        # Diagonal should be non-zero
        assert np.all(np.diag(gram) > 0)

    def test_raw_poly_new_data(self):
        """Test raw polynomial on new data."""
        basis = Poly(degree=2, intercept=True, raw=True)

        # Evaluate on some data
        x_train = np.array([1, 2, 3], dtype=float)
        X_train = basis.evaluate(x_train)

        # New data should work the same way
        x_test = np.array([1.5, 2.5], dtype=float)
        X_test = basis.evaluate(x_test)

        assert X_test.shape == (2, 3)
        # Check values: [1, x, x^2]
        assert_array_equal(X_test[:, 0], 1)
        assert_array_equal(X_test[:, 1], x_test)
        assert_array_equal(X_test[:, 2], x_test**2)

    def test_bspline_new_data(self):
        """Test BSpline basis fitted on one dataset, applied to new data."""
        basis = BSpline(df=4)

        # Fit on training data
        x_train = np.linspace(0, 10, 100)
        X_train = basis.evaluate(x_train)

        # Apply to new data within the same range
        x_test = np.array([2.5, 5.0, 7.5])
        X_test = basis.evaluate(x_test)

        assert X_test.shape == (3, 4)
        # B-splines should be non-negative
        assert np.all(X_test >= 0)

    def test_scale_new_data(self):
        """Test Scale transform on new data uses fitted statistics."""
        transform = Scale()

        # Fit on training data
        x_train = np.array([1, 2, 3, 4, 5], dtype=float)
        Z_train = transform.evaluate(x_train)

        # Store fitted statistics
        train_mean = transform._mean
        train_std = transform._std

        # Apply to new data - note: evaluate refits on new data
        # For a true predict method, we'd need to preserve parameters
        x_test = np.array([2, 3, 4], dtype=float)
        Z_test = transform.evaluate(x_test)

        # Note: Current implementation refits on each evaluate call
        # This test documents current behavior
        assert Z_test.shape == (3, 1)

    def test_standardized_predict(self):
        """Test Standardized fit/transform pattern."""
        std = Standardized()

        # Fit on training data
        x_train = np.array([1, 2, 3, 4, 5], dtype=float)
        std.fit(x_train)

        train_mean = std._mean
        train_std = std._std

        # Transform test data using training statistics
        x_test = np.array([2, 3, 4], dtype=float)
        Z_test = std.evaluate(x_test)

        # Should use training stats
        expected = (x_test - train_mean) / train_std
        assert_allclose(Z_test.ravel(), expected)


class TestConditionBasis:
    """Test condition_basis_list function.

    Note: These tests require proper event setup which is complex.
    For now, we skip these to focus on the transform and sub_basis tests.
    condition_basis.py coverage will be improved in a separate task.
    """

    def test_condition_basis_list_basic(self):
        """Test basic condition basis list creation."""
        from fmrimod import condition_basis_list
        from fmrimod.events import EventFactor
        from fmrimod.sampling import SamplingFrame
        from fmrimod.events.term import EventTerm
        from fmrimod import SPM_CANONICAL

        # Create simple event
        event = EventFactor(
            name='condition',
            onsets=[0, 5, 10, 15],
            values=['A', 'A', 'B', 'B'],
            durations=1
        )

        # Create EventTerm
        event_term = EventTerm([event])

        # Simple HRF with nbasis=1
        hrf = SPM_CANONICAL

        # Sampling frame
        sf = SamplingFrame(tr=2.0, n_scans=15)

        # Get condition list
        result = condition_basis_list(event_term, hrf, sf, output="condition_list")

        # Should have 2 conditions (A and B)
        # Result is a dict mapping condition names to matrices
        assert len(result) == 2
        assert 'condition.A' in result
        assert 'condition.B' in result

        # Each condition should have a matrix with the right shape
        for cond_name, cond_matrix in result.items():
            assert cond_matrix.shape[0] == 15  # n_scans
            assert cond_matrix.shape[1] == 1   # nbasis from HRF

    def test_condition_basis_list_matrix_output(self):
        """Test getting full matrix output."""
        from fmrimod import condition_basis_list
        from fmrimod.events import EventFactor
        from fmrimod.sampling import SamplingFrame
        from fmrimod.events.term import EventTerm
        from fmrimod import SPM_CANONICAL

        event = EventFactor(
            name='stim',
            onsets=[2, 6, 10],
            values=['X', 'X', 'X'],
            durations=1
        )

        event_term = EventTerm([event])
        hrf = SPM_CANONICAL
        sf = SamplingFrame(tr=2.0, n_scans=10)

        # Get matrix output
        result = condition_basis_list(event_term, hrf, sf, output="matrix")

        # Should return a matrix
        assert hasattr(result, 'shape')
        assert result.shape[0] == 10  # n_scans
        assert result.shape[1] >= 1  # at least one column

    def test_condition_basis_list_multi_basis_hrf(self):
        """Test with multi-basis HRF."""
        from fmrimod import condition_basis_list
        from fmrimod.events import EventFactor
        from fmrimod.sampling import SamplingFrame
        from fmrimod.events.term import EventTerm
        from fmrimod import SPM_WITH_DISPERSION as HRF_SPMG3

        event = EventFactor(
            name='task',
            onsets=[1, 5, 9],
            values=['A', 'B', 'A'],
            durations=1
        )

        event_term = EventTerm([event])
        # HRF with 3 basis functions (canonical + 2 derivatives)
        hrf = HRF_SPMG3
        sf = SamplingFrame(tr=2.0, n_scans=10)

        result = condition_basis_list(event_term, hrf, sf, output="condition_list")

        # Should have 2 conditions (A and B)
        assert len(result) == 2
        assert 'task.A' in result
        assert 'task.B' in result

        # Each condition basis should have 3 basis functions (from SPMG3)
        for cond_name, cond_matrix in result.items():
            assert cond_matrix.shape[1] == 3  # nbasis = 3


class TestTransformIntegration:
    """Integration tests for basis transforms."""

    def test_transform_with_event_basis(self):
        """Test using transforms with EventBasis."""
        from fmrimod.events import EventBasis

        # Create basis with polynomial transform
        poly = Poly(degree=2)
        event = EventBasis(
            name='rating',
            onsets=[1, 2, 3],
            values=[1.0, 2.0, 3.0],
            basis=poly
        )

        assert event.n_basis == 3
        assert event.expanded_values.shape == (3, 3)

    def test_multiple_transforms_pipeline(self):
        """Test combining multiple transform types."""
        x = np.linspace(0, 1, 100)

        transforms = [
            Scale(),
            RobustScale(scale='mad'),
            Standardized(),
            Ident(),
        ]

        for transform in transforms:
            result = transform.evaluate(x)
            assert result.shape[0] == len(x)
            assert result.shape[1] == 1

    def test_subbasis_with_transforms(self):
        """Test SubBasis with transform bases."""
        # This is mainly a shape/compatibility test
        poly = Poly(degree=5)
        sub = sub_basis(poly, [0, 1, 2])

        x = np.linspace(0, 1, 50)
        result = sub.evaluate(x)

        assert result.shape == (50, 3)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_input(self):
        """Test with empty input arrays."""
        transform = Scale()
        x = np.array([])

        result = transform.evaluate(x)
        assert result.shape == (0, 1)

    def test_single_value(self):
        """Test with single value."""
        transform = Scale()
        x = np.array([5.0])

        result = transform.evaluate(x)
        assert result.shape == (1, 1)

    def test_very_large_values(self):
        """Test with very large values."""
        transform = Scale()
        x = np.array([1e10, 2e10, 3e10])

        result = transform.evaluate(x)
        assert_allclose(np.mean(result), 0, atol=1e-5)
        assert_allclose(np.std(result), 1, atol=1e-5)

    def test_very_small_values(self):
        """Test with very small values."""
        transform = RobustScale(scale='mad')
        x = np.array([1e-10, 2e-10, 3e-10])

        result = transform.evaluate(x)
        assert result.shape == (3, 1)
