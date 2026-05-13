"""Tests for HRF core functionality."""

import numpy as np
import pytest

from fmrimod.hrf.core import HRF, as_hrf, bind_basis
from fmrimod.hrf.library import (
    BSPLINE_HRF,
    FOURIER_HRF,
    GAMMA_HRF,
    GAUSSIAN_HRF,
    SPM_CANONICAL,
    SPM_WITH_DERIVATIVE,
    SPM_WITH_DISPERSION,
)


class TestHRFBase:
    """Test base HRF functionality."""
    
    def test_spm_canonical_structure(self):
        """Test SPM canonical HRF has correct structure."""
        assert isinstance(SPM_CANONICAL, HRF)
        assert SPM_CANONICAL.name == "SPMG1"
        assert SPM_CANONICAL.nbasis == 1
        assert SPM_CANONICAL.span == 24.0
        assert SPM_CANONICAL.param_names == ["p1", "p2", "a1"]
        assert SPM_CANONICAL.params == {"p1": 5.0, "p2": 15.0, "a1": 0.0833}
    
    def test_spm_canonical_evaluation(self, time_grid):
        """Test SPM canonical HRF evaluation."""
        result = SPM_CANONICAL(time_grid)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == time_grid.shape
        assert np.all(np.isfinite(result))
        
        # Check negative times return zero
        neg_times = np.array([-5, -2, -1])
        neg_result = SPM_CANONICAL(neg_times)
        assert np.allclose(neg_result, 0)
        
        # Check approximate peak timing (should be around 5-6 seconds)
        peak_idx = np.argmax(result)
        peak_time = time_grid[peak_idx]
        assert 4 <= peak_time <= 7
    
    def test_spm_with_derivative(self, time_grid):
        """Test SPM with temporal derivative."""
        assert SPM_WITH_DERIVATIVE.name == "SPMG2"
        assert SPM_WITH_DERIVATIVE.nbasis == 2
        
        result = SPM_WITH_DERIVATIVE(time_grid)
        assert result.shape == (len(time_grid), 2)
        
        # Test single time point returns proper shape
        single_result = SPM_WITH_DERIVATIVE(5.0)
        assert single_result.shape == (1, 2)
    
    def test_spm_with_dispersion(self, time_grid):
        """Test SPM with dispersion derivative."""
        assert SPM_WITH_DISPERSION.name == "SPMG3"
        assert SPM_WITH_DISPERSION.nbasis == 3
        
        result = SPM_WITH_DISPERSION(time_grid)
        assert result.shape == (len(time_grid), 3)
    
    def test_gamma_hrf(self, time_grid):
        """Test Gamma HRF."""
        assert GAMMA_HRF.name == "gamma"
        assert GAMMA_HRF.param_names == ["shape", "rate"]
        
        result = GAMMA_HRF(time_grid)
        assert result.shape == time_grid.shape
        assert np.all(result >= 0)  # Gamma should be non-negative
    
    def test_gaussian_hrf(self, time_grid):
        """Test Gaussian HRF."""
        assert GAUSSIAN_HRF.name == "gaussian"
        assert GAUSSIAN_HRF.param_names == ["mean", "sd"]
        
        result = GAUSSIAN_HRF(time_grid)
        assert result.shape == time_grid.shape
        assert np.all(result >= 0)  # Gaussian PDF is non-negative
    
    def test_bspline_hrf(self, time_grid):
        """Test B-spline HRF."""
        assert BSPLINE_HRF.name == "bspline"
        assert BSPLINE_HRF.nbasis == 5
        
        result = BSPLINE_HRF(time_grid)
        assert result.shape == (len(time_grid), 5)
    
    def test_evaluate_with_duration(self):
        """Test HRF evaluation with block duration."""
        t = np.arange(0, 20, 0.2)
        
        # Zero duration should match direct evaluation
        result_zero = SPM_CANONICAL.evaluate(t, duration=0)
        result_direct = SPM_CANONICAL(t)
        assert np.allclose(result_zero, result_direct)
        
        # Non-zero duration should produce larger response
        result_block = SPM_CANONICAL.evaluate(t, duration=2)
        assert np.max(result_block) > np.max(result_direct)
        
        # Test summation vs max
        result_sum = SPM_CANONICAL.evaluate(t, duration=2, summate=True)
        result_max = SPM_CANONICAL.evaluate(t, duration=2, summate=False)
        assert not np.allclose(result_sum, result_max)
        
        # Test normalization
        result_norm = SPM_CANONICAL.evaluate(t, duration=2, normalize=True)
        assert np.abs(np.max(np.abs(result_norm)) - 1.0) < 1e-7
    
    def test_evaluate_precision(self):
        """Test evaluation precision parameter."""
        t = np.arange(0, 20, 0.2)
        
        result_fine = SPM_CANONICAL.evaluate(t, duration=2, precision=0.1)
        result_coarse = SPM_CANONICAL.evaluate(t, duration=2, precision=0.5)
        
        # Results should be similar but not identical
        assert not np.array_equal(result_fine, result_coarse)
        assert np.corrcoef(result_fine, result_coarse)[0, 1] > 0.99


class TestSPMGTypedFields:
    """SPMG HRFs expose their double-gamma parameters as typed dataclass fields.

    See bead ``bd-01KRGCYHKHM07TC5ANX3EEMPQE``; this regression suite locks the
    typed-field contract so subsequent type-end-to-end steps can't unwind it.
    """

    def test_spmg1_typed_fields(self):
        from dataclasses import fields

        from fmrimod.hrf.spm_hrf import SPMG1_HRF

        hrf = SPMG1_HRF()
        assert hrf.p1 == 5.0
        assert hrf.p2 == 15.0
        assert hrf.a1 == 0.0833

        field_names = {f.name for f in fields(hrf)}
        assert {"p1", "p2", "a1"}.issubset(field_names)

    def test_spmg1_constructor_overrides(self):
        from fmrimod.hrf.spm_hrf import SPMG1_HRF

        hrf = SPMG1_HRF(p1=6.0, p2=14.0, a1=0.05)
        assert hrf.p1 == 6.0
        assert hrf.p2 == 14.0
        assert hrf.a1 == 0.05
        # Back-compat: params dict mirrors the typed fields during transition.
        assert hrf.params == {"p1": 6.0, "p2": 14.0, "a1": 0.05}

    def test_spmg2_typed_fields_and_call_uses_them(self):
        from fmrimod.hrf.spm_hrf import SPMG2_HRF

        hrf = SPMG2_HRF(p1=6.0)
        assert hrf.p1 == 6.0
        # Mutating params dict must not change the basis: __call__ reads typed
        # fields, not the dict.
        hrf.params["p1"] = 99.0
        result_default_p1 = SPMG2_HRF(p1=6.0)(np.arange(0, 24, 1.0))
        result_mutated_dict = hrf(np.arange(0, 24, 1.0))
        assert np.allclose(result_default_p1, result_mutated_dict)

    def test_spmg3_typed_fields(self):
        from fmrimod.hrf.spm_hrf import SPMG3_HRF

        hrf = SPMG3_HRF(p1=4.5, p2=13.0, a1=0.07)
        assert hrf.p1 == 4.5
        assert hrf.p2 == 13.0
        assert hrf.a1 == 0.07
        assert hrf.nbasis == 3


class TestBasisHRFTypedFields:
    """Basis-set and parametric HRFs expose typed dataclass fields.

    See bead ``bd-01KRGCYQAN2YGB2TE200JNYD0T``; locks the contract that
    each kind has named fields rather than ``Dict[str, Any]`` keys.
    """

    def test_gamma_typed_fields(self):
        from fmrimod.hrf.library import GammaHRF

        hrf = GammaHRF(shape=8.0, rate=1.5)
        assert hrf.shape == 8.0
        assert hrf.rate == 1.5
        assert hrf.params == {"shape": 8.0, "rate": 1.5}

    def test_gaussian_typed_fields(self):
        from fmrimod.hrf.library import GaussianHRF

        hrf = GaussianHRF(mean=4.0, sd=1.5)
        assert hrf.mean == 4.0
        assert hrf.sd == 1.5

    def test_bspline_typed_fields(self):
        from fmrimod.hrf.library import BSplineHRF

        hrf = BSplineHRF(nbasis=7, degree=3, span=30.0)
        assert hrf.nbasis == 7
        assert hrf.degree == 3
        assert hrf.span == 30.0
        result = hrf(np.linspace(0, 30, 50))
        assert result.shape == (50, 7)

    def test_fir_typed_fields(self):
        from fmrimod.hrf.library import FIRHRF

        hrf = FIRHRF(nbasis=8, span=20.0)
        assert hrf.nbasis == 8
        result = hrf(np.linspace(0, 20, 40))
        assert result.shape == (40, 8)

    def test_lwu_typed_fields_and_call_uses_them(self):
        from fmrimod.hrf.library import LWUHRF

        hrf = LWUHRF(tau=7.0, sigma=3.0, rho=0.4)
        assert hrf.tau == 7.0
        assert hrf.sigma == 3.0
        assert hrf.rho == 0.4
        # __call__ reads typed fields, not the back-compat params mirror.
        hrf.params["tau"] = -999
        ref = LWUHRF(tau=7.0, sigma=3.0, rho=0.4)(np.arange(0, 30, 1.0))
        assert np.allclose(hrf(np.arange(0, 30, 1.0)), ref)

    def test_lwu_validates_sigma_at_construction(self):
        from fmrimod.hrf.library import LWUHRF

        with pytest.raises(ValueError, match="sigma must be > 0.05"):
            LWUHRF(sigma=0.01)

    def test_lwu_validates_rho_at_construction(self):
        from fmrimod.hrf.library import LWUHRF

        with pytest.raises(ValueError, match="rho must be between 0 and 1.5"):
            LWUHRF(rho=2.0)

    def test_lwu_basis_typed_fields(self):
        from fmrimod.hrf.library import LWUBasisHRF

        hrf = LWUBasisHRF(theta0=(7.0, 3.0, 0.5))
        assert hrf.theta0 == (7.0, 3.0, 0.5)
        result = hrf(np.linspace(0, 30, 60))
        assert result.shape == (60, 4)

    def test_lwu_basis_rejects_wrong_theta0_length(self):
        from fmrimod.hrf.library import LWUBasisHRF

        with pytest.raises(ValueError, match="theta0 must have length 3"):
            LWUBasisHRF(theta0=(1.0, 2.0))  # type: ignore[arg-type]


class TestDecoratorComposition:
    """Decorators return typed subclasses that preserve the base HRF.

    See bead ``bd-01KRGCYXE7CQTT86MW7FRR309A``; locks the contract that
    structural composition is introspectable rather than opaque.
    """

    def test_lagged_preserves_base(self):
        from fmrimod.hrf.decorators import LaggedHRF, lag_hrf

        lagged = lag_hrf(SPM_CANONICAL, 2.0)
        assert isinstance(lagged, LaggedHRF)
        assert lagged.base is SPM_CANONICAL
        assert lagged.lag == 2.0

    def test_blocked_preserves_base(self):
        from fmrimod.hrf.decorators import BlockedHRF, block_hrf

        blocked = block_hrf(SPM_CANONICAL, width=4.0)
        assert isinstance(blocked, BlockedHRF)
        assert blocked.base is SPM_CANONICAL
        assert blocked.width == 4.0

    def test_bound_basis_preserves_components(self):
        from fmrimod.hrf.core import BoundBasisHRF

        combined = SPM_CANONICAL + GAMMA_HRF  # __add__ -> bind_basis
        assert isinstance(combined, BoundBasisHRF)
        assert combined.components == (SPM_CANONICAL, GAMMA_HRF)
        assert combined.nbasis == SPM_CANONICAL.nbasis + GAMMA_HRF.nbasis

    def test_lag_validates_finiteness_at_construction(self):
        from fmrimod.hrf.decorators import LaggedHRF

        with pytest.raises(ValueError, match="lag must be finite"):
            LaggedHRF(base=SPM_CANONICAL, lag=float("inf"))

    def test_block_validates_width_at_construction(self):
        from fmrimod.hrf.decorators import BlockedHRF

        with pytest.raises(ValueError, match="width must be finite"):
            BlockedHRF(base=SPM_CANONICAL, width=float("inf"))

    def test_block_validates_precision_at_construction(self):
        from fmrimod.hrf.decorators import BlockedHRF

        with pytest.raises(ValueError, match="precision must be finite and positive"):
            BlockedHRF(base=SPM_CANONICAL, width=4.0, precision=0.0)

    def test_evaluate_block_matches_blocked_hrf(self):
        """HRF.evaluate(duration=d) now delegates to BlockedHRF.

        Both paths must agree numerically; this catches drift if either
        the inline quadrature or the BlockedHRF implementation diverges.
        """
        from fmrimod.hrf.decorators import BlockedHRF

        grid = np.linspace(0, 24, 80)
        via_evaluate = SPM_CANONICAL.evaluate(grid, duration=4.0)
        via_blocked = BlockedHRF(base=SPM_CANONICAL, width=4.0)(grid)
        assert np.allclose(via_evaluate, via_blocked)


class TestPenaltyTypeDispatch:
    """penalty_matrix dispatches on HRF type, not hrf.name.

    Regression for the bug where any decorator chain mutated the name
    string and dropped penalty selection back to ridge silently. See
    bead ``bd-01KRGCZ1VMBRS9BXWVM1DTDE4M``.
    """

    def test_spmg2_penalty_survives_lag(self):
        from fmrimod.hrf.penalty import penalty_matrix

        direct = penalty_matrix(SPM_WITH_DERIVATIVE)
        lagged = penalty_matrix(SPM_WITH_DERIVATIVE.lag(2.0))
        assert np.allclose(direct, lagged)
        # Sanity: SPMG2 penalty leaves canonical column unpenalized and
        # shrinks the derivative column.
        assert direct[0, 0] == 0.0
        assert direct[1, 1] == 2.0

    def test_spmg3_penalty_survives_block(self):
        from fmrimod.hrf.decorators import block_hrf
        from fmrimod.hrf.penalty import penalty_matrix

        direct = penalty_matrix(SPM_WITH_DISPERSION)
        blocked = penalty_matrix(block_hrf(SPM_WITH_DISPERSION, width=4.0))
        assert np.allclose(direct, blocked)

    def test_bspline_penalty_survives_normalize(self):
        from fmrimod.hrf.decorators import normalize_hrf
        from fmrimod.hrf.penalty import penalty_matrix

        direct = penalty_matrix(BSPLINE_HRF)
        normalized = penalty_matrix(normalize_hrf(BSPLINE_HRF))
        assert np.allclose(direct, normalized)
        assert direct.shape == (BSPLINE_HRF.nbasis, BSPLINE_HRF.nbasis)

    def test_fourier_penalty_uses_frequency(self):
        from fmrimod.hrf.penalty import penalty_matrix

        p = penalty_matrix(FOURIER_HRF)
        # Diagonal frequencies: ceil([1..nbasis]/2) -> [1, 1, 2, 2, 3]
        assert np.allclose(np.diag(p), [1, 1, 4, 4, 9])  # squared

    def test_bound_basis_penalty_is_block_diagonal(self):
        from fmrimod.hrf.penalty import penalty_matrix

        combined = SPM_WITH_DERIVATIVE + BSPLINE_HRF  # 2 + 5 = 7 cols
        p = penalty_matrix(combined)
        # Top-left 2x2 must match SPMG2; bottom-right 5x5 must match BSpline.
        assert np.allclose(p[:2, :2], penalty_matrix(SPM_WITH_DERIVATIVE))
        assert np.allclose(p[2:, 2:], penalty_matrix(BSPLINE_HRF))
        # Off-diagonal blocks zero.
        assert np.allclose(p[:2, 2:], 0)
        assert np.allclose(p[2:, :2], 0)

    def test_unknown_hrf_defaults_to_ridge(self):
        from fmrimod.hrf.penalty import penalty_matrix

        p = penalty_matrix(GAMMA_HRF)  # no specialised handler
        assert np.allclose(p, np.eye(GAMMA_HRF.nbasis))


class TestAsHRF:
    """Test as_hrf function."""
    
    def test_simple_function(self):
        """Test converting simple function to HRF."""
        def my_func(t):
            return t ** 2
        
        hrf_obj = as_hrf(my_func, name="test_sq", nbasis=1, span=10,
                        params={"power": 2})
        
        assert isinstance(hrf_obj, HRF)
        assert hrf_obj.name == "test_sq"
        assert hrf_obj.nbasis == 1
        assert hrf_obj.span == 10
        assert hrf_obj.param_names == ["power"]
        assert hrf_obj.params == {"power": 2}
        
        # Test evaluation
        assert hrf_obj(5) == 25
        assert np.array_equal(hrf_obj(np.array([1, 2, 3])), np.array([1, 4, 9]))
    
    def test_defaults(self):
        """Test as_hrf with default parameters."""
        def my_func(t):
            return np.sin(t)
        
        hrf_obj = as_hrf(my_func)
        assert hrf_obj.name == "my_func"
        assert hrf_obj.nbasis == 1
        assert hrf_obj.span == 24.0
        assert hrf_obj.params == {}
        assert hrf_obj.param_names is None
    
    def test_multi_basis_function(self):
        """Test as_hrf with multi-basis function."""
        def my_multi_func(t):
            t = np.asarray(t)
            return np.column_stack([t, t**2])
        
        hrf_obj = as_hrf(my_multi_func, nbasis=2)
        assert hrf_obj.nbasis == 2
        
        result = hrf_obj(3)
        assert np.array_equal(result, np.array([[3, 9]]))


class TestBindBasis:
    """Test bind_basis function."""
    
    def test_combine_single_basis(self):
        """Test combining single-basis HRFs."""
        f1 = lambda t: t
        f2 = lambda t: t**2
        f3 = lambda t: np.ones_like(t)
        
        hrf1 = as_hrf(f1, name="linear", span=10)
        hrf2 = as_hrf(f2, name="quadratic", span=12)
        hrf3 = as_hrf(f3, name="constant", span=8)
        
        combined = bind_basis(hrf1, hrf2, hrf3)
        
        assert combined.name == "linear + quadratic + constant"
        assert combined.nbasis == 3
        assert combined.span == 12  # max(10, 12, 8)
        
        # Test evaluation
        t_vals = np.array([0, 1, 2, 5])
        result = combined(t_vals)
        expected = np.column_stack([f1(t_vals), f2(t_vals), f3(t_vals)])
        assert np.array_equal(result, expected)
    
    def test_combine_with_multi_basis(self):
        """Test combining with multi-basis HRF."""
        f1 = lambda t: t
        f_multi = lambda t: np.column_stack([np.sin(t), np.cos(t)])
        
        hrf1 = as_hrf(f1, name="linear", nbasis=1, span=10)
        hrf_multi = as_hrf(f_multi, name="trig", nbasis=2, span=15)
        
        combined = bind_basis(hrf1, hrf_multi)
        
        assert combined.nbasis == 3
        assert combined.span == 15
        assert combined.name == "linear + trig"
        
        t_vals = np.array([0, 1, 2])
        result = combined(t_vals)
        assert result.shape == (3, 3)
    
    def test_single_hrf(self):
        """Test bind_basis with single HRF."""
        hrf = as_hrf(lambda t: t**2, name="square")
        bound = bind_basis(hrf)
        
        assert bound.name == "square"
        assert bound.nbasis == 1
        assert np.array_equal(bound(5), hrf(5))
    
    def test_empty_input(self):
        """Test bind_basis with no arguments."""
        with pytest.raises(ValueError, match="At least one HRF"):
            bind_basis()


class TestHRFFromCoefficients:
    """Test HRF from_coefficients method."""
    
    def test_single_basis(self):
        """Test from_coefficients with single basis."""
        weighted = SPM_CANONICAL.from_coefficients([2.5])
        
        t = np.array([0, 5, 10])
        expected = 2.5 * SPM_CANONICAL(t)
        assert np.allclose(weighted(t), expected)
    
    def test_multi_basis(self):
        """Test from_coefficients with multi-basis HRF."""
        coeffs = np.array([1.0, -0.5, 0.3])
        weighted = SPM_WITH_DISPERSION.from_coefficients(coeffs)
        
        t = np.array([0, 5, 10])
        basis_vals = SPM_WITH_DISPERSION(t)
        expected = basis_vals @ coeffs
        assert np.allclose(weighted(t), expected)
    
    def test_wrong_number_coefficients(self):
        """Test error with wrong number of coefficients."""
        with pytest.raises(ValueError, match="Number of coefficients"):
            SPM_WITH_DERIVATIVE.from_coefficients([1.0])  # Needs 2 coeffs


class TestHRFValidation:
    """Test HRF input validation."""
    
    def test_evaluate_empty_grid(self):
        """Test evaluate with empty grid."""
        with pytest.raises(ValueError, match="grid must contain"):
            SPM_CANONICAL.evaluate(np.array([]))
    
    def test_evaluate_nan_grid(self):
        """Test evaluate with NaN in grid."""
        with pytest.raises(ValueError, match="cannot contain NaN"):
            SPM_CANONICAL.evaluate(np.array([0, 5, np.nan, 10]))
    
    def test_evaluate_invalid_precision(self):
        """Test evaluate with invalid precision."""
        t = np.array([0, 5, 10])
        
        with pytest.raises(ValueError, match="precision must be positive"):
            SPM_CANONICAL.evaluate(t, precision=0)
        
        with pytest.raises(ValueError, match="precision must be positive"):
            SPM_CANONICAL.evaluate(t, precision=-0.5)


class TestHRFStringRepresentation:
    """Test HRF string representations."""
    
    def test_str_representation(self):
        """Test __str__ method."""
        str_repr = str(SPM_CANONICAL)
        assert "HRF(name='SPMG1'" in str_repr
        assert "nbasis=1" in str_repr
        assert "span=24.0" in str_repr
        assert "p1=5.0" in str_repr
    
    def test_repr(self):
        """Test __repr__ method."""
        repr_str = repr(GAMMA_HRF)
        assert "HRF(name='gamma'" in repr_str
        assert "shape=6.0" in repr_str
        assert "rate=1.0" in repr_str
