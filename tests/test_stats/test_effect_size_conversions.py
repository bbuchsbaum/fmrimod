"""Tests for parity conversion helpers used in meta-analysis."""

import numpy as np
import pytest

import fmrimod
from fmrimod.stats import r_to_z, t_to_d, z_to_r


def test_t_to_d_one_sample_formula_matches_reference():
    t = np.array([2.0, -1.5], dtype=np.float64)
    df = np.array([18.0, 9.0], dtype=np.float64)
    d, v = t_to_d(t, df)

    n = df + 1.0
    expected_d = t * np.sqrt(1.0 / n)
    expected_v = 1.0 / n + expected_d ** 2 / (2.0 * n)
    np.testing.assert_allclose(d, expected_d, atol=1e-12)
    np.testing.assert_allclose(v, expected_v, atol=1e-12)


def test_t_to_d_two_sample_formula_matches_reference():
    t = np.array([2.5], dtype=np.float64)
    df = np.array([30.0], dtype=np.float64)
    d, v = t_to_d(t, df, n=20.0)

    expected_d = 2.0 * t / np.sqrt(df)
    expected_v = 4.0 / 20.0 + expected_d ** 2 / (2.0 * df)
    np.testing.assert_allclose(d, expected_d, atol=1e-12)
    np.testing.assert_allclose(v, expected_v, atol=1e-12)


def test_r_to_z_and_z_to_r_roundtrip():
    r = np.array([0.1, -0.5, 0.8], dtype=np.float64)
    z, v = r_to_z(r, n=30.0)
    r_back = z_to_r(z)

    expected_v = np.full_like(r, 1.0 / (30.0 - 3.0))
    np.testing.assert_allclose(v, expected_v, atol=1e-12)
    np.testing.assert_allclose(r_back, r, atol=1e-12)


def test_conversion_helpers_validate_domains():
    with pytest.raises(ValueError, match="df must be > 0"):
        t_to_d(2.0, df=0.0)
    with pytest.raises(ValueError, match="n must be > 0"):
        t_to_d(2.0, df=10.0, n=0.0)
    with pytest.raises(ValueError, match="strictly between -1 and 1"):
        r_to_z(1.0, n=20.0)
    with pytest.raises(ValueError, match="n must be > 3"):
        r_to_z(0.2, n=3.0)


def test_top_level_conversion_wrappers_match_stats_module():
    d1, v1 = t_to_d(2.0, df=18.0)
    d2, v2 = fmrimod.t_to_d(2.0, df=18.0)
    np.testing.assert_allclose(d1, d2, atol=1e-12)
    np.testing.assert_allclose(v1, v2, atol=1e-12)

    z1, v1 = r_to_z(0.2, n=25.0)
    z2, v2 = fmrimod.r_to_z(0.2, n=25.0)
    np.testing.assert_allclose(z1, z2, atol=1e-12)
    np.testing.assert_allclose(v1, v2, atol=1e-12)

    np.testing.assert_allclose(fmrimod.z_to_r(z1), z_to_r(z1), atol=1e-12)

