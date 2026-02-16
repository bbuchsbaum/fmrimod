"""Tests for FmriLmConfig and option dataclasses."""

import numpy as np
import pytest

from fmrimod.model.config import (
    AROptions,
    FmriLmConfig,
    RobustOptions,
    SoftSubspaceOptions,
    VolumeWeightOptions,
    fmri_lm_control,
)


class TestRobustOptions:
    def test_defaults(self):
        opts = RobustOptions()
        assert opts.type is False
        assert not opts.enabled

    def test_huber(self):
        opts = RobustOptions(type="huber")
        assert opts.enabled
        assert opts.k_huber == 1.345

    def test_bisquare(self):
        opts = RobustOptions(type="bisquare")
        assert opts.enabled
        assert opts.c_tukey == 4.685

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="robust type"):
            RobustOptions(type="invalid")

    def test_max_iter_validation(self):
        with pytest.raises(ValueError, match="max_iter"):
            RobustOptions(max_iter=0)


class TestAROptions:
    def test_defaults(self):
        opts = AROptions()
        assert opts.struct == "iid"
        assert opts.ar_order == 0
        assert not opts.enabled

    def test_ar1(self):
        opts = AROptions(struct="ar1")
        assert opts.ar_order == 1
        assert opts.enabled

    def test_ar2(self):
        opts = AROptions(struct="ar2")
        assert opts.ar_order == 2

    def test_arp(self):
        opts = AROptions(struct="arp", p=3)
        assert opts.ar_order == 3

    def test_arp_missing_p(self):
        with pytest.raises(ValueError, match="p must be specified"):
            AROptions(struct="arp")

    def test_invalid_struct(self):
        with pytest.raises(ValueError):
            AROptions(struct="ar99")


class TestVolumeWeightOptions:
    def test_defaults(self):
        opts = VolumeWeightOptions()
        assert not opts.enabled

    def test_threshold_validation(self):
        with pytest.raises(ValueError, match="threshold"):
            VolumeWeightOptions(threshold=-1.0)


class TestSoftSubspaceOptions:
    def test_defaults(self):
        opts = SoftSubspaceOptions()
        assert not opts.enabled

    def test_enabled_without_matrix(self):
        with pytest.raises(ValueError, match="nuisance_matrix"):
            SoftSubspaceOptions(enabled=True)

    def test_enabled_with_matrix(self):
        mat = np.random.randn(100, 5)
        opts = SoftSubspaceOptions(enabled=True, nuisance_matrix=mat)
        assert opts.enabled


class TestFmriLmConfig:
    def test_defaults(self):
        cfg = FmriLmConfig()
        assert "OLS" in repr(cfg)

    def test_repr_ar(self):
        cfg = FmriLmConfig(ar=AROptions(struct="ar1"))
        assert "ar=ar1" in repr(cfg)

    def test_repr_robust(self):
        cfg = FmriLmConfig(robust=RobustOptions(type="huber"))
        assert "robust=huber" in repr(cfg)


class TestFmriLmControl:
    def test_defaults(self):
        cfg = fmri_lm_control()
        assert isinstance(cfg, FmriLmConfig)
        assert cfg.ar.struct == "iid"

    def test_ar_options(self):
        cfg = fmri_lm_control(ar_options={"struct": "ar1"})
        assert cfg.ar.struct == "ar1"

    def test_robust_options(self):
        cfg = fmri_lm_control(robust_options={"type": "bisquare", "max_iter": 5})
        assert cfg.robust.type == "bisquare"
        assert cfg.robust.max_iter == 5
