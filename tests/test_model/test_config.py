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

    def test_false_string_alias(self):
        opts = RobustOptions(type="FALSE")
        assert opts.type is False
        assert not opts.enabled

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="robust type"):
            RobustOptions(type="invalid")

    def test_max_iter_validation(self):
        with pytest.raises(ValueError, match="max_iter"):
            RobustOptions(max_iter=0)

    def test_scale_scope_local_alias_maps_to_voxel(self):
        opts = RobustOptions(scale_scope="local")
        assert opts.scale_scope == "voxel"

    def test_reestimate_phi_requires_boolean_scalar(self):
        with pytest.raises(ValueError, match="reestimate_phi must be a boolean scalar"):
            RobustOptions(reestimate_phi=1)
        with pytest.raises(ValueError, match="reestimate_phi must be a boolean scalar"):
            RobustOptions(reestimate_phi="yes")

    def test_reestimate_phi_must_be_boolean_scalar(self):
        for value in [1, "yes"]:
            with pytest.raises(ValueError, match="reestimate_phi must be a boolean scalar"):
                RobustOptions(reestimate_phi=value)


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

    def test_global_alias_is_accepted(self):
        opts = AROptions(**{"struct": "ar1", "global": True})
        assert opts.global_ar is True

    def test_global_alias_conflicts_with_global_ar(self):
        with pytest.raises(TypeError, match="only one of 'global' or 'global_ar'"):
            AROptions(global_ar=True, **{"global": True})

    def test_logical_flags_require_boolean_scalars(self):
        with pytest.raises(ValueError, match="global_ar must be a boolean scalar"):
            AROptions(global_ar=1)
        with pytest.raises(ValueError, match="voxelwise must be a boolean scalar"):
            AROptions(voxelwise="yes")
        with pytest.raises(ValueError, match="exact_first must be a boolean scalar"):
            AROptions(exact_first=1)

    def test_censor_accepts_auto_string(self):
        opts = AROptions(censor="auto")
        assert opts.censor == "auto"

    def test_censor_accepts_numeric_or_logical_vectors(self):
        opts_num = AROptions(censor=[1, 4, 9])
        opts_bool = AROptions(censor=[True, False, True])
        assert opts_num.censor == [1, 4, 9]
        assert opts_bool.censor == [True, False, True]

    def test_censor_rejects_invalid_string(self):
        with pytest.raises(ValueError, match="censor must be"):
            AROptions(censor="bad_mode")

    def test_censor_rejects_nonnumeric_nonlogical(self):
        with pytest.raises(ValueError, match="censor must be"):
            AROptions(censor={"bad": 1})

    def test_censor_accepts_auto_numeric_or_logical(self):
        assert AROptions(censor="auto").censor == "auto"
        assert AROptions(censor=[1, 5, 10]).censor == [1, 5, 10]
        assert AROptions(censor=[True, False, True]).censor == [True, False, True]

    def test_censor_rejects_invalid_string(self):
        with pytest.raises(ValueError, match="censor must be None, 'auto'"):
            AROptions(censor="bad")

    def test_censor_rejects_invalid_type(self):
        with pytest.raises(ValueError, match="censor must be None, 'auto'"):
            AROptions(censor={"bad": "type"})

    def test_logical_flags_must_be_boolean_scalars(self):
        for kwargs in [
            {"global_ar": 1},
            {"voxelwise": "yes"},
            {"exact_first": "no"},
        ]:
            with pytest.raises(ValueError, match="must be a boolean scalar"):
                AROptions(**kwargs)


class TestVolumeWeightOptions:
    def test_defaults(self):
        opts = VolumeWeightOptions()
        assert not opts.enabled

    def test_threshold_validation(self):
        with pytest.raises(ValueError, match="threshold"):
            VolumeWeightOptions(threshold=-1.0)

    def test_weights_must_be_1d(self):
        with pytest.raises(ValueError, match="1-D"):
            VolumeWeightOptions(weights=np.ones((5, 1)))

    def test_weights_must_be_finite(self):
        bad = np.array([1.0, np.nan, 0.5])
        with pytest.raises(ValueError, match="finite"):
            VolumeWeightOptions(weights=bad)

    def test_weights_must_be_nonnegative(self):
        bad = np.array([1.0, -0.1, 0.5])
        with pytest.raises(ValueError, match="non-negative"):
            VolumeWeightOptions(weights=bad)


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

    def test_r_compat_aliases(self):
        cfg = fmri_lm_control(
            robust_options={"type": "FALSE", "scale_scope": "local"},
            ar_options={"struct": "ar1", "global": True},
        )
        assert cfg.robust.type is False
        assert cfg.robust.scale_scope == "voxel"
        assert cfg.ar.global_ar is True

    def test_soft_subspace_lambda_alias(self):
        nuisance = np.ones((6, 1))
        cfg = fmri_lm_control(
            soft_subspace_options={
                "enabled": True,
                "nuisance_matrix": nuisance,
                "lambda": 0.25,
            }
        )
        assert cfg.soft_subspace.lam == 0.25
