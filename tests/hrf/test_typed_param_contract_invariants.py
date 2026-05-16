"""Regression guards for the typed HRF-parameter contract.

These pin the architectural invariants established by two closed beads
so a future change cannot silently rot them (the same silent-regression
class a prior categorical-ordering incident this codebase hit):

* bd-01KRGCZJ6JAA4BKRTNQ91P2PE5 — the base ``HRF.params`` /
  ``param_names`` dict+registry mirror is gone; HRF parameters are
  typed ``@dataclass`` fields on each subclass (MISSION
  non-negotiable #1: no untyped option containers in public
  signatures).
* bd-01KRRGDT8N87QJN2DJ6DXWVF67 — ``_NormalizedHRF`` makes ``base`` /
  ``norm_mode`` genuinely required (an invalid instance is
  unconstructible, not ``Optional[...] = None`` validated in
  ``__post_init__``).

Every assertion below was red before those beads landed (e.g.
``hasattr(GAMMA_HRF, "params")`` was True; ``_NormalizedHRF()`` raised
a post-init ``ValueError`` rather than a missing-argument
``TypeError``), so this is not a cheap pass.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from fmrimod.hrf.core import FunctionHRF, as_hrf
from fmrimod.hrf.library import GAMMA_HRF, SPM_CANONICAL, GammaHRF
from fmrimod.hrf.normalization import _NormalizedHRF, normalize


class TestHRFParamsDictRemoved:
    """The back-compat params/param_names mirror must stay gone."""

    def test_instances_have_no_params_or_param_names_attr(self) -> None:
        for hrf in (GAMMA_HRF, SPM_CANONICAL, GammaHRF(shape=6.0, rate=1.0)):
            assert not hasattr(hrf, "params")
            assert not hasattr(hrf, "param_names")

    def test_parameters_are_typed_dataclass_fields(self) -> None:
        field_names = {f.name for f in dataclasses.fields(GammaHRF)}
        assert {"shape", "rate"} <= field_names
        assert "params" not in field_names
        assert "param_names" not in field_names

    def test_constructors_reject_params_kwargs(self) -> None:
        # B1 removed the params=/param_names= constructor plumbing.
        with pytest.raises(TypeError):
            FunctionHRF(func=lambda t: np.asarray(t), params={"x": 1})  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            as_hrf(lambda t: np.asarray(t), params={"x": 1})  # type: ignore[call-arg]

    def test_str_renders_typed_fields_not_a_params_dict(self) -> None:
        s = str(SPM_CANONICAL)
        assert s.startswith("HRF(name='SPMG1'")
        assert "p1=5.0" in s
        assert "params=" not in s
        assert "param_names=" not in s


class TestNormalizedHRFRequiredFields:
    """``_NormalizedHRF`` requires base/norm_mode at construction."""

    def test_unconstructible_without_base_and_norm_mode(self) -> None:
        with pytest.raises(TypeError):
            _NormalizedHRF()  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            _NormalizedHRF(base=SPM_CANONICAL)  # type: ignore[call-arg]

    def test_no_optional_none_then_post_init_pattern(self) -> None:
        # The cheap-pass disqualifier: a __post_init__ that still
        # permits None construction. There must be no such hook, and
        # base/norm_mode are not Optional-defaulted dataclass fields.
        assert "__post_init__" not in vars(_NormalizedHRF)
        field_names = {f.name for f in dataclasses.fields(_NormalizedHRF)}
        assert "base" not in field_names
        assert "norm_mode" not in field_names

    def test_factory_produces_typed_required_attributes(self) -> None:
        h = normalize(SPM_CANONICAL, "spm")
        assert isinstance(h, _NormalizedHRF)
        assert h.base is SPM_CANONICAL
        assert h.norm_mode == "spm"
        assert np.ndim(h.norm_factor) == 0
        t = np.arange(0.0, 24.0, 1.0)
        assert np.all(np.isfinite(h(t)))
