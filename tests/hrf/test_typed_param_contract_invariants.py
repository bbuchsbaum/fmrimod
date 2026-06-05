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

from fmrimod.hrf.core import FunctionHRF, as_hrf, bind_basis
from fmrimod.hrf.decorators import block_hrf, lag_hrf
from fmrimod.hrf.library import (
    BSPLINE_HRF,
    GAMMA_HRF,
    SPM_CANONICAL,
    EmpiricalHRF,
    GammaHRF,
)
from fmrimod.hrf.normalization import _NormalizedHRF, normalize

try:  # multi-basis SPM kinds (typed-fields render path)
    from fmrimod.hrf.library import HRF_SPMG2  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - name guard
    HRF_SPMG2 = None


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
        assert "delay=6.0" in s
        assert "ratio=0.167" in s
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


def _str_contract_cases() -> list[tuple[str, object]]:
    """One instance per structurally distinct HRF kind.

    Covers single-basis, multi-basis, the HRF-valued-field skip path
    (decorator ``base`` / composite ``components``), the non-dataclass
    FunctionHRF adapter, a tuple-field carrier (EmpiricalHRF), and a
    normalized wrapper.
    """
    cases: list[tuple[str, object]] = [
        ("single_basis", GAMMA_HRF),
        ("spm_canonical", SPM_CANONICAL),
        ("function_adapter", as_hrf(lambda t: np.asarray(t), name="fn")),
        ("tuple_fields", EmpiricalHRF([0, 1, 2, 3], [0.0, 1.0, 0.5, 0.1])),
        ("decorator_lag", lag_hrf(SPM_CANONICAL, 2.0)),
        ("decorator_block", block_hrf(SPM_CANONICAL, width=3.0)),
        ("composite_bind", bind_basis(BSPLINE_HRF, GAMMA_HRF)),
        ("normalized", normalize(SPM_CANONICAL, "spm")),
    ]
    if HRF_SPMG2 is not None:
        cases.append(("multi_basis", HRF_SPMG2))
    return cases


class TestStrContractAcrossHRFHierarchy:
    """The Phase-A2 __str__ rewrite touched every subclass's repr.

    Pin the contract hierarchy-wide so a future change to __str__ or to
    any subclass's fields cannot silently break a specific kind's repr
    or reintroduce a params/param_names rendering.
    """

    @pytest.mark.parametrize(
        "hrf", [c[1] for c in _str_contract_cases()],
        ids=[c[0] for c in _str_contract_cases()],
    )
    def test_str_is_well_formed_and_dictless(self, hrf: object) -> None:
        s = str(hrf)
        assert s.startswith("HRF(name='")
        assert f"nbasis={hrf.nbasis}" in s  # type: ignore[attr-defined]
        assert "span=" in s
        assert "params=" not in s
        assert "param_names=" not in s
        # repr() delegates to / is consistent with the same contract.
        assert "params=" not in repr(hrf)

    @pytest.mark.parametrize(
        "hrf",
        [
            lag_hrf(SPM_CANONICAL, 2.0),
            block_hrf(SPM_CANONICAL, width=3.0),
            bind_basis(BSPLINE_HRF, GAMMA_HRF),
        ],
        ids=["lag", "block", "bind"],
    )
    def test_hrf_valued_fields_are_skipped_in_str(self, hrf: object) -> None:
        # The subtle A2 skip-logic: decorator ``base`` / composite
        # ``components`` (HRF / tuple-of-HRF valued) must NOT be rendered
        # (no nested "HRF(name=" beyond the single outer one, no
        # components= dump).
        s = str(hrf)
        assert s.count("HRF(name='") == 1
        assert "components=" not in s
        assert "base=" not in s
