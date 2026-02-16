"""Regression tests for regressor type-hint resolvability."""

from __future__ import annotations

import importlib
import typing

from fmrimod.regressor.core import Regressor, RegressorSet
from fmrimod.regressor.design import regressor_design


def test_regressor_type_hints_resolve() -> None:
    """Public regressor APIs should support ``typing.get_type_hints``."""
    typing.get_type_hints(Regressor.evaluate)
    typing.get_type_hints(RegressorSet.evaluate)
    typing.get_type_hints(regressor_design)


def test_regressor_type_hints_resolve_with_sparse_module_scope() -> None:
    """Type-hint resolution should not depend on ``csr_matrix`` globals."""
    regressor_core = importlib.import_module("fmrimod.regressor.core")
    regressor_design_mod = importlib.import_module("fmrimod.regressor.design")

    core_globals = dict(vars(regressor_core))
    core_globals.pop("csr_matrix", None)
    typing.get_type_hints(Regressor.evaluate, globalns=core_globals)
    typing.get_type_hints(RegressorSet.evaluate, globalns=core_globals)

    design_globals = dict(vars(regressor_design_mod))
    design_globals.pop("csr_matrix", None)
    typing.get_type_hints(regressor_design, globalns=design_globals)
