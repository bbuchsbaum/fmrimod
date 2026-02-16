"""Regression tests for forward annotation resolution."""

import importlib
import typing


def test_forward_type_hints_resolve_without_custom_namespace():
    # Ensure top-level package import works (exports symbols needed by annotations).
    importlib.import_module("fmrimod")

    modules = {
        "fmrimod.events.variable": ["EventVariable.bin_values"],
        "fmrimod.events.matrix": ["EventMatrix.split_columns", "EventMatrix.apply_transform"],
        "fmrimod.events.basis": ["EventBasis.to_matrix"],
        "fmrimod.formula.parser": ["parse_formula"],
        "fmrimod.utils.misc": ["single_trial_regressor", "hrf_toeplitz"],
        "fmrimod.ar.whitening": ["whiten_apply", "whiten"],
        "fmrimod.regressor.core": ["Regressor.evaluate", "RegressorSet.evaluate"],
        "fmrimod.regressor.neural_input": ["neural_input"],
    }

    for module_name, dotted_names in modules.items():
        mod = importlib.import_module(module_name)
        for dotted in dotted_names:
            obj = mod
            for part in dotted.split("."):
                obj = getattr(obj, part)
            typing.get_type_hints(obj)
