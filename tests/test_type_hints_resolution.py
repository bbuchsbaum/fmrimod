"""Regression tests for forward annotation resolution."""

import importlib
import typing


def test_forward_type_hints_resolve_for_recent_f821_hotspots():
    events_mod = importlib.import_module("fmrimod.events")
    formula_base_mod = importlib.import_module("fmrimod.formula.base")
    hrf_mod = importlib.import_module("fmrimod.hrf")
    regressor_mod = importlib.import_module("fmrimod.regressor")
    shared_ns = {
        **vars(events_mod),
        **vars(formula_base_mod),
        **vars(hrf_mod),
        **vars(regressor_mod),
    }

    modules = {
        "fmrimod.events.variable": ["EventVariable.bin_values"],
        "fmrimod.events.matrix": ["EventMatrix.split_columns", "EventMatrix.apply_transform"],
        "fmrimod.events.basis": ["EventBasis.to_matrix"],
        "fmrimod.formula.parser": ["parse_formula"],
        "fmrimod.utils.misc": ["single_trial_regressor", "hrf_toeplitz"],
        "fmrimod.ar.whitening": ["whiten_apply", "whiten"],
        "fmrimod.regressor.core": ["Regressor.evaluate", "RegressorSet.evaluate"],
    }

    for module_name, dotted_names in modules.items():
        mod = importlib.import_module(module_name)
        for dotted in dotted_names:
            obj = mod
            for part in dotted.split("."):
                obj = getattr(obj, part)
            typing.get_type_hints(obj, globalns={**vars(mod), **shared_ns})
