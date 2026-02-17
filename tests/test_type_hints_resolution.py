"""Regression tests for forward annotation resolution."""

import importlib
import typing


def test_forward_type_hints_resolve_for_recent_f821_hotspots():
    importlib.import_module("fmrimod")
    modules = {
        "fmrimod.events.variable": ["EventVariable.bin_values"],
        "fmrimod.events.matrix": ["EventMatrix.split_columns", "EventMatrix.apply_transform"],
        "fmrimod.events.basis": ["EventBasis.to_matrix"],
        "fmrimod.formula.parser": ["parse_formula"],
        "fmrimod.formula.dsl": [
            "HRFNamespace.__call__",
            "BasisNamespace.__call__",
            "HRFTransform.__init__",
            "BasisTransform.__init__",
        ],
        "fmrimod.formula.base": ["EventModelBuilder.build"],
        "fmrimod.formula.functional": ["hrf", "basis"],
        "fmrimod.utils.misc": ["single_trial_regressor", "hrf_toeplitz"],
        "fmrimod.ar.estimation": ["fit_noise"],
        "fmrimod.ar.whitening": ["whiten_apply", "whiten"],
        "fmrimod.regressor.core": ["Regressor.evaluate", "RegressorSet.evaluate"],
        "fmrimod.regressor.neural_input": ["neural_input"],
        "fmrimod.contrast.plot_contrasts": ["plot_contrasts_event_model"],
    }
    if importlib.util.find_spec("matplotlib") is not None:
        modules["fmrimod.plotting"] = [
            "plot_hrf",
            "plot_hrfs",
            "plot_regressor",
            "plot_regressors",
        ]

    for module_name, dotted_names in modules.items():
        mod = importlib.import_module(module_name)
        for dotted in dotted_names:
            obj = mod
            for part in dotted.split("."):
                obj = getattr(obj, part)
            typing.get_type_hints(obj)


def test_type_hints_for_public_api_resolve_without_custom_globalns():
    """Regression: these APIs should resolve without shared globalns injection."""
    cases = [
        ("fmrimod.events.variable", ["EventVariable.bin_values"]),
        ("fmrimod.events.matrix", ["EventMatrix.split_columns"]),
        ("fmrimod.events.basis", ["EventBasis.to_matrix"]),
        ("fmrimod.formula.parser", ["parse_formula"]),
        ("fmrimod.formula.dsl", ["HRFNamespace.__call__", "BasisNamespace.__call__"]),
    ]

    for module_name, dotted_names in cases:
        mod = importlib.import_module(module_name)
        for dotted in dotted_names:
            obj = mod
            for part in dotted.split("."):
                obj = getattr(obj, part)
            typing.get_type_hints(obj)
