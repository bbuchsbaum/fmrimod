from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


EXAMPLE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "design"
    / "fitlins_style_first_level.py"
)


def _load_example_module():
    spec = importlib.util.spec_from_file_location(
        "fitlins_style_first_level",
        EXAMPLE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_fitlins_style_first_level_example_runs():
    module = _load_example_module()

    result = module.run_example(seed=11)

    assert result.fit.betas.shape[0] == len(result.column_names)
    assert result.fit.betas.shape[1] == 32
    assert set(result.contrast_summary["contrast"]) == {
        "word_gt_pseudoword",
        "task_vs_baseline",
    }
    assert set(result.contrast_summary["test"]) == {"t"}
    assert np.isfinite(result.contrast_summary["max_abs_t"]).all()
    assert "fm.event_model('hrf(trial_type)', data=events, ...)" in set(
        result.comparison["fmrimod"]
    )


def test_fitlins_style_contrast_vector_uses_named_columns():
    module = _load_example_module()

    columns = [
        "trial_type_trial_type.word",
        "trial_type_trial_type.pseudoword",
        "constant",
    ]
    vector = module.contrast_vector(
        columns,
        {
            "trial_type_trial_type.word": 1.0,
            "trial_type_trial_type.pseudoword": -1.0,
        },
    )

    np.testing.assert_array_equal(vector, np.array([1.0, -1.0, 0.0]))
