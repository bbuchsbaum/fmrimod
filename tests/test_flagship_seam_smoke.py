"""First-ten-minutes smoke test for the public flagship seam."""

from __future__ import annotations

import ast
import time
from pathlib import Path

import numpy as np
import pandas as pd

import fmrimod as fm


THIS_FILE = Path(__file__)


def test_public_flagship_seam_imports_only_top_level_api() -> None:
    tree = ast.parse(THIS_FILE.read_text())
    forbidden_imports: list[str] = []
    fm_attrs: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "fmrimod"
        ):
            forbidden_imports.append(node.module or "")
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "fm"
        ):
            fm_attrs.add(node.attr)

    assert forbidden_imports == []
    checked_attrs = fm_attrs - {"__all__"}
    assert checked_attrs <= set(fm.__all__)
    assert not any(name.startswith("_") for name in checked_attrs)


def test_first_ten_minutes_public_flagship_seam_survives_intent_payload() -> None:
    start = time.perf_counter()
    rng = np.random.default_rng(23)
    n_scans = 60
    img = rng.standard_normal((2, 2, 2, n_scans))
    events = pd.DataFrame(
        {
            "onset": [6.0, 18.0, 30.0, 42.0],
            "duration": [2.0, 2.0, 2.0, 2.0],
            "trial_type": ["a", "b", "a", "b"],
        }
    )

    dataset = fm.fmri_dataset(img, tr=2.0, events=events)
    fit = fm.fmri_lm("hrf(trial_type)", dataset)
    result = fit.contrast(
        fm.column_contrast(
            pattern_A=r"^trial_type_trial_type\.a$",
            name="trial_type_a",
        )
    )

    assert result.intent is not None
    assert result.intent.kind == "contrast_spec"
    assert result.intent.basis_label is not None
    assert result.intent.weights is not None
    assert result.intent.design_id is not None
    assert result.intent.design_id.startswith("design:sha256:")
    assert result.intent.provenance_id is not None
    assert result.intent.provenance_id.startswith("fitprov:sha256:")
    assert result.touched_columns == ("trial_type_trial_type.a",)
    assert time.perf_counter() - start < 5.0
