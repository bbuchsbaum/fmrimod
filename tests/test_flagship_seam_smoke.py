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
    n_scans = 24
    tr = 1.0
    events = pd.DataFrame(
        {
            "onset": [2.0, 8.0, 14.0],
            "duration": [0.0, 0.0, 0.0],
            "trial_type": ["go", "go", "go"],
            "run": [1, 1, 1],
        }
    )
    data = np.random.default_rng(20260514).normal(
        scale=0.01,
        size=(n_scans, 4),
    )

    dataset = fm.fmri_dataset(data, tr=tr, events=events)
    model = fm.event_model(
        "trial_type",
        data=events,
        block="run",
        tr=tr,
        n_scans=n_scans,
    )
    fit = fm.fmri_lm(model, dataset)
    result = fit.contrast(fm.column_contrast("go", name="go"))

    intent = result.explain().to_dict()["intent"]
    assert intent["kind"] != "default"
    assert intent["basis_label"] is not None
    assert intent["provenance_id"] is not None
    assert intent["weights"]
    assert time.perf_counter() - start < 5.0
