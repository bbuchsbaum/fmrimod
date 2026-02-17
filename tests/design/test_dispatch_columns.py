"""Regression tests for fmrimod.dispatch column name extraction."""

import pandas as pd

from fmrimod import event_model
from fmrimod.dispatch import columns as dispatch_columns


def test_dispatch_columns_accepts_event_model_column_names_property():
    """dispatch.columns should support EventModel.column_names property access."""
    df = pd.DataFrame(
        {
            "onset": [1, 5, 9, 13],
            "condition": ["A", "B", "A", "B"],
            "duration": [1.0, 1.0, 1.0, 1.0],
        }
    )
    model = event_model("condition", data=df, tr=2.0, n_scans=20)

    cols = dispatch_columns(model)

    assert cols == model.column_names
    assert any("A" in c for c in cols)
    assert any("B" in c for c in cols)
