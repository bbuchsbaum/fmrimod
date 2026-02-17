"""Regression tests for fmrimod.dispatch design-matrix extraction."""

import numpy as np
import pandas as pd

from fmrimod import event_model
from fmrimod.dispatch import design_matrix as dispatch_design_matrix


def test_dispatch_design_matrix_accepts_event_model_property():
    """dispatch.design_matrix should work with EventModel.design_matrix property."""
    df = pd.DataFrame(
        {
            "onset": [1, 5, 9, 13],
            "condition": ["A", "B", "A", "B"],
            "duration": [1.0, 1.0, 1.0, 1.0],
        }
    )
    model = event_model("condition", data=df, tr=2.0, n_scans=20)

    X = dispatch_design_matrix(model)

    assert X.shape == model.design_matrix.shape


def test_dispatch_design_matrix_accepts_ndarray():
    """dispatch.design_matrix should return ndarray inputs unchanged."""
    X = np.arange(12, dtype=float).reshape(4, 3)
    out = dispatch_design_matrix(X)
    assert out is X


def test_dispatch_design_matrix_accepts_dataframe():
    """dispatch.design_matrix should convert DataFrame to ndarray."""
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    out = dispatch_design_matrix(df)
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, df.to_numpy())
