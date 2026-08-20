"""Degenerate parametric-modulator warnings (fmridesign #8)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fmrimod.design.event_model import event_model
from fmrimod.sampling import SamplingFrame


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "onset": np.linspace(0, 38, 20),
            "block": 1,
            "cond": ["A"] * 20,
            "modv": np.arange(1, 21, dtype=float),
            "modc": np.full(20, 2.0),
            "modz": np.zeros(20),
            "modna": np.full(20, np.nan),
        }
    )


def test_event_model_warns_for_degenerate_parametric_modulators():
    ev = _events()
    sframe = SamplingFrame(blocklens=[100], TR=2)

    event_model(
        "hrf(cond) + hrf(modv)",
        data=ev,
        sampling_frame=sframe,
    )

    with pytest.warns(UserWarning, match="zero variance"):
        event_model(
            "hrf(cond) + hrf(modc)",
            data=ev,
            sampling_frame=sframe,
        )

    with pytest.warns(UserWarning, match="all zero"):
        event_model(
            "hrf(cond) + hrf(modz)",
            data=ev,
            sampling_frame=sframe,
        )

    with pytest.warns(UserWarning, match="NA values detected"):
        event_model(
            "hrf(cond) + hrf(modna)",
            data=ev,
            sampling_frame=sframe,
        )
