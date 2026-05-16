"""Cheap-pass-disqualifying tests for the typed demo-dataset helper.

The helper (bd-01KRSFY2XK499A7P15QBEPXFCS) exists to replace the disowned
``design_colmap(built).query("condition == 'term.face'")`` rendered-name
lookup. These tests prove the injected effect lands on exactly the column
the *same typed semantic object the lesson uses* resolves to — not a string
match — so reverting to the string pattern would fail here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import fmrimod as fm
from fmrimod.contrast.semantic import SemanticContrast, condition
from fmrimod.simulate.demo import dataset_with_effect


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "onset": np.arange(16) * 8.0,
            "condition": (["face", "scene"] * 8),
            "duration": 2.0,
        }
    )


def _spec() -> object:
    from fmrimod.spec import drift, hrf

    return hrf("condition", basis="spmg1") + drift("poly", degree=2)


def test_injection_column_is_the_typed_semantic_resolution() -> None:
    """The injected column == the column SemanticContrast.resolve picks.

    This is the lesson's exact resolution path (fit.contrast resolves a
    SemanticContrast against fit.design_columns()). Asserting the injected
    effect appears on that resolved column — and nowhere else — disqualifies
    any string/rendered-name shortcut.
    """
    sf = fm.SamplingFrame(blocklens=130, TR=1.0)
    spec = _spec()
    on = condition("face", term="condition")
    ds = dataset_with_effect(
        spec, _events(), on=on, effect=2.0, sampling_frame=sf,
        n_signal=5, n_voxels=10, noise=0.3, seed=202,
    )
    fit = fm.fmri_lm(spec, ds)

    weights = SemanticContrast(positive=on).resolve(fit.design_columns())
    resolved_idx = int(np.flatnonzero(weights)[0])

    betas = np.asarray(fit.betas)
    signal = betas[resolved_idx, :5].mean()
    null = betas[resolved_idx, 5:].mean()
    assert signal == pytest.approx(2.0, abs=0.15), signal
    assert abs(null) < 0.15, null


def test_known_effect_recovered_by_the_lesson_contrast() -> None:
    """The typed face-minus-scene contrast recovers the injected effect."""
    sf = fm.SamplingFrame(blocklens=130, TR=1.0)
    spec = _spec()
    ds = dataset_with_effect(
        spec, _events(), on=condition("face", term="condition"),
        effect=2.0, sampling_frame=sf, n_signal=5, n_voxels=10,
        noise=0.3, seed=202,
    )
    fit = fm.fmri_lm(spec, ds)
    con = fit.contrast(
        condition("face", term="condition")
        - condition("scene", term="condition"),
        name="face_minus_scene",
    )
    est = np.asarray(con.estimate).ravel()
    assert est[:5].mean() == pytest.approx(2.0, abs=0.2), est[:5].mean()
    assert abs(est[5:].mean()) < 0.2, est[5:].mean()


def test_unresolvable_level_raises_not_silent() -> None:
    """A level the typed provenance cannot resolve must raise, not no-op.

    A rendered-name string query would mismatch differently; the point is
    the helper resolves through typed provenance and fails loudly.
    """
    sf = fm.SamplingFrame(blocklens=130, TR=1.0)
    with pytest.raises((ValueError, Exception)):
        dataset_with_effect(
            _spec(), _events(),
            on=condition("does_not_exist", term="condition"),
            effect=1.0, sampling_frame=sf,
        )


def test_requires_typed_condition_not_raw_string() -> None:
    """`on=` rejects a raw string, so the helper cannot degrade to the
    rendered-name string-keyed pattern it exists to replace."""
    sf = fm.SamplingFrame(blocklens=130, TR=1.0)
    with pytest.raises(TypeError):
        dataset_with_effect(
            _spec(), _events(), on="face", effect=1.0, sampling_frame=sf,
        )


def test_helper_is_not_publicly_exported() -> None:
    """S1 scope: importable for tutorials, but not promoted to the public
    fmrimod.simulate surface (steward gate before public promotion)."""
    import fmrimod.simulate as simulate

    assert "dataset_with_effect" not in getattr(simulate, "__all__", [])
    assert not hasattr(simulate, "dataset_with_effect")
