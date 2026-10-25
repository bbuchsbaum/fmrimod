"""Tests for plot_contrasts / plot_Fcontrasts (R fmridesign ``plot_contrasts``)."""

import subprocess
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex
from matplotlib.figure import Figure

import fmrimod as fm
from fmrimod.contrast import (
    Fcontrasts,
    contrast_set,
    pair_contrast,
    plot_contrasts,
    plot_Fcontrasts,
    unit_contrast,
)
from fmrimod.formula.base import Term
from fmrimod.visualization.design_stats import contrast_se
from fmrimod.visualization.style import INK, WARN


@pytest.fixture(autouse=True)
def _strict():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield
    plt.close("all")


@pytest.fixture(scope="module")
def events():
    rng = np.random.default_rng(2)
    return pd.DataFrame({
        "onset": np.concatenate([np.sort(rng.uniform(2, 170, 12)) for _ in range(2)]),
        "cond": np.tile(["face", "house", "scene"], 8),
        "load": np.tile(["low", "low", "high", "high"], 6),
        "run": np.repeat([1, 2], 12),
    })


@pytest.fixture(scope="module")
def sframe():
    return fm.SamplingFrame(blocklens=[90, 90], TR=2)


@pytest.fixture(scope="module")
def emc(events, sframe):
    t = Term("cond")
    t.contrast_specs = contrast_set(
        pair_contrast("cond == 'face'", "cond == 'house'", name="face_vs_house"),
        unit_contrast("cond == 'scene'", name="scene"),
    )
    return fm.event_model([t], data=events, block="run", sampling_frame=sframe)


def _texts(ax):
    return [t.get_text() for t in ax.texts]


# -- contrast_se --------------------------------------------------------


def test_contrast_se_matches_explicit_inverse():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    run = np.repeat([1, 2], 20)
    W = np.array([[1.0, -1.0, 0.0], [0.0, 0.0, 1.0]]).T
    Xa = np.column_stack([X, run == 1, run == 2]).astype(float)
    inv = np.linalg.inv(Xa.T @ Xa)
    expected = [np.sqrt(np.r_[w, 0, 0] @ inv @ np.r_[w, 0, 0]) for w in W.T]
    np.testing.assert_allclose(contrast_se(X, run, W), expected, rtol=1e-10)


def test_contrast_se_not_estimable_and_zero():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(30, 2))
    X = np.column_stack([A, A.sum(axis=1)])  # third column aliased
    W = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 0.0, 0.0]]).T
    se = contrast_se(X, None, W)
    assert np.isnan(se[0])  # the aliased column alone is not estimable
    assert np.isfinite(se[1])  # orthogonal to the null vector (1, 1, -1): estimable
    assert se[2] == 0.0


# -- EventModel ---------------------------------------------------------


def test_event_model_rows_weights_and_margins(emc):
    fig = plot_contrasts(emc)
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["face_vs_house", "scene"]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["face", "house", "scene"]
    texts = _texts(ax)
    assert texts[:6] == ["+1", "−1", "0", "0", "0", "+1"]
    zero = [t for t in ax.texts if t.get_text() == "0" and t.get_transform() == ax.transData]
    assert len(zero) == 3
    assert all(to_hex(t.get_color()).upper() == INK["zero"].upper() for t in zero)
    assert "sum w" in texts and "SE ×σ" in texts
    X = emc.design_matrix
    run = np.repeat([1, 2], 90)
    se = contrast_se(X, run, np.array([[1, -1, 0], [0, 0, 1]], dtype=float).T)
    assert f"{se[0]:.2g}" in texts and f"{se[1]:.2g}" in texts
    # The unbalanced unit contrast sums to 1, drawn in the warning colour.
    sum_one = [t for t in ax.texts if t.get_text() == "1"]
    assert sum_one and to_hex(sum_one[0].get_color()).upper() == WARN.upper()
    sub = fig.texts[1].get_text()
    assert sub == "2 contrasts over 3 regressors · 1 of 2 sum to zero"
    assert fig.texts[-1].get_text().startswith("SE ×σ: standard error")


def test_not_estimable_flagged(events, sframe):
    t1 = Term("cond")
    t1.contrast_specs = contrast_set(unit_contrast("cond == 'face'", name="face"))
    t2 = Term("load")
    t2.contrast_specs = contrast_set(pair_contrast("load == 'high'", "load == 'low'", name="hi_lo"))
    em = fm.event_model([t1, t2], data=events, block="run", sampling_frame=sframe)
    fig = plot_contrasts(em)
    ax = fig.axes[0]
    ne = [t for t in ax.texts if t.get_text() == "not est."]
    assert len(ne) == 1 and to_hex(ne[0].get_color()).upper() == WARN.upper()
    assert "1 not estimable" in fig.texts[1].get_text()
    assert {"cond", "load"} <= set(_texts(ax))  # term spanners


# -- dict ---------------------------------------------------------------


def test_dict_f_components_and_labels():
    d = {"main_effect": np.array([[1, 0], [-1, 1], [0, -1], [0, 0]]), "simple": np.array([1, 1, -1, -1])}
    fig = plot_contrasts(d, labels=["a", "b", "c", "d"])
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["main_effect   1", "2", "simple"]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["a", "b", "c", "d"]
    rules = [ln for ln in ax.lines if ln.get_color() == INK["muted"]]
    assert len(rules) == 1  # between the F-contrast and the next contrast
    assert "SE ×σ" not in _texts(ax)  # no design given
    assert "1 F-contrast" in fig.texts[1].get_text()


def test_dict_with_design_adds_se():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(30, 2))
    fig = plot_contrasts({"diff": np.array([1.0, -1.0])}, design=X)
    se = contrast_se(X, None, np.array([1.0, -1.0]))
    assert f"{se[0]:.2g}" in _texts(fig.axes[0])


def test_scales_and_limits():
    d = {"pos": np.array([2.0, 1.0, 0.0])}
    fig = plot_contrasts(d, scale_mode="one_sided")
    mesh = fig.axes[0].collections[0]
    assert mesh.norm.vmin == 0 and mesh.norm.vmax == 2
    fig = plot_contrasts(d, absolute_limits=True)
    mesh = fig.axes[0].collections[0]
    assert (mesh.norm.vmin, mesh.norm.vmax) == (-1, 1)
    np.testing.assert_array_equal(mesh.get_array().reshape(1, 3), [[2.0, 1.0, 0.0]])


def test_dict_errors_and_overrides():
    with pytest.raises(ValueError, match="No contrasts provided"):
        plot_contrasts({})
    with pytest.raises(ValueError, match="expected 4"):
        plot_contrasts({"a": np.array([1, -1, 0, 0]), "b": np.array([1, -1, 0])})
    with pytest.raises(ValueError, match="scale_mode"):
        plot_contrasts({"a": np.array([1, -1])}, scale_mode="log")
    fig = plot_contrasts({"a": np.array([1, -1])}, figsize=(6, 3), title="", subtitle="Mine")
    assert tuple(fig.get_size_inches()) == (6, 3)
    assert [t.get_text() for t in fig.texts] == ["Mine"]


def test_rotation_follows_rotate_x_text():
    d = {"t": np.r_[1.0, -1.0, np.zeros(18)]}
    labels = [f"long_regressor_name_{i}" for i in range(20)]
    rot = plot_contrasts(d, labels=labels).axes[0].get_xticklabels()[0].get_rotation()
    flat = plot_contrasts(d, labels=labels, rotate_x_text=False).axes[0].get_xticklabels()[0].get_rotation()
    assert rot == 90 and flat == 0


# -- plot_Fcontrasts ----------------------------------------------------


def test_plot_fcontrasts_event_model(events, sframe):
    em = fm.event_model("hrf(cond, load)", data=events, block="run", sampling_frame=sframe)
    fig = plot_Fcontrasts(em)
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    fc = Fcontrasts(em)
    n_rows = sum(np.asarray(v).shape[1] for v in fc.values())
    assert len(ax.get_yticklabels()) == n_rows
    assert fig.texts[0].get_text() == "F-contrast weights"
    assert "SE ×σ" in _texts(ax)


def test_plot_fcontrasts_dict_returns_figure():
    fig = plot_Fcontrasts({"a": np.eye(3)[:, :2], "b": np.array([[1], [-1], [0]])})
    assert isinstance(fig, Figure)
    assert [t.get_text() for t in fig.axes[0].get_yticklabels()] == ["a   1", "2", "b"]
    with pytest.raises(ValueError, match="No F-contrasts"):
        plot_Fcontrasts({})


def test_contrast_module_import_is_matplotlib_free():
    code = ("import sys; import fmrimod.contrast.plot_contrasts, fmrimod.contrast.fcontrast; "
            "print('matplotlib.pyplot' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
