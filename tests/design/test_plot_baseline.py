"""Tests for plot_baseline_model and plot_sampling_frame (R fmridesign plot_baseline.R)."""

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
from fmrimod.visualization import plot_baseline_model, plot_sampling_frame
from fmrimod.visualization.baseline_plot import drift_cutoff_note, motion_kind
from fmrimod.visualization.style import CATEGORICAL


@pytest.fixture(autouse=True)
def _strict():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield
    plt.close("all")


@pytest.fixture(scope="module")
def sframe():
    return fm.SamplingFrame(blocklens=[90, 90], TR=2)


@pytest.fixture(scope="module")
def motion_model(sframe):
    rng = np.random.default_rng(0)
    nuis = [pd.DataFrame(np.cumsum(rng.normal(size=(90, 6)) * 0.05, 0),
                         columns=["tx", "ty", "tz", "rx", "ry", "rz"]) for _ in range(2)]
    return fm.baseline_model(basis="poly", degree=3, sframe=sframe, nuisance_list=nuis)


def _lane_axes(fig):
    return [a for a in fig.axes if a.get_subplotspec() is not None]


def _data_lines(ax):
    return [ln for ln in ax.lines if len(ln.get_xdata()) > 2]


def test_motion_kind():
    assert [motion_kind(s) for s in ["tx", "trans_y", "x", "rot_z", "rx", "pitch", "csf"]] == [
        "translation", "translation", "translation", "rotation", "rotation", "rotation", None]


def test_cutoff_note_numbers():
    note = drift_cutoff_note("bs", np.array([4, 4]), np.array([180.0, 180.0]))
    assert "4 B-spline columns (degree-4 polynomial, no interior knots) per 180 s run" in note
    assert "cutoff ~90 s" in note  # 2 * 180 / 4
    assert drift_cutoff_note("cosine", np.array([4]), np.array([180.0])) is None


def test_drift_lane_keys_runs_and_caption(sframe):
    bm = fm.baseline_model(basis="bs", degree=4, sframe=sframe)
    fig = plot_baseline_model(bm)
    assert isinstance(fig, Figure)
    (ax,) = _lane_axes(fig)
    assert ax.get_title(loc="left") == "Drift"
    lines = _data_lines(ax)
    assert len(lines) == 8  # 4 basis functions x 2 runs, each within its own run
    for ln in lines:
        x = np.asarray(ln.get_xdata())
        assert x.max() <= 180 or x.min() >= 180
    assert {to_hex(ln.get_color()).upper() for ln in lines} == {c.upper() for c in CATEGORICAL[:4]}
    assert [t.get_text() for t in ax.texts] == ["drift 1", "drift 2", "drift 3", "drift 4"]
    texts = [t.get_text() for t in fig.texts]
    assert texts[0] == "Baseline model"
    assert texts[1] == "Drift: B-spline, 4 per run · 2 runs · 180 scans (TR 2 s)"
    assert "cutoff ~90 s" in texts[2]
    assert "Not shown (constant within each run): run intercepts (2 columns)." in texts[2]
    top = ax.child_axes[0]
    assert [t.get_text() for t in top.get_xticklabels()] == ["Run 1", "Run 2"]


def test_motion_split_into_translation_and_rotation(motion_model):
    fig = plot_baseline_model(motion_model)
    titles = [a.get_title(loc="left") for a in _lane_axes(fig)]
    assert titles == ["Drift", "Translation", "Rotation"]
    trans = _lane_axes(fig)[1]
    assert [t.get_text() for t in trans.texts] == ["tx", "ty", "tz"]
    assert "Translation and rotation are plotted in the units supplied." in fig.texts[-1].get_text()
    assert "6 nuisance per run" in fig.texts[1].get_text()
    assert all(a.get_ylabel() == "" for a in _lane_axes(fig))  # units differ: no shared y title


def test_term_selection_and_unit_scale(sframe):
    bm = fm.baseline_model(basis="bs", degree=4, sframe=sframe)
    fig = plot_baseline_model(bm, term="block")
    (ax,) = _lane_axes(fig)
    assert ax.get_title(loc="left") == "Intercept"
    assert plot_baseline_model(bm, term_name="dri") is not None  # unique partial match
    with pytest.raises(ValueError, match="not found"):
        plot_baseline_model(bm, term="nuisance")
    fig = plot_baseline_model(bm, drift_scale="unit")
    peaks = [np.abs(ln.get_ydata()).max() for ln in _data_lines(_lane_axes(fig)[0])]
    np.testing.assert_allclose(peaks, 1.0)
    assert "Drift scaled" in fig.texts[-1].get_text()


def test_block_x_run_columns(motion_model):
    fig = plot_baseline_model(motion_model, block_x="run")
    axes = _lane_axes(fig)
    assert len(axes) == 6
    assert axes[0].get_title() == "Run 1" and axes[1].get_title() == "Run 2"
    assert axes[0].get_ylabel() == "Drift"


def test_many_series_use_a_legend():
    sf = fm.SamplingFrame(blocklens=[100], TR=2)
    bm = fm.baseline_model(basis="cosine", degree=9, sframe=sf)
    fig = plot_baseline_model(bm)
    ax = _lane_axes(fig)[0]
    assert ax.get_legend() is not None and len(ax.get_legend().get_texts()) == 9
    assert not ax.texts


def test_baseline_method(sframe):
    bm = fm.baseline_model(basis="poly", degree=2, sframe=sframe)
    assert isinstance(bm.plot(title="B"), Figure)


# -- sampling frame -----------------------------------------------------


def test_timeline_bars_follow_per_run_tr():
    sf = fm.SamplingFrame(blocklens=[2, 2], tr=[1.0, 3.0])
    fig = plot_sampling_frame(sf)
    ax = fig.axes[0]
    np.testing.assert_allclose([p.get_width() for p in ax.patches], [2.0, 6.0])
    assert [t.get_text() for t in ax.get_yticklabels()] == ["Run 1", "Run 2"]
    assert fig.texts[1].get_text() == "2 runs · 4 scans · TR 1–3 s (varies by run) · total 0:08"


def test_timeline_flags_odd_run_and_counts_events(sframe):
    sf = fm.SamplingFrame(blocklens=[100, 100, 80], TR=2)
    fig = plot_sampling_frame(sf)
    bold = [t for t in fig.axes[0].texts if t.get_fontweight() == "bold"]
    assert len(bold) == 1 and bold[0].get_text().startswith("80 scans (−20)")
    ev = pd.DataFrame({"onset": [10.0, 50.0, 400.0], "run": [1, 2, 2]})
    fig = plot_sampling_frame(sf, events=ev)
    cap = fig.texts[-1].get_text()
    assert "2 event onsets" in cap and "1 onset outside" in cap and "No events in run 3" in cap


def test_grid_and_lane_styles(sframe):
    fig = plot_sampling_frame(sframe, style="grid")
    ax = fig.axes[0]
    assert len(ax.patches) == 180
    assert fig.texts[0].get_text() == "Scans per run"
    assert ax.get_xlabel() == "Scan (within run)"
    fig = sframe.plot(style="lane")
    assert {"Run 1", "Run 2"} <= {t.get_text() for t in fig.axes[0].texts}


def test_sampling_frame_errors_and_duck_typing():
    sf = fm.SamplingFrame(blocklens=[30, 20], TR=2)
    with pytest.raises(ValueError, match="style"):
        plot_sampling_frame(sf, style="bars")
    with pytest.raises(ValueError, match="tick_every must be positive"):
        plot_sampling_frame(sf, tick_every=0)

    class TrOnly:
        blocklens = np.array([30, 20])
        tr = np.array([2.0, 2.0])

    assert isinstance(plot_sampling_frame(TrOnly()), Figure)
    fig, ax = plt.subplots()
    assert plot_sampling_frame(sf, ax=ax, title="SF") is fig
