"""Tests for design_map (R fmridesign ``design_map``)."""

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

import fmrimod as fm
from fmrimod.model import FmriModel
from fmrimod.visualization import design_map, plot_model_summary
from fmrimod.visualization.heatmaps import display_columns
from fmrimod.visualization.style import WARN


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
def em(sframe):
    rng = np.random.default_rng(1)
    ev = pd.DataFrame({
        "onset": np.concatenate([np.sort(rng.uniform(2, 170, 12)) for _ in range(2)]),
        "cond": np.tile(["face", "house", "scene"], 8),
        "rt": rng.normal(1, 0.2, 24),
        "run": np.repeat([1, 2], 12),
    })
    return fm.event_model("hrf(cond) + hrf(cond, rt)", data=ev, block="run", sampling_frame=sframe)


def _main_ax(fig):
    return fig.axes[0]


def test_column_scaled_image_and_labels(em):
    fig = design_map(em)
    assert isinstance(fig, Figure)
    ax = _main_ax(fig)
    (im,) = ax.get_images()
    X = em.design_matrix
    np.testing.assert_allclose(np.asarray(im.get_array()), X / np.abs(X).max(axis=0))
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels[0].split("\n")[0] == "cond: face"
    lo, hi = X[:, 0].min(), X[:, 0].max()
    assert labels[0].split("\n")[1].endswith(f"to {hi:.2g}")
    assert lo < 0  # the undershoot exists but is shown as part of the range text
    assert [t.get_text() for t in ax.get_yticklabels()] == ["Run 1", "Run 2"]
    texts = [t.get_text() for t in fig.texts]
    assert texts[:2] == ["Design matrix", "6 regressors · 2 runs · 180 scans (TR 2 s)"]


def test_run_separator_and_scan_axis(em):
    ax = _main_ax(design_map(em))
    seps = [ln for ln in ax.lines if list(ln.get_ydata()) == [90.5, 90.5]]
    assert len(seps) == 1
    sec = [c for c in ax.child_axes if c.get_yticks().size]
    assert [int(v) for v in sec[0].get_yticks()] == [1, 91, 180]
    assert not [ln for ln in _main_ax(design_map(em, block_separators=False)).lines
                if list(ln.get_ydata()) == [90.5, 90.5]]


def test_term_spanners(em):
    ax = _main_ax(design_map(em))
    assert {"cond", "cond × rt"} <= {t.get_text() for t in ax.texts}


def test_magnitude_strip_relative_to_term(em):
    ax = _main_ax(design_map(em))
    X = em.design_matrix
    mx = np.abs(X).max(axis=0)
    rel = np.concatenate([mx[:3] / mx[:3].max(), mx[3:] / mx[3:].max()])
    bars = [p for p in ax.patches if isinstance(p, Rectangle) and p.get_y() > 180 and p.get_height() > 0
            and to_hex(p.get_facecolor()).upper() != WARN.upper()]
    heights = np.array([b.get_height() for b in sorted(bars, key=lambda b: b.get_x())])
    np.testing.assert_allclose(heights / heights.max(), rel / rel.max(), rtol=1e-9)


def test_near_flat_columns_flagged(sframe):
    rng = np.random.default_rng(0)
    nuis = [pd.DataFrame(rng.normal(size=(90, 2)) * [1.0, 1e-3], columns=["big", "tiny"]) for _ in range(2)]
    bm = fm.baseline_model(basis="poly", degree=2, sframe=sframe, nuisance_list=nuis)
    fig = design_map(bm)
    assert "near-flat: tiny · run 1, tiny · run 2" in fig.texts[1].get_text()
    outlines = [p for p in _main_ax(fig).patches if to_hex(p.get_edgecolor()).upper() == WARN.upper()
                and p.get_facecolor()[3] == 0]
    assert len(outlines) == 2


def test_baseline_grouped_by_term_then_run(sframe):
    bm = fm.baseline_model(basis="bs", degree=4, sframe=sframe)
    ci = display_columns(bm)
    assert list(ci["term"]) == ["drift"] * 8 + ["block"] * 2
    fig = design_map(bm)
    ax = _main_ax(fig)
    assert fig.texts[0].get_text() == "Baseline design matrix"
    texts = [t.get_text() for t in ax.texts]
    assert {"drift", "intercept"} <= set(texts) and texts.count("Run 1") == 1
    labels = [t.get_text().split("\n")[0] for t in ax.get_xticklabels()]
    assert labels[:4] == ["drift 1", "drift 2", "drift 3", "drift 4"]
    assert labels[-2:] == ["Run 1", "Run 2"]


def test_fmri_model_combines_event_and_baseline(em, sframe):
    ds = fm.fmri_dataset(np.zeros((180, 2)), tr=2.0, run_length=[90, 90])
    fmod = FmriModel(em, fm.baseline_model(basis="poly", degree=3, sframe=sframe), ds)
    ax = _main_ax(design_map(fmod))
    (im,) = ax.get_images()
    assert im.get_array().shape == (180, 6 + 6 + 2)
    assert [ln for ln in ax.lines if list(ln.get_ydata()) == [90.5, 90.5]]
    # More than 12 columns: no range line under the labels.
    assert all("\n" not in t.get_text() for t in ax.get_xticklabels())


def test_palette_and_time_axis(em):
    ax = _main_ax(design_map(em, palette="grey", y_axis="time", scale="none"))
    sec = [c for c in ax.child_axes if c.get_yticks().size][0]
    assert [t.get_text() for t in sec.get_yticklabels()] == ["0:00", "3:00", "5:58"]
    assert sec.get_ylabel() == "Time (min:s)"
    (im,) = ax.get_images()
    np.testing.assert_allclose(np.asarray(im.get_array()), em.design_matrix)


def test_errors_and_given_axes(em):
    with pytest.raises(ValueError, match="scale"):
        design_map(em, scale="rank")
    fig, ax = plt.subplots()
    assert design_map(em, ax=ax, title="DM") is fig
    assert ax.get_title(loc="left") == "DM"


def test_model_summary_returns_figures(em):
    out = plot_model_summary(em)
    assert set(out) == {"design_map", "correlation_map", "time_series"}
    assert all(isinstance(f, Figure) for f in out.values())
