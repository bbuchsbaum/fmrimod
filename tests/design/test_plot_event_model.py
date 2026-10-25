"""Tests for the event-model plot (R fmridesign ``plot.event_model``)."""

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
from fmrimod.visualization import plot_design_matrix, plot_event_model
from fmrimod.visualization._design_common import design_columns, structured_colours
from fmrimod.visualization.event_plot import _peak_breaks, event_table
from fmrimod.visualization.style import CATEGORICAL, WARN


@pytest.fixture(autouse=True)
def _strict():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield
    plt.close("all")


@pytest.fixture(scope="module")
def events():
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "onset": np.concatenate([np.sort(rng.uniform(2, 170, 12)) for _ in range(2)]),
        "cond": np.tile(["face", "house", "scene"], 8),
        "load": np.tile(["low", "low", "high", "high"], 6),
        "rt": rng.normal(1, 0.2, 24),
        "run": np.repeat([1, 2], 12),
    })


@pytest.fixture(scope="module")
def sframe():
    return fm.SamplingFrame(blocklens=[90, 90], TR=2)


@pytest.fixture(scope="module")
def em(events, sframe):
    return fm.event_model("hrf(cond) + hrf(cond, rt)", data=events, block="run", sampling_frame=sframe)


def _row_axes(fig):
    return [a for a in fig.axes if a.get_subplotspec() is not None]


def _figtexts(fig):
    return [t.get_text() for t in fig.texts]


def test_stacked_rows_labels_and_header(em):
    fig = plot_event_model(em)
    assert isinstance(fig, Figure)
    axes = _row_axes(fig)
    ci = design_columns(em)
    assert len(axes) == 6
    assert [a.get_ylabel() for a in axes] == list(ci["label"])
    texts = _figtexts(fig)
    assert texts[0] == "Event regressors"
    assert texts[1] == "6 regressors · 2 runs · 180 scans (TR 2 s) · 24 events"
    assert texts[2].startswith("Rows of the same term share a scale")
    # Top run labels on the first row only.
    top = list(axes[0].child_axes)
    assert [t.get_text() for t in top[0].get_xticklabels()] == ["Run 1", "Run 2"]
    assert not any(a.child_axes for a in axes[1:])


def test_stacked_lines_never_cross_runs_and_use_palette(em):
    fig = plot_event_model(em, show_events=False)
    palette = {c.upper() for c in CATEGORICAL}
    for ax in _row_axes(fig):
        data_lines = [ln for ln in ax.lines if len(ln.get_xdata()) > 2]
        assert len(data_lines) == 2  # one per run
        for ln in data_lines:
            x = np.asarray(ln.get_xdata())
            assert x.min() >= 0 and (x.max() <= 180 or x.min() >= 180)
            assert to_hex(ln.get_color()).upper() in palette


def test_rugs_match_event_onsets(em, events, sframe):
    fig = plot_event_model(em)
    first = _row_axes(fig)[0]
    ticks = [ln for ln in first.lines if list(ln.get_ydata()) == [0.0, 0.16]]
    face = events[events["cond"] == "face"]
    expected = np.sort(face["onset"].to_numpy() + np.where(face["run"] == 2, 180.0, 0.0))
    got = np.sort([ln.get_xdata()[0] for ln in ticks])
    np.testing.assert_allclose(got, expected)


def test_event_table_global_onsets(em, events, sframe):
    from fmrimod.visualization._design_common import timeline

    ci = design_columns(em)
    tl = timeline(sframe, 180)
    ev = event_table(em, ci, list(ci["label"]), tl)
    assert ev["event"].nunique() == 24
    house = ev[ev["row"] == ci["label"].iloc[1]]
    exp = events[events["cond"] == "house"]
    np.testing.assert_allclose(np.sort(house["onset"]), np.sort(exp["onset"] + 180.0 * (exp["run"] == 2)))


def test_y_scale_modes(em):
    term = [a.get_ylim() for a in _row_axes(plot_event_model(em, y_scale="term"))]
    assert term[0] == term[1] == term[2] and term[3] == term[4] == term[5]
    shared = [a.get_ylim() for a in _row_axes(plot_event_model(em, y_scale="shared"))]
    assert len(set(shared)) == 1
    row = [a.get_ylim() for a in _row_axes(plot_event_model(em, y_scale="row"))]
    assert len(set(row)) > 2


def test_peak_breaks_round_towards_zero():
    assert _peak_breaks(0.0, 0.2234, 0.3) == [0.0, 0.22]
    assert _peak_breaks(-0.021, 0.2234, 0.3) == [0.0, 0.22]  # trough too close to 0
    b = _peak_breaks(-0.0968, 0.0834, 0.3)
    assert b == [-0.096, 0.0, 0.083]


def test_factorial_structured_colours(events, sframe):
    em2 = fm.event_model("hrf(cond, load)", data=events, block="run", sampling_frame=sframe)
    ci = design_columns(em2)
    cols = structured_colours(ci, list(ci["label"]))
    assert len(cols) == 6
    # Same hue family for a condition's two load levels, distinct colours.
    assert cols["face × high"] != cols["face × low"]
    assert len(set(cols.values())) == 6
    fig = plot_event_model(em2)
    line_cols = {to_hex(ln.get_color()).upper() for a in _row_axes(fig) for ln in a.lines if len(ln.get_xdata()) > 2}
    assert line_cols == {c.upper() for c in cols.values()}


def test_overlay_one_axes_with_legend(em):
    fig = plot_event_model(em, style="overlay")
    ax = _row_axes(fig)[0]
    assert len(_row_axes(fig)) == 1
    assert len([ln for ln in ax.lines if len(ln.get_xdata()) > 2]) == 12  # 6 regressors x 2 runs
    assert [t.get_text() for t in ax.get_legend().get_texts()] == list(design_columns(em)["full_label"])
    assert ax.get_ylabel() == "Regressor amplitude (a.u.)"


def test_overlay_into_given_axes(em):
    fig, ax = plt.subplots()
    out = plot_event_model(em, style="overlay", ax=ax, title="Mine")
    assert out is fig and ax.get_title(loc="left") == "Mine"
    with pytest.raises(ValueError, match="stacked"):
        plot_event_model(em, style="stacked", ax=ax)


def test_heatmap_max_r_matches_corrcoef(em):
    fig = plot_event_model(em, style="heatmap")
    ax = _row_axes(fig)[0]
    X = em.design_matrix
    R = np.corrcoef(X.T)
    np.fill_diagonal(R, np.nan)
    expected = sorted(f"{v:.2f}".lstrip("0") for v in np.nanmax(np.abs(R), axis=0))
    vals = [t for t in ax.texts if t.get_text().startswith(".")]
    assert sorted(t.get_text() for t in vals) == expected
    for t in vals:
        flagged = float(t.get_text()) >= 0.5
        assert (to_hex(t.get_color()).upper() == WARN.upper()) == flagged
    assert any(t.get_text() == "max r" for t in ax.texts)
    assert len(ax.collections) >= 1  # the raster (plus onset ticks)


def test_auto_style_picks_heatmap_for_many_rows(events, sframe):
    emt = fm.event_model("trialwise()", data=events, block="run", sampling_frame=sframe)
    fig = plot_event_model(emt)
    axes = _row_axes(fig)
    assert len(axes) == 1
    labels = [t.get_text() for t in axes[0].get_yticklabels()]
    assert labels[0] == "trial 1" and len(labels) == 24


def test_term_selection_and_errors(em):
    fig = plot_event_model(em, term="cond")
    assert len(_row_axes(fig)) == 3
    with pytest.raises(ValueError, match="No columns found"):
        plot_event_model(em, term="nope")
    with pytest.raises(ValueError, match="style"):
        plot_event_model(em, style="bars")
    with pytest.raises(ValueError, match="time_range"):
        plot_event_model(em, time_range=(10, 0))


def test_block_x_run_draws_one_column_per_run(em):
    fig = plot_event_model(em, block_x="run")
    axes = _row_axes(fig)
    assert len(axes) == 12
    assert axes[0].get_title() == "Run 1" and axes[1].get_title() == "Run 2"
    assert axes[-1].get_xlim() == pytest.approx((0, 180))


def test_time_range_and_text_overrides(em):
    fig = plot_event_model(em, time_range=(0, 60), title="", subtitle="Zoom", caption="")
    assert _row_axes(fig)[-1].get_xlim() == (0, 60)
    assert _figtexts(fig) == ["Zoom"]


def test_label_mode_none_and_compact(em):
    fig = plot_event_model(em, label_mode="none")
    assert all(a.get_ylabel() == "" for a in _row_axes(fig))
    fig = plot_event_model(em, label_mode="compact", abbrev_min=6)
    assert all(len(a.get_ylabel()) <= 8 for a in _row_axes(fig))


def test_durations_drawn_as_bars(events, sframe):
    ev = events.assign(dur=np.where(np.arange(24) % 4 == 0, 6.0, 0.0))
    em = fm.event_model("hrf(cond)", data=ev, block="run", sampling_frame=sframe, durations="dur")
    fig = plot_event_model(em)
    assert "bars show event durations" in _figtexts(fig)[-1]
    bars = [ln for a in _row_axes(fig) for ln in a.lines if ln.get_linewidth() > 3]
    assert len(bars) == 6


def test_method_and_alias(em):
    assert isinstance(em.plot(), Figure)
    fig = plot_design_matrix(em, term_name="cond", separate_regressors=False)
    assert len(_row_axes(fig)) == 1


def test_spmg3_rows_per_basis(events, sframe):
    em3 = fm.event_model("hrf(cond, basis='spmg3')", data=events, block="run", sampling_frame=sframe)
    fig = plot_event_model(em3)
    labels = [a.get_ylabel() for a in _row_axes(fig)]
    assert labels[:3] == ["face · canonical", "temporal deriv.", "dispersion deriv."]
