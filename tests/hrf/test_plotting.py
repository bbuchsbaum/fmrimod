"""HRF and regressor plots (port of R fmrihrf's plot_hrfs / plot_regressors).

Structure (panels, lines, labels, palette colours) and the numbers the
figures draw are checked against independent computations; no pixels.
"""

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from fmrimod.hrf import lag_hrf
from fmrimod.hrf.generators import boxcar_generator
from fmrimod.hrf.library import (
    GAMMA_HRF,
    HRF_FIR,
    HRF_GAUSSIAN,
    HRF_SPMG1,
    HRF_SPMG2,
    HRF_SPMG3,
    SPM_CANONICAL,
)
from fmrimod.plotting import (
    _budget_breaks,
    _make_unique,
    _plot_precision,
    hrf_curves,
    plot_hrf,
    plot_hrfs,
    plot_regressor,
    plot_regressors,
    regressor_curves,
)
from fmrimod.regressor.core import regressor, regressor_set
from fmrimod.visualization.style import CATEGORICAL, INK, palette

pytestmark = pytest.mark.filterwarnings("error")


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _data_lines(ax):
    return [ln for ln in ax.get_lines() if not ln.get_label().startswith("_")]


def _hex(c):
    return to_hex(c).upper()


def _event_bars(ax):
    return [p for p in ax.patches if isinstance(p, Rectangle)]


def _fig_texts(fig):
    return [t.get_text() for t in fig.texts]


# --------------------------------------------------------------------------
# Data builders
# --------------------------------------------------------------------------


class TestHrfCurves:
    def test_long_frame_matches_evaluation(self):
        df = hrf_curves(HRF_SPMG1, HRF_GAUSSIAN)
        assert list(df.columns) == ["time", "series", "response"]
        assert list(df["series"].cat.categories) == ["SPMG1", "gaussian"]
        t = np.arange(0, 24.0 + 1e-9, 0.1)
        sub = df[df["series"] == "gaussian"]
        np.testing.assert_allclose(sub["time"], t)
        np.testing.assert_allclose(sub["response"], HRF_GAUSSIAN(t))

    def test_normalize_divides_each_curve_by_its_absolute_peak(self):
        t = np.linspace(0, 24, 97)
        df = hrf_curves(HRF_SPMG3, time=t, normalize=True)
        raw = HRF_SPMG3(t)
        for j, name in enumerate(df["series"].cat.categories):
            got = df.loc[df["series"] == name, "response"].to_numpy()
            np.testing.assert_allclose(got, raw[:, j] / np.max(np.abs(raw[:, j])))

    def test_basis_set_column_names(self):
        assert list(hrf_curves(HRF_SPMG3)["series"].cat.categories) == [
            "canonical", "temporal deriv.", "dispersion deriv."]
        assert list(hrf_curves(HRF_FIR)["series"].cat.categories) == [f"B{k}" for k in range(1, 13)]

    def test_several_sets_are_prefixed_by_hrf_name(self):
        cats = list(hrf_curves(HRF_SPMG2, HRF_FIR, basis="all")["series"].cat.categories)
        assert cats[:2] == ["SPMG2 · canonical", "SPMG2 · temporal deriv."]
        assert cats[2] == "fir B1"

    def test_basis_first_keeps_one_curve_per_hrf(self):
        df = hrf_curves(HRF_SPMG3, HRF_SPMG1, basis="first")
        assert list(df["series"].cat.categories) == ["SPMG3", "SPMG1"]

    def test_duplicate_names_follow_r(self):
        df = hrf_curves(HRF_SPMG1, HRF_SPMG1)
        assert list(df["series"].cat.categories) == ["SPMG1_1", "SPMG1_2"]

    def test_labels_per_hrf_or_per_curve(self):
        assert list(hrf_curves(HRF_SPMG1, GAMMA_HRF, labels=["A", "B"])["series"].cat.categories) == ["A", "B"]
        cats = hrf_curves(HRF_SPMG2, HRF_SPMG1, labels=["x", "y", "z"])["series"].cat.categories
        assert list(cats) == ["x", "y", "z"]
        with pytest.raises(ValueError, match="labels"):
            hrf_curves(HRF_SPMG1, GAMMA_HRF, labels=["only one", "two", "three", "four"])

    def test_mapping_input_names_the_curves(self):
        df = hrf_curves({"early": HRF_SPMG1, "late": lag_hrf(HRF_SPMG1, 3)})
        assert list(df["series"].cat.categories) == ["early", "late"]

    def test_rejects_non_hrf(self):
        with pytest.raises(TypeError, match="argument 2"):
            hrf_curves(HRF_SPMG1, "spmg1")
        with pytest.raises(ValueError):
            hrf_curves()


def test_make_unique_matches_r():
    # R: make.unique(c("a","a","a 1","a"), sep = " ")
    assert _make_unique(["a", "a", "a 1", "a"]) == ["a", "a 2", "a 1", "a 3"]
    assert _make_unique(["B1", "B1", "B1"]) == ["B1", "B1 1", "B1 2"]


class TestRegressorCurves:
    def test_default_grid_and_values(self):
        reg = regressor(onsets=[10, 30], hrf=HRF_SPMG1)
        df = regressor_curves(reg)
        grid = np.arange(0, 30 + reg.span + 1e-9, 0.25)
        np.testing.assert_allclose(df["time"], grid)
        np.testing.assert_allclose(df["response"], reg.evaluate(grid, precision=0.25))
        assert list(df["series"].cat.categories) == ["Regressor 1"]

    def test_regressor_set_gives_one_curve_per_level(self):
        rs = regressor_set(onsets=[5, 15, 25, 35], fac=["b", "a", "b", "a"])
        df = regressor_curves(rs)
        assert list(df["series"].cat.categories) == ["a", "b"]

    def test_mapping_and_basis(self):
        df = regressor_curves({"A": regressor([10], HRF_SPMG3)})
        assert list(df["series"].cat.categories) == ["canonical", "temporal deriv.", "dispersion deriv."]
        df = regressor_curves({"A": regressor([10], HRF_SPMG3), "B": regressor([20])}, basis="first")
        assert list(df["series"].cat.categories) == ["A", "B"]

    def test_plot_precision_rule(self):
        assert _plot_precision(None, np.arange(0, 10, 0.25)) == 0.25
        assert _plot_precision(None, np.arange(0, 10, 2.0)) == 0.33
        assert _plot_precision(None, np.arange(0, 1, 0.001)) == 0.01
        assert _plot_precision(None, np.array([3.0])) == 0.33
        assert _plot_precision(0.5, np.arange(0, 10, 0.1)) == 0.5


@pytest.mark.parametrize(
    "limits, kw, expected",
    [
        # Values from R: fmrihrf:::.budget_breaks(limits, stack, floor)
        ((-0.31, 0.96), {"floor": 0.13}, [0, 0.25, 0.5, 0.75]),
        ((-0.02, 0.21), {}, [0, 0.1, 0.2]),
        ((-0.12, 0.95), {}, [0, 0.25, 0.5, 0.75]),
        ((-0.5, 0.2), {}, [-0.4, -0.2, 0, 0.2]),
        ((-0.09, 0.19), {}, [-0.05, 0, 0.05, 0.1, 0.15]),
        ((-0.3, 1.0), {"stack": True, "floor": 0.19}, [0, 1]),
        ((-0.06, 0.19), {"stack": True, "floor": 0.19}, [0, 0.1]),
        ((-0.1, 0.1), {"stack": True}, [-0.1, 0, 0.1]),
        ((-0.0866, 0.0866), {"stack": True}, [-0.05, 0, 0.05]),
        ((0.0, 1.2), {"stack": True, "floor": 0.19}, [0.5, 1.0]),
    ],
)
def test_budget_breaks_match_r(limits, kw, expected):
    np.testing.assert_allclose(_budget_breaks(*limits, **kw), expected, atol=1e-12)


# --------------------------------------------------------------------------
# plot_hrfs
# --------------------------------------------------------------------------


class TestPlotHrfs:
    def test_overlay_returns_figure_with_legend_and_palette(self):
        fig = plot_hrfs(HRF_SPMG1, HRF_GAUSSIAN, GAMMA_HRF)
        assert isinstance(fig, Figure)
        (ax,) = fig.axes
        lines = _data_lines(ax)
        assert [ln.get_label() for ln in lines] == ["SPMG1", "gaussian", "gamma"]
        assert [_hex(ln.get_color()) for ln in lines] == list(CATEGORICAL[:3])
        assert [t.get_text() for t in ax.get_legend().get_texts()] == ["SPMG1", "gaussian", "gamma"]
        assert "HRF comparison" in _fig_texts(fig)
        assert ax.get_xlabel() == "Time (s)" and ax.get_ylabel() == "Response"

    def test_single_hrf_title_is_its_name(self):
        fig = plot_hrfs(GAMMA_HRF)
        assert "gamma" in _fig_texts(fig)
        assert fig.axes[0].get_legend() is None

    def test_title_and_subtitle_suppression(self):
        fig = plot_hrfs(HRF_SPMG1, GAMMA_HRF, title="", subtitle="note")
        assert "HRF comparison" not in _fig_texts(fig)
        assert "note" in _fig_texts(fig)

    def test_basis_set_gets_direct_labels_not_legend(self):
        fig = plot_hrfs(HRF_SPMG3)
        ax = fig.axes[0]
        assert ax.get_legend() is None
        texts = [t.get_text() for t in ax.texts]
        assert texts == ["canonical", "temporal deriv.", "dispersion deriv."]
        # Labels sit inside the panel.
        fig.canvas.draw()
        box = ax.get_window_extent()
        for t in ax.texts:
            bb = t.get_window_extent()
            assert box.x0 <= bb.x0 and bb.x1 <= box.x1 and bb.y1 <= box.y1

    def test_explicit_labels_use_legend(self):
        fig = plot_hrfs(HRF_SPMG2, labels=["Canonical", "Temporal derivative"])
        ax = fig.axes[0]
        assert len(ax.texts) == 0
        assert [t.get_text() for t in ax.get_legend().get_texts()] == ["Canonical", "Temporal derivative"]

    def test_auto_palette_is_ordered_for_long_basis_sets_and_many_curves(self):
        fig = plot_hrfs(HRF_FIR)
        assert [_hex(ln.get_color()) for ln in _data_lines(fig.axes[0])] == palette(12, "ordered")
        hrfs = [lag_hrf(HRF_SPMG1, k) for k in range(7)]
        fig = plot_hrfs(*hrfs)
        assert [_hex(ln.get_color()) for ln in _data_lines(fig.axes[0])] == palette(7, "ordered")
        fig = plot_hrfs(*hrfs[:3], palette="ordered")
        assert [_hex(ln.get_color()) for ln in _data_lines(fig.axes[0])] == palette(3, "ordered")

    def test_many_categorical_curves_vary_linetype(self):
        hrfs = [lag_hrf(HRF_SPMG1, k) for k in range(5)]
        lines = _data_lines(plot_hrfs(*hrfs).axes[0])
        assert len({str(ln._unscaled_dash_pattern) for ln in lines}) == 5
        lines = _data_lines(plot_hrfs(*hrfs[:4]).axes[0])
        assert {ln.get_linestyle() for ln in lines} == {"-"}

    def test_legend_reads_by_row(self):
        hrfs = [lag_hrf(HRF_SPMG1, k) for k in range(8)]
        fig = plot_hrfs(*hrfs, labels=[f"lag {k}" for k in range(8)], palette="ordered")
        leg = fig.axes[0].get_legend()
        ncol = leg._ncols
        assert 1 < ncol < 8
        order = [t.get_text() for t in leg.get_texts()]
        # Column-major storage of a by-row layout: first column is lag 0, lag ncol, ...
        assert order[0] == "lag 0" and order[1] == f"lag {ncol}"

    def test_reference_curve_is_dashed_grey_with_caption(self):
        fig = plot_hrfs(HRF_GAUSSIAN, GAMMA_HRF, normalize=True, reference=HRF_SPMG1)
        ax = fig.axes[0]
        ref = [ln for ln in ax.get_lines() if ln.get_label() == "_reference"]
        assert len(ref) == 1
        assert _hex(ref[0].get_color()) == INK["muted"]
        assert ref[0].get_linestyle() == "--"
        t = ref[0].get_xdata()
        y = HRF_SPMG1(t)
        np.testing.assert_allclose(ref[0].get_ydata(), y / np.max(np.abs(y)))
        assert "Dashed grey: SPMG1" in _fig_texts(fig)
        assert ax.get_ylabel() == "Response / peak"
        assert [t.get_text() for t in ax.get_legend().get_texts()] == ["gaussian", "gamma"]
        fig = plot_hrfs(HRF_GAUSSIAN, reference=HRF_SPMG1, reference_label="canonical")
        assert "Dashed grey: canonical" in _fig_texts(fig)
        with pytest.raises(TypeError):
            plot_hrfs(HRF_GAUSSIAN, reference="spmg1")

    def test_stack_has_one_panel_per_curve(self):
        fig = plot_hrfs(HRF_SPMG3, layout="stack")
        assert len(fig.axes) == 3
        assert [ax.get_title(loc="left") for ax in fig.axes] == [
            "canonical", "temporal deriv.", "dispersion deriv."]
        for ax in fig.axes:
            assert len(_data_lines(ax)) == 1
            assert ax.get_legend() is None
            assert len(ax.get_yticks()) <= 3
        assert fig.axes[-1].get_xlabel() == "Time (s)"
        assert fig.get_supylabel() == "Response"

    def test_stack_fixed_scales_share_y(self):
        fig = plot_hrfs(HRF_SPMG1, HRF_GAUSSIAN, layout="stack", scales="fixed")
        assert fig.axes[0].get_ylim() == fig.axes[1].get_ylim()
        fig = plot_hrfs(HRF_SPMG1, HRF_SPMG3, basis="first", layout="stack")
        assert len(fig.axes) == 2

    def test_overlay_tick_budget(self):
        ax = plot_hrfs(HRF_SPMG1, HRF_GAUSSIAN, GAMMA_HRF).axes[0]
        assert 2 <= len(ax.get_yticks()) <= 5
        lo, hi = ax.get_ylim()
        assert all(lo <= t <= hi for t in ax.get_yticks())

    def test_ax_mode(self):
        fig, (a1, a2) = plt.subplots(1, 2)
        out = plot_hrfs(HRF_SPMG1, GAMMA_HRF, ax=a1)
        assert out is fig
        assert a1.get_title(loc="left") == "HRF comparison"
        assert len(_data_lines(a1)) == 2 and len(_data_lines(a2)) == 0
        with pytest.raises(ValueError, match="ax="):
            plot_hrfs(HRF_SPMG3, layout="stack", ax=a2)

    def test_bad_choices(self):
        with pytest.raises(ValueError):
            plot_hrfs(HRF_SPMG1, layout="grid")
        with pytest.raises(ValueError):
            plot_hrfs(HRF_SPMG1, palette="rainbow")
        with pytest.raises(ValueError):
            plot_hrfs(HRF_SPMG1, basis="second")


# --------------------------------------------------------------------------
# plot_hrf / HRF.plot
# --------------------------------------------------------------------------


class TestPlotHrf:
    def test_single_basis_marks_peak(self):
        fig = plot_hrf(SPM_CANONICAL)
        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        t = np.arange(0, 24.0 + 1e-9, 0.1)
        y = SPM_CANONICAL(t)
        i = int(np.argmax(y))
        notes = [tx.get_text() for tx in ax.texts]
        assert notes == [f"Peak: {t[i]:.1f} s"]
        (pk,) = [ln for ln in ax.get_lines() if ln.get_label() == "_peak"]
        np.testing.assert_allclose([pk.get_xdata()[0], pk.get_ydata()[0]], [t[i], y[i]])
        assert _hex(_data_lines(ax)[0].get_color()) == CATEGORICAL[0]
        # Headroom above the peak for the note.
        assert ax.get_ylim()[1] > y[i] * 1.15
        assert "SPMG1" in _fig_texts(fig)

    def test_options(self):
        assert len(plot_hrf(SPM_CANONICAL, show_peak=False).axes[0].texts) == 0
        ax = plot_hrf(SPM_CANONICAL, normalize=True).axes[0]
        assert np.max(_data_lines(ax)[0].get_ydata()) == pytest.approx(1.0)
        t = np.linspace(0, 30, 200)
        ax = plot_hrf(SPM_CANONICAL, time=t).axes[0]
        np.testing.assert_allclose(_data_lines(ax)[0].get_xdata(), t)

    def test_multi_basis_one_panel_coloured_by_basis(self):
        fig = plot_hrf(HRF_SPMG2)
        assert len(fig.axes) == 1
        lines = _data_lines(fig.axes[0])
        assert [_hex(ln.get_color()) for ln in lines] == list(CATEGORICAL[:2])
        assert [t.get_text() for t in fig.axes[0].texts] == ["canonical", "temporal deriv."]

    def test_existing_axes_and_method(self):
        fig, ax = plt.subplots()
        assert plot_hrf(SPM_CANONICAL, ax=ax) is fig
        assert ax.get_title(loc="left") == "SPMG1"
        out = SPM_CANONICAL.plot(title="Canonical")
        assert isinstance(out, Figure)
        assert "Canonical" in _fig_texts(out)


# --------------------------------------------------------------------------
# plot_regressors
# --------------------------------------------------------------------------


@pytest.fixture
def two_regs():
    reg1 = regressor(onsets=[10, 30, 50, 70], hrf=HRF_SPMG1, duration=[0, 4, 0, 8])
    reg2 = regressor(onsets=[20, 40, 60], hrf=HRF_SPMG1)
    return reg1, reg2


class TestPlotRegressors:
    def test_overlay_marks_first_regressor_in_grey(self, two_regs):
        fig = plot_regressors({"A": two_regs[0], "B": two_regs[1]})
        ax = fig.axes[0]
        assert [ln.get_label() for ln in _data_lines(ax)] == ["A", "B"]
        bars = _event_bars(ax)
        assert len(bars) == 4
        assert {_hex(b.get_facecolor()) for b in bars} == {INK["text2"]}
        # Bars span onset to onset + max(duration, 0.4% of the time range).
        grid = np.arange(0, 70 + two_regs[0].span + 1e-9, 0.25)
        min_w = 0.004 * (grid[-1] - grid[0])
        widths = sorted(round(b.get_width(), 6) for b in bars)
        assert widths == sorted(round(max(d, min_w), 6) for d in [0, 4, 0, 8])
        assert sorted(b.get_x() for b in bars) == [10, 30, 50, 70]
        # Bars hang below the curves: from lo - 0.16 r to lo - 0.06 r.
        y = np.column_stack([r.evaluate(grid, precision=0.25) for r in two_regs])
        lo, hi = min(y.min(), 0), max(y.max(), 0)
        r = hi - lo
        b = bars[0]
        assert b.get_y() == pytest.approx(lo - 0.16 * r)
        assert b.get_y() + b.get_height() == pytest.approx(lo - 0.06 * r)
        # No tick label inside the event band.
        ylo, yhi = ax.get_ylim()
        assert min(ax.get_yticks()) >= ylo + 0.13 * (yhi - ylo) - 1e-12
        assert "Regressor comparison" in _fig_texts(fig)
        assert any(t.startswith("Bars:") for t in _fig_texts(fig))

    def test_show_onsets_true_colours_by_series(self, two_regs):
        ax = plot_regressors(*two_regs, show_onsets=True).axes[0]
        colours = [_hex(b.get_facecolor()) for b in _event_bars(ax)]
        assert colours.count(CATEGORICAL[0]) == 4 and colours.count(CATEGORICAL[1]) == 3
        assert [ln.get_label() for ln in _data_lines(ax)] == ["Regressor 1", "Regressor 2"]

    def test_show_onsets_false(self, two_regs):
        fig = plot_regressors(*two_regs, show_onsets=False)
        assert _event_bars(fig.axes[0]) == []
        assert not any(t.startswith("Bars:") for t in _fig_texts(fig))

    def test_stack_marks_each_panel(self, two_regs):
        fig = plot_regressors({"A": two_regs[0], "B": two_regs[1]}, layout="stack")
        assert len(fig.axes) == 2
        assert [len(_event_bars(ax)) for ax in fig.axes] == [4, 3]
        for ax in fig.axes:
            ylo, yhi = ax.get_ylim()
            assert min(ax.get_yticks()) >= ylo + 0.19 * (yhi - ylo) - 1e-12
            assert len(ax.get_yticks()) <= 3

    def test_samples_drawn_as_points(self, two_regs):
        st = np.arange(0, 90, 2.0)
        ax = plot_regressors(two_regs[0], samples=st, labels=["SPMG1"]).axes[0]
        (pts,) = [ln for ln in ax.get_lines() if ln.get_label() == "_samples"]
        np.testing.assert_allclose(pts.get_xdata(), st)
        grid = np.arange(0, 70 + two_regs[0].span + 1e-9, 0.25)
        np.testing.assert_allclose(pts.get_ydata(), two_regs[0].evaluate(st, precision=0.25))
        assert len(grid) == len(_data_lines(ax)[0].get_xdata())

    def test_marks_outside_grid_are_dropped(self, two_regs):
        ax = plot_regressors(two_regs[0], grid=np.arange(0, 40, 0.5)).axes[0]
        assert sorted(b.get_x() for b in _event_bars(ax)) == [10, 30]

    def test_regressor_set(self):
        rs = regressor_set(onsets=[5, 15, 25, 35], fac=["x", "y", "x", "y"], duration=2)
        ax = plot_regressors(rs).axes[0]
        assert [ln.get_label() for ln in _data_lines(ax)] == ["x", "y"]

    def test_single_curve_title_is_its_label(self, two_regs):
        assert "SPMG1" in _fig_texts(plot_regressors(two_regs[0], labels=["SPMG1"]))

    def test_errors(self, two_regs):
        with pytest.raises(ValueError):
            plot_regressors()
        with pytest.raises(TypeError):
            plot_regressors(two_regs[0], HRF_SPMG1)
        with pytest.raises(ValueError):
            plot_regressors(two_regs[0], show_onsets="all")


# --------------------------------------------------------------------------
# plot_regressor / Regressor.plot
# --------------------------------------------------------------------------


class TestPlotRegressor:
    def test_title_and_bars(self):
        reg = regressor(onsets=[10, 30, 50])
        fig = plot_regressor(reg)
        assert isinstance(fig, Figure)
        assert "SPMG1 regressor, 3 events" in _fig_texts(fig)
        assert len(_event_bars(fig.axes[0])) == 3
        assert "SPMG1 regressor, 1 event" in _fig_texts(plot_regressor(regressor([10])))

    def test_no_onsets(self):
        fig = plot_regressor(regressor(onsets=[10, 30]), show_onsets=False)
        assert _event_bars(fig.axes[0]) == []

    def test_custom_grid(self):
        grid = np.arange(0, 60, 0.5)
        ax = plot_regressor(regressor(onsets=[10, 30]), grid=grid).axes[0]
        np.testing.assert_allclose(_data_lines(ax)[0].get_xdata(), grid)

    def test_multi_basis_stacks_by_default(self):
        reg = regressor([10, 40], HRF_SPMG3)
        fig = plot_regressor(reg)
        assert len(fig.axes) == 3
        assert [len(_event_bars(ax)) for ax in fig.axes] == [2, 2, 2]
        assert len(plot_regressor(reg, layout="overlay").axes) == 1

    def test_trial_varying_and_empty(self):
        reg = regressor(onsets=[10, 30], hrf=[boxcar_generator(width=4), boxcar_generator(width=8)])
        assert "trial-varying HRF regressor, 2 events" in _fig_texts(plot_regressor(reg))
        fig = plot_regressor(regressor([], HRF_SPMG1))
        assert "SPMG1 regressor, 0 events" in _fig_texts(fig)
        assert _event_bars(fig.axes[0]) == []

    def test_method_and_axes(self):
        reg = regressor(onsets=[10, 30])
        assert isinstance(reg.plot(), Figure)
        fig, ax = plt.subplots()
        assert reg.plot(ax=ax) is fig
        assert ax.get_title(loc="left") == "SPMG1 regressor, 2 events"
        assert len(_data_lines(ax)) == 1


def test_long_frame_is_a_dataframe():
    assert isinstance(regressor_curves(regressor([5])), pd.DataFrame)
