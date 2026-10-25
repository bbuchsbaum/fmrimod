"""Fit plots (R fmrireg ``autoplot.fmri_lm`` / ``plot.fmri_model``).

Every number the figures show comes from a pure function in
:mod:`fmrimod.visualization.fit_plots`; these tests check those numbers
against independent computations (statsmodels OLS, explicit matrices,
Monte Carlo) and the figures' structure. No pixel snapshots.
"""

from __future__ import annotations

import warnings

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from matplotlib.collections import PatchCollection, PathCollection
from matplotlib.colors import to_hex
from scipy import signal, stats

import fmrimod as fm
from fmrimod.accessors import fitted_hrf
from fmrimod.spec import drift, hrf
from fmrimod.visualization import fit_plots as fp
from fmrimod.visualization._helpers import column_info
from fmrimod.visualization.style import CATEGORICAL, INK, colour_map

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _events(nruns: int, n: int, tr: float, conds: list[str], seed: int, spacing: float = 12.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(nruns):
        on = np.arange(8, n * tr - 20, spacing)
        cond = np.resize(conds, len(on))
        rng.shuffle(cond)
        for o, c in zip(on, cond, strict=True):
            rows.append({"onset": o, "duration": 0.0, "trial_type": c, "run": r + 1, "rt": rng.normal()})
    return pd.DataFrame(rows)


def _fit(formula, nruns=2, n=80, tr=2.0, nvox=30, conds=("face", "house", "scene"), seed=0, ar_noise=0.0, **kw):
    ev = _events(nruns, n, tr, list(conds), seed)
    rng = np.random.default_rng(seed + 1)
    ds0 = fm.matrix_dataset(rng.normal(size=(n * nruns, nvox)), tr=tr, run_length=[n] * nruns, event_table=ev)
    X = fm.fmri_lm(formula, ds0).model.design_matrix_array()
    B = rng.normal(scale=2.0, size=(X.shape[1], nvox))
    noise = rng.normal(size=(n * nruns, nvox))
    if ar_noise:
        for t in range(1, noise.shape[0]):
            noise[t] += ar_noise * noise[t - 1]
    ds = fm.matrix_dataset(X @ B + noise + 50.0, tr=tr, run_length=[n] * nruns, event_table=ev)
    return fm.fmri_lm(formula, ds, **kw)


@pytest.fixture(scope="module")
def fit2():
    return _fit(hrf("trial_type") + hrf("rt") + drift(), nruns=2)


@pytest.fixture(scope="module")
def fit1():
    return _fit(hrf("trial_type") + drift(), nruns=1, n=120)


@pytest.fixture(scope="module")
def fit_ar():
    return _fit(hrf("trial_type", basis="spmg3") + drift(), nruns=2, ar_noise=0.4, ar="ar1")


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _no_warnings(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return fn(*args, **kwargs)


def _fig_texts(fig) -> list[str]:
    return [t.get_text() for t in fig.findobj(mpl.text.Text)]


# --------------------------------------------------------------------------
# Pure computations: estimates view
# --------------------------------------------------------------------------


def test_critical_t_matches_r_qt():
    # R: qt(1 - 0.001 / 2, 100)
    assert fp.critical_t(100.0) == pytest.approx(3.390491, abs=1e-6)


def test_sina_offsets_are_deterministic_bounded_and_density_scaled():
    rng = np.random.default_rng(3)
    x = rng.normal(size=400)
    off = fp.sina_offsets(x, width=0.38)
    np.testing.assert_array_equal(off, fp.sina_offsets(x, width=0.38))
    assert np.all(np.abs(off) <= 0.38 + 1e-12)
    # offsets = (u - 1/2) * 2 * width * density / max(density) with u the
    # golden-ratio sequence by rank; recover the density and compare with an
    # independent Gaussian KDE at the same (Silverman, bw.nrd0) bandwidth.
    rank = np.argsort(np.argsort(x, kind="stable"), kind="stable")
    u = (np.arange(1, x.size + 1) * 0.6180339887) % 1.0
    scale = (u[rank] - 0.5) * 2 * 0.38
    ok = np.abs(scale) > 0.05
    dens_rel = off[ok] / scale[ok]
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    bw = 0.9 * min(np.std(x, ddof=1), iqr / 1.34) * x.size ** -0.2
    kde = stats.gaussian_kde(x, bw_method=bw / np.std(x, ddof=1))(x)
    np.testing.assert_allclose(dens_rel, (kde / kde.max())[ok], atol=0.01)
    assert np.all(fp.sina_offsets(np.ones(10)) == 0)
    assert np.all(fp.sina_offsets([1.0, 2.0]) == 0)


def test_share_beyond_counts_finite_voxels_and_flags_aliased_columns():
    t = np.array([[4.0, np.nan], [-5.0, np.nan], [1.0, np.nan], [3.5, np.nan]])
    pos, neg, est = fp.share_beyond(t, 3.29)
    assert pos[0] == pytest.approx(0.5) and neg[0] == pytest.approx(0.25)
    assert est.tolist() == [True, False]
    assert np.isnan(pos[1])


def test_coefficient_intervals_match_statsmodels_ols(fit1):
    v = 4
    X = fit1.model.design_matrix_array()
    y = fm.get_data_matrix(fit1.model.dataset)[:, v]
    ols = sm.OLS(y, X).fit()
    cols = fit1.model.event_column_indices
    tab = fp.coefficient_intervals(fit1.betas[cols, v], fit1.se()[cols, v], fit1.tstat()[cols, v],
                                   fit1.residual_df, 0.9)
    ci = ols.conf_int(alpha=0.1)[cols]
    np.testing.assert_allclose(tab["est"], ols.params[cols], rtol=1e-8)
    np.testing.assert_allclose(tab["lo"], ci[:, 0], rtol=1e-7)
    np.testing.assert_allclose(tab["hi"], ci[:, 1], rtol=1e-7)
    np.testing.assert_allclose(tab["p"], ols.pvalues[cols], rtol=1e-6)
    assert tab["txt"].str.match(r"^t = −?\d+\.\d, p (<|=) \.\d+$").all()


def test_peak_voxel_is_the_largest_absolute_event_t(fit2):
    t = fit2.tstat()[fit2.model.event_column_indices]
    assert fp.peak_voxel(fit2) == int(np.argmax(np.abs(t).max(axis=0)))


# --------------------------------------------------------------------------
# Pure computations: HRF view
# --------------------------------------------------------------------------


def test_hrf_band_single_voxel_matches_statsmodels_covariance(fit1):
    v = 7
    sample = np.arange(0, 20.25, 0.5)
    curves, _, aliased, terms = fp.hrf_curves(fit1, [v], sample, level=0.95)
    assert aliased == []
    X = fit1.model.design_matrix_array()
    y = fm.get_data_matrix(fit1.model.dataset)[:, v]
    ols = sm.OLS(y, X).fit()
    cov = ols.cov_params()
    q = stats.t.isf(0.025, ols.df_resid)
    (td,) = terms
    for cond, idx in td["blocks"].items():
        G = td["G"]
        mean = G @ ols.params[idx]
        half = q * np.sqrt(np.diag(G @ cov[np.ix_(idx, idx)] @ G.T))
        c = curves[curves["condition"] == cond]
        np.testing.assert_allclose(c["mean"], mean, rtol=1e-8, atol=1e-12)
        np.testing.assert_allclose(c["hi"] - c["mean"], half, rtol=1e-7, atol=1e-12)
        pk = int(np.argmax(np.abs(mean)))
        assert bool(c["sig"].iloc[0]) == bool(mean[pk] - half[pk] > 0 or mean[pk] + half[pk] < 0)


def test_hrf_means_agree_with_fitted_hrf_accessor(fit_ar):
    sample = np.arange(0, 20.25, 1.0)
    curves, _, _, terms = fp.hrf_curves(fit_ar, [2], sample)
    ref = fitted_hrf(fit_ar, sample)["trial_type"]["pred"]  # (conds*time, V)
    conds = list(terms[0]["blocks"])
    for k, cond in enumerate(conds):
        np.testing.assert_allclose(curves.loc[curves["condition"] == cond, "mean"],
                                   ref[k * sample.size:(k + 1) * sample.size, 2], rtol=1e-10)
    assert (curves["nbasis"] == 3).all()


def test_hrf_several_voxels_gives_mean_and_t_interval_of_the_mean(fit2):
    vox = list(range(12))
    sample = np.arange(0, 20.25, 1.0)
    curves, lines, _, terms = fp.hrf_curves(fit2, vox, sample, level=0.9, max_lines=5)
    td = terms[0]
    cond, idx = next(iter(td["blocks"].items()))
    H = td["G"] @ fit2.betas[np.ix_(idx, vox)]
    c = curves[curves["condition"] == cond]
    np.testing.assert_allclose(c["mean"], H.mean(axis=1))
    half = stats.t.isf(0.05, len(vox) - 1) * H.std(axis=1, ddof=1) / np.sqrt(len(vox))
    np.testing.assert_allclose(c["hi"] - c["mean"], half)
    assert lines.groupby(["condition", "voxel"]).ngroups == 5 * curves["condition"].nunique()


# --------------------------------------------------------------------------
# Pure computations: time course and residuals
# --------------------------------------------------------------------------


def test_voxel_refit_matches_statsmodels_joint_and_runwise(fit2):
    y = fm.get_data_matrix(fit2.model.dataset)[:, 3]
    Xe, Xb = fp._design_parts(fit2.model)
    run_id = np.repeat([1, 2], 80)
    joint = fp.voxel_refit(y, Xe, Xb, run_id, runwise=False)
    full = sm.OLS(y, np.hstack([Xe, Xb])).fit()
    base = sm.OLS(y, Xb).fit()
    np.testing.assert_allclose(joint["fitted"], full.fittedvalues, rtol=1e-9)
    assert joint["partial_r2"] == pytest.approx(1 - full.ssr / base.ssr, rel=1e-9)
    per = fp.voxel_refit(y, Xe, Xb, run_id, runwise=True)
    for r in (1, 2):
        rows = run_id == r
        xb = Xb[rows][:, (Xb[rows] != 0).any(axis=0)]
        ref = sm.OLS(y[rows], np.hstack([Xe[rows], xb])).fit()
        np.testing.assert_allclose(per["fitted"][rows], ref.fittedvalues, rtol=1e-9)
    np.testing.assert_allclose(per["resid"], y - per["fitted"])
    flat = fp.voxel_refit(Xb[:, 0] * 3.0, Xe, Xb, run_id, runwise=False)
    assert np.isnan(flat["partial_r2"])


def test_acf_cols_matches_numpy():
    rng = np.random.default_rng(5)
    E = rng.normal(size=(60, 3))
    A = fp.acf_cols(E, 5)
    for j in range(3):
        e = E[:, j] - E[:, j].mean()
        full = np.correlate(e, e, mode="full")[e.size - 1:]
        np.testing.assert_allclose(A[:, j], full[1:6] / full[0])


def _explicit_null(X: np.ndarray, lag_max: int):
    n = X.shape[0]
    Q = fp._orth_basis(X)
    M = np.eye(n) - Q @ Q.T
    tr_m = np.trace(M)
    A = []
    for k in range(1, lag_max + 1):
        S = np.eye(n, k=k)
        A.append((S + S.T) / 2)
    expect = np.array([np.trace(a @ M) / tr_m for a in A])
    cov = np.array([[2 * np.trace(A[j] @ M @ A[k] @ M) / tr_m**2 for k in range(lag_max)]
                    for j in range(lag_max)])
    return Q, expect, cov


def test_acf_null_moments_match_explicit_residual_maker(fit1):
    X = fit1.model.design_matrix_array()
    Q, expect, cov = _explicit_null(X, 6)
    np.testing.assert_allclose(fp.acf_null_cov(Q, 6), cov, rtol=1e-10, atol=1e-14)
    res = fp.residual_acf(np.zeros((X.shape[0], 1)) + 1e-3 * np.arange(X.shape[0])[:, None],
                          X, np.ones(X.shape[0]), runwise=False, lag_max=6)
    np.testing.assert_allclose(res["expect"], expect, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(res["cov"], cov, rtol=1e-10, atol=1e-14)


def test_white_noise_band_and_portmanteau_are_calibrated(fit1):
    """Monte Carlo: OLS residual autocorrelations of white noise centre on
    the design-induced expectation with the stated covariance, and the
    generalised Ljung-Box test rejects at its nominal rate."""
    X = fit1.model.design_matrix_array()
    rng = np.random.default_rng(11)
    Y = rng.normal(size=(X.shape[0], 4000))
    res = fp.residual_acf(Y, X, np.ones(X.shape[0]), runwise=False, lag_max=8)
    A = res["acf"]
    se_mean = np.sqrt(np.diag(res["cov"]) / A.shape[1])
    assert np.all(np.abs(A.mean(axis=1) - res["expect"]) < 4 * se_mean + 2e-3)
    emp = np.cov(A)
    d = np.diag(res["cov"])
    # treating e'e as its expectation overstates the variances by ~2% at n = 120
    np.testing.assert_allclose(np.diag(emp), d, rtol=0.1)  # + 3 MC sd (2.2% each)
    mc_sd = np.sqrt(np.outer(d, d) / A.shape[1])  # sampling sd of a covariance estimate
    off = ~np.eye(d.size, dtype=bool)
    assert np.all(np.abs(emp - res["cov"])[off] < 4 * mc_sd[off])
    lo, hi = fp.acf_band(res["expect"], res["cov"])
    inside = ((lo[:, None] <= A) & (hi[:, None] >= A)).mean(axis=1)
    assert np.all(np.abs(inside - 0.95) < 0.02)
    _, p = fp.portmanteau(A, res["expect"], res["cov"])
    assert abs(np.mean(p < 0.05) - 0.05) < 0.015
    # an i.i.d. 1/n band centred on zero would be visibly miscalibrated here
    assert np.mean(np.abs(A[0]) > 1.96 / np.sqrt(X.shape[0])) > 0.06


def test_portmanteau_is_the_quadratic_form_on_the_null_covariance():
    rng = np.random.default_rng(2)
    L = 4
    B = rng.normal(size=(L, L))
    cov = B @ B.T / 50 + np.eye(L) * 0.01
    expect = rng.normal(scale=0.01, size=L)
    acf = rng.normal(scale=0.1, size=(L, 3))
    q, p = fp.portmanteau(acf, expect, cov)
    for j in range(3):
        d = acf[:, j] - expect
        assert q[j] == pytest.approx(d @ np.linalg.inv(cov) @ d)
        assert p[j] == pytest.approx(stats.chi2.sf(q[j], L))


def test_ar_whiten_matches_lfilter_and_drops_burn_in():
    rng = np.random.default_rng(4)
    E = rng.normal(size=(50, 2))
    phi = [0.5, -0.2]
    ref = signal.lfilter([1.0, -0.5, 0.2], [1.0], E, axis=0)[2:]
    np.testing.assert_allclose(fp.ar_whiten(E, phi), ref)
    np.testing.assert_array_equal(fp.ar_whiten(E, []), E)


def test_prewhitened_acf_is_the_acf_of_gls_residuals(fit1):
    X = fit1.model.design_matrix_array()
    Y = fm.get_data_matrix(fit1.model.dataset)[:, :3]
    phi = [np.array([0.3])]
    res = fp.residual_acf(Y, X, np.ones(X.shape[0]), runwise=False, lag_max=5, phi=phi)
    Yw, Xw = fp.ar_whiten(Y, phi[0]), fp.ar_whiten(X, phi[0])
    E = Yw - Xw @ np.linalg.lstsq(Xw, Yw, rcond=None)[0]
    np.testing.assert_allclose(res["acf"], fp.acf_cols(E, 5), atol=1e-10)


def test_fit_ar_phi_reads_run_pooled_and_voxelwise_fits(fit_ar):
    phi, pooled = fp._fit_ar_phi(fit_ar, 2)
    assert not pooled and len(phi) == 2
    np.testing.assert_allclose(np.concatenate(phi), np.ravel(fit_ar.ar_params))
    vfit = _fit(hrf("trial_type") + drift(), nruns=2, ar_noise=0.3, ar="ar1_voxelwise")
    phi_v, pooled_v = fp._fit_ar_phi(vfit, 2)
    assert pooled_v and len(phi_v) == 2
    np.testing.assert_allclose(phi_v[1], np.median(np.asarray(vfit.ar_params)[1], axis=-1))
    assert fp._fit_ar_phi(_fit(hrf("trial_type") + drift(), nruns=1), 1) == (None, False)


def test_row_breaks_are_round_and_inside_the_limits():
    np.testing.assert_allclose(fp.row_breaks(-0.02, 0.19), [0.0, 0.1])
    np.testing.assert_allclose(fp.row_breaks(-0.5, 0.45), [-0.4, 0.0, 0.4])
    np.testing.assert_allclose(fp.row_breaks(0.0, 0.0), [0.0])


def test_place_curve_labels_avoids_other_curves_and_labels():
    t = np.linspace(0, 20, 81)
    curves = pd.concat([
        pd.DataFrame({"condition": "a", "time": t, "mean": np.exp(-((t - 5) ** 2) / 4),
                      "lo": np.nan, "hi": np.nan}),
        pd.DataFrame({"condition": "b", "time": t, "mean": 0.9 * np.exp(-((t - 5.5) ** 2) / 4),
                      "lo": np.nan, "hi": np.nan}),
    ])
    labs = pd.DataFrame({"condition": ["a", "b"], "txt": ["a", "b"], "peak_t": [5.0, 5.5],
                         "peak_v": [1.0, 0.9]})
    pos = fp.place_curve_labels(curves, labs, widths=[1.5, 1.5], height=0.05)
    (xa, ya), (xb, yb) = pos.to_numpy()
    overlap_x = xa < xb + 1.5 and xb < xa + 1.5
    assert not (overlap_x and abs(ya - yb) < 0.1)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def test_betas_distribution_structure(fit2):
    fig = _no_warnings(fp.plot_fit, fit2)
    (ax,) = fig.axes
    info = column_info(fit2)
    ev = info[info["kind"] == "event"]
    assert [t.get_text() for t in ax.get_yticklabels()] == list(ev["full_label"])
    texts = _fig_texts(fig)
    assert any(t.startswith("Regressor t-statistics across 30 voxels") for t in texts)
    assert any(t.startswith("t > 3.") for t in texts)
    assert not any("trial_type_trial_type" in t for t in texts)
    cmap = colour_map(list(ev["label"]))
    point_colours = {to_hex(c.get_facecolor()[0]).upper() for c in ax.collections
                     if isinstance(c, PathCollection)}
    assert point_colours == {v.upper() for v in cmap.values()}
    crit = fp.critical_t(fit2.residual_df)
    pos, _, _ = fp.share_beyond(fit2.tstat()[fit2.model.event_column_indices].T, crit)
    assert fp._pct(pos[0]) in texts
    assert fp.plot_fit(fit2, "estimates").axes[0].get_yticklabels()[0].get_text() == ev["full_label"].iloc[0]


def test_betas_distribution_bins_many_voxels(fit2, monkeypatch):
    monkeypatch.setattr(fp, "BIN_ABOVE", 10)
    fig = _no_warnings(fp.plot_fit, fit2)
    assert any(isinstance(c, PatchCollection) for c in fig.axes[0].collections)
    assert not any(isinstance(c, PathCollection) for c in fig.axes[0].collections)


def test_coefficient_plot_has_one_panel_per_term_and_intervals(fit2):
    fig = _no_warnings(fp.plot_fit, fit2, voxel=3, level=0.9)
    assert len(fig.axes) == 2  # trial_type and rt
    assert [a.get_title(loc="left") for a in fig.axes] == ["trial_type", "rt"]
    texts = _fig_texts(fig)
    assert any(t.startswith("Regressor estimates, voxel 3") for t in texts)
    t = fit2.tstat()[0, 3]
    assert any(t_.startswith(f"t = {t:.1f}".replace("-", "−")) for t_ in texts)
    fig2 = _no_warnings(fp.plot_fit, fit2, voxel=[1, 2])
    assert len(fig2.axes) == 4


def test_contrasts_views(fit2):
    with pytest.raises(ValueError, match="no t contrasts"):
        fp.plot_fit(fit2, "contrasts")
    fit2.contrast("face - house", name="face_vs_house")
    try:
        fig = _no_warnings(fp.plot_fit, fit2, "contrasts")
        assert [t.get_text() for t in fig.axes[0].get_yticklabels()] == ["face_vs_house"]
        fig = _no_warnings(fp.plot_fit, fit2, "contrasts", voxel=0)
        assert fig.axes[0].get_xlabel() == "Contrast estimate"
    finally:
        fit2.contrasts.clear()


def test_hrf_view_structure(fit2):
    fig = _no_warnings(fp.plot_fit, fit2, "hrf")
    assert len([a for a in fig.axes if a.get_visible()]) == 2
    texts = _fig_texts(fig)
    v = fp.peak_voxel(fit2)
    assert any(t.startswith(f"Voxel {v} (largest |t|), SPMG1 basis") for t in texts)
    info = column_info(fit2)
    for cond in ("face", "house", "scene"):
        assert any(t == cond or t == f"{cond} (n.s.)" for t in texts)
    cmap = colour_map(list(info.loc[info["kind"] == "event", "label"]))
    line_colours = {to_hex(line.get_color()).upper() for line in fig.axes[0].lines if line.get_label() in cmap}
    assert line_colours <= {c.upper() for c in cmap.values()}
    fig = _no_warnings(fp.plot_fit, fit2, "hrf", voxel=list(range(10)), direct_labels=False)
    assert fig.axes[0].get_legend() is not None
    assert any("Mean of 10 voxels" in t for t in _fig_texts(fig))


def test_hrf_multibasis_labels_report_latency(fit_ar):
    fig = _no_warnings(fp.plot_fit, fit_ar, "hrf")
    texts = _fig_texts(fig)
    assert any(t.endswith(" s") and "  " in t for t in texts) or any("(n.s.)" in t for t in texts)
    assert any("labels give its estimated latency" in t for t in texts)


def test_timecourse_panels_never_cross_run_boundaries(fit2):
    fig = _no_warnings(fp.plot_fit, fit2, "timecourse", voxel=2)
    axes = [a for a in fig.axes if a.get_ylabel()]
    assert [a.get_ylabel() for a in axes] == ["Signal", "Onsets", "Residual"]
    times = fit2.model.sampling_frame.samples
    boundary = (times[79] + times[80]) / 2
    for a in (axes[0], axes[2]):
        for line in a.lines:
            x = np.asarray(line.get_xdata(), dtype=float)
            if x.size > 2:
                assert x.max() < boundary or x.min() > boundary
    texts = _fig_texts(fig)
    assert {"observed", "fitted", "face", "house", "scene", "Run 1", "Run 2"} <= set(texts)
    y = fm.get_data_matrix(fit2.model.dataset)[:, 2]
    Xe, Xb = fp._design_parts(fit2.model)
    r2 = fp.voxel_refit(y, Xe, Xb, np.repeat([1, 2], 80), True)["partial_r2"]
    assert f"Partial R² of events given baseline = {r2:.2f}" in texts
    with pytest.raises(ValueError, match="single index"):
        fp.plot_fit(fit2, "timecourse", voxel=[1, 2])


def test_residual_views(fit2, fit_ar):
    fig = _no_warnings(fp.plot_fit, fit2, "residuals")
    texts = _fig_texts(fig)
    assert any(t.startswith("Residual autocorrelation, 30 voxels") for t in texts)
    assert fig.axes[0].get_legend() is None
    fig = _no_warnings(fp.plot_fit, fit_ar, "residuals", voxel=1)
    texts = _fig_texts(fig)
    assert any(t.startswith("Portmanteau test, lags 1–10: OLS p") and "prewhitened p" in t for t in texts)
    assert [t.get_text() for t in fig.axes[0].get_legend().get_texts()] == ["OLS", "prewhitened"]
    assert any("lag 1 by run" in t for t in texts)


def test_views_need_the_dataset(fit2, monkeypatch):
    monkeypatch.setattr(type(fit2.model), "dataset", property(lambda self: None))
    with pytest.raises(ValueError, match="does not carry its dataset"):
        fp.plot_fit(fit2, "timecourse")
    with pytest.raises(ValueError, match="does not carry its dataset"):
        fp.plot_fit(fit2, "residuals")


def test_argument_validation(fit2):
    with pytest.raises(ValueError, match="kind must be one of"):
        fp.plot_fit(fit2, "nope")
    with pytest.raises(ValueError, match="level"):
        fp.plot_fit(fit2, level=1.5)
    with pytest.raises(ValueError, match="integer indices between 0 and 29"):
        fp.plot_fit(fit2, voxel=30)
    with pytest.raises(ValueError, match="direct_labels"):
        fp.plot_fit(fit2, "hrf", direct_labels="yes")


def test_text_overrides_and_ax(fit1):
    fig = fp.plot_fit(fit1, title="", subtitle="Custom", caption="")
    texts = _fig_texts(fig)
    assert "Custom" in texts and not any(t.startswith("Regressor t-statistics") for t in texts)
    fig0, ax = plt.subplots()
    out = fp.plot_fit(fit1, "residuals", ax=ax)
    assert out is fig0 and ax.get_title(loc="left").startswith("Residual autocorrelation")


def test_fit_and_model_methods(fit1, fit2):
    assert isinstance(fit1.plot(), mpl.figure.Figure)
    assert isinstance(fit1.plot(kind="hrf", voxel=0), mpl.figure.Figure)
    fig = _no_warnings(fit2.model.plot)
    labels = [a.get_ylabel() for a in fig.axes if a.get_ylabel()]
    assert labels == ["face", "house", "scene", "rt", "cosine"]
    assert any("2 run intercepts not shown" in t for t in _fig_texts(fig))
    cosine_ax = fig.axes[labels.index("cosine")]
    assert {to_hex(line.get_color()).upper() for line in cosine_ax.lines
            if np.asarray(line.get_xdata()).size > 2} == {INK["muted"].upper()}
    times = fit2.model.sampling_frame.samples
    boundary = (times[79] + times[80]) / 2
    for a in fig.axes:
        for line in a.lines:
            x = np.asarray(line.get_xdata(), dtype=float)
            if x.size > 2:
                assert x.max() < boundary or x.min() > boundary
    face_ax = fig.axes[0]
    assert to_hex(face_ax.lines[0].get_color()).upper() == CATEGORICAL[0].upper()
    events_only = fit2.model.plot(baseline=False)
    assert [a.get_ylabel() for a in events_only.axes if a.get_ylabel()] == ["face", "house", "scene", "rt"]


def test_unidentified_marks_every_column_of_a_dependent_set():
    rng = np.random.default_rng(8)
    a, b, c = rng.normal(size=(3, 40))
    X = np.column_stack([a, b, a + b, c, np.zeros(40)])
    assert fp._unidentified(X).tolist() == [True, True, True, False, True]
    assert not fp._unidentified(rng.normal(size=(40, 4))).any()


def test_aliased_conditions_are_flagged_not_drawn():
    ev = _events(2, 80, 2.0, ["face", "house"], seed=3)
    ev["copy"] = np.where(ev["trial_type"] == "face", "face", "other")
    rng = np.random.default_rng(4)
    ds = fm.matrix_dataset(rng.normal(size=(160, 12)) + 10, tr=2.0, run_length=[80, 80], event_table=ev)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # fmri_lm warns about the rank deficiency
        fit = fm.fmri_lm(hrf("trial_type") + hrf("copy") + drift(), ds)
    alias = fp.aliased_columns(fit)
    assert alias[fit.model.event_column_indices].all()
    assert not alias[fit.model.baseline_column_indices].any()
    fig = _no_warnings(fp.plot_fit, fit)
    assert _fig_texts(fig).count("aliased") == 4
    assert not any(isinstance(c, PathCollection) for c in fig.axes[0].collections)
    fig = _no_warnings(fp.plot_fit, fit, voxel=0)
    assert _fig_texts(fig).count("not estimable (aliased)") == 4
    with pytest.raises(ValueError, match="estimable response"):
        fp.plot_fit(fit, "hrf")


def test_hrf_lists_aliased_conditions_in_the_caption(fit2, monkeypatch):
    mask = np.zeros(fit2.n_coefficients, dtype=bool)
    mask[1] = True  # house
    monkeypatch.setattr(fp, "aliased_columns", lambda fit: mask)
    _, _, aliased, _ = fp.hrf_curves(fit2, [0], np.arange(0, 20.25, 1.0))
    assert aliased == ["house"]
    fig = _no_warnings(fp.plot_fit, fit2, "hrf", voxel=0)
    assert any("Not estimable (aliased), not drawn: house." in t for t in _fig_texts(fig))
