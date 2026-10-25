"""Tests for correlation_map and the design diagnostics behind it."""

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from scipy.stats import spearmanr

import fmrimod as fm
from fmrimod.visualization import correlation_map, design_correlations, design_vif
from fmrimod.visualization.design_stats import vif, within_run_correlations
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
    rng = np.random.default_rng(4)
    on = np.concatenate([np.sort(rng.uniform(2, 170, 12)) for _ in range(2)])
    ev = pd.DataFrame({"onset": on, "cond": np.tile(["a", "b", "c"], 8), "run": np.repeat([1, 2], 12)})
    # A near-duplicate of condition "a" one second later: strongly correlated.
    dup = ev[ev["cond"] == "a"].assign(cond="a2", onset=lambda d: d["onset"] + 1.0)
    return fm.event_model("hrf(cond)", data=pd.concat([ev, dup], ignore_index=True), block="run",
                          sampling_frame=sframe)


def _independent_vif(X, run):
    """1 / (1 - R^2) regressing each column on the others plus run dummies."""
    D = np.column_stack([(run == r).astype(float) for r in np.unique(run)])
    out = []
    for j in range(X.shape[1]):
        A = np.column_stack([np.delete(X, j, axis=1), D])
        y = X[:, j]
        res = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
        yc = y - D @ np.linalg.lstsq(D, y, rcond=None)[0]
        out.append((yc @ yc) / (res @ res))
    return np.array(out)


def test_design_correlations_match_corrcoef(em):
    R = design_correlations(em)
    X = em.design_matrix
    np.testing.assert_allclose(R.to_numpy(), np.corrcoef(X.T), atol=1e-12)
    assert list(R.index) == ["a", "a2", "b", "c"]
    S = design_correlations(em, method="spearman")
    np.testing.assert_allclose(S.to_numpy(), spearmanr(X).statistic, atol=1e-12)


def test_design_vif_matches_regression(em, sframe):
    X = em.design_matrix
    run = np.repeat([1, 2], 90)
    np.testing.assert_allclose(design_vif(em).to_numpy(), _independent_vif(X, run), rtol=1e-8)


def test_vif_aliased_and_constant():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(50, 2))
    X = np.column_stack([A, A.sum(axis=1), np.ones(50)])
    v = vif(X)
    assert np.isinf(v[:3]).all() and np.isnan(v[3])


def test_within_run_correlations_use_own_run(sframe):
    bm = fm.baseline_model(basis="poly", degree=2, sframe=sframe)
    R = design_correlations(bm, within_run=True)
    assert list(R.index) == ["drift 1 · run 1", "drift 2 · run 1", "drift 1 · run 2", "drift 2 · run 2"]
    X = bm.design_matrix[:, :2][:90]
    np.testing.assert_allclose(R.iloc[0, 1], np.corrcoef(X.T)[0, 1], atol=1e-12)
    assert np.isnan(R.iloc[0, 2])  # different runs share no scans
    # design_vif defaults to within-run for a baseline model when None.
    assert len(design_vif(bm, within_run=None)) == 4


def test_within_run_hidden_mask():
    run = np.repeat([1, 2], 10)
    X = np.zeros((20, 3))
    X[:10, 0] = np.arange(10)
    X[10:, 1] = np.arange(10) ** 2
    X[:, 2] = np.sin(np.arange(20))
    R, hidden, col_run = within_run_correlations(X, run)
    assert list(col_run) == [1, 2, 0]
    assert hidden[0, 1] and hidden[1, 0] and not hidden[0, 2]
    np.testing.assert_allclose(R[0, 2], np.corrcoef(X[:10, 0], X[:10, 2] - X[:10, 2].mean())[0, 1])


def test_map_cells_vif_and_flags(em):
    fig = correlation_map(em)
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    R = np.corrcoef(em.design_matrix.T)
    cells = [t for t in ax.texts if t.get_text() and t.get_text()[0] in ".−0" and "VIF" not in t.get_text()]
    assert len(cells) == 6  # lower triangle of 4 columns
    v = design_vif(em).to_numpy()
    diag = [t for t in ax.texts if t.get_text().startswith("VIF")]
    assert len(diag) == 4
    for t in diag:
        k = int(round(t.get_position()[0]))
        flagged = v[k] >= 5
        assert (to_hex(t.get_color()).upper() == WARN.upper()) == flagged
    sub = fig.texts[1].get_text()
    i, j = np.unravel_index(np.nanargmax(np.abs(np.tril(R, -1))), R.shape)
    assert sub.startswith(f"Max abs r = {abs(R[i, j]):.2f}: a vs a2")
    n_flag = int((np.abs(R[np.tril_indices(4, -1)]) >= 0.5).sum())
    assert f"{n_flag} pair" in sub
    outlined = [p for p in ax.patches if to_hex(p.get_edgecolor()).upper() == to_hex("#1C232E").upper()]
    assert len(outlined) == n_flag


def test_baseline_map_within_run_and_titles(sframe):
    bm = fm.baseline_model(basis="poly", degree=3, sframe=sframe)
    fig = correlation_map(bm)
    assert fig.texts[0].get_text() == "Baseline regressor correlations"
    assert "within runs" in fig.texts[-1].get_text()
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert "intercept · run 1" not in labels and len(labels) == 6
    cells = [t for t in ax.texts if t.get_text() and "VIF" not in t.get_text() and t.get_text() not in ("drift",)]
    assert len(cells) == 6  # 3 pairs within each run; cross-run pairs are not drawn


def test_options_and_errors(em):
    fig = correlation_map(em, half_matrix=False, limits="data", annot=False, method="spearman")
    ax = fig.axes[0]
    assert not [t for t in ax.texts if t.get_text().startswith(".")]
    assert "Spearman" in fig.texts[-1].get_text()
    with pytest.raises(ValueError, match="annot"):
        correlation_map(em, annotate=True, annot=True)
    with pytest.raises(ValueError, match="limits"):
        correlation_map(em, limits="wide")
    fig2, ax2 = plt.subplots()
    assert correlation_map(em, ax=ax2) is fig2
