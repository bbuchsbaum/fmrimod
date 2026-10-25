"""design_colmap basis and run metadata (inventory v2 DES-16).

R fmridesign 0d53fb2/453c62a record ``basis_name`` as the ``hrf(basis=)``
family and label FIR columns ``lag_00, lag_01, ...``; 1ecaf91/fa40860 mark
columns shared by all runs (global intercept, global constant drift) as not
block-diagonal and give per-run baseline columns their run.
"""

import numpy as np
import pandas as pd
import pytest

from fmrimod import event_model
from fmrimod.baseline import baseline_model
from fmrimod.design_colmap import design_colmap
from fmrimod.sampling import SamplingFrame


def _events():
    return pd.DataFrame({"onset": [2.0, 20.0, 40.0, 60.0], "cond": ["A", "B", "A", "B"]})


@pytest.mark.parametrize(
    "formula,family",
    [
        ("onset ~ hrf(cond, basis='fir', nbasis=3)", "fir"),
        ("onset ~ hrf(cond, basis='spmg3')", "spmg3"),
        ("onset ~ hrf(cond, basis='spmg3', lag=2)", "spmg3"),
        ("onset ~ hrf(cond, basis='tent')", "tent"),
        ("onset ~ hrf(cond, basis='spm')", "spmg1"),
    ],
)
def test_basis_name_is_the_declared_family(formula, family):
    em = event_model(formula, data=_events(), tr=2.0, n_scans=50)
    assert set(design_colmap(em).basis_name) == {family}


def test_fir_columns_are_labelled_by_lag():
    em = event_model(
        "onset ~ hrf(cond, basis='fir', nbasis=3)", data=_events(), tr=2.0, n_scans=50
    )
    cm = design_colmap(em)
    assert list(cm.basis_label[:3]) == ["lag_00", "lag_01", "lag_02"]


def test_spmg_labels_survive_a_lag():
    em = event_model(
        "onset ~ hrf(cond, basis='spmg3', lag=2)", data=_events(), tr=2.0, n_scans=50
    )
    assert list(design_colmap(em).basis_label[:3]) == ["canonical", "derivative", "dispersion"]


def _two_runs():
    return SamplingFrame(blocklens=[40, 40], tr=2.0)


def test_global_intercept_and_per_run_drift():
    cm = design_colmap(baseline_model(basis="poly", degree=1, sframe=_two_runs(),
                                      intercept="global"))
    rows = cm.set_index("name")
    assert not bool(rows.loc["constant_global", "is_block_diagonal"])
    assert pd.isna(rows.loc["constant_global", "run"])
    drift = cm[cm.role == "drift"]
    assert list(drift.run) == [1, 2]
    assert drift.is_block_diagonal.all()


def test_global_constant_drift_is_shared():
    cm = design_colmap(baseline_model(basis="constant", degree=1, sframe=_two_runs(),
                                      intercept="global"))
    assert list(cm.name) == ["base_constant"]
    assert not bool(cm.is_block_diagonal.iloc[0])
    assert pd.isna(cm.run.iloc[0])


def test_runwise_intercepts_and_nuisance_carry_their_run():
    rng = np.random.default_rng(0)
    nuis = [pd.DataFrame({"trans_x": rng.normal(size=40)}) for _ in range(2)]
    cm = design_colmap(baseline_model(basis="poly", degree=1, sframe=_two_runs(),
                                      intercept="runwise", nuisance_list=nuis))
    assert list(cm.run) == [1, 2, 1, 2, 1, 2]
    assert cm.is_block_diagonal.all()


def test_single_run_global_intercept_is_not_block_diagonal():
    cm = design_colmap(baseline_model(basis="poly", degree=1,
                                      sframe=SamplingFrame(blocklens=[40], tr=2.0),
                                      intercept="global"))
    rows = cm.set_index("name")
    assert not bool(rows.loc["constant_global", "is_block_diagonal"])
    assert bool(rows.loc["base_poly1_block_1", "is_block_diagonal"])
    assert rows.loc["base_poly1_block_1", "run"] == 1
