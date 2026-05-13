"""Regression tests for parity-exercise pain points (May 2026).

The two new parity stress tests (``tier_a_parametric_modulation`` and
``tier_a_fir_unconstrained_hrf``) surfaced four real issues in the typed
spec/design path:

1. ``hrf("trial_type", modulators=["rt"])`` accepted the kwarg but the
   lowered EventModelTerm never consumed it, so the design contained only
   the main effect (silent footgun).
2. For categorical-by-continuous interactions, ``column_facts[i]["level"]``
   carried the placeholder ``"trial_type:rt_1"`` string instead of the
   underlying categorical level — so typed contrast resolvers could not
   find the per-condition parametric columns.
3. ``OmnibusContrast`` and ``DesignColumns.where(...)`` had no way to
   filter by ``basis_ix``, leaving multi-basis HRFs (FIR, spmg3) without a
   typed surface for "lag k only" hypotheses.
4. For multi-basis HRFs with multiple categorical levels, the column
   *names* and ``column_facts`` were laid out in basis-major order while
   the convolver actually produced columns in condition-major order. The
   declared ``(level, basis_ix)`` tuple at column k pointed at the wrong
   realised column, silently corrupting any contrast that selected by
   level or basis on a multi-basis × multi-condition design.

Each test below pins a specific behavioural claim of the corresponding
fix so a future regression would fail visibly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import fmrimod as fm
from fmrimod.contrast import OmnibusContrast
from fmrimod.spec import hrf


TR = 2.0
N_SCANS = 60


def _events_with_parametric() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 12
    onsets = np.linspace(8.0, N_SCANS * TR - 24.0, n, dtype=np.float64)
    return pd.DataFrame(
        {
            "onset": onsets,
            "duration": np.zeros(n, dtype=np.float64),
            "trial_type": (["A", "B"] * (n // 2)),
            "rt": rng.uniform(0.4, 1.2, n).astype(np.float64),
            "run": 1,
        }
    )


def _zero_dataset(events: pd.DataFrame) -> fm.FmriDataset:
    return fm.fmri_dataset(np.zeros((N_SCANS, 1)), tr=TR, events=events)


def test_modulators_kwarg_expands_into_parametric_terms() -> None:
    """hrf(..., modulators=[...]) lowers to main + parametric terms.

    Fix #1: previously the kwarg was accepted but downstream never read,
    so the design carried only the main effect. The expansion must be
    bitwise-equal to the explicit hrf(A) + hrf(A, rt) composition.
    """
    events = _events_with_parametric()
    ds = _zero_dataset(events)

    modulator_fit = fm.fmri_lm(
        hrf("trial_type", basis="spm", modulators=["rt"]), ds
    )
    explicit_fit = fm.fmri_lm(
        hrf("trial_type", basis="spm") + hrf("trial_type", "rt", basis="spm"),
        ds,
    )

    X_mod = modulator_fit.model.design_matrix_array(run=0)
    X_exp = explicit_fit.model.design_matrix_array(run=0)
    assert X_mod.shape == X_exp.shape
    np.testing.assert_array_equal(X_mod, X_exp)

    # Four task columns: main_A, main_B, param_A, param_B.
    task = [c for c in modulator_fit.design_columns().columns if c.role == "task"]
    assert len(task) == 4
    assert sorted({c.term for c in task}) == ["trial_type", "trial_type:rt"]


def test_mixed_interaction_columns_carry_categorical_level() -> None:
    """Parametric columns expose the underlying categorical level.

    Fix #2: ``column_facts[i]["level"]`` for a categorical-by-continuous
    interaction column is the categorical level string (``"A"``/``"B"``),
    not the placeholder ``"trial_type:rt_1"`` tag. This makes typed
    contrast resolution (e.g. ``OmnibusContrast(..., levels=("A",))``)
    find the correct per-condition parametric column.
    """
    events = _events_with_parametric()
    ds = _zero_dataset(events)
    fit = fm.fmri_lm(hrf("trial_type", "rt", basis="spm"), ds)

    param_columns = fit.design_columns().where(term="trial_type:rt")
    assert len(param_columns) == 2
    assert sorted({c.level for c in param_columns.columns}) == ["A", "B"]

    # Typed omnibus over only the A parametric column resolves to a single
    # row pointing at the correct column index.
    weights = OmnibusContrast("trial_type:rt", levels=("A",)).resolve(
        fit.design_columns()
    )
    assert weights.shape == (1, fit.model.design_matrix_array(run=0).shape[1])
    a_index = next(c.index for c in param_columns.columns if c.level == "A")
    assert int(np.argmax(weights[0])) == a_index


def test_where_filters_by_basis_ix() -> None:
    """DesignColumns.where(basis_ix=...) selects matching FIR/SPMG3 lags.

    Fix #3 part 1: ``where`` accepts a typed ``basis_ix`` kwarg so
    multi-basis HRF users can address a specific lag without iterating
    DesignColumn objects manually.
    """
    events = _events_with_parametric()
    ds = _zero_dataset(events)
    fit = fm.fmri_lm(hrf("trial_type", basis="fir"), ds)

    lag3 = fit.design_columns().where(term="trial_type", basis_ix=3)
    assert sorted({c.level for c in lag3.columns}) == ["A", "B"]
    assert {c.basis_ix for c in lag3.columns} == {3}


def test_omnibus_filters_by_basis_ix() -> None:
    """OmnibusContrast(basis_ix=...) builds the lag-subset F matrix.

    Fix #3 part 2: a 1-row contrast for "lag 3, condition A" stays
    typed. Mismatched basis indices raise DesignProvenanceError.
    """
    from fmrimod.contrast.errors import DesignProvenanceError

    events = _events_with_parametric()
    ds = _zero_dataset(events)
    fit = fm.fmri_lm(hrf("trial_type", basis="fir"), ds)

    lag3_A = OmnibusContrast(
        "trial_type", levels=("A",), basis_ix=(3,)
    ).resolve(fit.design_columns())
    assert lag3_A.shape == (1, fit.model.design_matrix_array(run=0).shape[1])

    # Basis 1..12 → asking for 13 must fail loudly, not silently drop.
    try:
        OmnibusContrast("trial_type", basis_ix=(13,)).resolve(
            fit.design_columns()
        )
    except DesignProvenanceError as exc:
        assert "basis_ix" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected DesignProvenanceError for basis_ix=13")


def test_fir_column_metadata_aligned_with_realised_content() -> None:
    """Multi-basis × multi-condition designs label columns by content.

    Fix #4: the convolver emits each categorical level's basis block as a
    contiguous (n_samples, nb) chunk via ``np.hstack``, so the realised
    layout is condition-major. The column names and ``column_facts`` now
    match: column k of the design carries level ``c.level`` and basis
    ``c.basis_ix`` for the corresponding DesignColumn.

    The probe uses singleton onsets — one A onset and one B onset — and
    checks which time points each column is non-zero at. For a 2 s FIR
    boxcar over 12 lags, basis k of an event at ``t0`` is non-zero in
    ``[t0 + 2(k-1), t0 + 2k)``; on a mid-TR sample grid that uniquely
    identifies the (level, basis_ix) tuple per column.
    """
    events = pd.DataFrame(
        {
            "onset": [10.0, 20.0],
            "duration": [0.0, 0.0],
            "trial_type": ["A", "B"],
            "run": [1, 1],
        }
    )
    ds = fm.fmri_dataset(np.zeros((30, 1)), tr=TR, events=events)
    fit = fm.fmri_lm(hrf("trial_type", basis="fir"), ds)

    X = np.asarray(fit.model.event_model.design_matrix)
    facts = list(fit.model.event_model.column_facts)
    sample_times = np.arange(X.shape[0], dtype=np.float64) * TR + TR / 2.0
    onset_for = {"A": 10.0, "B": 20.0}

    for k in range(X.shape[1]):
        fact = facts[k]
        level = fact["level"]
        basis_ix = fact["basis_ix"]
        assert level in onset_for, f"unexpected level {level!r}"
        assert isinstance(basis_ix, int) and 1 <= basis_ix <= 12

        # Predict which sample mid-TR time falls inside the boxcar window.
        boxcar_start = onset_for[level] + 2.0 * (basis_ix - 1)
        boxcar_end = boxcar_start + 2.0
        expected_active = (
            (sample_times >= boxcar_start) & (sample_times < boxcar_end)
        )
        actual_active = X[:, k] > 0
        np.testing.assert_array_equal(
            actual_active,
            expected_active,
            err_msg=(
                f"column {k} declared (level={level!r}, basis_ix={basis_ix}) "
                f"but its non-zero mask does not match the predicted FIR "
                f"window [{boxcar_start}, {boxcar_end})"
            ),
        )
