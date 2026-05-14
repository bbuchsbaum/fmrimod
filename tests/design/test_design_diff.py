"""Tests for the typed ``design_diff`` primitive.

Bead: ``bd-01KRK97QMGJMH62H7NEXD14QGY``. Pins the sum-type contract and
the self-consistency property (``design_diff(a, a) == NoDiff()``) on
small handwritten fixtures, one per variant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from fmrimod import baseline_model
from fmrimod.design.diff import (
    BaselineDiff,
    Composite,
    EventDiff,
    HRFDiff,
    HRFKindChange,
    HRFParameterChange,
    NoDiff,
    SamplingDiff,
    TermChanged,
    TermFieldChange,
    design_diff,
)
from fmrimod.design.event_model import EventModel
from fmrimod.events.factor import EventFactor
from fmrimod.events.variable import EventVariable
from fmrimod.formula import Term
from fmrimod.hrf.library import SPMG1_HRF, GammaHRF
from fmrimod.model.fmri_model import FmriModel
from fmrimod.sampling import SamplingFrame

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _FakeDataset:
    """Minimal DatasetProtocol-compatible double for the FmriModel constructor."""

    sframe: SamplingFrame
    run_lengths: list[int]

    @property
    def n_runs(self) -> int:
        return len(self.run_lengths)

    @property
    def n_timepoints(self) -> list[int]:
        return list(self.run_lengths)

    def get_sampling_frame(self) -> SamplingFrame:
        return self.sframe

    def get_data(self, run: int) -> np.ndarray:  # pragma: no cover - never called
        return np.zeros((self.run_lengths[run], 1))


def _events_df(condition: list[str], onset: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "onset": onset,
        "condition": condition,
        "duration": [1.0] * len(onset),
    })


def _build_model(
    df: pd.DataFrame,
    *,
    n_scans: int = 60,
    extra_terms: list = (),
    include_rt_variable: bool = False,
    condition_hrf="spmg1",
    bmodel_basis: str = "constant",
) -> FmriModel:
    """Build an FmriModel directly from typed Term + Event objects.

    Bypasses the string-formula factory so HRF kind / parameters can be
    pinned per-term without touching the call site's argument shape.
    """

    sframe = SamplingFrame(tr=2.0, n_scans=n_scans)
    onsets = df["onset"].to_numpy(dtype=np.float64)
    durations = df["duration"].to_numpy(dtype=np.float64)
    events = {
        "condition": EventFactor(
            name="condition",
            onsets=onsets,
            values=df["condition"].to_list(),
            durations=durations,
        ),
    }
    terms = [Term("condition", hrf=condition_hrf), *extra_terms]
    if include_rt_variable:
        events["RT"] = EventVariable(
            name="RT",
            onsets=onsets,
            values=df["RT"].to_numpy(dtype=np.float64),
            durations=durations,
        )
        terms.append(Term("RT", hrf=condition_hrf))

    em = EventModel(terms=terms, events=events, sampling_info=sframe)
    bm = baseline_model(basis=bmodel_basis, sframe=sframe, intercept="runwise")
    ds = _FakeDataset(sframe=sframe, run_lengths=[n_scans])
    return FmriModel(em, bm, ds)


def _diff_for(a: FmriModel, b: FmriModel):
    return design_diff(a, b)


# ---------------------------------------------------------------------------
# NoDiff / self-consistency
# ---------------------------------------------------------------------------


class TestNoDiff:
    def test_identical_models_compare_equivalent(self) -> None:
        df = _events_df(["A", "B", "A", "B"], [5.0, 15.0, 25.0, 35.0])
        a = _build_model(df)
        b = _build_model(df)

        assert isinstance(_diff_for(a, b), NoDiff)

    def test_model_compared_against_itself_is_NoDiff(self) -> None:
        df = _events_df(["A", "B"], [10.0, 30.0])
        a = _build_model(df)

        assert isinstance(_diff_for(a, a), NoDiff)


# ---------------------------------------------------------------------------
# HRFDiff isolation
# ---------------------------------------------------------------------------


class TestHRFDiff:
    def test_parameter_tweak_returns_HRFDiff_not_EventDiff(self) -> None:
        df = _events_df(["A", "B"], [10.0, 30.0])
        a = _build_model(df, condition_hrf=SPMG1_HRF(p1=5.0, p2=15.0, a1=0.0833))
        b = _build_model(df, condition_hrf=SPMG1_HRF(p1=6.0, p2=15.0, a1=0.0833))

        diff = _diff_for(a, b)
        assert isinstance(diff, HRFDiff)
        # No kind change - same SPMG1_HRF class.
        assert diff.kind_changes == ()
        # One parameter change on the single 'condition' term.
        assert len(diff.parameter_changes) == 1
        pc = diff.parameter_changes[0]
        assert isinstance(pc, HRFParameterChange)
        assert pc.parameter == "p1"
        assert pc.a_value == 5.0
        assert pc.b_value == 6.0

    def test_kind_change_returns_HRFKindChange_not_parameter_change(self) -> None:
        df = _events_df(["A", "B"], [10.0, 30.0])
        a = _build_model(df, condition_hrf=SPMG1_HRF())
        b = _build_model(df, condition_hrf=GammaHRF(shape=6.0, rate=1.0))

        diff = _diff_for(a, b)
        # Kind change discriminates above parameter change; structural
        # comparison short-circuits parameters once the kind differs.
        # The diff may also pick up a column-name change because the
        # term column tags include the HRF kind.
        if isinstance(diff, HRFDiff):
            hrf_diff = diff
        elif isinstance(diff, Composite):
            hrf_diff = next(
                (p for p in diff.parts if isinstance(p, HRFDiff)), None,
            )
            assert hrf_diff is not None, (
                "expected an HRFDiff part inside the Composite"
            )
        else:
            pytest.fail(f"unexpected diff variant {type(diff).__name__}")
        assert len(hrf_diff.kind_changes) == 1
        kc = hrf_diff.kind_changes[0]
        assert isinstance(kc, HRFKindChange)
        assert kc.a_kind == "SPMG1_HRF"
        assert kc.b_kind == "GammaHRF"
        assert hrf_diff.parameter_changes == ()


# ---------------------------------------------------------------------------
# EventDiff: added / removed / changed
# ---------------------------------------------------------------------------


class TestEventDiff:
    def test_term_added_returns_TermAdded(self) -> None:
        df = pd.DataFrame({
            "onset": [5.0, 15.0, 25.0, 35.0],
            "condition": ["A", "B", "A", "B"],
            "RT": [1.2, 0.8, 1.1, 0.9],
            "duration": [1.0] * 4,
        })
        a = _build_model(df, include_rt_variable=False)
        b = _build_model(df, include_rt_variable=True)

        diff = _diff_for(a, b)
        # The added 'RT' term also changes the realized column set; the
        # diff may be reported as either a bare EventDiff or a Composite
        # of (EventDiff, ColumnsDiff) depending on whether column-name
        # caching has been triggered.
        event = (
            diff if isinstance(diff, EventDiff)
            else next(
                (p for p in diff.parts if isinstance(p, EventDiff)),
                None,
            )
            if isinstance(diff, Composite)
            else None
        )
        assert event is not None
        added_names = {t.name for t in event.added}
        assert "RT" in added_names
        assert event.removed == ()

    def test_term_removed_returns_TermRemoved(self) -> None:
        df = pd.DataFrame({
            "onset": [5.0, 15.0, 25.0, 35.0],
            "condition": ["A", "B", "A", "B"],
            "RT": [1.2, 0.8, 1.1, 0.9],
            "duration": [1.0] * 4,
        })
        a = _build_model(df, include_rt_variable=True)
        b = _build_model(df, include_rt_variable=False)

        diff = _diff_for(a, b)
        event = (
            diff if isinstance(diff, EventDiff)
            else next(
                (p for p in diff.parts if isinstance(p, EventDiff)),
                None,
            )
            if isinstance(diff, Composite)
            else None
        )
        assert event is not None
        removed_names = {t.name for t in event.removed}
        assert "RT" in removed_names

    def test_term_normalize_flag_change_returns_TermFieldChange(self) -> None:
        df = _events_df(["A", "B"], [10.0, 30.0])
        a = _build_model(df)
        b = _build_model(df)
        # Mutate one term's normalize flag on the b model.
        b.event_model.terms[0].normalize = True

        diff = _diff_for(a, b)
        # ``b`` keeps the same term name and HRF, so the change must
        # surface as a TermFieldChange under EventDiff.changed.
        assert isinstance(diff, EventDiff), (
            f"expected EventDiff for a single field tweak, got {type(diff).__name__}"
        )
        assert len(diff.changed) == 1
        tc = diff.changed[0]
        assert isinstance(tc, TermChanged)
        assert tc.name == "condition"
        assert len(tc.changes) == 1
        change = tc.changes[0]
        assert isinstance(change, TermFieldChange)
        assert change.field == "normalize"
        assert change.a_value is False
        assert change.b_value is True


# ---------------------------------------------------------------------------
# BaselineDiff
# ---------------------------------------------------------------------------


class TestBaselineDiff:
    def test_baseline_basis_change_returns_BaselineDiff(self) -> None:
        df = _events_df(["A", "B"], [10.0, 30.0])
        a = _build_model(df, bmodel_basis="constant")
        b = _build_model(df, bmodel_basis="poly")

        diff = _diff_for(a, b)
        # The baseline value and the realized baseline columns both changed.
        assert isinstance(diff, Composite)
        baseline = next(
            (p for p in diff.parts if isinstance(p, BaselineDiff)), None,
        )
        assert baseline is not None
        assert baseline.a_repr != baseline.b_repr


# ---------------------------------------------------------------------------
# ColumnsDiff / SamplingDiff
# ---------------------------------------------------------------------------


class TestColumnsAndSampling:
    def test_sampling_difference_returns_SamplingDiff(self) -> None:
        df = _events_df(["A", "B"], [10.0, 30.0])
        a = _build_model(df, n_scans=60)
        b = _build_model(df, n_scans=80)

        diff = _diff_for(a, b)
        sampling = (
            diff if isinstance(diff, SamplingDiff)
            else next(
                (p for p in diff.parts if isinstance(p, SamplingDiff)),
                None,
            )
            if isinstance(diff, Composite)
            else None
        )
        assert sampling is not None
        assert sampling.a_n_timepoints == (60,)
        assert sampling.b_n_timepoints == (80,)


# ---------------------------------------------------------------------------
# Composite handling
# ---------------------------------------------------------------------------


class TestComposite:
    def test_multiple_disjoint_changes_collapse_into_Composite(self) -> None:
        df = pd.DataFrame({
            "onset": [10.0, 30.0],
            "condition": ["A", "B"],
            "RT": [1.0, 1.5],
            "duration": [1.0, 1.0],
        })
        a = _build_model(df, include_rt_variable=False, bmodel_basis="constant")
        b = _build_model(df, include_rt_variable=True, bmodel_basis="poly")

        diff = _diff_for(a, b)
        assert isinstance(diff, Composite)
        kinds = {type(p).__name__ for p in diff.parts}
        # Both the event and baseline dimensions changed; columns shifted too.
        assert "EventDiff" in kinds
        assert "BaselineDiff" in kinds


# ---------------------------------------------------------------------------
# Variant types are typed (no dict[str, Any])
# ---------------------------------------------------------------------------


class TestVariantTyping:
    def test_no_diff_variants_use_dict_str_Any(self) -> None:
        """Every diff variant is a frozen dataclass - disqualifies the
        ``dict[str, Any]`` cheap pass named in the bead body."""

        import dataclasses

        from fmrimod.design import diff as diff_module

        for name in [
            "NoDiff", "EventDiff", "HRFDiff", "BaselineDiff",
            "ColumnsDiff", "SamplingDiff", "Composite",
            "TermAdded", "TermRemoved", "TermChanged", "TermFieldChange",
            "HRFKindChange", "HRFParameterChange",
        ]:
            cls = getattr(diff_module, name)
            assert dataclasses.is_dataclass(cls), (
                f"{name} must be a dataclass, not a free dict"
            )
            assert cls.__dataclass_params__.frozen, (
                f"{name} must be frozen so diff values are hashable / immutable"
            )
