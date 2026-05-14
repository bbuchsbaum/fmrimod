"""Tests for typed realized-design column provenance."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fmrimod.design import DesignColumns
from fmrimod.design.event_model import EventModel
from fmrimod.design_colmap import design_colmap, design_meta
from fmrimod.events import EventFactor
from fmrimod.formula.base import Term
from fmrimod.model.fmri_model import FmriModel
from fmrimod.sampling import SamplingFrame


class _Baseline:
    def __init__(self, n_scans: int):
        self.design_matrix = pd.DataFrame({"intercept": np.ones(n_scans)})
        self.column_names = ["intercept"]


class _Dataset:
    def __init__(self, n_scans: int):
        self.n_timepoints = n_scans
        self.n_runs = 1

    def get_sampling_frame(self):
        return SamplingFrame(tr=2.0, n_scans=self.n_timepoints)


def _event_model_with_hrf() -> EventModel:
    n_scans = 30
    events = {
        "condition": EventFactor(
            name="condition",
            onsets=np.array([2.0, 8.0, 14.0, 20.0]),
            values=np.array(["face", "scene", "face", "scene"]),
            durations=1.0,
        )
    }
    return EventModel(
        terms=[Term("condition", hrf="spmg2")],
        events=events,
        sampling_info=SamplingFrame(tr=2.0, n_scans=n_scans),
    )


def test_event_model_emits_declared_condition_and_basis_facts() -> None:
    model = _event_model_with_hrf()
    facts = model.column_facts

    assert len(facts) == len(model.column_names)
    assert [fact["name"] for fact in facts] == model.column_names

    face = [fact for fact in facts if fact["level"] == "face"]
    assert {fact["condition"] for fact in face} == {"condition.face"}
    assert {fact["basis_ix"] for fact in face} == {1, 2}
    assert all(fact["provenance"]["condition"] == "declared" for fact in face)
    assert all(fact["provenance"]["basis_ix"] == "declared" for fact in face)


def test_design_meta_matches_r_multibasis_factor_contract() -> None:
    model = _event_model_with_hrf()
    meta = design_meta(model)

    assert meta.shape[0] == model.design_matrix.shape[1]
    assert meta["col"].tolist() == list(range(1, len(model.column_names) + 1))
    assert meta["name"].tolist() == model.column_names
    assert meta["basis_ix"].tolist() == [1, 2, 1, 2]
    assert meta["basis_total"].tolist() == [2, 2, 2, 2]
    assert meta["basis_label"].tolist() == [
        "canonical",
        "derivative",
        "canonical",
        "derivative",
    ]


def test_design_colmap_reads_declared_metadata_instead_of_reparsing_names() -> None:
    n_scans = 30
    events = {
        "condition": EventFactor(
            name="condition",
            onsets=np.array([2.0, 8.0, 14.0, 20.0]),
            values=np.array(["face", "scene", "face", "scene"]),
            durations=1.0,
        )
    }
    model = EventModel(
        terms=[Term("condition", hrf="spmg2").set(prefix="stim")],
        events=events,
        sampling_info=SamplingFrame(tr=2.0, n_scans=n_scans),
    )

    meta = design_meta(model)
    colmap = design_colmap(model)

    assert colmap["name"].tolist() == model.column_names
    assert colmap["condition"].tolist() == meta["condition"].tolist()
    assert set(colmap["condition"]) == {"condition.face", "condition.scene"}
    assert set(colmap["term_tag"]) == {"stim_condition"}
    assert "stim_condition_condition.face" not in set(colmap["condition"])


def test_fmri_model_design_columns_exposes_declared_event_provenance() -> None:
    event_model = _event_model_with_hrf()
    model = FmriModel(
        event_model,
        _Baseline(n_scans=30),
        _Dataset(n_scans=30),
    )

    columns = model.design_columns()

    assert isinstance(columns, DesignColumns)
    assert columns.names == tuple(model.design_matrix(run=0).columns)

    face_basis_1 = columns.where(term="condition", level="face").where(
        model_source="event"
    )[0]
    assert face_basis_1.provenance_for("condition") == "declared"
    assert face_basis_1.provenance_for("basis_ix") == "declared"

    baseline = columns.where(model_source="baseline").one()
    assert baseline.name == "intercept"
    assert baseline.provenance_for("condition") == "missing"
