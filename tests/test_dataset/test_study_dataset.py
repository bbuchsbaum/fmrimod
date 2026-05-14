"""Tests for the canonical StudyDataset to GroupDataset substrate."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from fmrimod.dataset import (
    FmriDatasetError,
    StudyDataset,
    fmri_group,
    matrix_dataset,
    study_dataset,
    study_to_group,
)
from fmrimod.glm.contrasts import ContrastIntent, ContrastResult
from fmrimod.group import GroupDataset, reduce


def _dataset(offset: float = 0.0):
    data = np.arange(12, dtype=np.float64).reshape(4, 3) + offset
    return matrix_dataset(data, tr=2.0)


def _contrast(
    name: str,
    beta: list[float],
    se: list[float],
    *,
    intent: ContrastIntent | None = None,
) -> ContrastResult:
    beta_arr = np.asarray(beta, dtype=np.float64)
    se_arr = np.asarray(se, dtype=np.float64)
    return ContrastResult(
        name=name,
        estimate=beta_arr,
        stat=beta_arr / se_arr,
        se=se_arr,
        p_value=np.full(beta_arr.shape, 0.5, dtype=np.float64),
        df=20.0,
        stat_type="t",
        intent=intent,
    )


def _intent() -> ContrastIntent:
    return ContrastIntent(
        kind="contrast_spec",
        name="faces",
        term="trial_type",
        levels=("face", "object"),
        rows=1,
        basis_label="hrf:canonical",
        weights=((1.0, -1.0, 0.0),),
        design_id="design:sha256:abc123",
        provenance_id="fitprov:sha256:def456",
    )


def test_study_dataset_validates_subject_table_and_normalizes_list_cells() -> None:
    ds1 = _dataset()
    ds2 = _dataset(10)
    study = study_dataset(
        pd.DataFrame(
            {
                "subject_id": ["s1", "s2"],
                "age": [20, 30],
                "dataset": [[ds1], ds2],
            }
        ),
        space="mni152",
        metadata={"cohort": "demo"},
    )

    assert isinstance(study, StudyDataset)
    assert study.subject_ids == ("s1", "s2")
    assert study.datasets == (ds1, ds2)
    assert study.get_subject_dataset("s1") is ds1
    assert study.col_data.index.tolist() == ["s1", "s2"]
    assert study.col_data["age"].tolist() == [20, 30]
    assert study.metadata["cohort"] == "demo"


def test_study_dataset_rejects_invalid_subject_metadata() -> None:
    ds = _dataset()
    with pytest.raises(FmriDatasetError, match="duplicates"):
        study_dataset(
            pd.DataFrame(
                {
                    "subject_id": ["s1", "s1"],
                    "dataset": [ds, ds],
                }
            )
        )
    with pytest.raises(FmriDatasetError, match="length-one"):
        study_dataset(
            pd.DataFrame(
                {
                    "subject_id": ["s1"],
                    "dataset": [[ds, ds]],
                }
            )
        )


def test_study_dataset_materializes_native_group_dataset() -> None:
    study = fmri_group(
        pd.DataFrame(
            {
                "participant": ["s1", "s2"],
                "age": [20, 30],
                "dataset": [_dataset(), _dataset(10)],
            }
        ),
        id="participant",
        space="roi",
        mask_strategy="intersect",
    )
    contrasts = {
        "s1": {"faces": _contrast("faces", [1.0, 2.0], [0.1, 0.2])},
        "s2": {"faces": _contrast("faces", [3.0, 4.0], [0.1, 0.2])},
    }

    group = study.to_group_dataset(contrasts, sample_labels=["r1", "r2"])

    assert isinstance(group, GroupDataset)
    assert group.shape == (2, 2, 1)
    assert group.subjects == ("s1", "s2")
    assert group.contrasts == ("faces",)
    assert group.space.labels == ("r1", "r2")
    assert group.metadata["source_format"] == "study_dataset"
    assert group.metadata["space"] == "roi"
    assert group.metadata["mask_strategy"] == "intersect"
    assert group.col_data is not None
    assert group.col_data["age"].tolist() == [20, 30]
    np.testing.assert_allclose(group.assay("beta")[:, :, 0], [[1.0, 3.0], [2.0, 4.0]])
    np.testing.assert_allclose(group.assay("se")[:, :, 0], [[0.1, 0.1], [0.2, 0.2]])


def test_study_dataset_carries_contrast_intent_payload() -> None:
    study = study_dataset(
        pd.DataFrame(
            {
                "subject_id": ["s1", "s2"],
                "dataset": [_dataset(), _dataset(10)],
            }
        )
    )
    intent = _intent()
    contrasts = {
        "s1": {"faces": _contrast("faces", [1.0, 2.0], [0.1, 0.2], intent=intent)},
        "s2": {
            "faces": _contrast(
                "faces",
                [3.0, 4.0],
                [0.1, 0.2],
                intent=ContrastIntent.from_dict(intent.to_dict()),
            )
        },
    }

    group = study.to_group_dataset(contrasts, sample_labels=["r1", "r2"])
    reduced = reduce(group, method="meta:fe")

    assert group.contrast_data is not None
    assert reduced.contrast_data is not None
    assert "contrast_intent" in group.contrast_data
    assert "contrast_intent" in reduced.contrast_data
    assert json.loads(group.contrast_data.loc["faces", "contrast_intent"]) == (
        intent.to_dict()
    )
    assert group.contrast_data.loc["faces", "contrast_intent"] == (
        reduced.contrast_data.loc["faces", "contrast_intent"]
    )


def test_study_to_group_feeds_native_fixed_effects_reducer() -> None:
    study = study_dataset(
        pd.DataFrame(
            {
                "subject_id": ["s1", "s2"],
                "dataset": [_dataset(), _dataset(10)],
            }
        )
    )
    group = study_to_group(
        study,
        {
            "s1": _contrast("task", [1.0, 2.0], [0.1, 0.2]),
            "s2": _contrast("task", [3.0, 4.0], [0.1, 0.2]),
        },
    )

    reduced = reduce(group, method="meta:fe")

    assert reduced.metadata["reduce_method"] == "meta:fe"
    np.testing.assert_allclose(reduced.assay("beta_g")[:, 0, 0], [2.0, 3.0])
    np.testing.assert_allclose(
        reduced.assay("se_g")[:, 0, 0],
        [np.sqrt(1.0 / 200.0), np.sqrt(1.0 / 50.0)],
    )


def test_study_dataset_rejects_missing_or_misaligned_contrasts() -> None:
    study = study_dataset(
        pd.DataFrame(
            {
                "subject_id": ["s1", "s2"],
                "dataset": [_dataset(), _dataset(10)],
            }
        )
    )
    with pytest.raises(FmriDatasetError, match="missing subjects"):
        study.to_group_dataset({"s1": _contrast("task", [1.0], [0.1])})
    with pytest.raises(FmriDatasetError, match="same sample length"):
        study.to_group_dataset(
            {
                "s1": _contrast("task", [1.0, 2.0], [0.1, 0.1]),
                "s2": _contrast("task", [1.0], [0.1]),
            }
        )
