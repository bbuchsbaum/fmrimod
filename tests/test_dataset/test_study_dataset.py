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


def test_study_lowering_routes_voxel_estimates_through_subject_masks() -> None:
    """Voxel estimates are matched by voxel id, never by position.

    R ``e998bd4`` (``3ae565e:R/study_backend.R:120-198``): under ``intersect``
    each subject's columns are routed through its own mask, non-identical
    masks are refused under ``identical``, and masks over different grids are
    refused.
    """
    m1 = np.array([1, 1, 1, 1, 0, 0], dtype=bool)
    m2 = np.array([0, 0, 1, 1, 1, 1], dtype=bool)
    rng = np.random.default_rng(0)
    d1 = matrix_dataset(rng.normal(size=(10, 6)), tr=2.0, mask=m1)
    d2 = matrix_dataset(rng.normal(size=(10, 6)), tr=2.0, mask=m2)
    # Each estimate encodes the voxel id it belongs to.
    est1 = np.flatnonzero(m1).astype(float)
    est2 = np.flatnonzero(m2).astype(float) + 1000
    results = {
        "s1": _contrast("c", est1.tolist(), (est1 + 1).tolist()),
        "s2": _contrast("c", est2.tolist(), (est2 + 1).tolist()),
    }
    table = pd.DataFrame({"subject_id": ["s1", "s2"], "dataset": [d1, d2]})

    group = study_dataset(table, mask_strategy="intersect").to_group_dataset(results)
    beta = group.assay("beta")[:, :, 0]
    assert group.shape == (2, 2, 1)
    np.testing.assert_array_equal(beta[:, 0], [2, 3])
    np.testing.assert_array_equal(beta[:, 1] - 1000, [2, 3])
    np.testing.assert_array_equal(group.assay("se")[:, 1, 0] - 1000, [3, 4])
    assert group.row_data["voxel_index"].tolist() == [2, 3]

    union = study_dataset(table, mask_strategy="union").to_group_dataset(results)
    beta_u = union.assay("beta")[:, :, 0]
    assert union.shape == (6, 2, 1)
    np.testing.assert_array_equal(beta_u[:, 0], [0, 1, 2, 3, np.nan, np.nan])
    np.testing.assert_array_equal(beta_u[:, 1] - 1000, [np.nan, np.nan, 2, 3, 4, 5])

    with pytest.raises(FmriDatasetError, match="masks differ"):
        study_dataset(table, mask_strategy="subject_specific").to_group_dataset(
            results
        )

    other_grid = matrix_dataset(
        rng.normal(size=(10, 8)), tr=2.0, mask=np.repeat([True, False], 4)
    )
    mixed = pd.DataFrame({"subject_id": ["s1", "s3"], "dataset": [d1, other_grid]})
    with pytest.raises(FmriDatasetError, match="grid"):
        study_dataset(mixed, mask_strategy="intersect").to_group_dataset(
            {"s1": results["s1"], "s3": results["s1"]}
        )


def test_study_subject_ids_are_canonical_keys() -> None:
    # fmridataset e5ae9b8 / 57ff4a2: whole numbers format exactly, fractional
    # numbers and missing keys are refused. Inventory v2 DS-13. The fractional
    # case rules out a blanket str(int(v)).
    ds = _dataset()
    study = study_dataset(pd.DataFrame({"subject_id": [100000.0, 2.0], "dataset": [ds, ds]}))
    assert study.subject_ids == ("100000", "2")
    with pytest.raises(FmriDatasetError, match="missing"):
        study_dataset(pd.DataFrame({"subject_id": [100000.0, np.nan], "dataset": [ds, ds]}))
    with pytest.raises(FmriDatasetError, match="whole"):
        study_dataset(pd.DataFrame({"subject_id": [1.5], "dataset": [ds]}))
    assert study_dataset(
        pd.DataFrame({"subject_id": ["sub-01", "01"], "dataset": [ds, ds]})
    ).subject_ids == ("sub-01", "01")


def test_study_lowering_emits_the_common_voxel_space_for_image_data() -> None:
    # Inventory v2 DS-02 (the VoxelSpace part; steward decision D1, approved
    # 2026-10-24). Voxel-routed estimates of datasets with a known affine
    # lower to the subjects' common packed VoxelSpace, so group results keep
    # their grid and world transform. Matrix data has no geometry and keeps
    # sample labels, as do explicit sample_labels.
    neuroim = pytest.importorskip("neuroim")
    from fmrimod.dataset import fmri_dataset
    from fmrimod.group import SampleLabelSpace, VoxelSpace

    shape = (3, 2, 1)
    affine = np.array([[-2.0, 0, 0, 6.0], [0, 2.0, 0, 0], [0, 0, 2.0, 0], [0, 0, 0, 1]])
    rng = np.random.default_rng(1)

    def subject(mask):
        vec = neuroim.DenseNeuroVec(rng.normal(size=shape + (8,)) + 5,
                                    neuroim.NeuroSpace.from_affine(affine, shape + (8,)))
        return fmri_dataset(vec, mask=mask.reshape(shape), tr=2.0)

    m1 = np.array([1, 1, 1, 1, 0, 0], dtype=bool)
    m2 = np.array([0, 0, 1, 1, 1, 1], dtype=bool)
    table = pd.DataFrame({"subject_id": ["s1", "s2"], "dataset": [subject(m1), subject(m2)]})
    results = {
        "s1": _contrast("c", np.flatnonzero(m1).astype(float).tolist(), [1.0] * 4),
        "s2": _contrast("c", (np.flatnonzero(m2) + 1000.0).tolist(), [1.0] * 4),
    }

    group = study_dataset(table, mask_strategy="intersect").to_group_dataset(results)
    assert isinstance(group.space, VoxelSpace)
    assert tuple(group.space.shape) == shape
    np.testing.assert_array_equal(group.space.mask_idx, [2, 3])
    np.testing.assert_allclose(group.space.affine, affine)
    np.testing.assert_array_equal(group.assay("beta")[:, 1, 0] - 1000, [2, 3])

    labelled = study_dataset(table, mask_strategy="intersect").to_group_dataset(
        results, sample_labels=["a", "b"]
    )
    assert isinstance(labelled.space, SampleLabelSpace)

    flat = pd.DataFrame({"subject_id": ["s1", "s2"], "dataset": [
        matrix_dataset(rng.normal(size=(8, 6)), tr=2.0, mask=m1),
        matrix_dataset(rng.normal(size=(8, 6)), tr=2.0, mask=m2),
    ]})
    no_geometry = study_dataset(flat, mask_strategy="intersect").to_group_dataset(results)
    assert isinstance(no_geometry.space, SampleLabelSpace)
