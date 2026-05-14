"""ContrastIntent payload equality across the first-level -> group seam."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from fmrimod.contrast import OmnibusContrast
from fmrimod.dataset import group_data_from_fmrilm
from fmrimod.glm.contrasts import ContrastIntent, ContrastResult
from fmrimod.group import AdapterContractError, group_dataset_from_group_data, reduce


def _seam_fit():
    import fmrimod as fm
    from fmrimod.spec import hrf

    rng = np.random.default_rng(13)
    bold = rng.standard_normal((60, 6))
    events = pd.DataFrame(
        {
            "onset": [6.0, 18.0, 30.0, 42.0],
            "duration": [2.0, 2.0, 2.0, 2.0],
            "trial_type": ["a", "b", "a", "b"],
        }
    )
    dataset = fm.fmri_dataset(bold, tr=2.0, events=events)
    return fm.fmri_lm(hrf("trial_type", norm="spm"), dataset)


def _intent_payload(result: ContrastResult) -> dict[str, object]:
    assert isinstance(result.intent, ContrastIntent)
    payload = result.intent.to_dict()
    assert payload["basis_label"] == "hrf_norm:spm"
    assert isinstance(payload["weights"], list)
    assert str(payload["design_id"]).startswith("design:sha256:")
    assert str(payload["provenance_id"]).startswith("fitprov:sha256:")
    return payload


def _group_payload(group, contrast_name: str) -> dict[str, object]:
    assert group.contrast_data is not None
    payload = group.contrast_data.loc[contrast_name, "contrast_intent"]
    return json.loads(str(payload))


def _rehydrate_contrast_result(result: ContrastResult) -> ContrastResult:
    assert isinstance(result.intent, ContrastIntent)
    payload = json.loads(json.dumps(result.intent.to_dict(), sort_keys=True))
    return ContrastResult(
        name=result.name,
        estimate=np.array(result.estimate, copy=True),
        stat=np.array(result.stat, copy=True),
        se=None if result.se is None else np.array(result.se, copy=True),
        p_value=np.array(result.p_value, copy=True),
        df=result.df,
        stat_type=result.stat_type,
        intent=ContrastIntent.from_dict(payload),
        touched_columns=tuple(result.touched_columns),
        touched_column_details=tuple(
            dict(item) for item in result.touched_column_details
        ),
        caveats=tuple(result.caveats),
    )


class _SerializedFit:
    """Minimal serialized-fit shape accepted by group_data_from_fmrilm."""

    def __init__(self, result: ContrastResult) -> None:
        n_voxels = int(np.asarray(result.stat).size)
        self.betas = np.empty((0, n_voxels), dtype=np.float64)
        self.tstat = np.empty((0, n_voxels), dtype=np.float64)
        self.se = np.empty((0, n_voxels), dtype=np.float64)
        self.n_voxels = n_voxels
        self.contrasts = {result.name: result}


def test_typed_fmrilm_contrast_intent_reaches_group_result_by_payload() -> None:
    fit_a = _seam_fit()
    fit_b = _seam_fit()
    request = OmnibusContrast(term="trial_type", levels=("a", "b"))
    first = fit_a.contrast(request)
    second = fit_b.contrast(request)
    assert first.name == second.name
    assert _intent_payload(first) == _intent_payload(second)

    group_input = group_data_from_fmrilm(
        [fit_a, fit_b],
        contrast=first.name,
        stat=("p",),
        subjects=["s1", "s2"],
    )
    group = group_dataset_from_group_data(group_input)
    reduced = reduce(group, method="combine:fisher")

    expected = _intent_payload(first)
    assert _group_payload(group, first.name) == expected
    assert _group_payload(reduced, first.name) == expected


def test_serialized_first_level_intent_reaches_group_result_by_payload() -> None:
    fit = _seam_fit()
    first = fit.contrast(OmnibusContrast(term="trial_type", levels=("a", "b")))
    serialized_a = _rehydrate_contrast_result(first)
    serialized_b = _rehydrate_contrast_result(first)
    assert serialized_a.intent is not first.intent
    assert serialized_b.intent is not first.intent

    group_input = group_data_from_fmrilm(
        [_SerializedFit(serialized_a), _SerializedFit(serialized_b)],
        contrast=first.name,
        stat=("p",),
        subjects=["s1", "s2"],
    )
    group = group_dataset_from_group_data(group_input)
    reduced = reduce(group, method="combine:fisher")

    expected = _intent_payload(first)
    assert _group_payload(group, first.name) == expected
    assert _group_payload(reduced, first.name) == expected


def test_group_materialization_rejects_mixed_intent_payloads() -> None:
    fit = _seam_fit()
    first = fit.contrast(OmnibusContrast(term="trial_type", levels=("a", "b")))
    serialized_a = _rehydrate_contrast_result(first)
    serialized_b = _rehydrate_contrast_result(first)
    assert isinstance(serialized_b.intent, ContrastIntent)
    serialized_b.intent = ContrastIntent.from_dict(
        {
            **serialized_b.intent.to_dict(),
            "design_id": "design:sha256:different",
        }
    )

    group_input = group_data_from_fmrilm(
        [_SerializedFit(serialized_a), _SerializedFit(serialized_b)],
        contrast=first.name,
        stat=("p",),
        subjects=["s1", "s2"],
    )

    with pytest.raises(AdapterContractError, match="payloads differ"):
        group_dataset_from_group_data(group_input)
