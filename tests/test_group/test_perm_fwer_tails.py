"""Wave 2: permutation FWER tails (fmrigds #22) and onesample weights (#21).

Cheap pass disqualified:
- accepting ``alternative`` while still ranking on ``|t|``
- setting ``p_fwer == p_perm`` without a max-statistic FWER
- accepting a weights argument that is ignored
"""

from __future__ import annotations

import numpy as np
import pytest

from fmrimod.group import (
    AdapterContractError,
    SampleLabelSpace,
    group_dataset,
    perm_onesample,
    perm_twosample,
)
from fmrimod.group._reducers_kernels import (
    _perm_tail_count,
    _perm_tail_score,
    _weighted_onesample_stats,
)


def _onesample_dataset(beta: np.ndarray, **assays: np.ndarray):
    n_subj = beta.shape[0]
    payload = {"beta": beta.reshape(1, n_subj, -1)}
    for name, arr in assays.items():
        payload[name] = np.asarray(arr, dtype=np.float64)
        if payload[name].ndim == 2:
            payload[name] = payload[name].reshape(1, n_subj, -1)
        elif payload[name].ndim == 1:
            payload[name] = payload[name].reshape(1, n_subj, 1)
    return group_dataset(
        payload,
        space=SampleLabelSpace(["roi=r1"]),
        subjects=[f"s{i}" for i in range(n_subj)],
        contrasts=[f"c{i}" for i in range(payload["beta"].shape[2])],
    )


def test_greater_fwer_uses_positive_tail_not_abs_t() -> None:
    """One-sided greater FWER must rank on +t, not |t|.

    Feature 0 is a large negative mean; feature 1 is a modest positive
    mean; feature 2 is a large positive mean whose nulls inflate FWER
    for the weaker features. Under ``alternative='greater'`` the
    negative feature is not extreme, so its FWER p-value is large.
    Two-sided / |t| ranking would treat it as more extreme than the
    modest positive feature.
    """
    beta = np.array(
        [
            [-3.0, 0.8, 5.0],
            [-3.2, 0.9, 5.2],
            [-2.8, 1.1, -5.0],
            [-3.1, 0.7, -4.8],
            [-2.9, 1.0, -5.1],
        ]
    )
    signs = np.array(
        [
            [1, 1, 1, 1, 1],
            [-1, 1, 1, 1, 1],
            [1, -1, 1, 1, 1],
            [1, 1, -1, 1, 1],
            [1, 1, 1, -1, 1],
            [1, 1, 1, 1, -1],
            [-1, -1, 1, 1, 1],
            [1, 1, -1, -1, 1],
            # Aligns feature 2 to all-positive without making feature 1
            # more extreme, so max-stat FWER exceeds uncorrected p.
            [1, 1, -1, -1, -1],
        ],
        dtype=np.int8,
    )
    ds = _onesample_dataset(beta)
    greater = perm_onesample(ds, signs=signs, alternative="greater")
    two_sided = perm_onesample(ds, signs=signs, alternative="two.sided")

    t_neg = greater.assay("t_g")[0, 0, 0]
    t_pos = greater.assay("t_g")[0, 0, 1]
    assert t_neg < 0 < t_pos
    # |t_neg| > t_pos, so |t| ranking would call the negative feature more extreme.
    assert abs(t_neg) > t_pos

    p_fwer_neg = greater.assay("p_fwer")[0, 0, 0]
    p_fwer_pos = greater.assay("p_fwer")[0, 0, 1]
    p_perm_neg = greater.assay("p_perm")[0, 0, 0]
    p_perm_pos = greater.assay("p_perm")[0, 0, 1]
    assert p_fwer_neg > p_fwer_pos
    assert p_fwer_neg > two_sided.assay("p_fwer")[0, 0, 0]
    assert p_fwer_neg >= p_perm_neg - 1e-12
    assert p_fwer_pos >= p_perm_pos - 1e-12
    # FWER is not a copy of the uncorrected permutation p.
    assert np.any(greater.assay("p_fwer") > greater.assay("p_perm") + 1e-12)


def test_onesample_fwer_matches_tail_oracle() -> None:
    beta = np.array(
        [
            [0.4, -0.2],
            [0.8, 0.1],
            [1.6, -0.3],
            [0.5, 0.4],
            [1.2, -0.1],
        ]
    )
    weights = np.array([1.0, 2.0, 4.0, 1.0, 3.0])
    signs = np.array(
        [
            [1, 1, 1, 1, 1],
            [-1, 1, 1, 1, 1],
            [1, -1, 1, 1, 1],
            [1, 1, -1, 1, 1],
            [1, 1, 1, -1, 1],
            [1, 1, 1, 1, -1],
        ],
        dtype=np.int8,
    )
    ds = _onesample_dataset(beta)
    w_full = np.repeat(weights[:, np.newaxis], 2, axis=1)

    for alternative in ("two.sided", "less", "greater"):
        observed = np.array(
            [
                _weighted_onesample_stats(beta[:, j], weights, min_subjects=2)[2]
                for j in range(2)
            ]
        )
        null = np.empty((signs.shape[0], 2))
        for i, row in enumerate(signs):
            for j in range(2):
                null[i, j] = _weighted_onesample_stats(
                    row.astype(np.float64) * beta[:, j],
                    weights,
                    min_subjects=2,
                )[2]
        obs_score = np.asarray(_perm_tail_score(observed, alternative))
        null_score = np.asarray(_perm_tail_score(null, alternative))
        max_null = np.max(null_score, axis=1)
        expected_perm = np.array(
            [
                (_perm_tail_count(null[:, j], observed[j], alternative) + 1.0)
                / (signs.shape[0] + 1.0)
                for j in range(2)
            ]
        )
        expected_fwer = (np.sum(max_null[:, None] >= obs_score, axis=0) + 1.0) / (
            signs.shape[0] + 1.0
        )
        out = perm_onesample(
            ds,
            signs=signs,
            alternative=alternative,
            weights="custom",
            custom_weights=w_full,
        )
        np.testing.assert_allclose(out.assay("p_perm")[0, 0, :], expected_perm)
        np.testing.assert_allclose(out.assay("p_fwer")[0, 0, :], expected_fwer)
        assert np.all(out.assay("p_fwer") >= out.assay("p_perm") - 1e-12)


def test_onesample_inverse_variance_weights_change_estimate() -> None:
    beta = np.array([0.2, 0.4, 1.8, 0.3, 0.5])
    se = np.array([0.10, 0.12, 1.50, 0.11, 0.09])
    signs = np.ones((7, 5), dtype=np.int8)
    signs[1:, 0] = -1
    ds = _onesample_dataset(beta, se=se.reshape(5, 1))

    equal = perm_onesample(ds, signs=signs, weights="equal")
    inverse = perm_onesample(ds, signs=signs, weights="1/var")
    assert inverse.assay("beta_g")[0, 0, 0] != equal.assay("beta_g")[0, 0, 0]
    # The high-SE outlier (1.8) should be down-weighted.
    assert inverse.assay("beta_g")[0, 0, 0] < equal.assay("beta_g")[0, 0, 0]
    np.testing.assert_allclose(
        equal.assay("beta_g")[0, 0, 0],
        float(np.mean(beta)),
    )


def test_onesample_equal_weights_match_historical_unweighted() -> None:
    ds = _onesample_dataset(np.array([1.0, 2.0, 3.0]))
    signs = np.array(
        [
            [1, 1, 1],
            [-1, 1, 1],
            [1, -1, 1],
            [1, 1, -1],
        ],
        dtype=np.int8,
    )
    out = perm_onesample(ds, signs=signs, weights="equal")
    y = np.array([1.0, 2.0, 3.0])
    expected_t = np.mean(y) / (np.std(y, ddof=1) / np.sqrt(3))
    np.testing.assert_allclose(out.assay("beta_g")[0, 0, 0], 2.0)
    np.testing.assert_allclose(out.assay("t_g")[0, 0, 0], expected_t)


def test_onesample_n_eff_and_custom_weights_require_payload() -> None:
    ds = _onesample_dataset(np.array([1.0, 2.0, 3.0]), se=np.ones((3, 1)))
    with pytest.raises(AdapterContractError, match="n_eff"):
        perm_onesample(ds, signs=np.ones((3, 3), dtype=np.int8), weights="n_eff")
    with pytest.raises(AdapterContractError, match="custom_weights"):
        perm_onesample(ds, signs=np.ones((3, 3), dtype=np.int8), weights="custom")

    n_eff = np.array([2.0, 4.0, 8.0])
    ds_n = _onesample_dataset(
        np.array([1.0, 2.0, 3.0]),
        se=np.ones((3, 1)),
        n_eff=n_eff,
    )
    out = perm_onesample(ds_n, signs=np.ones((4, 3), dtype=np.int8), weights="n_eff")
    custom = perm_onesample(
        ds_n,
        signs=np.ones((4, 3), dtype=np.int8),
        weights="custom",
        custom_weights=n_eff,
    )
    np.testing.assert_allclose(out.assay("beta_g"), custom.assay("beta_g"))


def test_twosample_greater_fwer_dominates_perm() -> None:
    ds = group_dataset(
        {"beta": np.array([[[1.0, 4.0], [1.5, 3.5], [4.0, -2.0], [5.0, -1.5]]])},
        space=SampleLabelSpace(["r1"]),
        subjects=["s1", "s2", "s3", "s4"],
        contrasts=["c1", "c2"],
    )
    group_mat = np.array(
        [
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 1, 0],
        ],
        dtype=np.int8,
    )
    for alternative in ("two.sided", "less", "greater"):
        out = perm_twosample(
            ds,
            group=[0, 0, 1, 1],
            group_mat=group_mat,
            alternative=alternative,
            variance="pooled",
        )
        p_fwer = out.assay("p_fwer")
        p_perm = out.assay("p_perm")
        assert np.all(np.isfinite(p_fwer))
        assert np.all(p_fwer >= p_perm - 1e-12)


def test_twosample_rejects_non_equal_weights() -> None:
    ds = group_dataset(
        {"beta": np.array([[[1.0], [2.0], [4.0], [5.0]]])},
        space=SampleLabelSpace(["r1"]),
        subjects=["s1", "s2", "s3", "s4"],
        contrasts=["c1"],
    )
    with pytest.raises(AdapterContractError, match="only weights"):
        perm_twosample(
            ds,
            group=[0, 0, 1, 1],
            group_mat=np.array([[0, 0, 1, 1]], dtype=np.int8),
            weights="1/var",
        )
