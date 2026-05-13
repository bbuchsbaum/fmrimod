"""Tests for ITEM helper layer in ``fmrimod.single.item``."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse

from fmrimod.single.item import (
    ItemBundle,
    ItemCovarianceBlockResult,
    ItemCovarianceMatrixResult,
    ItemCvResult,
    _item_simple_hash,
    _score_fold,
    item_build_design,
    item_compute_u,
    item_cv,
    item_fit,
    item_from_lsa,
    item_predict,
    item_slice_fold,
)
from fmrimod.single.lsa import lsa_single_trial


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260218)


def test_item_compute_u_dense_matches_closed_form(rng: np.random.Generator) -> None:
    X_t = rng.standard_normal((30, 6))
    ridge = 1e-8
    u_res = item_compute_u(X_t, ridge=ridge)

    assert isinstance(u_res, ItemCovarianceMatrixResult)
    U = u_res.U
    assert U.shape == (6, 6)
    assert_allclose(U, U.T, atol=1e-8)

    expected = np.linalg.solve(X_t.T @ X_t + ridge * np.eye(6), np.eye(6))
    assert_allclose(U, expected, atol=1e-6)

    eigvals = np.linalg.eigvalsh(U)
    assert np.min(eigvals) > -1e-8


def test_item_compute_u_block_v_matches_dense(rng: np.random.Generator) -> None:
    X_t = rng.standard_normal((24, 5))
    V_blocks = [np.eye(12) * 2.0, np.eye(12) * 3.0]

    V_dense = np.zeros((24, 24), dtype=np.float64)
    V_dense[:12, :12] = V_blocks[0]
    V_dense[12:, 12:] = V_blocks[1]

    u_block = item_compute_u(X_t, V=V_blocks, v_type="cov", ridge=1e-6)
    u_dense = item_compute_u(X_t, V=V_dense, v_type="cov", ridge=1e-6)

    assert isinstance(u_block, ItemCovarianceMatrixResult)
    assert isinstance(u_dense, ItemCovarianceMatrixResult)
    assert_allclose(u_block.U, u_dense.U, atol=1e-6)


def test_item_compute_u_accepts_sparse_precision(rng: np.random.Generator) -> None:
    X_t = rng.standard_normal((30, 5))
    V_prec = sparse.diags([2.0] * 30, format="csc")

    u_res = item_compute_u(X_t, V=V_prec, v_type="precision", ridge=1e-6)
    assert isinstance(u_res, ItemCovarianceMatrixResult)
    assert u_res.U.shape == (5, 5)
    assert_allclose(u_res.U, u_res.U.T, atol=1e-8)
    assert np.all(np.isfinite(u_res.U))


def test_item_slice_fold_matrix_and_by_run_paths(rng: np.random.Generator) -> None:
    n_time = 60
    n_trials = 12
    n_vox = 4

    X_t = rng.standard_normal((n_time, n_trials))
    run_id = np.repeat(np.arange(1, 4), 4)
    labels = np.array(["A", "B"] * 6)

    bundle = item_build_design(X_t=X_t, T_target=labels, run_id=run_id)
    u_matrix = item_compute_u(X_t, ridge=1e-6)
    assert isinstance(u_matrix, ItemCovarianceMatrixResult)
    bundle = replace(
        bundle,
        Gamma=rng.standard_normal((n_trials, n_vox)),
        U=u_matrix.U,
    )

    fold = item_slice_fold(bundle, test_run=2)
    train_idx = np.where(run_id != 2)[0]
    test_idx = np.where(run_id == 2)[0]

    assert_allclose(fold.train_idx, train_idx)
    assert_allclose(fold.test_idx, test_idx)
    assert_allclose(fold.Gamma_train, bundle.Gamma[train_idx, :])
    assert_allclose(fold.Gamma_test, bundle.Gamma[test_idx, :])
    assert bundle.U is not None
    assert_allclose(fold.U_train, bundle.U[np.ix_(train_idx, train_idx)])
    assert_allclose(fold.U_test, bundle.U[np.ix_(test_idx, test_idx)])

    by_run = item_compute_u(X_t, ridge=1e-6, run_id=run_id, output="by_run")
    assert isinstance(by_run, ItemCovarianceBlockResult)
    bundle_by_run = replace(bundle, U=None, U_by_run=by_run.U_by_run)

    fold_blocks = item_slice_fold(bundle_by_run, test_run=2)
    assert bundle_by_run.U_by_run is not None
    assert_allclose(fold_blocks.U_test, bundle_by_run.U_by_run["2"])
    assert fold_blocks.U_train.shape == (train_idx.size, train_idx.size)


def test_item_bundle_is_frozen(rng: np.random.Generator) -> None:
    bundle = item_build_design(
        X_t=rng.standard_normal((20, 4)),
        T_target=np.array(["A", "B", "A", "B"]),
    )

    with pytest.raises(FrozenInstanceError):
        bundle.Gamma = rng.standard_normal((4, 2))  # type: ignore[misc]


def test_item_fit_fallback_and_predict_shape(rng: np.random.Generator) -> None:
    n_trials = 24
    Gamma = rng.standard_normal((n_trials, 4))
    Gamma = np.column_stack([Gamma, Gamma[:, 0]])  # induce rank deficiency
    T_train = rng.standard_normal((n_trials, 2))
    U_train = np.eye(n_trials)

    with pytest.warns(UserWarning, match="requested 'chol' but used"):
        fit = item_fit(Gamma, T_train, U_train, ridge=0.0, method="chol")

    assert fit.W_hat.shape == (Gamma.shape[1], T_train.shape[1])
    assert np.all(np.isfinite(fit.W_hat))

    pred = item_predict(Gamma, fit)
    assert pred.shape == (n_trials, T_train.shape[1])


def test_item_cv_deterministic_regression(rng: np.random.Generator) -> None:
    n_runs = 4
    trials_per_run = 20
    n_trials = n_runs * trials_per_run

    run_id = np.repeat(np.arange(1, n_runs + 1), trials_per_run)
    Gamma = rng.standard_normal((n_trials, 6))
    T_target = rng.standard_normal((n_trials, 2))
    U = np.eye(n_trials)

    res1 = item_cv(
        Gamma=Gamma,
        T_target=T_target,
        U=U,
        run_id=run_id,
        mode="regression",
        metric="correlation",
        ridge=1e-4,
        method="svd",
    )
    res2 = item_cv(
        Gamma=Gamma,
        T_target=T_target,
        U=U,
        run_id=run_id,
        mode="regression",
        metric="correlation",
        ridge=1e-4,
        method="svd",
    )

    assert isinstance(res1, ItemCvResult)
    assert res1.folds == res2.folds
    assert res1.aggregate == res2.aggregate
    assert_allclose(res1.predictions.T_hat, res2.predictions.T_hat, atol=1e-12)


def test_item_cv_enforces_trial_hash(rng: np.random.Generator) -> None:
    n_trials = 15
    n_features = 4

    X_t = rng.standard_normal((75, n_trials))
    run_id = np.repeat(np.arange(1, 4), 5)
    trial_id = np.asarray([f"trial_{i:02d}" for i in range(n_trials)])
    trial_hash = _item_simple_hash(trial_id)

    bundle = item_build_design(
        X_t=X_t,
        T_target=rng.standard_normal(n_trials),
        run_id=run_id,
        trial_id=trial_id,
        trial_hash=trial_hash,
    )
    bundle = replace(
        bundle,
        Gamma=rng.standard_normal((n_trials, n_features)),
        U=np.eye(n_trials),
    )

    res = item_cv(bundle, mode="regression", metric="correlation", check_hash=True)
    assert isinstance(res, ItemCvResult)

    bad_trial_id = bundle.trial_id.copy()
    bad_trial_id[0] = "tampered_trial"
    bundle_bad = replace(bundle, trial_id=bad_trial_id)

    with pytest.raises(ValueError, match="Trial hash mismatch"):
        item_cv(bundle_bad, mode="regression", metric="correlation", check_hash=True)


def test_item_from_lsa_alignment_and_nuisance_alias(rng: np.random.Generator) -> None:
    n_time = 90
    n_trials = 12
    n_vox = 5

    X_t = rng.standard_normal((n_time, n_trials))
    confounds = np.column_stack([np.ones(n_time), np.linspace(-1.0, 1.0, n_time)])

    beta = rng.standard_normal((n_trials, n_vox))
    beta_z = rng.standard_normal((confounds.shape[1], n_vox))
    Y = X_t @ beta + confounds @ beta_z + rng.standard_normal((n_time, n_vox)) * 0.05

    run_id = np.repeat(np.arange(1, 4), 4)
    labels = np.array(["A", "B"] * 6)

    bundle_conf = item_from_lsa(
        Y=Y,
        X_t=X_t,
        T_target=labels,
        run_id=run_id,
        confounds=confounds,
        solver="svd",
    )
    bundle_nuis = item_from_lsa(
        Y=Y,
        X_t=X_t,
        T_target=labels,
        run_id=run_id,
        nuisance=confounds,
        solver="svd",
    )

    assert isinstance(bundle_conf, ItemBundle)
    assert bundle_conf.Gamma is not None
    assert bundle_conf.U is not None
    assert bundle_conf.Gamma.shape == (n_trials, n_vox)
    assert bundle_conf.U.shape == (n_trials, n_trials)
    assert_allclose(bundle_conf.run_id, run_id)
    assert_allclose(bundle_conf.Gamma, bundle_nuis.Gamma, atol=1e-12)

    with pytest.warns(UserWarning, match="Both confounds and nuisance"):
        item_from_lsa(
            Y=Y,
            X_t=X_t,
            T_target=labels,
            run_id=run_id,
            confounds=confounds,
            nuisance=confounds * 0.5,
            solver="svd",
        )


def test_item_cv_classification_near_chance_under_null(
    rng: np.random.Generator,
) -> None:
    n_runs = 4
    trials_per_run = 45
    n_trials = n_runs * trials_per_run

    run_id = np.repeat(np.arange(1, n_runs + 1), trials_per_run)
    Gamma = rng.standard_normal((n_trials, 8))
    labels = rng.choice(np.array(["A", "B", "C"]), size=n_trials, replace=True)
    U = np.eye(n_trials)

    res = item_cv(
        Gamma=Gamma,
        T_target=labels,
        U=U,
        run_id=run_id,
        mode="classification",
        metric="accuracy",
        ridge=1e-4,
        method="svd",
    )

    chance = 1.0 / 3.0
    assert res.aggregate.mean > (chance - 0.15)
    assert res.aggregate.mean < (chance + 0.15)


def test_item_cv_regression_improves_with_snr(rng: np.random.Generator) -> None:
    n_runs = 5
    trials_per_run = 36
    n_trials = n_runs * trials_per_run
    n_features = 6
    n_targets = 2

    run_id = np.repeat(np.arange(1, n_runs + 1), trials_per_run)
    Gamma = rng.standard_normal((n_trials, n_features))
    W_true = rng.standard_normal((n_features, n_targets))
    signal = Gamma @ W_true
    T_low_snr = signal + rng.standard_normal((n_trials, n_targets)) * 2.0
    T_high_snr = signal + rng.standard_normal((n_trials, n_targets)) * 0.4
    U = np.eye(n_trials)

    res_low = item_cv(
        Gamma=Gamma,
        T_target=T_low_snr,
        U=U,
        run_id=run_id,
        mode="regression",
        metric="correlation",
        ridge=1e-4,
        method="svd",
    )
    res_high = item_cv(
        Gamma=Gamma,
        T_target=T_high_snr,
        U=U,
        run_id=run_id,
        mode="regression",
        metric="correlation",
        ridge=1e-4,
        method="svd",
    )

    assert res_high.aggregate.mean > res_low.aggregate.mean


def test_item_fit_matches_closed_form_identity_u(rng: np.random.Generator) -> None:
    """Oracle test: U=I and ridge=0 should match OLS closed form."""
    n_trials = 50
    n_features = 7
    n_targets = 3

    Gamma = rng.standard_normal((n_trials, n_features))
    T_train = rng.standard_normal((n_trials, n_targets))
    fit = item_fit(Gamma, T_train, np.eye(n_trials), ridge=0.0, method="svd")

    expected = np.linalg.solve(Gamma.T @ Gamma, Gamma.T @ T_train)
    assert_allclose(fit.W_hat, expected, atol=1e-10, rtol=1e-10)


def test_item_fit_matches_ridge_closed_form_identity_u(
    rng: np.random.Generator,
) -> None:
    """Oracle test: ridge path should match analytic ridge solution."""
    n_trials = 60
    n_features = 6
    n_targets = 2
    lam = 1e-3

    Gamma = rng.standard_normal((n_trials, n_features))
    T_train = rng.standard_normal((n_trials, n_targets))
    fit = item_fit(Gamma, T_train, np.eye(n_trials), ridge=lam, method="svd")

    expected = np.linalg.solve(
        Gamma.T @ Gamma + lam * np.eye(n_features),
        Gamma.T @ T_train,
    )
    assert_allclose(fit.W_hat, expected, atol=1e-10, rtol=1e-10)


def test_item_compute_u_trial_permutation_equivariance(
    rng: np.random.Generator,
) -> None:
    """Metamorphic: column permutation should induce matching U permutation."""
    X_t = rng.standard_normal((70, 9))
    ridge = 1e-6
    u_res = item_compute_u(X_t, ridge=ridge)
    assert isinstance(u_res, ItemCovarianceMatrixResult)
    U = u_res.U

    perm = rng.permutation(X_t.shape[1])
    u_perm = item_compute_u(X_t[:, perm], ridge=ridge)
    assert isinstance(u_perm, ItemCovarianceMatrixResult)
    U_perm = u_perm.U

    expected = U[np.ix_(perm, perm)]
    assert_allclose(U_perm, expected, atol=1e-8)


def test_item_cv_trial_permutation_invariance(rng: np.random.Generator) -> None:
    """Metamorphic: consistent trial permutation should preserve CV outputs."""
    n_runs = 5
    trials_per_run = 16
    n_trials = n_runs * trials_per_run

    run_id = np.repeat(np.arange(1, n_runs + 1), trials_per_run)
    Gamma = rng.standard_normal((n_trials, 8))
    T_target = rng.standard_normal((n_trials, 2))
    U = np.eye(n_trials)

    res = item_cv(
        Gamma=Gamma,
        T_target=T_target,
        U=U,
        run_id=run_id,
        mode="regression",
        metric="correlation",
        ridge=1e-4,
        method="svd",
    )

    perm = rng.permutation(n_trials)
    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(n_trials)

    res_perm = item_cv(
        Gamma=Gamma[perm, :],
        T_target=T_target[perm, :],
        U=U[np.ix_(perm, perm)],
        run_id=run_id[perm],
        mode="regression",
        metric="correlation",
        ridge=1e-4,
        method="svd",
    )

    assert res.aggregate.metric == res_perm.aggregate.metric
    assert_allclose(res.aggregate.mean, res_perm.aggregate.mean, atol=1e-12)
    assert_allclose(res.aggregate.sd, res_perm.aggregate.sd, atol=1e-12)

    T_hat_back = np.empty_like(res_perm.predictions.T_hat)
    T_hat_back[perm, :] = res_perm.predictions.T_hat
    assert_allclose(T_hat_back, res.predictions.T_hat, atol=1e-10)


def test_item_cv_run_label_invariance(rng: np.random.Generator) -> None:
    """Metamorphic: relabeling runs should not change decoded predictions."""
    n_runs = 4
    trials_per_run = 18
    n_trials = n_runs * trials_per_run

    run_id = np.repeat(np.arange(1, n_runs + 1), trials_per_run)
    run_id_str = np.asarray([f"run_{r}" for r in run_id])
    Gamma = rng.standard_normal((n_trials, 7))
    T_target = rng.standard_normal((n_trials, 2))
    U = np.eye(n_trials)

    res_num = item_cv(
        Gamma=Gamma,
        T_target=T_target,
        U=U,
        run_id=run_id,
        mode="regression",
        metric="correlation",
        ridge=1e-4,
        method="svd",
    )
    res_str = item_cv(
        Gamma=Gamma,
        T_target=T_target,
        U=U,
        run_id=run_id_str,
        mode="regression",
        metric="correlation",
        ridge=1e-4,
        method="svd",
    )

    assert_allclose(res_num.aggregate.mean, res_str.aggregate.mean, atol=1e-12)
    assert_allclose(res_num.aggregate.sd, res_str.aggregate.sd, atol=1e-12)
    assert_allclose(res_num.predictions.T_hat, res_str.predictions.T_hat, atol=1e-10)


def test_item_compute_u_adversarial_collinearity_and_scale(
    rng: np.random.Generator,
) -> None:
    """Adversarial edge case: collinear design with ill-scaled covariance."""
    X_base = rng.standard_normal((80, 6))
    X_t = np.column_stack([X_base, X_base[:, 0] * (1.0 + 1e-12)])

    # Strongly ill-conditioned covariance matrix.
    diag_vals = np.logspace(-8, 8, num=X_t.shape[0])
    V = np.diag(diag_vals)
    u_res = item_compute_u(X_t, V=V, v_type="cov", ridge=1e-8, method="chol")

    assert isinstance(u_res, ItemCovarianceMatrixResult)
    U = u_res.U
    assert_allclose(U, U.T, atol=1e-8)
    assert np.all(np.isfinite(U))
    assert u_res.diagnostics["solver_path"] in {"chol", "svd", "pinv"}
    assert np.isfinite(u_res.diagnostics["condition_number"]) or np.isinf(
        u_res.diagnostics["condition_number"]
    )


def test_item_from_lsa_matches_direct_lsa_gamma(
    rng: np.random.Generator,
) -> None:
    """Differential test: item_from_lsa Gamma should equal direct lsa output."""
    n_time = 75
    n_trials = 10
    n_vox = 4

    X_t = rng.standard_normal((n_time, n_trials))
    confounds = np.column_stack([np.ones(n_time), np.linspace(-1.0, 1.0, n_time)])
    beta = rng.standard_normal((n_trials, n_vox))
    beta_z = rng.standard_normal((confounds.shape[1], n_vox))
    Y = X_t @ beta + confounds @ beta_z + rng.standard_normal((n_time, n_vox)) * 0.03

    run_id = np.repeat(np.arange(1, 3), n_trials // 2)
    labels = np.array(["A", "B"] * (n_trials // 2))

    bundle = item_from_lsa(
        Y=Y,
        X_t=X_t,
        T_target=labels,
        run_id=run_id,
        confounds=confounds,
        solver="svd",
    )
    direct = lsa_single_trial(Y, X_t, confounds=confounds).betas

    assert bundle.Gamma is not None
    assert_allclose(bundle.Gamma, direct, atol=1e-12)


def test_item_scoring_contracts_for_ties_and_metrics() -> None:
    """Contract test for deterministic ties and metric formulas."""
    T_true_cls = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    T_hat_cls = np.array(
        [
            [0.1, 0.9, 0.9],  # tie between class 2/3 -> choose first max
            [0.5, 0.5, 0.5],  # full tie -> choose class 1
            [0.2, 0.2, 0.6],
        ]
    )
    class_levels = ["A", "B", "C"]

    score_acc = _score_fold(
        T_true=T_true_cls,
        T_hat=T_hat_cls,
        mode="classification",
        metric="accuracy",
        class_levels=class_levels,
    )
    score_bal = _score_fold(
        T_true=T_true_cls,
        T_hat=T_hat_cls,
        mode="classification",
        metric="balanced_accuracy",
        class_levels=class_levels,
    )

    assert score_acc.pred_labels is not None
    assert score_acc.pred_labels.tolist() == ["B", "A", "C"]
    assert_allclose(score_acc.metric, 1.0, atol=1e-12)
    assert_allclose(score_bal.metric, 1.0, atol=1e-12)

    T_true_reg = np.array([[0.0], [1.0], [2.0], [3.0]])
    T_hat_reg = np.array([[0.0], [2.0], [2.0], [4.0]])
    score_rmse = _score_fold(
        T_true=T_true_reg,
        T_hat=T_hat_reg,
        mode="regression",
        metric="rmse",
        class_levels=None,
    )
    assert_allclose(score_rmse.metric, np.sqrt(0.5), atol=1e-12)
