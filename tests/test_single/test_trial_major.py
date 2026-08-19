"""Wave 2: trial-major multi-basis layout and OASIS-only K>1.

Guards fmrilss ``f851fb0``. Cheap pass disqualified: accepting a
caller-supplied already-trial-major matrix only, or silently treating
K columns as independent trials under LSS.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from fmrimod import event_model
from fmrimod.single import (
    OasisConfig,
    canonicalize_trial_major,
    estimate_single_trial,
    estimate_single_trial_from_dataset,
    oasis_single_trial,
    prepare_trialwise_design,
)


def test_canonicalize_trial_major_remaps_basis_major_columns() -> None:
    """A basis-major layout must be rewritten before OASIS K>1.

    Cheap pass: only accepting an already-trial-major caller matrix.
    """
    rng = np.random.default_rng(7)
    n, n_trials, K, v = 80, 4, 2, 3
    # Orthonormal trial-major columns: t1b1, t1b2, t2b1, t2b2, ...
    X_true, _ = np.linalg.qr(rng.standard_normal((n, n_trials * K)))
    true_betas = rng.standard_normal((n_trials * K, v))
    Y = X_true @ true_betas + rng.standard_normal((n, v)) * 0.01

    # Permute to basis-major: all basis-1 columns, then all basis-2 columns.
    basis_major_idx = [
        trial * K + basis for basis in range(K) for trial in range(n_trials)
    ]
    X_basis_major = X_true[:, basis_major_idx]
    trials = [f"trial_{t + 1}" for _basis in range(K) for t in range(n_trials)]
    bases = [b + 1 for b in range(K) for _t in range(n_trials)]

    X_canon, mapping = canonicalize_trial_major(
        X_basis_major, trial=trials, basis=bases
    )
    assert_allclose(X_canon, X_true)
    assert list(mapping["basis"]) == [1, 2] * n_trials
    assert mapping["trial"].tolist() == [
        t for t in range(1, n_trials + 1) for _ in range(K)
    ]

    recovered = oasis_single_trial(
        Y, X_canon, config=OasisConfig(K=K, ridge_mode="none")
    )
    scrambled = oasis_single_trial(
        Y, X_basis_major, config=OasisConfig(K=K, ridge_mode="none")
    )
    assert recovered.betas.shape == (n_trials * K, v)
    assert np.max(np.abs(recovered.betas - true_betas)) < 0.1
    assert np.max(np.abs(scrambled.betas - true_betas)) > 0.2


def test_event_model_spmg3_is_trial_major_and_oasis_only() -> None:
    events = pd.DataFrame(
        {
            "onset": [8.0, 24.0, 40.0, 56.0],
            "duration": [0.0, 0.0, 0.0, 0.0],
        }
    )
    model = event_model(
        "trialwise(basis='spmg3')",
        data=events,
        tr=2.0,
        n_scans=50,
    )
    prepared = prepare_trialwise_design(model)
    assert prepared.K == 3
    assert prepared.n_trials == 4
    assert prepared.X.shape[1] == 12
    assert len(prepared.trial_labels) == 4
    assert any("_b" in name for name in model.column_names)
    # First three output columns belong to trial 1, bases 1..3.
    assert prepared.trial_basis_map["trial"].tolist()[:3] == [1, 1, 1]
    assert prepared.trial_basis_map["basis"].tolist()[:3] == [1, 2, 3]


def test_non_oasis_k_gt_1_raises() -> None:
    """Silently treating K columns as trials is the current bug."""
    rng = np.random.default_rng(3)
    Y = rng.standard_normal((60, 4))
    X = rng.standard_normal((60, 8))
    with pytest.raises(ValueError, match="require method='oasis'"):
        estimate_single_trial(Y, X, method="lss", n_basis=2)
    with pytest.raises(ValueError, match="require method='oasis'"):
        estimate_single_trial(
            Y, X, method="lsa", oasis_config=OasisConfig(K=2, ridge_mode="none")
        )


def test_from_dataset_multi_basis_requires_oasis() -> None:
    rng = np.random.default_rng(4)
    onsets = np.array([8.0, 24.0, 40.0, 56.0])
    events = pd.DataFrame(
        {"onset": onsets, "duration": np.zeros_like(onsets), "run": 1}
    )
    ds = __import__("fmrimod").fmri_dataset(
        rng.standard_normal((50, 3)),
        tr=2.0,
        events=events,
    )
    with pytest.raises(ValueError, match="require method='oasis'"):
        estimate_single_trial_from_dataset(ds, "trialwise(basis='spmg3')", method="lss")
    result = estimate_single_trial_from_dataset(
        ds, "trialwise(basis='spmg3')", method="oasis"
    )
    assert result.betas.shape[0] == 12
    assert result.extra.K == 3
