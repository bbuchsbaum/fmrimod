"""Trial-major design preparation for single-trial estimation.

Mirrors R ``fmrilss::.prepare_fmridesign_lss``: trialwise columns are
canonicalized to trial-major / basis-within-trial order, other event
terms become fixed regressors, and ``K > 1`` is an OASIS-only contract.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

_BASIS_SUFFIX = re.compile(r"_b(\d+)$")


@dataclass(frozen=True)
class PreparedTrialDesign:
    """Canonical trialwise target extracted from an EventModel."""

    X: NDArray[np.float64]
    fixed: NDArray[np.float64] | None
    K: int
    n_trials: int
    trial_labels: list[str]
    column_names: list[str]
    trial_basis_map: pd.DataFrame


def canonicalize_trial_major(
    X: NDArray[np.float64],
    *,
    trial: Sequence[Any],
    basis: Sequence[int],
) -> tuple[NDArray[np.float64], pd.DataFrame]:
    """Reorder columns to trial-major, basis-within-trial order.

    Parameters
    ----------
    X
        Design matrix whose columns are described by ``trial`` / ``basis``.
    trial
        Trial identity for each column (any hashable).
    basis
        1-based basis index for each column.

    Returns
    -------
    X_out, map
        Trial-major matrix and a map with ``trial``, ``basis``,
        ``source_column``, and ``output_name``.
    """

    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be a 2-D matrix")
    trial_ids = list(trial)
    basis_ids = [int(b) for b in basis]
    if len(trial_ids) != X.shape[1] or len(basis_ids) != X.shape[1]:
        raise ValueError("trial and basis must have one entry per X column")
    if any(b < 1 for b in basis_ids):
        raise ValueError("basis indices must be 1-based positive integers")

    trial_levels: list[Any] = []
    seen: set[Any] = set()
    for tid in trial_ids:
        if tid not in seen:
            trial_levels.append(tid)
            seen.add(tid)
    K = max(basis_ids)
    expected = [(tid, b) for tid in trial_levels for b in range(1, K + 1)]
    current = list(zip(trial_ids, basis_ids, strict=True))
    if len(set(current)) != len(current):
        raise ValueError(
            "Trial/basis columns do not form a complete rectangular mapping"
        )
    try:
        order = [current.index(key) for key in expected]
    except ValueError as exc:
        raise ValueError(
            "Trial/basis columns do not form a complete rectangular mapping"
        ) from exc
    if len(order) != X.shape[1]:
        raise ValueError(
            "Trial/basis columns do not form a complete rectangular mapping"
        )

    X_out = np.ascontiguousarray(X[:, order])
    names = [str(tid) if K == 1 else f"{tid}:basis_{b}" for tid, b in expected]
    mapping = pd.DataFrame(
        {
            "output_row": np.arange(len(names), dtype=int),
            "column": names,
            "trial": np.repeat(np.arange(1, len(trial_levels) + 1), K),
            "trial_name": [tid for tid, _ in expected],
            "basis": [b for _, b in expected],
            "source_column": order,
            "output_name": names,
        }
    )
    return X_out, mapping


def prepare_trialwise_design(event_model: object) -> PreparedTrialDesign:
    """Extract trialwise columns from an EventModel in trial-major order.

    Non-trialwise event terms are returned as ``fixed``. Multi-basis
    identity is inferred from column names (``_bN``), ``column_facts``,
    or the trialwise term's HRF ``nbasis``.
    """

    terms = getattr(event_model, "terms", None)
    if not terms:
        raise ValueError("event_model has no terms")
    column_indices = getattr(event_model, "column_indices", None)
    column_names = getattr(event_model, "column_names", None)
    if column_indices is None or column_names is None:
        raise ValueError("event_model lacks column metadata")

    trialwise_terms = [term for term in terms if getattr(term, "_is_trialwise", False)]
    if len(trialwise_terms) != 1:
        raise ValueError(
            "prepare_trialwise_design() requires exactly one trialwise event term"
        )
    trial_term = trialwise_terms[0]
    term_name = getattr(trial_term, "name", None)
    target_idx = list(column_indices.get(term_name, []))
    if not target_idx:
        raise ValueError("No columns were mapped to the trialwise event term")

    X_all = np.ascontiguousarray(
        np.asarray(event_model.design_matrix, dtype=np.float64)
    )
    target_names = [str(column_names[i]) for i in target_idx]
    facts = _facts_for_indices(event_model, target_idx)
    trial_ids, basis_ids, K = _infer_trial_basis(target_names, facts, trial_term)

    X_target = X_all[:, target_idx]
    X, mapping = canonicalize_trial_major(X_target, trial=trial_ids, basis=basis_ids)
    mapping = mapping.copy()
    mapping["source_column"] = [target_idx[int(i)] for i in mapping["source_column"]]

    fixed_idx = [i for i in range(X_all.shape[1]) if i not in set(target_idx)]
    fixed = X_all[:, fixed_idx] if fixed_idx else None
    n_trials = int(X.shape[1] // K)
    trial_labels = [str(name) for name in mapping["trial_name"].iloc[::K].tolist()]
    return PreparedTrialDesign(
        X=X,
        fixed=fixed,
        K=K,
        n_trials=n_trials,
        trial_labels=trial_labels,
        column_names=list(mapping["output_name"]),
        trial_basis_map=mapping,
    )


def _facts_for_indices(
    event_model: object, indices: Sequence[int]
) -> list[dict[str, Any]]:
    facts = getattr(event_model, "column_facts", None)
    if not facts:
        return []
    by_index = {int(row.get("index", -1)): row for row in facts}
    return [by_index[i] for i in indices if i in by_index]


def _infer_trial_basis(
    names: list[str],
    facts: list[dict[str, Any]],
    trial_term: object,
) -> tuple[list[Any], list[int], int]:
    suffix_basis = [_basis_from_name(name) for name in names]
    has_suffix = [b is not None for b in suffix_basis]
    if any(has_suffix) and not all(has_suffix):
        raise ValueError("Inconsistent basis suffixes in trialwise columns")

    expected_K = _hrf_nbasis(trial_term)
    if all(has_suffix):
        basis_ids = [int(b) for b in suffix_basis]
        trial_ids = [_BASIS_SUFFIX.sub("", name) for name in names]
        K = max(basis_ids)
        if expected_K > 1 and expected_K != K:
            raise ValueError(
                "Trialwise basis names disagree with the event-term HRF metadata"
            )
        return trial_ids, basis_ids, K

    fact_basis = [
        int(row["basis_ix"]) for row in facts if row.get("basis_ix") is not None
    ]
    if len(fact_basis) == len(names) and max(fact_basis) > 1:
        K = max(fact_basis)
        trial_ids = [row.get("condition", names[i]) for i, row in enumerate(facts)]
        return trial_ids, fact_basis, K

    K = expected_K if expected_K > 1 else 1
    if K > 1:
        if len(names) % K != 0:
            raise ValueError(f"ncol(X)={len(names)} is not divisible by inferred K={K}")
        n_trials = len(names) // K
        trial_ids = [names[i * K] for i in range(n_trials) for _ in range(K)]
        basis_ids = [b for _ in range(n_trials) for b in range(1, K + 1)]
        return trial_ids, basis_ids, K

    return names, [1] * len(names), 1


def _basis_from_name(name: str) -> int | None:
    match = _BASIS_SUFFIX.search(name)
    return int(match.group(1)) if match else None


def _hrf_nbasis(term: object) -> int:
    hrf = getattr(term, "hrf", None)
    if hrf is None:
        return 1
    nbasis = getattr(hrf, "nbasis", None)
    if nbasis is not None:
        try:
            return max(int(nbasis), 1)
        except (TypeError, ValueError):
            return 1
    if isinstance(hrf, str):
        lowered = hrf.lower()
        if lowered in {"spmg3", "spm_with_dispersion"}:
            return 3
        if lowered in {"spmg2", "spm_with_derivative"}:
            return 2
    return 1
