"""Tests for authored semantic contrasts resolved through fit.contrast."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fmrimod.contrast import DesignProvenanceError, condition
from fmrimod.design import DesignColumn, DesignColumns
from fmrimod.glm.matrix import fit_glm_from_matrix


def _declared_column(
    name: str,
    index: int,
    *,
    level: str,
    basis_ix: int = 1,
) -> DesignColumn:
    return DesignColumn(
        name=name,
        index=index,
        role="task",
        model_source="matrix-fixture",
        term="trial_type",
        term_tag="trial_type",
        condition=f"trial_type.{level}",
        level=level,
        basis_ix=basis_ix,
        basis_name="identity",
        basis_total=1,
        provenance={
            "role": "declared",
            "term": "declared",
            "condition": "declared",
            "level": "declared",
            "basis_ix": "declared",
            "basis_name": "derived",
            "basis_total": "derived",
        },
    )


def _weak_column(name: str, index: int, *, level: str) -> DesignColumn:
    return DesignColumn(
        name=name,
        index=index,
        role="matrix",
        model_source="matrix-fixture",
        term="trial_type",
        term_tag="trial_type",
        condition=f"trial_type.{level}",
        level=level,
        provenance={
            "role": "inferred",
            "term": "inferred",
            "condition": "inferred",
            "level": "inferred",
        },
    )


class _DeclaredMatrixSource:
    def __init__(self, design: pd.DataFrame, columns: DesignColumns) -> None:
        self._design = design
        self._columns = columns

    def design_matrix(self) -> pd.DataFrame:
        return self._design

    def design_columns(self) -> DesignColumns:
        return self._columns


def _fit_with_order(order: tuple[str, ...]):
    rng = np.random.default_rng(44)
    n = 90
    base = {
        "intercept": np.ones(n),
        "gain": np.linspace(-1.0, 1.0, n),
        "loss": np.cos(np.linspace(0.0, 2.0 * np.pi, n)),
    }
    design = pd.DataFrame({name: base[name] for name in order})
    beta_lookup = {"intercept": 0.4, "gain": 1.5, "loss": -0.75}
    beta = np.array([[beta_lookup[name]] for name in order], dtype=np.float64)
    y = design.to_numpy(dtype=np.float64) @ beta
    y = y + rng.normal(scale=0.02, size=(n, 3))
    columns = DesignColumns(
        tuple(_declared_column(name, idx, level=name) for idx, name in enumerate(order))
    )
    source = _DeclaredMatrixSource(design, columns)
    return fit_glm_from_matrix(design, y, model=source)


def test_authored_condition_contrast_survives_column_permutation() -> None:
    canonical = _fit_with_order(("intercept", "gain", "loss"))
    permuted = _fit_with_order(("loss", "intercept", "gain"))
    contrast = condition("gain", term="trial_type") - condition(
        "loss",
        term="trial_type",
    )

    canonical_result = canonical.contrast(contrast)
    permuted_result = permuted.contrast(contrast)

    np.testing.assert_allclose(canonical_result.estimate, permuted_result.estimate)
    np.testing.assert_allclose(canonical_result.stat, permuted_result.stat)
    assert canonical_result.touched_columns == ("gain", "loss")
    assert permuted_result.touched_columns == ("loss", "gain")

    explanation = permuted_result.explain().to_dict()
    assert explanation["intent"]["kind"] == "semantic_contrast"
    assert explanation["intent"]["positive"]["level"] == "gain"
    assert explanation["intent"]["negative"]["level"] == "loss"
    assert explanation["intent"]["term"] == "trial_type"
    assert [column["index"] for column in explanation["design_columns"]] == [0, 2]


def test_authored_condition_contrast_refuses_weak_level_provenance() -> None:
    columns = DesignColumns(
        (
            _weak_column("gain", 0, level="gain"),
            _weak_column("loss", 1, level="loss"),
        )
    )
    contrast = condition("gain", term="trial_type") - condition(
        "loss",
        term="trial_type",
    )

    with pytest.raises(DesignProvenanceError, match="weaker than declared"):
        contrast.resolve(columns)


def test_authored_condition_contrast_refuses_implicit_multi_basis_average() -> None:
    columns = DesignColumns(
        (
            _declared_column("gain_b1", 0, level="gain"),
            _declared_column("gain_b2", 1, level="gain", basis_ix=2),
            _declared_column("loss_b1", 2, level="loss"),
        )
    )
    contrast = condition("gain", term="trial_type") - condition(
        "loss",
        term="trial_type",
    )

    with pytest.raises(DesignProvenanceError, match="ambiguous across basis"):
        contrast.resolve(columns)

    resolved = (
        condition("gain", term="trial_type", basis_ix=1)
        - condition("loss", term="trial_type", basis_ix=1)
    ).resolve(columns)
    np.testing.assert_allclose(resolved, [1.0, 0.0, -1.0])
