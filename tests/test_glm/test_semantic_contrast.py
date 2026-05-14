"""Tests for authored semantic contrasts resolved through fit.contrast."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fmrimod.contrast import DesignProvenanceError, cell, condition
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


def _declared_factorial_column(
    name: str,
    index: int,
    *,
    level: str,
    term: str = "task:valence",
) -> DesignColumn:
    return DesignColumn(
        name=name,
        index=index,
        role="task",
        model_source="matrix-fixture",
        term=term,
        term_tag=term.replace(":", "_"),
        condition=level,
        level=level,
        basis_ix=1,
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


def test_one_sided_semantic_contrast_resolves_declared_level() -> None:
    fit = _fit_with_order(("intercept", "gain", "loss"))
    contrast = condition("gain", term="trial_type")

    from fmrimod.contrast import SemanticContrast

    result = fit.contrast(SemanticContrast(contrast, name="gain"))

    explanation = result.explain().to_dict()
    assert result.touched_columns == ("gain",)
    assert explanation["intent"]["kind"] == "semantic_contrast"
    assert explanation["intent"]["positive"]["level"] == "gain"
    assert explanation["intent"]["negative"] is None
    assert explanation["intent"]["levels"] == ["gain"]
    assert explanation["intent"]["weights"] == ((0.0, 1.0, 0.0),)


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


def test_factorial_cell_linear_contrast_resolves_by_declared_levels() -> None:
    columns = DesignColumns(
        (
            _declared_factorial_column(
                "task_valence_task.encode_valence.emotional",
                0,
                level="task.encode_valence.emotional",
            ),
            _declared_factorial_column(
                "task_valence_task.encode_valence.neutral",
                1,
                level="task.encode_valence.neutral",
            ),
            _declared_factorial_column(
                "task_valence_task.recall_valence.emotional",
                2,
                level="task.recall_valence.emotional",
            ),
            _declared_factorial_column(
                "task_valence_task.recall_valence.neutral",
                3,
                level="task.recall_valence.neutral",
            ),
            DesignColumn(
                name="intercept",
                index=4,
                role="baseline",
                model_source="matrix-fixture",
            ),
        )
    )

    def task_cell(task: str, valence: str):
        return cell("task:valence", task=task, valence=valence)

    contrast = 0.5 * (
        task_cell("recall", "emotional")
        + task_cell("recall", "neutral")
        - task_cell("encode", "emotional")
        - task_cell("encode", "neutral")
    )

    np.testing.assert_allclose(
        contrast.resolve(columns),
        [-0.5, -0.5, 0.5, 0.5, 0.0],
    )
    assert contrast.intent()["kind"] == "semantic_linear_contrast"
    assert contrast.intent()["term"] == "task:valence"


def test_fit_contrast_accepts_factorial_cell_arithmetic() -> None:
    rng = np.random.default_rng(20260514)
    n = 96
    design = pd.DataFrame(
        {
            "task_valence_task.encode_valence.emotional": rng.normal(size=n),
            "task_valence_task.encode_valence.neutral": rng.normal(size=n),
            "task_valence_task.recall_valence.emotional": rng.normal(size=n),
            "task_valence_task.recall_valence.neutral": rng.normal(size=n),
            "intercept": np.ones(n),
        }
    )
    beta = np.array([[0.2], [0.1], [0.8], [0.5], [1.0]])
    y = design.to_numpy(dtype=np.float64) @ beta
    y = y + rng.normal(scale=0.05, size=(n, 4))
    levels = (
        "task.encode_valence.emotional",
        "task.encode_valence.neutral",
        "task.recall_valence.emotional",
        "task.recall_valence.neutral",
    )
    columns = DesignColumns(
        tuple(
            _declared_factorial_column(name, index, level=level)
            for index, (name, level) in enumerate(zip(design.columns[:4], levels))
        )
        + (
            DesignColumn(
                name="intercept",
                index=4,
                role="baseline",
                model_source="matrix-fixture",
            ),
        )
    )
    fit = fit_glm_from_matrix(design, y, model=_DeclaredMatrixSource(design, columns))

    def task_cell(task: str, valence: str):
        return cell("task:valence", task=task, valence=valence)

    authored = 0.5 * (
        task_cell("recall", "emotional")
        + task_cell("recall", "neutral")
        - task_cell("encode", "emotional")
        - task_cell("encode", "neutral")
    )
    raw = fit.contrast(np.array([-0.5, -0.5, 0.5, 0.5, 0.0]), name="raw_task")
    semantic = fit.contrast(authored, name="task_main")

    np.testing.assert_allclose(semantic.stat, raw.stat, atol=1e-12)
    np.testing.assert_allclose(semantic.estimate, raw.estimate, atol=1e-12)
    assert semantic.intent["kind"] == "semantic_linear_contrast"
    assert semantic.touched_columns == tuple(design.columns[:4])
