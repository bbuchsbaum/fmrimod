"""Contract tests for explicit parity report gate semantics."""

from __future__ import annotations

import json

import numpy as np

from cross_testing.harness import (
    Caveat,
    ParityCase,
    ParityTolerance,
    PipelineOutput,
    render,
    run,
)


def _candidate(_: object) -> PipelineOutput:
    return PipelineOutput(
        arrays={
            "exact": np.array([1.0, 2.0, 3.0]),
            "scaled": np.array([1.0, 2.0, 3.0]),
        }
    )


def _reference(_: object) -> PipelineOutput:
    return PipelineOutput(
        arrays={
            "exact": np.array([1.0, 2.0, 3.0]),
            "scaled": np.array([2.0, 4.0, 6.0]),
        }
    )


def test_report_exposes_gate_and_caveat_scoped_rows(tmp_path):
    case = ParityCase(
        name="gate_contract",
        fmrimod_pipeline=_candidate,
        reference_pipeline=_reference,
        inputs=None,
        tolerances={
            "scaled": ParityTolerance(
                check_allclose=False,
                min_pearson=0.99,
                min_spearman=0.99,
            )
        },
        declared_caveats=(
            Caveat(
                caveat_id="scaled-reference-convention",
                quantity="scaled",
                reason="reference uses a different scale convention",
                expected="rank and correlation are stable",
                link="docs/contracts/CAVEATS.md#scaled-reference-convention",
            ),
        ),
    )

    result = run(case)
    assert result.status == "pass_with_caveats"
    assert result.deltas["exact"].gate == "allclose+pearson+spearman"
    assert result.deltas["scaled"].gate == "pearson+spearman"

    json_path, md_path = render(result, tmp_path)
    payload = json.loads(json_path.read_text())
    assert payload["deltas"]["scaled"]["gate"] == "pearson+spearman"
    assert payload["deltas"]["scaled"]["failed_gates"] == []

    markdown = md_path.read_text()
    assert "| quantity | shape | gate | caveat |" in markdown
    assert "caveat-bypassed:pearson+spearman" in markdown
    assert "scaled-reference-convention" in markdown
