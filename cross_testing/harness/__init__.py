"""Reusable parity harness for external-reference workflow comparisons."""

from cross_testing.harness.compare import (
    ArrayDelta,
    Caveat,
    ColumnMap,
    ParityCase,
    ParityResult,
    ParityTolerance,
    PipelineOutput,
    run,
)
from cross_testing.harness.report import render

__all__ = [
    "ArrayDelta",
    "Caveat",
    "ColumnMap",
    "ParityCase",
    "ParityResult",
    "ParityTolerance",
    "PipelineOutput",
    "render",
    "run",
]
