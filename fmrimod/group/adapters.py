"""Adapter protocol for native group-analysis data sources."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import pandas as pd
from numpy.typing import NDArray

from .space import GroupSpace


@dataclass(frozen=True)
class GroupProbe:
    """Semantic probe result for a group-analysis data source."""

    assays: tuple[str, ...]
    shape: tuple[int, int, int]
    subjects: tuple[str, ...]
    contrasts: tuple[str, ...]
    space: GroupSpace
    metadata: Mapping[str, object] = field(default_factory=dict)
    columns: Mapping[str, object] = field(default_factory=dict)
    col_data: pd.DataFrame | None = None
    row_data: pd.DataFrame | None = None
    contrast_data: pd.DataFrame | None = None


class GroupAdapter(Protocol):
    """Protocol implemented by concrete group-analysis source adapters."""

    def probe(self) -> GroupProbe:
        """Return axis, assay, space, and metadata without full materialization."""

    def read(
        self,
        assays: Sequence[str] | None = None,
        *,
        sample: Sequence[int] | NDArray[Any] | None = None,
        subject: Sequence[int] | NDArray[Any] | None = None,
        contrast: Sequence[int] | NDArray[Any] | None = None,
    ) -> Mapping[str, NDArray[Any]]:
        """Read assay arrays, optionally restricted by axis indices."""

