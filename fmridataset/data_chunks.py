"""Compatibility re-exports for typed dataset chunks."""

from __future__ import annotations

import sys

from fmrimod.dataset.data_chunks import (
    ChunkIterator,
    DataChunk,
    collect_chunks,
    data_chunk,
    data_chunks,
    exec_strategy,
    voxel_index_chunks,
)

__all__ = [
    "DataChunk",
    "ChunkIterator",
    "data_chunk",
    "data_chunks",
    "collect_chunks",
    "exec_strategy",
    "voxel_index_chunks",
]

_parent = sys.modules.get(__package__)
if _parent is not None:
    setattr(_parent, "data_chunks", data_chunks)
