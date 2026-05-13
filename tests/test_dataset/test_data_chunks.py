"""Tests for canonical typed dataset chunks."""

from __future__ import annotations

import numpy as np
import pytest

import fmrimod
from fmridataset import (
    DataChunk as FacadeDataChunk,
)
from fmridataset import (
    collect_chunks as facade_collect_chunks,
)
from fmridataset import (
    data_chunks as facade_data_chunks,
)
from fmridataset import (
    exec_strategy as facade_exec_strategy,
)
from fmridataset import (
    voxel_index_chunks as facade_voxel_index_chunks,
)
from fmrimod.dataset import (
    ChunkIterator,
    DataChunk,
    collect_chunks,
    data_chunk,
    data_chunks,
    exec_strategy,
    voxel_index_chunks,
)


def test_data_chunk_constructor_and_repr() -> None:
    chunk = data_chunk(
        np.ones((2, 3)),
        voxel_ind=np.array([0, 2, 4]),
        row_ind=np.array([1, 3]),
        chunk_num=2,
    )

    assert isinstance(chunk, DataChunk)
    assert "DataChunk" in repr(chunk)
    assert chunk.chunk_num == 2


def test_data_chunks_split_voxels_with_data_and_indices() -> None:
    mat = np.arange(40, dtype=float).reshape(10, 4)
    ds = fmrimod.matrix_dataset(mat, tr=2.0)
    iterator = data_chunks(ds, nchunks=2)

    assert isinstance(iterator, ChunkIterator)
    chunks = collect_chunks(iterator)
    assert [chunk.chunk_num for chunk in chunks] == [1, 2]
    np.testing.assert_array_equal(chunks[0].data, mat[:, [0, 1]])
    np.testing.assert_array_equal(chunks[0].voxel_ind, np.array([0, 1]))
    np.testing.assert_array_equal(chunks[0].row_ind, np.arange(10))
    np.testing.assert_array_equal(chunks[1].data, mat[:, [2, 3]])


def test_data_chunks_can_iterate_runwise() -> None:
    mat = np.arange(40, dtype=float).reshape(10, 4)
    ds = fmrimod.matrix_dataset(mat, tr=2.0, run_length=[4, 6])

    chunks = list(data_chunks(ds, runwise=True))
    assert len(chunks) == 2
    np.testing.assert_array_equal(chunks[0].data, mat[:4])
    np.testing.assert_array_equal(chunks[0].row_ind, np.arange(4))
    np.testing.assert_array_equal(chunks[1].data, mat[4:])
    np.testing.assert_array_equal(chunks[1].row_ind, np.arange(4, 10))


def test_exec_strategy_and_legacy_index_helper_are_distinct() -> None:
    mat = np.arange(30, dtype=float).reshape(10, 3)
    ds = fmrimod.matrix_dataset(mat, tr=2.0)

    voxelwise = list(exec_strategy("voxelwise")(ds))
    assert len(voxelwise) == 3
    assert all(isinstance(chunk, DataChunk) for chunk in voxelwise)

    index_chunks = voxel_index_chunks(ds, nchunks=2)
    assert [chunk.tolist() for chunk in index_chunks] == [[0, 1], [2]]
    assert not isinstance(index_chunks[0], DataChunk)

    with pytest.raises(ValueError, match="nchunks is required"):
        exec_strategy("chunkwise")


def test_fmridataset_chunk_facade_identity() -> None:
    assert FacadeDataChunk is DataChunk
    assert facade_data_chunks is data_chunks
    assert facade_collect_chunks is collect_chunks
    assert facade_exec_strategy is exec_strategy
    assert facade_voxel_index_chunks is voxel_index_chunks
