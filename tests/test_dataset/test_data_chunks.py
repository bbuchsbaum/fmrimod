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


def test_data_chunks_voxel_ind_is_linear_index_on_3d_mask() -> None:
    """``voxel_ind`` is the in-mask voxel's linear index, not its x index.

    R legacy ``3ae565e:R/data_chunks.R`` uses ``which(mask > 0)``. The voxels
    sit at (1, 1, 1) and (2, 0, 1), whose x indices ``[1, 2]`` differ from
    their linear indices ``[7, 9]``.
    """
    ni = pytest.importorskip("neuroim")
    shape = (3, 2, 2)
    mask = np.zeros(shape, dtype=bool)
    mask[1, 1, 1] = True
    mask[2, 0, 1] = True
    space = ni.NeuroSpace(dim=shape + (5,), spacing=(1, 1, 1, 1), origin=(0, 0, 0, 0))
    vec = ni.DenseNeuroVec(np.random.default_rng(0).normal(size=shape + (5,)), space)
    ds = fmrimod.dataset.fmri_dataset(vec, mask=mask, tr=2.0)

    chunk = data_chunks(ds, nchunks=1).collect()[0]
    np.testing.assert_array_equal(chunk.voxel_ind, np.flatnonzero(mask))
    np.testing.assert_array_equal(chunk.voxel_ind, [7, 9])
    np.testing.assert_array_equal(
        chunk.data, ds.get_data_matrix()[:, [0, 1]]
    )
    runwise = data_chunks(ds, runwise=True).collect()[0]
    np.testing.assert_array_equal(runwise.voxel_ind, [7, 9])


def test_data_chunks_on_an_empty_mask_plans_zero_chunks() -> None:
    # fmridataset a814330: an emptied feature axis plans zero blocks instead of
    # crashing. Inventory v2 DS-12. nchunks must report 0, so catching the
    # error inside collect() cannot pass.
    import warnings

    from fmrimod.dataset import data_chunks, matrix_dataset

    ds = matrix_dataset(np.zeros((6, 3)), tr=2.0, mask=np.zeros(3, dtype=bool))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        chunks = data_chunks(ds, nchunks=2)
    assert chunks.nchunks == 0
    assert chunks.collect() == []
