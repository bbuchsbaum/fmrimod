"""Tests for dataset protocols, adapters, and chunking."""

import numpy as np
import pytest

from fmrimod.sampling import SamplingFrame
from fmrimod.dataset.protocols import DatasetProtocol
from fmrimod.dataset.adapters.numpy_adapter import NumpyAdapter
from fmrimod.dataset.fmri_dataset import FmriDataset
from fmrimod.dataset.chunking import VoxelChunker, BlockChunker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def single_run_data():
    """Single run: 100 timepoints, 50 voxels."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((100, 50))


@pytest.fixture
def multi_run_data():
    """Two runs: 80 and 120 timepoints, 50 voxels."""
    rng = np.random.default_rng(42)
    return [rng.standard_normal((80, 50)), rng.standard_normal((120, 50))]


@pytest.fixture
def sframe_single():
    return SamplingFrame(blocklens=[100], tr=2.0)


@pytest.fixture
def sframe_multi():
    return SamplingFrame(blocklens=[80, 120], tr=2.0)


# ---------------------------------------------------------------------------
# NumpyAdapter tests
# ---------------------------------------------------------------------------

class TestNumpyAdapter:
    def test_single_run(self, single_run_data, sframe_single):
        adapter = NumpyAdapter(single_run_data, sframe_single)
        assert adapter.n_runs == 1
        assert adapter.n_voxels == 50
        assert adapter.n_timepoints == [100]

    def test_multi_run(self, multi_run_data, sframe_multi):
        adapter = NumpyAdapter(multi_run_data, sframe_multi)
        assert adapter.n_runs == 2
        assert adapter.n_voxels == 50
        assert adapter.n_timepoints == [80, 120]

    def test_get_data(self, multi_run_data, sframe_multi):
        adapter = NumpyAdapter(multi_run_data, sframe_multi)
        d0 = adapter.get_data(0)
        assert d0.shape == (80, 50)
        np.testing.assert_array_equal(d0, multi_run_data[0])

        d1 = adapter.get_data(1)
        assert d1.shape == (120, 50)

    def test_get_data_out_of_range(self, single_run_data, sframe_single):
        adapter = NumpyAdapter(single_run_data, sframe_single)
        with pytest.raises(IndexError):
            adapter.get_data(1)

    def test_mask_default(self, single_run_data, sframe_single):
        adapter = NumpyAdapter(single_run_data, sframe_single)
        mask = adapter.get_mask()
        assert mask.dtype == bool
        assert np.all(mask)

    def test_mask_custom(self, single_run_data, sframe_single):
        mask = np.ones((50, 1, 1), dtype=bool)
        mask[0] = False
        adapter = NumpyAdapter(single_run_data, sframe_single, mask=mask)
        np.testing.assert_array_equal(adapter.get_mask(), mask)

    def test_mismatched_timepoints(self, sframe_single):
        bad_data = np.zeros((50, 50))  # 50 timepoints but frame expects 100
        with pytest.raises(ValueError, match="timepoints"):
            NumpyAdapter(bad_data, sframe_single)

    def test_mismatched_run_count(self, sframe_multi):
        data = [np.zeros((80, 50))]  # 1 array but frame has 2 blocks
        with pytest.raises(ValueError, match="does not match"):
            NumpyAdapter(data, sframe_multi)

    def test_satisfies_protocol(self, single_run_data, sframe_single):
        adapter = NumpyAdapter(single_run_data, sframe_single)
        assert isinstance(adapter, DatasetProtocol)


# ---------------------------------------------------------------------------
# FmriDataset tests
# ---------------------------------------------------------------------------

class TestFmriDataset:
    def test_basic(self, multi_run_data, sframe_multi):
        adapter = NumpyAdapter(multi_run_data, sframe_multi)
        ds = FmriDataset(adapter)
        assert ds.n_runs == 2
        assert ds.n_voxels == 50
        assert repr(ds).startswith("FmriDataset")

    def test_get_all_data(self, multi_run_data, sframe_multi):
        adapter = NumpyAdapter(multi_run_data, sframe_multi)
        ds = FmriDataset(adapter)
        all_data = ds.get_all_data()
        assert all_data.shape == (200, 50)

    def test_censor_concat(self, multi_run_data, sframe_multi):
        adapter = NumpyAdapter(multi_run_data, sframe_multi)
        censor = np.zeros(200, dtype=bool)
        censor[0] = True
        ds = FmriDataset(adapter, censor=censor)
        assert ds.censor is not None
        assert len(ds.censor) == 2
        assert ds.censor[0][0] == True
        assert np.sum(ds.censor[0]) == 1
        assert np.sum(ds.censor[1]) == 0

    def test_censor_per_run(self, multi_run_data, sframe_multi):
        adapter = NumpyAdapter(multi_run_data, sframe_multi)
        c0 = np.zeros(80, dtype=bool)
        c1 = np.zeros(120, dtype=bool)
        c1[-1] = True
        ds = FmriDataset(adapter, censor=[c0, c1])
        assert ds.get_censor(0) is not None
        assert np.sum(ds.get_censor(1)) == 1

    def test_censor_bad_length(self, multi_run_data, sframe_multi):
        adapter = NumpyAdapter(multi_run_data, sframe_multi)
        with pytest.raises(ValueError, match="length"):
            FmriDataset(adapter, censor=np.zeros(50, dtype=bool))

    def test_event_table(self, single_run_data, sframe_single):
        import pandas as pd
        adapter = NumpyAdapter(single_run_data, sframe_single)
        events = pd.DataFrame({"onset": [1.0, 5.0], "condition": ["A", "B"]})
        ds = FmriDataset(adapter, event_table=events)
        assert ds.event_table is not None
        assert len(ds.event_table) == 2


# ---------------------------------------------------------------------------
# Chunking tests
# ---------------------------------------------------------------------------

class TestVoxelChunker:
    def test_basic(self):
        chunker = VoxelChunker(100, chunk_size=30)
        assert chunker.n_chunks == 4  # ceil(100/30)

    def test_iter(self):
        data = np.arange(200).reshape(10, 20).astype(float)
        chunker = VoxelChunker(20, chunk_size=7)
        chunks = list(chunker.iter_chunks(data))
        assert len(chunks) == 3
        assert chunks[0][0].shape == (10, 7)
        assert chunks[2][0].shape == (10, 6)
        np.testing.assert_array_equal(chunks[0][1], np.arange(7))


class TestBlockChunker:
    def test_basic(self):
        labels = np.array([0, 0, 1, 1, 1, 2])
        chunker = BlockChunker(labels)
        assert chunker.n_chunks == 3

    def test_iter(self):
        data = np.arange(30).reshape(5, 6).astype(float)
        labels = np.array([0, 0, 1, 1, 1, 2])
        chunker = BlockChunker(labels)
        chunks = list(chunker.iter_chunks(data))
        assert chunks[0][0].shape == (5, 2)
        assert chunks[1][0].shape == (5, 3)
        assert chunks[2][0].shape == (5, 1)
