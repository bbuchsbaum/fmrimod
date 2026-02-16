"""Tests for SamplingFrame class."""

import numpy as np
import pytest

from fmrimod.sampling import SamplingFrame


class TestSamplingFrameBasic:
    """Test basic SamplingFrame functionality."""
    
    def test_single_block_scalar_inputs(self):
        """Test SamplingFrame with single block and scalar inputs."""
        sf = SamplingFrame(blocklens=100, tr=2.0)
        
        assert sf.n_blocks == 1
        assert sf.n_scans == 100
        assert np.array_equal(sf.blocklens, [100])
        assert np.array_equal(sf.tr, [2.0])
        assert np.array_equal(sf.start_time, [1.0])  # Default is TR/2
        assert sf.precision == 0.1
    
    def test_single_block_with_start_time(self):
        """Test single block with custom start time."""
        sf = SamplingFrame(blocklens=50, tr=1.5, start_time=10.0)
        
        assert sf.n_scans == 50
        assert np.array_equal(sf.start_time, [10.0])
        
        # Check samples
        expected_samples = 10.0 + np.arange(50) * 1.5
        assert np.allclose(sf.samples, expected_samples)
    
    def test_multi_block_same_tr(self):
        """Test multiple blocks with same TR."""
        sf = SamplingFrame(blocklens=[50, 75, 100], tr=2.0)
        
        assert sf.n_blocks == 3
        assert sf.n_scans == 225
        assert np.array_equal(sf.blocklens, [50, 75, 100])
        assert np.array_equal(sf.tr, [2.0, 2.0, 2.0])
        assert np.array_equal(sf.start_time, [1.0, 1.0, 1.0])  # Default is TR/2
    
    def test_multi_block_different_tr(self):
        """Test multiple blocks with different TRs."""
        sf = SamplingFrame(
            blocklens=[50, 75, 100],
            tr=[2.0, 2.0, 1.5],
            start_time=[0.0, 110.0, 280.0]
        )
        
        assert sf.n_blocks == 3
        assert sf.n_scans == 225
        assert np.array_equal(sf.tr, [2.0, 2.0, 1.5])
        
        # Check block IDs
        blockids = sf.blockids
        assert len(blockids) == 225
        assert np.sum(blockids == 0) == 50
        assert np.sum(blockids == 1) == 75
        assert np.sum(blockids == 2) == 100

    def test_legacy_tr_keyword_alias(self):
        """Test constructor with legacy ``TR`` keyword alias."""
        sf = SamplingFrame(blocklens=[20], TR=2.0)

        assert np.array_equal(sf.blocklens, [20])
        assert np.array_equal(sf.tr, [2.0])
        assert np.array_equal(sf.start_time, [1.0])


    def test_conflicting_tr_and_TR_raise(self):
        """Reject ambiguous constructor calls with conflicting TR aliases."""
        with pytest.raises(TypeError, match="different values"):
            SamplingFrame(blocklens=20, tr=2.0, TR=3.0)

    def test_matching_tr_and_TR_ndarrays_allowed(self):
        """Allow matching ndarray aliases for ``tr`` and ``TR``."""
        sf = SamplingFrame(
            blocklens=[10, 15],
            tr=np.array([2.0, 3.0]),
            TR=np.array([2.0, 3.0]),
        )
        np.testing.assert_allclose(sf.tr, [2.0, 3.0])

    def test_conflicting_tr_and_TR_ndarrays_raise(self):
        """Reject conflicting ndarray aliases for ``tr`` and ``TR``."""
        with pytest.raises(TypeError, match="different values"):
            SamplingFrame(
                blocklens=[10, 15],
                tr=np.array([2.0, 3.0]),
                TR=np.array([2.0, 4.0]),
            )


class TestSamplingFrameProperties:
    """Test SamplingFrame property methods."""
    
    def test_blockids(self, sampling_frame_multi_block):
        """Test blockids property."""
        sf = sampling_frame_multi_block
        blockids = sf.blockids
        
        assert len(blockids) == sf.n_scans
        assert blockids.dtype == np.int32
        
        # Check correct assignment
        assert np.all(blockids[:50] == 0)
        assert np.all(blockids[50:125] == 1)
        assert np.all(blockids[125:] == 2)
    
    def test_samples(self, sampling_frame_multi_block):
        """Test samples property."""
        sf = sampling_frame_multi_block
        samples = sf.samples
        
        assert len(samples) == sf.n_scans
        assert samples.dtype == np.float64
        
        # Check first block (50 scans, TR=2.0, start=0.0)
        block0_expected = 0.0 + np.arange(50) * 2.0
        assert np.allclose(samples[:50], block0_expected)
        
        # Check second block (75 scans, TR=2.0, start=110.0)
        # Cumulative time from block 1: 50 * 2.0 = 100.0
        block1_expected = 100.0 + 110.0 + np.arange(75) * 2.0
        assert np.allclose(samples[50:125], block1_expected)
        
        # Check third block (100 scans, TR=1.5, start=280.0)
        # Cumulative time from blocks 1 & 2: 100.0 + 150.0 = 250.0
        block2_expected = 250.0 + 280.0 + np.arange(100) * 1.5
        assert np.allclose(samples[125:], block2_expected)

    def test_sample_times_with_block_filter(self, sampling_frame_multi_block):
        """sample_times should use strict 0-based block IDs."""
        sf = sampling_frame_multi_block

        block0 = sf.sample_times(global_time=True, blockids=[0])
        assert np.allclose(block0, np.arange(50) * 2.0)

        block2_relative = sf.sample_times(global_time=False, blockids=[2])
        assert np.allclose(block2_relative, 280.0 + np.arange(100) * 1.5)

        with pytest.raises(ValueError, match="must be in range"):
            sf.sample_times(global_time=True, blockids=[3])

        with pytest.raises(ValueError, match="must be in range"):
            sf.sample_times(global_time=True, blockids=[-1])
    
    def test_acquisition_onsets(self, sampling_frame_multi_block):
        """Test acquisition_onsets property."""
        sf = sampling_frame_multi_block
        acq_onsets = sf.acquisition_onsets

        assert len(acq_onsets) == sf.n_scans

        # acquisition_onsets returns global times (cumulative + start + scan*TR)
        # Block 0: cum=0, start=0.0 → 0 + np.arange(50)*2.0
        assert np.allclose(acq_onsets[:50], 0.0 + np.arange(50) * 2.0)
        # Block 1: cum=100, start=110.0 → 210 + np.arange(75)*2.0
        assert np.allclose(acq_onsets[50:125], 210.0 + np.arange(75) * 2.0)
        # Block 2: cum=250, start=280.0 → 530 + np.arange(100)*1.5
        assert np.allclose(acq_onsets[125:], 530.0 + np.arange(100) * 1.5)
    
    def test_global_onsets(self, sampling_frame_multi_block):
        """Test global_onsets method for converting block-relative to global times."""
        sf = sampling_frame_multi_block

        # Test single onset in first block
        onsets = [5.0]
        blockids = [0]
        global_times = sf.global_onsets(onsets, blockids)
        assert np.isclose(global_times[0], 5.0)

        # Test single onset in second block
        # Block 1 duration: 50 scans * 2.0 TR = 100.0 seconds
        onsets = [3.0]
        blockids = [1]
        global_times = sf.global_onsets(onsets, blockids)
        expected = 3.0 + 100.0
        assert np.isclose(global_times[0], expected)

        # Test single onset in third block
        # Block 1-2 duration: (50 * 2.0) + (75 * 2.0) = 100.0 + 150.0 = 250.0 seconds
        onsets = [7.5]
        blockids = [2]
        global_times = sf.global_onsets(onsets, blockids)
        expected = 7.5 + 250.0
        assert np.isclose(global_times[0], expected)

        # Test multiple onsets across different blocks
        onsets = [2.0, 5.0, 1.5, 10.0]
        blockids = [0, 0, 1, 2]
        global_times = sf.global_onsets(onsets, blockids)
        expected = [2.0, 5.0, 1.5 + 100.0, 10.0 + 250.0]
        assert np.allclose(global_times, expected)

        # Test error handling
        with pytest.raises(ValueError, match="must have the same length"):
            sf.global_onsets([1.0, 2.0], [1])

        with pytest.raises(ValueError, match="must be in range"):
            sf.global_onsets([1.0], [-1])

        with pytest.raises(ValueError, match="must be in range"):
            sf.global_onsets([1.0], [3])

    def test_global_scan_times(self, sampling_frame_multi_block):
        """Test global_scan_times property (backward compatibility)."""
        sf = sampling_frame_multi_block

        # global_scan_times should be identical to samples
        assert np.array_equal(sf.global_scan_times, sf.samples)
    
    def test_block_samples(self, sampling_frame_multi_block):
        """Test block_samples method."""
        sf = sampling_frame_multi_block
        
        # Test each block
        block0 = sf.block_samples(0)
        assert len(block0) == 50
        assert np.allclose(block0, np.arange(50) * 2.0)
        
        block1 = sf.block_samples(1)
        assert len(block1) == 75
        assert np.allclose(block1, 110.0 + np.arange(75) * 2.0)
        
        block2 = sf.block_samples(2)
        assert len(block2) == 100
        assert np.allclose(block2, 280.0 + np.arange(100) * 1.5)
        
        # Test invalid block index
        with pytest.raises(ValueError, match="block_idx must be in range"):
            sf.block_samples(3)
        
        with pytest.raises(ValueError, match="block_idx must be in range"):
            sf.block_samples(-1)


class TestSamplingFrameValidation:
    """Test input validation for SamplingFrame."""
    
    def test_invalid_blocklens(self):
        """Test validation of blocklens."""
        with pytest.raises(ValueError, match="blocklens must be positive"):
            SamplingFrame(blocklens=0, tr=2.0)
        
        with pytest.raises(ValueError, match="blocklens must be positive"):
            SamplingFrame(blocklens=[50, -10, 100], tr=2.0)
    
    def test_invalid_tr(self):
        """Test validation of TR."""
        with pytest.raises(ValueError, match="tr must be positive"):
            SamplingFrame(blocklens=100, tr=0)

        with pytest.raises(ValueError, match="tr must be positive"):
            SamplingFrame(blocklens=100, tr=-1.5)

        with pytest.raises(ValueError, match="tr must be positive"):
            SamplingFrame(blocklens=[50, 100], tr=[2.0, -1.0])
    
    def test_invalid_precision(self):
        """Test validation of precision."""
        with pytest.raises(ValueError, match="precision must be positive"):
            SamplingFrame(blocklens=100, tr=2.0, precision=0)
        
        with pytest.raises(ValueError, match="precision must be positive"):
            SamplingFrame(blocklens=100, tr=2.0, precision=-0.1)
    
    def test_mismatched_lengths(self):
        """Test validation of array length matching."""
        # TR length mismatch
        with pytest.raises(ValueError, match="Length of tr"):
            SamplingFrame(blocklens=[50, 100], tr=[2.0, 2.0, 1.5])
        
        # start_time length mismatch
        with pytest.raises(ValueError, match="Length of start_time"):
            SamplingFrame(blocklens=[50, 100], tr=2.0, start_time=[0.0, 10.0, 20.0])


class TestSamplingFrameStringRepresentation:
    """Test string representations of SamplingFrame."""
    
    def test_str_single_block(self):
        """Test string representation for single block."""
        sf = SamplingFrame(blocklens=100, tr=2.0)
        str_repr = str(sf)
        
        assert "1 block(s)" in str_repr
        assert "100 total scans" in str_repr
        assert "Block 0: 100 scans, tr=2.0s, start=1.0s" in str_repr
        assert "Precision: 0.1s" in str_repr
    
    def test_str_multi_block(self, sampling_frame_multi_block):
        """Test string representation for multiple blocks."""
        sf = sampling_frame_multi_block
        str_repr = str(sf)
        
        assert "3 block(s)" in str_repr
        assert "225 total scans" in str_repr
        assert "Block 0: 50 scans" in str_repr
        assert "Block 1: 75 scans" in str_repr
        assert "Block 2: 100 scans" in str_repr
    
    def test_repr(self, sampling_frame_multi_block):
        """Test repr representation."""
        sf = sampling_frame_multi_block
        repr_str = repr(sf)
        
        assert "SamplingFrame(" in repr_str
        assert "blocklens=[50, 75, 100]" in repr_str
        assert "tr=[2.0, 2.0, 1.5]" in repr_str
        assert "start_time=[0.0, 110.0, 280.0]" in repr_str
        assert "precision=0.1" in repr_str


class TestSamplingFrameSerialization:
    """Test serialization methods."""
    
    def test_to_dict(self, sampling_frame_multi_block):
        """Test to_dict method."""
        sf = sampling_frame_multi_block
        data = sf.to_dict()
        
        assert isinstance(data, dict)
        assert data['blocklens'] == [50, 75, 100]
        assert data['tr'] == [2.0, 2.0, 1.5]
        assert data['start_time'] == [0.0, 110.0, 280.0]
        assert data['precision'] == 0.1
        assert data['n_blocks'] == 3
        assert data['n_scans'] == 225
    
    def test_from_dict(self):
        """Test from_dict method."""
        data = {
            'blocklens': [50, 75],
            'tr': [2.0, 1.5],
            'start_time': [0.0, 105.0],
            'precision': 0.2
        }
        
        sf = SamplingFrame.from_dict(data)
        
        assert np.array_equal(sf.blocklens, [50, 75])
        assert np.array_equal(sf.tr, [2.0, 1.5])
        assert np.array_equal(sf.start_time, [0.0, 105.0])
        assert sf.precision == 0.2
    
    def test_round_trip(self, sampling_frame_multi_block):
        """Test round-trip serialization."""
        sf1 = sampling_frame_multi_block
        data = sf1.to_dict()
        sf2 = SamplingFrame.from_dict(data)
        
        assert np.array_equal(sf1.blocklens, sf2.blocklens)
        assert np.array_equal(sf1.tr, sf2.tr)
        assert np.array_equal(sf1.start_time, sf2.start_time)
        assert sf1.precision == sf2.precision


class TestSamplingFrameConcatenation:
    """Test concatenation of sampling frames."""
    
    def test_concatenate_simple(self):
        """Test simple concatenation."""
        sf1 = SamplingFrame(blocklens=50, tr=2.0)
        sf2 = SamplingFrame(blocklens=75, tr=1.5)
        
        sf_concat = sf1.concatenate(sf2)
        
        assert sf_concat.n_blocks == 2
        assert sf_concat.n_scans == 125
        assert np.array_equal(sf_concat.blocklens, [50, 75])
        assert np.array_equal(sf_concat.tr, [2.0, 1.5])
        
        # Check start times - second block should start after first ends
        expected_start2 = 50 * 2.0  # End of first block
        # First block has start_time=1.0 (TR/2), second block starts after first ends
        # and gets its default start_time (TR/2 = 0.75) added to the end time
        assert np.allclose(sf_concat.start_time, [1.0, 101.75])
    
    def test_concatenate_with_start_times(self):
        """Test concatenation preserving relative timing."""
        sf1 = SamplingFrame(blocklens=[30, 40], tr=2.0, start_time=[0.0, 65.0])
        sf2 = SamplingFrame(blocklens=50, tr=1.5, start_time=10.0)
        
        sf_concat = sf1.concatenate(sf2)
        
        assert sf_concat.n_blocks == 3
        assert sf_concat.n_scans == 120
        
        # The third block should start after the end of sf1
        last_scan_sf1 = sf1.samples[-1]
        last_tr_sf1 = sf1.tr[sf1.blockids[-1]]
        expected_start3 = last_scan_sf1 + last_tr_sf1 + 10.0  # Original offset preserved
        
        assert np.allclose(sf_concat.start_time[2], expected_start3)
    
    def test_concatenate_precision_mismatch(self):
        """Test error on precision mismatch."""
        sf1 = SamplingFrame(blocklens=50, tr=2.0, precision=0.1)
        sf2 = SamplingFrame(blocklens=50, tr=2.0, precision=0.2)
        
        with pytest.raises(ValueError, match="different precision"):
            sf1.concatenate(sf2)
