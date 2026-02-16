"""Tests for event utilities - split_onsets, split_by_block."""

import pytest
import numpy as np
import pandas as pd

from fmrimod.events.factor import EventFactor
from fmrimod.events.variable import EventVariable
from fmrimod.utils import split_onsets, split_by_block


class TestSplitOnsets:
    """Test split_onsets function."""

    def test_split_onsets_by_values(self):
        """Test splitting onsets by event values."""
        event = EventFactor(
            name='condition',
            onsets=[5, 10, 15, 20, 25, 30],
            values=['A', 'B', 'A', 'B', 'A', 'B'],
            durations=1.0
        )

        onset_groups = split_onsets(event, by='values')

        # Should return dict
        assert isinstance(onset_groups, dict)

        # Should have group for each unique value
        assert 'A' in onset_groups
        assert 'B' in onset_groups

        # Check onset arrays
        np.testing.assert_array_equal(onset_groups['A'], [5, 15, 25])
        np.testing.assert_array_equal(onset_groups['B'], [10, 20, 30])

    def test_split_onsets_continuous_by_values(self):
        """Test splitting continuous variable by values."""
        event = EventVariable(
            name='rating',
            onsets=[5, 10, 15, 20],
            values=[1.0, 2.0, 1.0, 2.0],
            durations=1.0,
            center=False  # Don't center for this test
        )

        onset_groups = split_onsets(event, by='values')

        assert 1.0 in onset_groups
        assert 2.0 in onset_groups
        np.testing.assert_array_equal(onset_groups[1.0], [5, 15])
        np.testing.assert_array_equal(onset_groups[2.0], [10, 20])

    def test_split_onsets_by_callable(self):
        """Test splitting onsets using callable function."""
        event = EventFactor(
            name='condition',
            onsets=[5, 10, 15, 20, 25, 30],
            values=['A', 'B', 'A', 'B', 'A', 'B'],
            durations=1.0
        )

        # Split by early/late
        def early_late(values):
            return ['early' if v == 'A' else 'late' for v in values]

        onset_groups = split_onsets(event, by=early_late)

        assert 'early' in onset_groups
        assert 'late' in onset_groups
        np.testing.assert_array_equal(onset_groups['early'], [5, 15, 25])
        np.testing.assert_array_equal(onset_groups['late'], [10, 20, 30])

    def test_split_onsets_external_values(self):
        """Test splitting with external values array."""
        event = EventFactor(
            name='condition',
            onsets=[5, 10, 15, 20],
            values=['A', 'B', 'A', 'B'],
            durations=1.0
        )

        # External grouping variable
        blocks = np.array(['block1', 'block1', 'block2', 'block2'])

        onset_groups = split_onsets(event, by='values', values=blocks)

        assert 'block1' in onset_groups
        assert 'block2' in onset_groups
        np.testing.assert_array_equal(onset_groups['block1'], [5, 10])
        np.testing.assert_array_equal(onset_groups['block2'], [15, 20])

    def test_split_onsets_single_group(self):
        """Test split_onsets with single group."""
        event = EventFactor(
            name='condition',
            onsets=[5, 10, 15],
            values=['A', 'A', 'A'],
            durations=1.0
        )

        onset_groups = split_onsets(event, by='values')

        assert len(onset_groups) == 1
        assert 'A' in onset_groups
        np.testing.assert_array_equal(onset_groups['A'], [5, 10, 15])

    def test_split_onsets_length_mismatch(self):
        """Test error when values length doesn't match onsets."""
        event = EventFactor(
            name='condition',
            onsets=[5, 10, 15],
            values=['A', 'B', 'A'],
            durations=1.0
        )

        # Wrong length values
        wrong_values = ['X', 'Y']

        with pytest.raises(ValueError, match="Length mismatch"):
            split_onsets(event, by='values', values=wrong_values)

    def test_split_onsets_no_values_attribute(self):
        """Test error when event has no values attribute."""
        # Create minimal event without values
        class MinimalEvent:
            def __init__(self):
                self.onsets = [5, 10, 15]

        event = MinimalEvent()

        with pytest.raises(ValueError, match="no 'values' attribute"):
            split_onsets(event, by='values')


class TestSplitByBlock:
    """Test split_by_block function."""

    def test_split_by_block_labels(self):
        """Test splitting events by block labels."""
        event = EventFactor(
            name='condition',
            onsets=[5, 10, 15, 20, 25, 30],
            values=['A', 'B', 'A', 'B', 'A', 'B'],
            durations=1.0
        )
        event.block = ['run1', 'run1', 'run2', 'run2', 'run3', 'run3']

        blocks = split_by_block(event, block_var='block')

        # Should return dict
        assert isinstance(blocks, dict)

        # Should have entry for each block
        assert 'run1' in blocks
        assert 'run2' in blocks
        assert 'run3' in blocks

        # Each should be EventFactor
        assert isinstance(blocks['run1'], EventFactor)

        # Check onsets
        np.testing.assert_array_equal(blocks['run1'].onsets, [5, 10])
        np.testing.assert_array_equal(blocks['run2'].onsets, [15, 20])
        np.testing.assert_array_equal(blocks['run3'].onsets, [25, 30])

    def test_split_by_block_array(self):
        """Test splitting with block array."""
        event = EventFactor(
            name='condition',
            onsets=[5, 10, 15, 20],
            values=['A', 'B', 'A', 'B'],
            durations=1.0
        )

        block_labels = np.array([1, 1, 2, 2])

        blocks = split_by_block(event, block_var=block_labels)

        assert 1 in blocks
        assert 2 in blocks
        np.testing.assert_array_equal(blocks[1].onsets, [5, 10])
        np.testing.assert_array_equal(blocks[2].onsets, [15, 20])

    def test_split_by_block_timing(self):
        """Test splitting by block timing windows."""
        event = EventFactor(
            name='condition',
            onsets=[5, 15, 25, 105, 115, 125],
            values=['A', 'B', 'A', 'B', 'A', 'B'],
            durations=1.0
        )

        block_labels = ['block1', 'block2']
        block_onsets = [0, 100]
        block_durations = [100, 100]

        blocks = split_by_block(
            event,
            block_var=block_labels,
            block_onsets=block_onsets,
            block_durations=block_durations
        )

        assert 'block1' in blocks
        assert 'block2' in blocks

        # Events in first 100s
        np.testing.assert_array_equal(blocks['block1'].onsets, [5, 15, 25])

        # Events in second 100s
        np.testing.assert_array_equal(blocks['block2'].onsets, [105, 115, 125])

    def test_split_by_block_variable(self):
        """Test splitting EventVariable by block."""
        event = EventVariable(
            name='rating',
            onsets=[5, 10, 15, 20],
            values=[1.0, 2.0, 3.0, 4.0],
            durations=1.0,
            center=False  # Don't center for this test
        )

        block_labels = [1, 1, 2, 2]

        blocks = split_by_block(event, block_var=block_labels)

        assert isinstance(blocks[1], EventVariable)
        assert isinstance(blocks[2], EventVariable)

        # Check that blocks have correct number of events
        assert len(blocks[1].onsets) == 2
        assert len(blocks[2].onsets) == 2

        # Values may be centered in the new blocks (default behavior)
        assert len(blocks[1].values) == 2
        assert len(blocks[2].values) == 2

    def test_split_by_block_preserves_durations(self):
        """Test that block splitting preserves durations."""
        event = EventFactor(
            name='condition',
            onsets=[5, 10, 15, 20],
            values=['A', 'B', 'A', 'B'],
            durations=[1.0, 2.0, 1.5, 2.5]
        )

        block_labels = [1, 1, 2, 2]

        blocks = split_by_block(event, block_var=block_labels)

        np.testing.assert_array_equal(blocks[1].durations, [1.0, 2.0])
        np.testing.assert_array_equal(blocks[2].durations, [1.5, 2.5])

    def test_split_by_block_empty_block(self):
        """Test that empty blocks are not included."""
        event = EventFactor(
            name='condition',
            onsets=[5, 10],
            values=['A', 'B'],
            durations=1.0
        )

        # Block 3 has no events
        block_labels = [1, 2]

        blocks = split_by_block(event, block_var=block_labels)

        # Should only have blocks with events
        assert 1 in blocks
        assert 2 in blocks
        assert len(blocks) == 2

    def test_split_by_block_timing_windows(self):
        """Test splitting by block timing windows."""
        event = EventFactor(
            name='condition',
            onsets=[5, 15, 105, 115],  # Events in two blocks
            values=['A', 'B', 'A', 'B'],
            durations=1.0
        )

        block_labels = ['block1', 'block2']
        block_onsets = [0, 100]
        block_durations = [50, 50]  # 0-50 and 100-150

        blocks = split_by_block(
            event,
            block_var=block_labels,
            block_onsets=block_onsets,
            block_durations=block_durations
        )

        # Should have blocks for events that fall within windows
        assert len(blocks) == 2
        assert 'block1' in blocks
        assert 'block2' in blocks

    def test_split_by_block_length_mismatch(self):
        """Test error on length mismatch."""
        event = EventFactor(
            name='condition',
            onsets=[5, 10, 15],
            values=['A', 'B', 'A'],
            durations=1.0
        )

        # Wrong length block labels
        block_labels = [1, 2]

        with pytest.raises(ValueError, match="Length mismatch"):
            split_by_block(event, block_var=block_labels)

    def test_split_by_block_missing_attribute(self):
        """Test error when block attribute doesn't exist."""
        event = EventFactor(
            name='condition',
            onsets=[5, 10, 15],
            values=['A', 'B', 'A'],
            durations=1.0
        )

        with pytest.raises(AttributeError, match="no attribute"):
            split_by_block(event, block_var='nonexistent')


class TestEventUtilsEdgeCases:
    """Test edge cases for event utilities."""

    def test_split_onsets_single_value(self):
        """Test split_onsets with single unique value."""
        event = EventFactor(
            name='condition',
            onsets=[5, 10, 15],
            values=['A', 'A', 'A'],
            durations=1.0
        )

        onset_groups = split_onsets(event, by='values')
        assert isinstance(onset_groups, dict)
        assert len(onset_groups) == 1
        assert 'A' in onset_groups

    def test_split_by_block_single_block(self):
        """Test split_by_block with single block."""
        event = EventFactor(
            name='condition',
            onsets=[5, 10, 15],
            values=['A', 'B', 'A'],
            durations=1.0
        )

        blocks = split_by_block(event, block_var=[1, 1, 1])
        assert isinstance(blocks, dict)
        assert len(blocks) == 1
        assert 1 in blocks

    def test_split_onsets_many_groups(self):
        """Test split_onsets with many unique values."""
        n_events = 100
        n_groups = 20

        event = EventFactor(
            name='condition',
            onsets=np.arange(n_events),
            values=[f'group_{i % n_groups}' for i in range(n_events)],
            durations=1.0
        )

        onset_groups = split_onsets(event, by='values')

        # Should have all groups
        assert len(onset_groups) == n_groups

        # Each group should have 5 events (100 / 20)
        for group_onsets in onset_groups.values():
            assert len(group_onsets) == 5
