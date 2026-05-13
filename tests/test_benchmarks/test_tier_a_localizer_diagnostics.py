"""Focused regression checks for the localizer parity solver path."""

from __future__ import annotations

# ruff: noqa: E402
import pytest

pytest.importorskip("nilearn")

from benchmarks.parity.tier_a_localizer_fixed_effects.workflow import make_case
from cross_testing.harness import run


def test_localizer_t_stat_passes_without_declared_caveat() -> None:
    result = run(make_case(max_voxels=512))

    assert result.status == "pass"
    assert result.caveats == ()
    assert result.deltas["effect_audio_gt_visual"].passes
    assert result.deltas["t_audio_gt_visual"].passes
    assert result.deltas["t_audio_gt_visual"].failed_gates == ()
