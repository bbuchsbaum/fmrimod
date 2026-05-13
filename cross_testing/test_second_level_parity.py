"""Nilearn SecondLevelModel parity workflow tests."""

from __future__ import annotations

import pytest

from cross_testing.harness import render, run


pytestmark = pytest.mark.parity

pytest.importorskip("nilearn")
pytest.importorskip("nibabel")


def test_second_level_synthetic_case_passes_and_renders(tmp_path):
    from benchmarks.parity.tier_c_second_level.workflow import make_case

    result = run(make_case())
    assert result.status == "pass_with_caveats"
    assert all(delta.passes for delta in result.deltas.values())

    json_path, md_path = render(result, tmp_path)
    assert json_path.exists()
    assert md_path.exists()
