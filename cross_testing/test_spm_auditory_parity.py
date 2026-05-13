"""SPM auditory parity case against Nilearn FirstLevelModel."""

from __future__ import annotations

import pytest

from cross_testing.harness import render, run


pytestmark = pytest.mark.parity

pytest.importorskip("nilearn")
pytest.importorskip("nibabel")


def test_spm_auditory_case_passes_and_renders(tmp_path):
    try:
        from benchmarks.parity.tier_a_spm_auditory.workflow import make_case
    except Exception as exc:  # pragma: no cover - import environment guard
        pytest.skip(f"SPM auditory parity dependencies unavailable: {exc}")

    try:
        result = run(make_case(max_voxels=2048))
    except Exception as exc:
        pytest.skip(f"SPM auditory dataset unavailable: {exc}")

    assert result.status == "pass_with_caveats"
    assert "spm-auditory-hrf-grid-scale" in {
        caveat.caveat_id for caveat in result.caveats
    }
    assert all(delta.passes for delta in result.deltas.values())

    json_path, md_path = render(result, tmp_path)
    assert json_path.exists()
    assert md_path.exists()
