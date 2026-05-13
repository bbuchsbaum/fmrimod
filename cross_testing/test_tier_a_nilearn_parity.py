"""Tier A Nilearn first-level parity workflow tests."""

from __future__ import annotations

import pytest

from cross_testing.harness import render, run


pytestmark = pytest.mark.parity

pytest.importorskip("nilearn")
pytest.importorskip("nibabel")


@pytest.mark.parametrize(
    "module_name",
    [
        "benchmarks.parity.tier_a_fiac.workflow",
        "benchmarks.parity.tier_a_localizer_fixed_effects.workflow",
    ],
)
def test_tier_a_nilearn_case_passes_and_renders(module_name, tmp_path):
    try:
        module = __import__(module_name, fromlist=["make_case"])
    except Exception as exc:  # pragma: no cover - import environment guard
        pytest.skip(f"Tier A parity dependencies unavailable for {module_name}: {exc}")

    try:
        result = run(module.make_case(max_voxels=2048))
    except Exception as exc:
        pytest.skip(f"Tier A parity dataset unavailable for {module_name}: {exc}")

    assert result.status in {"pass", "pass_with_caveats"}
    assert all(delta.passes for delta in result.deltas.values())

    json_path, md_path = render(result, tmp_path)
    assert json_path.exists()
    assert md_path.exists()
