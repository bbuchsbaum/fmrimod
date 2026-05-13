"""FitLins CLI derivative-tree parity tests."""

from __future__ import annotations

import json
import shutil

import pytest


pytestmark = pytest.mark.parity


def test_fitlins_cli_derivative_tree_parity(tmp_path):
    if shutil.which("uv") is None:
        pytest.skip("uv is required to run the local FitLins CLI fixture")

    from benchmarks.parity.tier_b_fitlins_bids.workflow import (
        render_fitlins_cli_derivative_report,
        run_fitlins_cli_derivative_parity,
    )

    result = run_fitlins_cli_derivative_parity(tmp_path / "work")
    assert result.status == "pass_with_caveats"
    assert len(result.deltas) == 6
    assert all(delta.passes for delta in result.deltas)
    assert {delta.caveat_id for delta in result.deltas if delta.caveat_id} == {
        "fitlins-ar1-coefficient-binning"
    }
    assert any(delta.gate.startswith("caveat-bypassed:") for delta in result.deltas)
    assert result.design_columns == [
        "trial_type.ice_cream",
        "trial_type.cake",
        "food_sweats",
        "intercept",
    ]
    assert any(path.endswith("_stat-t_statmap.nii.gz") for path in result.fitlins_output_files)

    json_path, md_path = render_fitlins_cli_derivative_report(result, tmp_path / "report")
    payload = json.loads(json_path.read_text())
    assert payload["status"] == "pass_with_caveats"
    assert "| map | gate |" in md_path.read_text()
