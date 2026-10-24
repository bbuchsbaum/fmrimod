"""Contract tests for the fmrimod-vs-Nilearn ergonomics scorecard.

Three layers:

1. Every scenario's fmrimod and Nilearn bodies run and agree numerically, so
   the authoring/speed comparison is between equivalent analyses.
2. The measurement itself is honest: docstrings, comments and parity knobs
   are not charged, and token counts do not depend on line wrapping.
3. Known API gaps found by the scorecard are pinned as strict xfails. When a
   gap is fixed the probe XPASSes and fails the run: shorten the matching
   scenario body to the new idiom, drop its friction note, regenerate the
   report, and delete the probe.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("nilearn")
pytest.importorskip("nibabel")

from benchmarks.ergonomics import workflow  # noqa: E402
from benchmarks.ergonomics.inputs import make_group_inputs, make_inputs  # noqa: E402
from benchmarks.ergonomics.metrics import authoring_cost  # noqa: E402
from benchmarks.ergonomics.scenarios import SCENARIOS  # noqa: E402

pytestmark = pytest.mark.benchmark

REPORT = Path(workflow.REPORT_DIR) / "ergonomics_report.json"


# ---------------------------------------------------------------------------
# 1. scenarios run and agree
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.scenario_id)
def test_scenario_paths_agree(scenario) -> None:
    result = workflow.run_scenario(scenario, repeats=1)

    assert result.agrees, (
        f"{scenario.scenario_id}: r={result.correlation:.5f} "
        f"< {scenario.min_correlation}"
    )
    for side in (result.fmrimod, result.nilearn):
        assert side.authoring.tokens > 0
        assert side.runtime.warm_seconds > 0
        # The profiler hook must have seen the library calls the body made.
        assert side.runtime.docs, f"{side.library}: no library callables recorded"
    assert set(result.verdicts) == {"authoring", "speed", "noise", "docs"}


def test_scenario_ids_are_unique_and_bodies_are_self_contained() -> None:
    ids = [s.scenario_id for s in SCENARIOS]
    assert len(ids) == len(set(ids))
    for scenario in SCENARIOS:
        for func in (scenario.fmrimod, scenario.nilearn):
            cost = authoring_cost(func)
            # Imports live in the body so their cost is counted.
            assert cost.imports >= 1, func.__name__


# ---------------------------------------------------------------------------
# 2. measurement honesty
# ---------------------------------------------------------------------------


def _compact(x):
    """Docstring is not charged."""
    import math

    # comments are not charged either
    y = math.hypot(x, 1.0)
    return round(y, 2)  # parity-knob


# fmt: off
def _wrapped(x):
    import math

    y = math.hypot(
        x, 1.0
    )
    return round(y, 2)  # parity-knob
# fmt: on


def test_authoring_cost_ignores_docstring_comments_and_wrapping() -> None:
    compact, wrapped = authoring_cost(_compact), authoring_cost(_wrapped)
    # `import math` (2) + `y = math.hypot(x, 1.0)` (10); the knob line is free.
    assert compact.tokens == 12
    assert compact.statements == 2
    assert compact.parity_knobs == 1
    assert compact.imported_names == ("math",)
    # Wrapping changes physical lines only.
    assert (wrapped.tokens, wrapped.statements) == (compact.tokens, compact.statements)
    assert wrapped.lines > compact.lines


def test_score_margins() -> None:
    assert workflow._lower_is_better(90, 100, 0.05) == "fmrimod"
    assert workflow._lower_is_better(97, 100, 0.05) == "tie"
    assert workflow._lower_is_better(110, 100, 0.05) == "nilearn"


# ---------------------------------------------------------------------------
# Checked-in report stays in step with the scenario code
# ---------------------------------------------------------------------------


def test_checked_in_report_matches_scenario_bodies() -> None:
    report = json.loads(REPORT.read_text())
    assert report["schema_version"] == workflow.SCHEMA_VERSION
    recorded = {
        (row["scenario_id"], lib): row[lib]["authoring"]
        for row in report["scenarios"]
        if row["size"] == "small"
        for lib in ("fmrimod", "nilearn")
    }
    for scenario in SCENARIOS:
        for lib in ("fmrimod", "nilearn"):
            current = authoring_cost(getattr(scenario, lib)).as_dict()
            current["imported_names"] = list(current["imported_names"])
            assert recorded[(scenario.scenario_id, lib)] == current, (
                f"{scenario.scenario_id}/{lib} changed; regenerate with "
                "`python -m benchmarks.ergonomics.workflow --large`"
            )
    assert all(row["agreement"]["agrees"] for row in report["scenarios"])


# ---------------------------------------------------------------------------
# 3. known gaps (strict xfail: an XPASS means the gap closed)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def one_run():
    return make_inputs(seed=11)


def _fit(inp, **lm_kwargs):
    import fmrimod as fm
    from fmrimod.spec import drift, hrf

    ds = fm.fmri_dataset(inp.img, mask=inp.mask_img, tr=inp.tr, events=inp.events[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fm.fmri_lm(
            hrf("trial_type") + drift("cosine", cutoff=128), ds, **lm_kwargs
        )


@pytest.mark.xfail(
    strict=True, raises=KeyError, reason="s01: no 'face - house' strings"
)
def test_gap_string_level_contrast(one_run) -> None:
    from fmrimod.contrast import condition

    fit = _fit(one_run)
    expected = fit.contrast(
        condition("face", term="trial_type") - condition("house", term="trial_type")
    )
    np.testing.assert_allclose(fit.contrast("face - house").stat, expected.stat)


@pytest.mark.xfail(strict=True, raises=ValueError, reason="s01: cosine needs cutoff=")
def test_gap_cosine_drift_has_default_cutoff(one_run) -> None:
    import fmrimod as fm
    from fmrimod.spec import drift, hrf

    ds = fm.fmri_dataset(
        one_run.img, mask=one_run.mask_img, tr=one_run.tr, events=one_run.events[0]
    )
    fm.fmri_lm(hrf("trial_type") + drift("cosine"), ds)


@pytest.mark.xfail(strict=True, reason="s01: fmri_lm needs a prebuilt dataset")
def test_gap_fmri_lm_accepts_images(one_run) -> None:
    import fmrimod as fm
    from fmrimod.spec import hrf

    fm.fmri_lm(
        hrf("trial_type"),
        one_run.img,
        tr=one_run.tr,
        events=one_run.events[0],
        mask=one_run.mask_img,
    )


@pytest.mark.xfail(
    strict=True,
    raises=AttributeError,
    reason="s02 bug: per-run event lists crash fmri_lm",
)
def test_gap_per_run_event_list_fits() -> None:
    import fmrimod as fm
    from fmrimod.spec import hrf

    inp = make_inputs(seed=12, n_runs=2, n_scans=80)
    ds = fm.fmri_dataset(inp.imgs, mask=inp.mask_img, tr=inp.tr, events=inp.events)
    fm.fmri_lm(hrf("trial_type"), ds)


@pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason="s07: typed trialwise() not accepted for single trial",
)
def test_gap_typed_trialwise_single_trial(one_run) -> None:
    import fmrimod as fm
    from fmrimod.spec import trialwise

    ds = fm.fmri_dataset(
        one_run.img, mask=one_run.mask_img, tr=one_run.tr, events=one_run.events[0]
    )
    fm.estimate_single_trial_from_dataset(ds, trialwise(), method="lss")


@pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason="s08: fmri_group_lm needs group_dataset_from_contrasts first",
)
def test_gap_group_lm_accepts_contrast_mapping() -> None:
    import fmrimod as fm
    from fmrimod.contrast import condition

    subjects = make_group_inputs(n_subjects=3, shape=(6, 6, 5), n_scans=80)
    per_subject = {
        f"sub-{i}": _fit(sub).contrast(condition("face", term="trial_type"))
        for i, sub in enumerate(subjects)
    }
    fm.fmri_group_lm(per_subject)


@pytest.mark.skipif(
    sys.version_info >= (3, 12),
    reason="entry_points() no longer returns SelectableGroups",
)
@pytest.mark.xfail(strict=True, reason="first fit warns: SelectableGroups deprecated")
def test_gap_first_fit_is_warning_free() -> None:
    code = textwrap.dedent(
        """
        import warnings
        import numpy as np, pandas as pd
        import fmrimod as fm
        from fmrimod.spec import hrf
        events = pd.DataFrame({"onset": [4.0, 20.0], "duration": 1.0,
                               "trial_type": ["a", "b"]})
        ds = fm.fmri_dataset(np.random.default_rng(0).normal(size=(40, 3)),
                             tr=1.0, events=events)
        # Record rather than raise: engine discovery swallows exceptions.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fm.fmri_lm(hrf("trial_type"), ds)
        assert not caught, [str(w.message) for w in caught]
        """
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)
