"""Contract tests for the fmrimod-vs-Nilearn ergonomics scorecard.

Three layers:

1. Every scenario's fmrimod and Nilearn bodies run and agree numerically, so
   the authoring/speed comparison is between equivalent analyses.
2. The measurement itself is honest: docstrings, comments and parity knobs
   are not charged, and token counts do not depend on line wrapping.
3. API gaps the scorecard found and that were then closed are pinned as
   regression tests asserting the short idiom equals the long one. Pin a
   newly found gap as a strict xfail probe; when it XPASSes, shorten the
   scenario body, drop its friction note, regenerate the report, and turn
   the probe into a regression test here.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
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
# 3. gaps closed after round 1 (each short form must equal the long form)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def one_run():
    return make_inputs(seed=11)


def _dataset(inp, **kwargs):
    import fmrimod as fm

    return fm.fmri_dataset(
        inp.img, mask=inp.mask_img, tr=inp.tr, events=inp.events[0], **kwargs
    )


def test_string_level_contrast_equals_condition_refs(one_run) -> None:
    import fmrimod as fm
    from fmrimod.contrast import condition
    from fmrimod.spec import drift, hrf

    fit = fm.fmri_lm(hrf("trial_type", modulators=["rt"]) + drift(), _dataset(one_run))
    typed = fit.contrast(
        condition("face", term="trial_type") - condition("house", term="trial_type")
    )
    parsed = fit.contrast("face - house")
    np.testing.assert_allclose(parsed.stat, typed.stat)
    assert parsed.name == "face - house"
    slope = fit.contrast("trial_type:rt.face")
    np.testing.assert_allclose(
        slope.stat, fit.contrast(condition("face", term="trial_type:rt")).stat
    )
    with pytest.raises(KeyError, match="Unknown contrast name"):
        fit.contrast("nonsense")
    with pytest.raises(ValueError, match="matched no declared"):
        fit.contrast("face - hose")


def test_cosine_drift_default_cutoff_is_128s(one_run) -> None:
    import fmrimod as fm
    from fmrimod.spec import drift, hrf

    ds = _dataset(one_run)
    default = fm.fmri_lm(hrf("trial_type") + drift(), ds)
    explicit = fm.fmri_lm(hrf("trial_type") + drift("cosine", cutoff=128), ds)
    np.testing.assert_allclose(default.betas, explicit.betas)


def test_fmri_lm_accepts_images_directly() -> None:
    import fmrimod as fm
    from fmrimod.spec import hrf

    inp = make_inputs(seed=12, n_runs=2, n_scans=80)
    shorthand = fm.fmri_lm(
        hrf("trial_type"), inp.imgs, tr=inp.tr, events=inp.events, mask=inp.mask_img
    )
    ds = fm.fmri_dataset(inp.imgs, mask=inp.mask_img, tr=inp.tr, events=inp.events_df)
    np.testing.assert_allclose(shorthand.betas, fm.fmri_lm(hrf("trial_type"), ds).betas)
    with pytest.raises(ValueError, match="prebuilt dataset"):
        fm.fmri_lm(hrf("trial_type"), ds, tr=2.0)


def test_per_run_event_list_equals_stacked_table() -> None:
    import fmrimod as fm

    inp = make_inputs(seed=12, n_runs=2, n_scans=80)
    ds = fm.fmri_dataset(inp.imgs, mask=inp.mask_img, tr=inp.tr, events=inp.events)
    pd.testing.assert_frame_equal(ds.event_table, inp.events_df)
    bad = [inp.events[0].assign(run=2), inp.events[1]]
    with pytest.raises(ValueError, match="run"):
        fm.fmri_dataset(inp.imgs, mask=inp.mask_img, tr=inp.tr, events=bad)


def test_confounds_take_per_run_tables_and_default_to_all_columns() -> None:
    import fmrimod as fm
    from fmrimod.spec import confounds, hrf

    inp = make_inputs(seed=13, n_runs=2, n_scans=80, with_confounds=True)
    ds = fm.fmri_dataset(inp.imgs, mask=inp.mask_img, tr=inp.tr, events=inp.events)
    stacked = pd.concat(inp.confounds, ignore_index=True)
    short = fm.fmri_lm(hrf("trial_type") + confounds(source=inp.confounds), ds)
    long = fm.fmri_lm(
        hrf("trial_type") + confounds(*stacked.columns, source=stacked), ds
    )
    np.testing.assert_allclose(short.betas, long.betas)


def test_typed_trialwise_equals_formula_string(one_run) -> None:
    import fmrimod as fm
    from fmrimod.spec import drift, trialwise

    ds = _dataset(one_run)
    typed = fm.estimate_single_trial_from_dataset(ds, trialwise(), method="lss")
    legacy = fm.estimate_single_trial_from_dataset(ds, "trialwise()", method="lss")
    np.testing.assert_allclose(typed.betas, legacy.betas)
    with pytest.raises(ValueError, match="baseline terms"):
        fm.estimate_single_trial_from_dataset(ds, trialwise() + drift(), method="lss")


def test_group_lm_accepts_contrast_results_directly() -> None:
    import fmrimod as fm
    from fmrimod.spec import hrf

    subjects = make_group_inputs(n_subjects=3, shape=(6, 6, 5), n_scans=80)
    per_subject = {
        f"sub-{i}": fm.fmri_lm(hrf("trial_type"), _dataset(sub)).contrast("face")
        for i, sub in enumerate(subjects)
    }
    via_dataset = fm.fmri_group_lm(fm.group_dataset_from_contrasts(per_subject))
    np.testing.assert_allclose(
        fm.fmri_group_lm(per_subject).t_stat(), via_dataset.t_stat()
    )
    as_list = fm.fmri_group_lm(list(per_subject.values()))
    np.testing.assert_allclose(as_list.t_stat(), via_dataset.t_stat())


def test_design_matrix_of_spec_equals_fitted_design(one_run) -> None:
    import fmrimod as fm
    from fmrimod.spec import drift, hrf

    spec = hrf("trial_type") + drift()
    fit = fm.fmri_lm(spec, _dataset(one_run))
    X = fm.design_matrix(
        spec, events=one_run.events[0], n_scans=one_run.img.shape[-1], tr=one_run.tr
    )
    pd.testing.assert_frame_equal(X, fit.model.design_matrix(run=0))


def test_fir_span_sets_bin_width(one_run) -> None:
    from fmrimod.hrf import hrf_fir_generator
    from fmrimod.spec import hrf

    term = hrf("trial_type", basis="fir", nbasis=8, span=16.0)
    assert term.hrf.span == hrf_fir_generator(nbasis=8, span=16.0).span == 16.0
    with pytest.raises(ValueError, match="flexible bases"):
        hrf("trial_type", basis="spm", span=16.0)


def test_first_fit_is_warning_free() -> None:
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
