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


# ---------------------------------------------------------------------------
# 4. gaps closed after round 3 (critic scenarios s10-s13)
# ---------------------------------------------------------------------------


def test_contrast_z_matches_reference_conversion(one_run) -> None:
    import nibabel as nib
    from scipy import stats

    import fmrimod as fm
    from fmrimod.contrast import OmnibusContrast
    from fmrimod.glm.contrasts import z_from_stat
    from fmrimod.spec import drift, hrf

    fit = fm.fmri_lm(hrf("trial_type") + drift(), _dataset(one_run))
    t = fit.contrast("face - house")
    np.testing.assert_allclose(t.z, stats.norm.isf(stats.t.sf(t.stat, t.df)))
    f = fit.contrast(OmnibusContrast(term="trial_type"))
    np.testing.assert_allclose(f.z, stats.norm.isf(stats.f.sf(f.stat, *f.df)))
    # Extreme statistics keep full precision through the small tail.
    np.testing.assert_allclose(
        z_from_stat(np.array([40.0, -40.0]), 100, "t"), [16.799, -16.799], atol=1e-3
    )
    path = t.to_nifti(one_run.out_dir / "z.nii.gz", kind="z")
    written = nib.load(path)
    np.testing.assert_allclose(written.affine, one_run.img.affine)
    np.testing.assert_allclose(
        np.asarray(written.dataobj)[one_run.mask], t.z, rtol=1e-6, atol=1e-6
    )


def test_single_trial_accepts_images_directly(one_run) -> None:
    import fmrimod as fm
    from fmrimod.spec import trialwise

    short = fm.estimate_single_trial_from_dataset(
        one_run.img,
        trialwise(),
        tr=one_run.tr,
        events=one_run.events[0],
        mask=one_run.mask_img,
    )
    long = fm.estimate_single_trial_from_dataset(_dataset(one_run), trialwise())
    np.testing.assert_allclose(short.betas, long.betas)


def test_group_lm_reads_nifti_maps_with_covariates() -> None:
    import nibabel as nib

    import fmrimod as fm
    from benchmarks.ergonomics.inputs import make_group_map_inputs

    maps = make_group_map_inputs(n_per_group=6)
    data = np.stack([np.asarray(nib.load(p).dataobj)[maps.mask] for p in maps.paths])
    table = maps.participants
    design = pd.DataFrame(
        {
            "Intercept": 1.0,
            "patient": (table["group"] == "patient").astype(float),
            "age": table["age"],
        }
    )
    reference = fm.fmri_group_lm(data, intercept_only=False, design_matrix=design)
    from_files = fm.fmri_group_lm(
        maps.paths, formula="~ group + age", covariates=table, mask=maps.mask_img
    )
    np.testing.assert_allclose(
        from_files.t_stat("group[T.patient]"), reference.t_stat("patient")
    )
    # Rows are matched on participant_id, not position.
    shuffled = fm.fmri_group_lm(
        maps.paths,
        formula="~ group + age",
        covariates=table.sample(frac=1.0, random_state=1),
        subjects=list(table["participant_id"]),
    )
    np.testing.assert_allclose(
        shuffled.t_stat("group[T.patient]"), reference.t_stat("patient")
    )
    out = from_files.to_nifti(
        Path(maps.paths[0]).with_name("group_z.nii.gz"), "group[T.patient]", kind="z"
    )
    assert nib.load(out).shape == maps.mask.shape


def test_dataset_confounds_feed_sourceless_confounds_terms() -> None:
    import fmrimod as fm
    from benchmarks.ergonomics.inputs import MOTION
    from fmrimod.spec import confounds, hrf

    inp = make_inputs(seed=14, n_runs=2, n_scans=80, with_confounds=True)
    tables = [c.assign(fd=np.r_[np.nan, np.ones(len(c) - 1)]) for c in inp.confounds]
    ds = fm.fmri_dataset(
        inp.imgs, mask=inp.mask_img, tr=inp.tr, events=inp.events, confounds=tables
    )
    bound = fm.fmri_lm(hrf("trial_type") + confounds(*MOTION), ds)
    explicit = fm.fmri_lm(hrf("trial_type") + confounds(source=inp.confounds), ds)
    np.testing.assert_allclose(bound.betas, explicit.betas)
    with pytest.raises(ValueError, match="missing values"):
        fm.fmri_lm(hrf("trial_type") + confounds("fd"), ds)
    with pytest.raises(KeyError, match="not in the dataset"):
        fm.fmri_lm(hrf("trial_type") + confounds("nope"), ds)
    with pytest.raises(ValueError, match="rows"):
        fm.fmri_dataset(
            inp.imgs,
            mask=inp.mask_img,
            tr=inp.tr,
            events=inp.events,
            confounds=[tables[0].iloc[:10], tables[1]],
        )


def test_bids_datasets_loads_fmriprep_layout() -> None:
    import fmrimod as fm
    from benchmarks.ergonomics.inputs import make_bids_inputs

    bids = make_bids_inputs(n_subjects=2, n_runs=2, shape=(6, 6, 5), n_scans=60)
    datasets = fm.bids_datasets(bids.root, bids.task, space=bids.space)
    assert list(datasets) == ["01", "02"]
    ds = datasets["01"]
    assert ds.n_runs == 2
    assert list(ds.event_table["run"].unique()) == [1, 2]
    assert ds.n_voxels == int(bids.mask.sum())
    assert "global_signal" in ds.confounds[0].columns
    one = fm.bids_datasets(bids.root, bids.task, subjects=["sub-02"])
    assert list(one) == ["02"]
    with pytest.raises(FileNotFoundError, match="03"):
        fm.bids_datasets(bids.root, bids.task, subjects=["03"])
    with pytest.raises(FileNotFoundError):
        fm.bids_datasets(bids.root, "nosuchtask")


# ---------------------------------------------------------------------------
# 5. proposal items after round 3 (additive; no default changed)
# ---------------------------------------------------------------------------


def test_ar1_voxelwise_shorthand_equals_aroptions() -> None:
    import fmrimod as fm
    from fmrimod.model.config import AROptions
    from fmrimod.spec import drift, hrf

    inp = make_inputs(seed=15, ar=0.4, n_scans=100)
    ds = _dataset(inp)
    spec = hrf("trial_type") + drift()
    short = fm.fmri_lm(spec, ds, ar="ar1_voxelwise")
    long = fm.fmri_lm(spec, ds, ar=AROptions(struct="ar1", voxelwise=True))
    np.testing.assert_allclose(short.betas, long.betas)
    pooled = fm.fmri_lm(spec, ds, ar="ar1")
    assert not np.allclose(pooled.betas, short.betas)
    with pytest.raises(ValueError, match="unknown ar"):
        fm.fmri_lm(spec, ds, ar="ar1_global")


def test_group_accessors_resolve_unambiguous_levels() -> None:
    import fmrimod as fm
    from benchmarks.ergonomics.inputs import make_group_map_inputs

    maps = make_group_map_inputs(n_per_group=5)
    res = fm.fmri_group_lm(
        maps.paths, formula="~ group + age", covariates=maps.participants
    )
    np.testing.assert_array_equal(res.t_stat("patient"), res.t_stat("group[T.patient]"))
    np.testing.assert_array_equal(res.effect("patient"), res.effect("group[T.patient]"))
    with pytest.raises(KeyError, match="control"):
        res.t_stat("control")


def test_group_covariates_read_from_participants_tsv(tmp_path) -> None:
    import fmrimod as fm
    from benchmarks.ergonomics.inputs import make_group_map_inputs

    maps = make_group_map_inputs(n_per_group=5)
    table = maps.participants.assign(handedness=["R"] * 9 + ["n/a"])
    path = tmp_path / "participants.tsv"
    table.to_csv(path, sep="\t", index=False)
    from_path = fm.fmri_group_lm(maps.paths, formula="~ group + age", covariates=path)
    from_frame = fm.fmri_group_lm(maps.paths, formula="~ group + age", covariates=table)
    np.testing.assert_allclose(from_path.t_stat("age"), from_frame.t_stat("age"))


def test_pool_modulators_gives_one_shared_slope(one_run) -> None:
    import fmrimod as fm
    from fmrimod.spec import as_spec, hrf
    from fmrimod.spec.serialize import from_dict, to_dict

    pooled = hrf("trial_type", modulators=["rt"], pool_modulators=True)
    fit = fm.fmri_lm(pooled, _dataset(one_run))
    slopes = fit.design_columns().where(term="trial_type:rt").columns
    assert len(slopes) == 1 and slopes[0].level == "rt"
    events = one_run.events[0]
    X = fit.model.design_matrix(run=0).to_numpy()
    per_level = fm.fmri_lm(
        hrf("trial_type", modulators=["rt"]), _dataset(one_run)
    ).model.design_matrix(run=0)
    assert per_level.shape[1] == X.shape[1] + events["trial_type"].nunique() - 1
    assert fit.contrast("rt").stat.shape == (one_run.n_voxels,)
    payload = to_dict(as_spec(pooled))
    assert from_dict(payload).events[0].pool_modulators is True
    plain = to_dict(as_spec(hrf("trial_type", modulators=["rt"])))
    assert "pool_modulators" not in str(plain)  # Spec/v1 payloads unchanged


def test_short_column_names_view(one_run) -> None:
    import fmrimod as fm
    from fmrimod.spec import confounds, drift, hrf

    inp = make_inputs(seed=16, n_runs=2, n_scans=80, with_confounds=True)
    spec = (
        hrf("trial_type", modulators=["rt"]) + confounds(source=inp.confounds) + drift()
    )
    fit = fm.fmri_lm(spec, inp.imgs, tr=inp.tr, events=inp.events, mask=inp.mask_img)
    short = fit.model.design_matrix(names="short")
    long = fit.model.design_matrix()
    np.testing.assert_array_equal(short.to_numpy(), long.to_numpy())
    assert list(short.columns[:4]) == ["face", "house", "rt:face", "rt:house"]
    assert {"drift1_run1", "constant_run2", "trans_x_run2"} <= set(short.columns)
    assert short.columns.is_unique
    frame = fm.design_matrix(
        spec, events=inp.events_df, n_scans=[80, 80], tr=inp.tr, names="short"
    )
    assert list(frame.columns) == list(short.columns)
    one = fm.design_matrix(
        hrf("trial_type") + drift(),
        events=one_run.events[0],
        n_scans=160,
        tr=one_run.tr,
        names="short",
    )
    assert list(one.columns[:2]) == ["face", "house"]
    assert "constant" in one.columns


@pytest.mark.parametrize("struct", ["ar1", "ar2"])
@pytest.mark.parametrize("n_runs", [1, 2])
def test_voxelwise_ar_fast_paths_equal_per_voxel_reference(
    monkeypatch, struct, n_runs
) -> None:
    import fmrimod as fm
    import fmrimod.ar.integration as integration
    from fmrimod.model.config import AROptions
    from fmrimod.spec import drift, hrf

    inp = make_inputs(seed=17, ar=0.4, n_runs=n_runs, n_scans=100, shape=(8, 8, 6))
    ds = fm.fmri_dataset(inp.imgs, mask=inp.mask_img, tr=inp.tr, events=inp.events)
    spec = hrf("trial_type") + drift()
    ar = AROptions(struct=struct, voxelwise=True)
    fast = fm.fmri_lm(spec, ds, ar=ar)

    def per_voxel_phi(rm, order, noise_pools, bin_width=None):
        phi = np.zeros((order, rm.resid.shape[1]))
        for v in range(rm.resid.shape[1]):
            phi[:, v] = integration._estimate_phi([rm.resid[:, v : v + 1]], order)
        return phi

    monkeypatch.setattr(integration, "_voxelwise_phi", per_voxel_phi)
    monkeypatch.setattr(integration, "_solve_voxelwise_batched", lambda rm, phi: None)
    reference = fm.fmri_lm(spec, ds, ar=ar)
    np.testing.assert_allclose(fast.betas, reference.betas, rtol=1e-9, atol=1e-9)
    a, b = fast.contrast("face - house"), reference.contrast("face - house")
    np.testing.assert_allclose(a.stat, b.stat, rtol=1e-9, atol=1e-9)
    assert a.df == b.df


def test_voxelwise_yule_walker_defers_degenerate_voxels() -> None:
    from fmrimod.ar.integration import _estimate_phi, _voxelwise_yule_walker

    rng = np.random.default_rng(3)
    x = rng.normal(size=(120, 50))
    x[:, 0] = 0.0  # no variance
    x[:, 1] = np.arange(120.0)  # trend: PACF at the clamp
    x[:, 2] = np.nan
    finite = np.all(np.isfinite(x), axis=0)
    phi, ok = _voxelwise_yule_walker(x, 1, finite)
    assert not ok[:3].any() and ok[3:].all()
    reference = np.array([_estimate_phi([x[:, v : v + 1]], 1)[0] for v in range(3, 50)])
    np.testing.assert_allclose(phi[0, 3:], reference, atol=1e-13)


# ---------------------------------------------------------------------------
# 6. round 5: thresholding, binned AR, spine promotion
# ---------------------------------------------------------------------------


def test_height_thresholds_match_nilearn_rules() -> None:
    from nilearn.glm import fdr_threshold
    from scipy import stats

    from fmrimod.glm.thresholding import height_threshold

    rng = np.random.default_rng(5)
    z = np.concatenate([rng.normal(size=900), rng.normal(4.0, 1.0, size=100)])
    ours = height_threshold(z, alpha=0.05, method="fdr", height=None, two_sided=True)
    assert ours == pytest.approx(fdr_threshold(np.abs(z), 0.025), abs=1e-9)
    bonf = height_threshold(
        z, alpha=0.05, method="bonferroni", height=None, two_sided=True
    )
    assert bonf == pytest.approx(stats.norm.isf(0.025 / z.size))
    null = height_threshold(
        rng.normal(size=50), alpha=0.001, method="fdr", height=None, two_sided=True
    )
    assert null == np.inf
    with pytest.raises(ValueError, match="height"):
        height_threshold(z, alpha=0.05, method="none", height=None, two_sided=True)


def test_contrast_threshold_matches_nilearn_threshold_stats_img() -> None:
    from benchmarks.ergonomics.scenarios import (
        _masked,
        fmrimod_fdr_clusters,
        nilearn_fdr_clusters,
    )

    inp = make_inputs(seed=15, shape=(14, 14, 10), noise_sd=2.0)
    ours, table = fmrimod_fdr_clusters(inp)
    theirs, their_table = nilearn_fdr_clusters(inp)
    reference = _masked(theirs, inp)
    np.testing.assert_array_equal(ours.z != 0, reference != 0)
    # Values differ only by the small HRF-sampling gap between the designs.
    assert np.corrcoef(ours.z, reference)[0, 1] > 0.9999
    np.testing.assert_allclose(ours.z, reference, rtol=0.02)
    top = their_table.iloc[0]
    assert (
        table.loc[0, ["x", "y", "z"]].to_numpy() == top[["X", "Y", "Z"]].to_numpy()
    ).all()
    assert table.loc[0, "volume_mm3"] == pytest.approx(float(top["Cluster Size (mm3)"]))


def test_threshold_cluster_extent_and_one_sided_f(one_run) -> None:
    import nibabel as nib

    import fmrimod as fm
    from fmrimod.contrast import OmnibusContrast
    from fmrimod.spec import drift, hrf

    fit = fm.fmri_lm(hrf("trial_type") + drift(), _dataset(one_run))
    t = fit.contrast("face - house")
    loose = t.threshold(method="none", height=1.0)
    strict = t.threshold(method="none", height=1.0, cluster_size=40)
    assert strict.n_suprathreshold <= loose.n_suprathreshold
    assert (strict.clusters()["size"] >= 40).all()
    f = fit.contrast(OmnibusContrast(term="trial_type")).threshold(0.05)
    assert f.two_sided is False and (f.z >= 0).all()
    out = strict.to_nifti(one_run.out_dir / "thr.nii.gz")
    assert nib.load(out).shape == one_run.mask.shape


def test_group_threshold_uses_group_z() -> None:
    import fmrimod as fm
    from benchmarks.ergonomics.inputs import make_group_map_inputs

    maps = make_group_map_inputs(n_per_group=8)
    res = fm.fmri_group_lm(
        maps.paths, formula="~ group + age", covariates=maps.participants
    )
    hits = res.threshold("patient", 0.05)
    expected = np.abs(res.z_score("patient")) > hits.threshold
    np.testing.assert_array_equal(hits.z != 0, expected)
    assert not hits.clusters().empty


def test_binned_ar_truncates_onto_nilearn_grid(monkeypatch) -> None:
    import fmrimod as fm
    import fmrimod.ar.integration as integration
    from fmrimod.model.config import AROptions
    from fmrimod.spec import drift, hrf

    seen = []
    original = integration._voxelwise_phi

    def spy(*args, **kwargs):
        phi = original(*args, **kwargs)
        seen.append(phi)
        return phi

    monkeypatch.setattr(integration, "_voxelwise_phi", spy)
    inp = make_inputs(seed=18, ar=0.4, n_scans=100, shape=(8, 8, 6))
    ds = _dataset(inp)
    binned = fm.fmri_lm(hrf("trial_type") + drift(), ds, ar="ar1_binned")
    phi = seen[-1]
    np.testing.assert_allclose(phi * 100, np.round(phi * 100), atol=1e-9)
    assert np.unique(phi).size < phi.size
    explicit = fm.fmri_lm(
        hrf("trial_type") + drift(),
        ds,
        ar=AROptions(struct="ar1", voxelwise=True, bin_width=0.01),
    )
    np.testing.assert_allclose(binned.betas, explicit.betas)
    with pytest.raises(ValueError, match="voxelwise"):
        AROptions(struct="ar1", bin_width=0.01)
    with pytest.raises(ValueError, match="not both"):
        AROptions(struct="ar1", voxelwise=True, bin_width=0.01, noise_pools=10)


# ---------------------------------------------------------------------------
# 7. round 5 critic findings (independent review)
# ---------------------------------------------------------------------------


def test_multirun_fit_refuses_events_without_run_labels() -> None:
    import fmrimod as fm
    from fmrimod.spec import hrf

    inp = make_inputs(seed=19, n_runs=2, n_scans=60, shape=(6, 6, 5))
    with pytest.raises(ValueError, match="1 events table"):
        fm.fmri_dataset(inp.imgs, mask=inp.mask_img, tr=2.0, events=[inp.events[0]])
    ds = fm.fmri_dataset(inp.imgs, mask=inp.mask_img, tr=2.0, events=inp.events[0])
    with pytest.raises(ValueError, match="no `run` column"):
        fm.fmri_lm(hrf("trial_type"), ds)
    bad = inp.events_df.assign(run=inp.events_df["run"] + 5)
    with pytest.raises(ValueError, match="run numbers 1..2"):
        fm.fmri_lm(hrf("trial_type"), inp.imgs, tr=2.0, events=bad, mask=inp.mask_img)


def test_group_maps_join_covariates_by_file_subject_id() -> None:
    import fmrimod as fm
    from benchmarks.ergonomics.inputs import make_group_map_inputs

    maps = make_group_map_inputs(n_per_group=6)
    ordered = fm.fmri_group_lm(
        maps.paths, formula="~ group + age", covariates=maps.participants
    )
    shuffled = fm.fmri_group_lm(
        maps.paths,
        formula="~ group + age",
        covariates=maps.participants.sample(frac=1.0, random_state=3),
    )
    np.testing.assert_allclose(shuffled.t_stat("patient"), ordered.t_stat("patient"))
    with pytest.raises(ValueError, match="no row for subjects"):
        fm.fmri_group_lm(
            maps.paths, formula="~ group", covariates=maps.participants.iloc[:-1]
        )


def test_group_lm_intersects_differing_first_level_masks(one_run) -> None:
    import nibabel as nib

    import fmrimod as fm
    from fmrimod.spec import hrf

    smaller = one_run.mask.copy()
    smaller[tuple(np.argwhere(smaller)[0])] = False
    smaller_img = nib.Nifti1Image(smaller.astype(np.uint8), one_run.img.affine)
    kw = {"tr": one_run.tr, "events": one_run.events[0]}
    a = fm.fmri_lm(hrf("trial_type"), one_run.img, mask=one_run.mask_img, **kw)
    b = fm.fmri_lm(hrf("trial_type"), one_run.img, mask=smaller_img, **kw)
    with pytest.warns(UserWarning, match="masks differ"):
        res = fm.fmri_group_lm({"a": a.contrast("face"), "b": b.contrast("face")})
    assert res.t_stat().shape == (int(smaller.sum()),)
    assert res.spatial is not None and res.spatial.mask.sum() == smaller.sum()


def test_fits_and_group_results_pickle(one_run) -> None:
    import pickle

    import fmrimod as fm
    from fmrimod.spec import drift, hrf

    fit = fm.fmri_lm(hrf("trial_type") + drift(), _dataset(one_run))
    clone = pickle.loads(pickle.dumps(fit))
    np.testing.assert_array_equal(clone.betas, fit.betas)
    np.testing.assert_allclose(
        clone.contrast("face - house").stat, fit.contrast("face - house").stat
    )
    group = fm.fmri_group_lm(np.random.default_rng(0).normal(size=(5, 7)))
    np.testing.assert_allclose(
        pickle.loads(pickle.dumps(group)).t_stat(), group.t_stat()
    )


def test_nilearn_keywords_get_fmrimod_spelling(one_run) -> None:
    import fmrimod as fm
    from fmrimod.spec import hrf

    with pytest.raises(TypeError, match="ar1_voxelwise"):
        fm.fmri_lm(hrf("trial_type"), _dataset(one_run), noise_model="ar1")
    with pytest.raises(TypeError, match="drift"):
        fm.fmri_lm(hrf("trial_type"), _dataset(one_run), high_pass=0.01)
    with pytest.raises(TypeError, match="model first"):
        fm.fmri_lm(one_run.img, tr=2.0, events=one_run.events[0])


def test_non_estimable_contrast_warns(one_run) -> None:
    import warnings

    import fmrimod as fm
    from fmrimod.spec import hrf

    events = one_run.events[0]
    twin = events[events.trial_type == "face"].assign(trial_type="faceB")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fm.fmri_lm(
            hrf("trial_type"),
            one_run.img,
            tr=one_run.tr,
            events=pd.concat([events, twin], ignore_index=True),
            mask=one_run.mask_img,
        )
    with pytest.warns(UserWarning, match="not estimable"):
        fit.contrast("face")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fit.contrast("face + faceB")
        fit.contrast("house")


def test_to_img_and_contrast_spellings(one_run) -> None:
    import fmrimod as fm
    from fmrimod.contrast import OmnibusContrast
    from fmrimod.spec import drift, hrf

    fit = fm.fmri_lm(hrf("trial_type") + drift(), _dataset(one_run))
    res = fit.contrast("face - house")
    img = res.to_img()
    np.testing.assert_allclose(np.asarray(img.dataobj)[one_run.mask], res.z)
    assert "ContrastResult(" in repr(res) and len(repr(res)) < 200
    rows = fit.contrast(["face", "house"])
    omnibus = fit.contrast(OmnibusContrast(term="trial_type"))
    np.testing.assert_allclose(rows.stat, omnibus.stat)
    by_name = fit.contrast("trial_type_trial_type.face - trial_type_trial_type.house")
    np.testing.assert_allclose(by_name.stat, res.stat)
    assert fit.contrast("drift1").stat.shape == res.stat.shape
    batch = fit.compute_contrasts({"fh": "face - house"})
    np.testing.assert_allclose(batch["fh"].stat, res.stat)
