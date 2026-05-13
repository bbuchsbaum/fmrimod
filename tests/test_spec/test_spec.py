"""Tests for the typed Spec / Term object tree."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import fmrimod as fm
from fmrimod.glm.fmri_lm import FmriLm

# ``hrf`` and ``trialwise`` are intentionally NOT at top level because they
# collide with the existing ``fmrimod.hrf`` and ``fmrimod.trialwise``
# submodules.  Import from ``fmrimod.spec`` directly.
from fmrimod.spec import (  # noqa: F401
    Confounds,
    Drift,
    HrfTerm,
    Intercept,
    Spec,
    Term,
    as_spec,
    hrf,
    is_spec,
    legacy_formula_to_spec,
    trialwise,
)
from fmrimod.spec import (
    compile as compile_spec,
)

# -- Term construction -----------------------------------------------------


def test_hrf_builder_returns_frozen_hrfterm():
    term = hrf("trial_type", basis="spm")
    assert isinstance(term, HrfTerm)
    assert term.variables == ("trial_type",)
    assert term.hrf == "spm"
    # Frozen — attribute assignment should fail.
    with pytest.raises(Exception):
        term.variables = ("other",)  # type: ignore[misc]


def test_hrf_builder_carries_convolution_options():
    term = hrf("trial_type", basis="spmg1", normalize=True, summate=False)
    assert term.normalize is True
    assert term.summate is False


def test_hrf_builder_interaction_term():
    term = hrf("trial_type", "block")
    assert term.variables == ("trial_type", "block")


def test_hrf_builder_requires_at_least_one_variable():
    with pytest.raises(ValueError, match="at least one variable"):
        hrf()


def test_drift_builder_defaults():
    d = fm.drift("cosine", cutoff=128)
    assert isinstance(d, Drift)
    assert d.basis == "cosine"
    assert d.cutoff == 128
    assert d.degree == 1


def test_intercept_builder_per_run_default():
    i = fm.intercept()
    assert isinstance(i, Intercept)
    assert i.per == "run"


def test_confounds_builder_requires_columns():
    with pytest.raises(ValueError, match="at least one column"):
        fm.confounds()


# -- Composition -----------------------------------------------------------


def test_term_plus_term_yields_spec():
    spec = hrf("a") + fm.drift("cosine", cutoff=128)
    assert isinstance(spec, Spec)
    assert len(spec.events) == 1
    assert len(spec.baseline) == 1


def test_spec_plus_term_extends_in_place_immutably():
    base = hrf("a") + fm.drift("cosine", cutoff=128)
    extended = base + fm.intercept(per="run")
    assert len(base.baseline) == 1  # original unchanged
    assert len(extended.baseline) == 2


def test_spec_plus_spec_concatenates_buckets():
    s1 = hrf("a") + fm.drift("cosine", cutoff=128)
    s2 = hrf("b") + fm.intercept()
    combined = s1 + s2
    assert len(combined.events) == 2
    assert len(combined.baseline) == 2


def test_routing_event_vs_baseline_terms():
    spec = (
        hrf("a")
        + hrf("b")
        + fm.drift("cosine", cutoff=128)
        + fm.intercept(per="run")
    )
    assert all(isinstance(t, HrfTerm) for t in spec.events)
    assert any(isinstance(t, Drift) for t in spec.baseline)
    assert any(isinstance(t, Intercept) for t in spec.baseline)


def test_iterating_spec_yields_events_then_baseline():
    spec = hrf("a") + fm.drift() + fm.intercept()
    order = list(spec)
    assert isinstance(order[0], HrfTerm)
    assert isinstance(order[1], Drift)
    assert isinstance(order[2], Intercept)


def test_is_spec_and_as_spec_helpers():
    term = hrf("a")
    spec = hrf("a") + fm.drift()
    assert is_spec(term)
    assert is_spec(spec)
    assert not is_spec("hrf(a)")
    coerced = as_spec([hrf("a"), fm.drift()])
    assert isinstance(coerced, Spec)
    assert len(coerced) == 2


# -- Compilation to EventModel + BaselineModel -----------------------------


@pytest.fixture
def synthetic_run():
    rng = np.random.default_rng(2024)
    tr = 2.0
    n_scans = 60
    onsets = np.arange(0.0, n_scans * tr, 24.0)
    n_events = len(onsets)
    half = n_events // 2
    trial_types = np.array(
        ["listening"] * half + ["rest"] * (n_events - half), dtype=object
    )
    rng.shuffle(trial_types)
    events = pd.DataFrame(
        {
            "onset": onsets,
            "trial_type": trial_types,
            "duration": np.full(n_events, 12.0),
            "run": np.ones(n_events, dtype=int),
        }
    )
    Y = rng.normal(100.0, 1.0, (n_scans, 8)).astype(np.float64)
    return events, Y, tr


def test_compile_returns_event_and_baseline_models(synthetic_run):
    events, Y, tr = synthetic_run
    ds = fm.fmri_dataset(Y, tr=tr, events=events)
    sf = ds.sampling_frame
    spec = hrf("trial_type") + fm.intercept(per="run")
    em, bm = compile_spec(spec, data=events, sampling_frame=sf, block="run", durations="duration")
    assert em is not None
    assert bm is not None


def test_compile_typed_spec_bypasses_string_formula_parser(synthetic_run, monkeypatch):
    events, Y, tr = synthetic_run
    ds = fm.fmri_dataset(Y, tr=tr, events=events)

    def fail_parse(*_args, **_kwargs):
        raise AssertionError("typed Spec lowering should not parse formula strings")

    monkeypatch.setattr("fmrimod.formula.parser.parse_formula", fail_parse)

    em, bm = compile_spec(
        hrf("trial_type", norm="spm") + fm.intercept(per="run"),
        data=events,
        sampling_frame=ds.sampling_frame,
        block="run",
        durations="duration",
    )

    assert em is not None
    assert bm is not None
    assert em.design_matrix.shape[0] == Y.shape[0]


def test_compile_typed_spec_preserves_hrf_options(synthetic_run):
    events, _Y, tr = synthetic_run
    ds = fm.fmri_dataset(np.zeros((60, 2)), tr=tr, events=events)

    spec = (
        hrf(
            "trial_type",
            basis="spmg1",
            normalize=True,
            summate=False,
            durations="duration",
            subset={"trial_type": "listening"},
            prefix="stim",
            id="stim_term",
            lag=1.25,
        )
        + fm.intercept(per="run")
    )
    em, _bm = compile_spec(
        spec,
        data=events,
        sampling_frame=ds.sampling_frame,
        block="run",
        durations="duration",
    )

    lowered = em.terms[0]
    assert lowered.events == ["trial_type"]
    assert lowered.hrf == "spmg1"
    assert lowered.name == "stim_term"
    assert lowered.normalize is True
    assert lowered.summate is False
    assert lowered.kwargs["durations"] == "duration"
    assert lowered.kwargs["subset"] == {"trial_type": "listening"}
    assert lowered.kwargs["prefix"] == "stim"
    assert lowered.kwargs["lag"] == 1.25


def test_legacy_formula_to_spec_preserves_hrf_options():
    spec = legacy_formula_to_spec(
        (
            "onset ~ hrf(trial_type, basis='spmg1', normalize=True, "
            "summate=False, id='stim_term', prefix='stim', lag=1.25, "
            "durations='duration', subset='trial_type == \"listening\"')"
        )
    )

    assert isinstance(spec, Spec)
    term = spec.events[0]
    assert isinstance(term, HrfTerm)
    assert term.variables == ("trial_type",)
    assert term.hrf == "spmg1"
    assert term.id == "stim_term"
    assert term.normalize is True
    assert term.summate is False
    assert term.prefix == "stim"
    assert term.lag == 1.25
    assert term.durations == "duration"
    assert term.subset == 'trial_type == "listening"'


def test_legacy_functional_terms_to_spec_preserve_hrf_options():
    from fmrimod.formula import hrf as formula_hrf
    from fmrimod.formula import term as formula_term

    spec = legacy_formula_to_spec(
        [
            formula_term("trial_type")
            | formula_hrf(
                "spmg1",
                normalize=True,
                summate=False,
                prefix="stim",
                durations="duration",
            )
        ]
    )

    term = spec.events[0]
    assert isinstance(term, HrfTerm)
    assert term.variables == ("trial_type",)
    assert term.hrf == "spmg1"
    assert term.normalize is True
    assert term.summate is False
    assert term.prefix == "stim"
    assert term.durations == "duration"


def test_compile_baseline_only_spec_uses_default_constant_intercept(synthetic_run):
    events, Y, tr = synthetic_run
    ds = fm.fmri_dataset(Y, tr=tr, events=events)
    sf = ds.sampling_frame
    em, bm = compile_spec(
        Spec(),
        data=events,
        sampling_frame=sf,
        block="run",
        durations="duration",
    )
    # No event terms — em is None; bm still produced with default settings.
    assert em is None
    assert bm is not None


def test_compile_rejects_multiple_drift_terms(synthetic_run):
    events, _Y, tr = synthetic_run
    sf = fm.SamplingFrame(blocklens=[60], tr=tr)
    spec = hrf("trial_type") + fm.drift("cosine", cutoff=128) + fm.drift("poly", degree=2)
    with pytest.raises(ValueError, match="at most one Drift"):
        compile_spec(spec, data=events, sampling_frame=sf, block="run", durations="duration")


# -- End-to-end: fmri_lm accepts Spec --------------------------------------


def test_fmri_lm_accepts_typed_spec(synthetic_run):
    events, Y, tr = synthetic_run
    ds = fm.fmri_dataset(Y, tr=tr, events=events)

    spec = hrf("trial_type") + fm.intercept(per="run")
    fit = fm.fmri_lm(spec, ds)
    assert isinstance(fit, FmriLm)
    assert fit.n_voxels == Y.shape[1]


def test_fmri_lm_accepts_single_term(synthetic_run):
    events, Y, tr = synthetic_run
    ds = fm.fmri_dataset(Y, tr=tr, events=events)

    # Single Term — should be coerced to a Spec internally.
    fit = fm.fmri_lm(hrf("trial_type"), ds)
    assert isinstance(fit, FmriLm)


def test_fmri_lm_spec_with_drift_produces_more_columns(synthetic_run):
    events, Y, tr = synthetic_run
    ds = fm.fmri_dataset(Y, tr=tr, events=events)

    base = fm.fmri_lm(hrf("trial_type") + fm.intercept(per="run"), ds)
    with_drift = fm.fmri_lm(
        hrf("trial_type") + fm.drift("poly", degree=2) + fm.intercept(per="run"),
        ds,
    )
    assert with_drift.n_coefficients > base.n_coefficients


def test_fmri_lm_string_formula_path_still_works(synthetic_run):
    """String-formula path stays the canonical sugar — covered by older
    tests, but re-asserted here to lock the coexistence guarantee."""
    events, Y, tr = synthetic_run
    ds = fm.fmri_dataset(Y, tr=tr, events=events)
    fit = fm.fmri_lm("hrf(trial_type)", ds)
    assert isinstance(fit, FmriLm)


def test_top_level_spec_builders_visible_via_fm():
    # ``fm.drift``, ``fm.intercept``, ``fm.confounds`` are bound at the top
    # level. ``hrf`` and ``trialwise`` are NOT — the existing
    # ``fmrimod.hrf`` and ``fmrimod.trialwise`` submodules own those slots.
    assert callable(fm.drift)
    assert callable(fm.intercept)
    assert callable(fm.confounds)
    # The Spec ``hrf`` / ``trialwise`` builders are reachable via the
    # ``fmrimod.spec`` namespace.
    from fmrimod.spec import hrf as spec_hrf
    from fmrimod.spec import trialwise as spec_trialwise
    assert callable(spec_hrf)
    assert callable(spec_trialwise)


def test_hrf_submodule_still_importable_despite_top_level_binding():
    """Critical: the spec ``hrf`` binding must not break access to
    ``fmrimod.hrf`` as a submodule for everyone else."""
    from fmrimod.hrf import HRF, HRF_SPMG1, bind_basis  # noqa: F401
    assert HRF_SPMG1 is not None
