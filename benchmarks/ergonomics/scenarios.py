"""Paired fmrimod / Nilearn implementations of common analysis tasks.

Each scenario is written twice, as an analyst would write it with each
library: start from images + events (+ confounds) and stop at the natural
result object of that library. The function bodies are the thing being
measured (statements, tokens, keyword arguments, imports, wall time), so:

* imports live inside the function body — knowing what to import is part of
  the cost of an API;
* no shared helpers are called from inside a measured body;
* a line ending in ``# parity-knob`` exists only to make the two outputs
  numerically comparable (e.g. aligning a sampling-grid convention). Such
  lines are excluded from the ergonomic counts and reported separately, so
  neither library is charged for the other's defaults.

Converting results to plain arrays for the agreement check happens in the
``extract_*`` adapters, which are not measured.

When an fmrimod body here looks clumsy, that is a finding: fix the API, then
shorten the body. Do not hide the clumsiness in a helper.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from benchmarks.ergonomics.inputs import Inputs, make_group_inputs, make_inputs

# --------------------------------------------------------------------------
# S01  two-condition t contrast, one run
# --------------------------------------------------------------------------


def fmrimod_two_condition_t(inp: Inputs):
    import fmrimod as fm
    from fmrimod.spec import drift, hrf

    fit = fm.fmri_lm(
        hrf("trial_type") + drift(),
        inp.img,
        tr=inp.tr,
        events=inp.events[0],
        mask=inp.mask_img,
        slice_timing_offset=0.0,  # parity-knob
    )
    return fit.contrast("face - house")


def nilearn_two_condition_t(inp: Inputs):
    from nilearn.glm.first_level import FirstLevelModel

    model = FirstLevelModel(
        t_r=inp.tr,
        mask_img=inp.mask_img,
        hrf_model="spm",
        noise_model="ols",  # parity-knob
        high_pass=1 / 128,  # parity-knob
        signal_scaling=False,  # parity-knob
    )
    model.fit(inp.img, events=inp.events[0])
    return model.compute_contrast("face - house", output_type="stat")


# --------------------------------------------------------------------------
# S02  three runs, motion confounds, fixed-effects contrast across runs
# --------------------------------------------------------------------------


def fmrimod_multirun_confounds(inp: Inputs):
    import fmrimod as fm
    from fmrimod.spec import confounds, drift, hrf

    fit = fm.fmri_lm(
        hrf("trial_type") + confounds(source=inp.confounds) + drift(),
        inp.imgs,
        tr=inp.tr,
        events=inp.events,
        mask=inp.mask_img,
        slice_timing_offset=0.0,  # parity-knob
    )
    return fit.contrast("face - house")


def nilearn_multirun_confounds(inp: Inputs):
    from nilearn.glm.first_level import FirstLevelModel

    model = FirstLevelModel(
        t_r=inp.tr,
        mask_img=inp.mask_img,
        hrf_model="spm",
        noise_model="ols",  # parity-knob
        high_pass=1 / 128,  # parity-knob
        signal_scaling=False,  # parity-knob
    )
    model.fit(inp.imgs, events=inp.events, confounds=inp.confounds)
    return model.compute_contrast("face - house", output_type="stat")


# --------------------------------------------------------------------------
# S03  parametric modulation by reaction time
# --------------------------------------------------------------------------


def fmrimod_parametric_rt(inp: Inputs):
    import fmrimod as fm
    from fmrimod.spec import drift, hrf

    fit = fm.fmri_lm(
        hrf("trial_type", modulators=["rt"]) + drift(),
        inp.img,
        tr=inp.tr,
        events=inp.events[0],
        mask=inp.mask_img,
        slice_timing_offset=0.0,  # parity-knob
    )
    return fit.contrast("trial_type:rt.face + trial_type:rt.house")


def nilearn_parametric_rt(inp: Inputs):
    import pandas as pd
    from nilearn.glm.first_level import FirstLevelModel

    events = inp.events[0].assign(modulation=1.0)
    rt = events.assign(trial_type="rt", modulation=events["rt"] - events["rt"].mean())
    all_events = pd.concat([events, rt], ignore_index=True)
    model = FirstLevelModel(
        t_r=inp.tr,
        mask_img=inp.mask_img,
        hrf_model="spm",
        noise_model="ols",  # parity-knob
        high_pass=1 / 128,  # parity-knob
        signal_scaling=False,  # parity-knob
    )
    model.fit(inp.img, events=all_events)
    return model.compute_contrast("rt", output_type="stat")


# --------------------------------------------------------------------------
# S04  omnibus F test across three conditions
# --------------------------------------------------------------------------


def fmrimod_omnibus_f(inp: Inputs):
    import fmrimod as fm
    from fmrimod.contrast import OmnibusContrast
    from fmrimod.spec import drift, hrf

    fit = fm.fmri_lm(
        hrf("trial_type") + drift(),
        inp.img,
        tr=inp.tr,
        events=inp.events[0],
        mask=inp.mask_img,
        slice_timing_offset=0.0,  # parity-knob
    )
    return fit.contrast(OmnibusContrast(term="trial_type"))


def nilearn_omnibus_f(inp: Inputs):
    from nilearn.glm.first_level import FirstLevelModel

    model = FirstLevelModel(
        t_r=inp.tr,
        mask_img=inp.mask_img,
        hrf_model="spm",
        noise_model="ols",  # parity-knob
        high_pass=1 / 128,  # parity-knob
        signal_scaling=False,  # parity-knob
    )
    model.fit(inp.img, events=inp.events[0])
    columns = model.design_matrices_[0].columns
    conditions = inp.events[0]["trial_type"].unique()
    matrix = np.array([columns == c for c in conditions], dtype=float)
    return model.compute_contrast(matrix, stat_type="F", output_type="stat")


# --------------------------------------------------------------------------
# S05  FIR basis: omnibus F over the FIR delays of one condition
# --------------------------------------------------------------------------

FIR_DELAYS = 8


def fmrimod_fir_shape(inp: Inputs):
    import fmrimod as fm
    from fmrimod.contrast import OmnibusContrast
    from fmrimod.spec import drift, hrf

    fir = hrf("trial_type", basis="fir", nbasis=FIR_DELAYS, span=FIR_DELAYS * inp.tr)
    fit = fm.fmri_lm(
        fir + drift(),
        inp.img,
        tr=inp.tr,
        events=inp.events[0],
        mask=inp.mask_img,
        slice_timing_offset=0.0,  # parity-knob
    )
    return fit.contrast(OmnibusContrast(term="trial_type", levels=("face",)))


def nilearn_fir_shape(inp: Inputs):
    from nilearn.glm.first_level import FirstLevelModel

    model = FirstLevelModel(
        t_r=inp.tr,
        mask_img=inp.mask_img,
        hrf_model="fir",
        fir_delays=range(FIR_DELAYS),
        noise_model="ols",  # parity-knob
        high_pass=1 / 128,  # parity-knob
        signal_scaling=False,  # parity-knob
    )
    model.fit(inp.img, events=inp.events[0])
    columns = model.design_matrices_[0].columns
    matrix = np.array(
        [columns == f"face_delay_{d}" for d in range(FIR_DELAYS)], dtype=float
    )
    return model.compute_contrast(matrix, stat_type="F", output_type="stat")


# --------------------------------------------------------------------------
# S06  AR(1) prewhitened t contrast
# --------------------------------------------------------------------------


def fmrimod_ar1_t(inp: Inputs):
    import fmrimod as fm
    from fmrimod.spec import drift, hrf

    fit = fm.fmri_lm(
        hrf("trial_type") + drift(),
        inp.img,
        tr=inp.tr,
        events=inp.events[0],
        mask=inp.mask_img,
        ar="ar1",
        slice_timing_offset=0.0,  # parity-knob
    )
    return fit.contrast("face - house")


def nilearn_ar1_t(inp: Inputs):
    from nilearn.glm.first_level import FirstLevelModel

    model = FirstLevelModel(
        t_r=inp.tr,
        mask_img=inp.mask_img,
        hrf_model="spm",
        noise_model="ar1",
        high_pass=1 / 128,  # parity-knob
        signal_scaling=False,  # parity-knob
    )
    model.fit(inp.img, events=inp.events[0])
    return model.compute_contrast("face - house", output_type="stat")


# --------------------------------------------------------------------------
# S07  LSS beta series (one beta map per trial)
# --------------------------------------------------------------------------


def fmrimod_lss_beta_series(inp: Inputs):
    import fmrimod as fm
    from fmrimod.spec import trialwise

    ds = fm.fmri_dataset(
        inp.img,
        mask=inp.mask_img,
        tr=inp.tr,
        events=inp.events[0],
        slice_timing_offset=0.0,  # parity-knob
    )
    return fm.estimate_single_trial_from_dataset(
        ds, trialwise(), method="lss", include_intercept=True
    )


def nilearn_lss_beta_series(inp: Inputs):
    # Follows Nilearn's "Beta-Series Modeling ... LSS" example: relabel one
    # trial at a time and refit the whole first-level model.
    from nilearn.glm.first_level import FirstLevelModel

    events = inp.events[0].assign(trial_type="other")
    model = FirstLevelModel(
        t_r=inp.tr,
        mask_img=inp.mask_img,
        hrf_model="spm",
        noise_model="ols",  # parity-knob
        drift_model=None,  # parity-knob
        signal_scaling=False,  # parity-knob
    )
    beta_maps = []
    for i in range(len(events)):
        trial_events = events.copy()
        trial_events.loc[i, "trial_type"] = "target"
        model.fit(inp.img, events=trial_events)
        beta_maps.append(model.compute_contrast("target", output_type="effect_size"))
    return beta_maps


# --------------------------------------------------------------------------
# S08  first level for N subjects, then one-sample group t
# --------------------------------------------------------------------------


def fmrimod_group_one_sample(subjects: list[Inputs]):
    import fmrimod as fm
    from fmrimod.spec import drift, hrf

    spec = hrf("trial_type") + drift()
    per_subject = [
        fm.fmri_lm(
            spec,
            sub.img,
            tr=sub.tr,
            events=sub.events[0],
            mask=sub.mask_img,
            slice_timing_offset=0.0,  # parity-knob
        ).contrast("face - house")
        for sub in subjects
    ]
    return fm.fmri_group_lm(per_subject)


def nilearn_group_one_sample(subjects: list[Inputs]):
    import pandas as pd
    from nilearn.glm.first_level import FirstLevelModel
    from nilearn.glm.second_level import SecondLevelModel

    effect_maps = []
    for sub in subjects:
        model = FirstLevelModel(
            t_r=sub.tr,
            mask_img=sub.mask_img,
            hrf_model="spm",
            noise_model="ols",  # parity-knob
            high_pass=1 / 128,  # parity-knob
            signal_scaling=False,  # parity-knob
        )
        model.fit(sub.img, events=sub.events[0])
        effect_maps.append(
            model.compute_contrast("face - house", output_type="effect_size")
        )
    design = pd.DataFrame({"intercept": [1.0] * len(effect_maps)})
    group = SecondLevelModel(mask_img=subjects[0].mask_img)
    group.fit(effect_maps, design_matrix=design)
    return group.compute_contrast("intercept", output_type="stat")


# --------------------------------------------------------------------------
# S09  design matrix only (no data): build and inspect
# --------------------------------------------------------------------------

DESIGN_SCANS = 160


def fmrimod_design_only(inp: Inputs):
    import fmrimod as fm
    from fmrimod.spec import drift, hrf

    return fm.design_matrix(
        hrf("trial_type") + drift(),
        events=inp.events[0],
        n_scans=DESIGN_SCANS,
        tr=inp.tr,
        start_time=0.0,  # parity-knob
    )


def nilearn_design_only(inp: Inputs):
    from nilearn.glm.first_level import make_first_level_design_matrix

    frame_times = np.arange(DESIGN_SCANS) * inp.tr
    return make_first_level_design_matrix(
        frame_times,
        inp.events[0],
        hrf_model="spm",
        high_pass=1 / 128,  # parity-knob
    )


# --------------------------------------------------------------------------
# Unmeasured adapters: natural result -> masked float array
# --------------------------------------------------------------------------


def _masked(img: Any, inp: Inputs) -> np.ndarray:
    return np.asarray(img.get_fdata(), dtype=np.float64)[inp.mask]


def extract_fmrimod_stat(result: Any, inp: Inputs) -> np.ndarray:
    return np.asarray(result.stat, dtype=np.float64).ravel()


def extract_nilearn_stat(result: Any, inp: Inputs) -> np.ndarray:
    return _masked(result, inp)


def extract_fmrimod_lss(result: Any, inp: Inputs) -> np.ndarray:
    return np.asarray(result.betas, dtype=np.float64).ravel()


def extract_nilearn_lss(result: Any, inp: Inputs) -> np.ndarray:
    return np.stack([_masked(img, inp) for img in result]).ravel()


def extract_fmrimod_group(result: Any, subjects: list[Inputs]) -> np.ndarray:
    return np.asarray(result.t_stat(), dtype=np.float64).ravel()


def extract_nilearn_group(result: Any, subjects: list[Inputs]) -> np.ndarray:
    return _masked(result, subjects[0])


def extract_fmrimod_design(result: Any, inp: Inputs) -> np.ndarray:
    names = list(result.columns)
    task = [c for c in names if c.startswith("trial_type")]
    return np.asarray(result[task], dtype=np.float64).ravel()


def extract_nilearn_design(result: Any, inp: Inputs) -> np.ndarray:
    return np.asarray(result[["face", "house"]], dtype=np.float64).ravel()


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    """One task implemented idiomatically in both libraries."""

    scenario_id: str
    title: str
    fmrimod: Callable[[Any], Any]
    nilearn: Callable[[Any], Any]
    make_inputs: Callable[[], Any]
    extract_fmrimod: Callable[[Any, Any], np.ndarray]
    extract_nilearn: Callable[[Any, Any], np.ndarray]
    min_correlation: float
    notes: tuple[str, ...] = field(default_factory=tuple)
    large_inputs: Callable[[], Any] | None = None


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        "s01_two_condition_t",
        "Two-condition t contrast, one run",
        fmrimod_two_condition_t,
        nilearn_two_condition_t,
        lambda: make_inputs(seed=1),
        extract_fmrimod_stat,
        extract_nilearn_stat,
        0.999,
        large_inputs=lambda: make_inputs(seed=1, shape=(34, 34, 28), n_scans=300),
        notes=(
            "Default noise model differs: Nilearn `FirstLevelModel` is AR(1) by default, fmrimod `fmri_lm` is OLS (aligned here with a parity knob; now stated in the `fmri_lm` docstring).",
            "Default sampling grid is the TR midpoint (`slice_timing_offset=TR/2`); Nilearn/SPM/FitLins use frame start. Any cross-tool comparison needs `slice_timing_offset=0.0`.",
            "Drift is opt-in: `fmri_lm` fits no high-pass unless the spec has `drift()`; Nilearn applies a 100 s cosine by default (fmrimod's `drift()` default is SPM's 128 s).",
        ),
    ),
    Scenario(
        "s02_multirun_confounds",
        "Three runs + six motion confounds, fixed effects",
        fmrimod_multirun_confounds,
        nilearn_multirun_confounds,
        lambda: make_inputs(seed=2, n_runs=3, with_confounds=True),
        extract_fmrimod_stat,
        extract_nilearn_stat,
        0.999,
    ),
    Scenario(
        "s03_parametric_rt",
        "Parametric modulation by reaction time",
        fmrimod_parametric_rt,
        nilearn_parametric_rt,
        lambda: make_inputs(seed=3, rt_effect=2.0),
        extract_fmrimod_stat,
        extract_nilearn_stat,
        0.999,
        notes=(
            "Modulators expand per condition, so a pooled RT effect is the sum of per-level slopes (`'trial_type:rt.face + trial_type:rt.house'`); there is no single pooled-modulator spelling.",
        ),
    ),
    Scenario(
        "s04_omnibus_f",
        "Omnibus F over three conditions",
        fmrimod_omnibus_f,
        nilearn_omnibus_f,
        lambda: make_inputs(seed=4, conditions=("face", "house", "tool"), spacing=9.0),
        extract_fmrimod_stat,
        extract_nilearn_stat,
        0.999,
    ),
    Scenario(
        "s05_fir_shape",
        "FIR basis, F over one condition's delays",
        fmrimod_fir_shape,
        nilearn_fir_shape,
        lambda: make_inputs(seed=5, spacing=14.0),
        extract_fmrimod_stat,
        extract_nilearn_stat,
        0.99,
        notes=(
            "`hrf(..., basis='fir', nbasis=N)` without `span=` still covers 24 s whatever N is; one bin per scan needs `span=N * tr` (Nilearn's `fir_delays` are in scans).",
        ),
    ),
    Scenario(
        "s06_ar1_t",
        "AR(1) prewhitened t contrast",
        fmrimod_ar1_t,
        nilearn_ar1_t,
        lambda: make_inputs(seed=6, ar=0.4),
        extract_fmrimod_stat,
        extract_nilearn_stat,
        0.99,
        large_inputs=lambda: make_inputs(
            seed=6, ar=0.4, shape=(34, 34, 28), n_scans=300
        ),
        notes=(
            "AR(1) estimators differ (fmrimod's pooled estimate vs Nilearn's quantised voxelwise AR(1)), so agreement is r≈0.996 rather than exact. Parity of the AR path itself is tier_a_ar1_prewhitening's job.",
        ),
    ),
    Scenario(
        "s07_lss_beta_series",
        "LSS beta series, one map per trial",
        fmrimod_lss_beta_series,
        nilearn_lss_beta_series,
        lambda: make_inputs(seed=7, spacing=10.0, trial_effects=True),
        extract_fmrimod_lss,
        extract_nilearn_lss,
        0.995,
        notes=(
            "Single-trial estimation has no image shorthand: unlike `fmri_lm`, `estimate_single_trial_from_dataset` needs an explicit `fmri_dataset(...)` first.",
        ),
    ),
    Scenario(
        "s08_group_one_sample",
        "Eight subjects: first level, then one-sample group t",
        fmrimod_group_one_sample,
        nilearn_group_one_sample,
        lambda: make_group_inputs(n_subjects=8, shape=(8, 8, 6)),
        extract_fmrimod_group,
        extract_nilearn_group,
        0.999,
    ),
    Scenario(
        "s09_design_only",
        "Build and inspect a design matrix without data",
        fmrimod_design_only,
        nilearn_design_only,
        lambda: make_inputs(seed=9),
        extract_fmrimod_design,
        extract_nilearn_design,
        0.999,
        notes=(
            "Column names repeat the term (`trial_type_trial_type.face`, `base_cosine1_block_1`) where Nilearn uses `face`, `drift_1`. Deferred: renaming is a public-contract change (parity receipts, BIDS export).",
        ),
    ),
)


def scenario_by_id(scenario_id: str) -> Scenario:
    for scenario in SCENARIOS:
        if scenario.scenario_id == scenario_id:
            return scenario
    raise KeyError(scenario_id)
