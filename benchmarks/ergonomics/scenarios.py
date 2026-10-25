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
import pandas as pd

from benchmarks.ergonomics.inputs import (
    MOTION,
    BidsInputs,
    GroupMapInputs,
    Inputs,
    make_bids_inputs,
    make_group_inputs,
    make_group_map_inputs,
    make_inputs,
)

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
        hrf("trial_type", modulators=["rt"], pool_modulators=True) + drift(),
        inp.img,
        tr=inp.tr,
        events=inp.events[0],
        mask=inp.mask_img,
        slice_timing_offset=0.0,  # parity-knob
    )
    return fit.contrast("rt")


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
        ar="ar1_voxelwise",
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

    return fm.estimate_single_trial_from_dataset(
        inp.img,
        trialwise(),
        tr=inp.tr,
        events=inp.events[0],
        mask=inp.mask_img,
        method="lss",
        include_intercept=True,
        slice_timing_offset=0.0,  # parity-knob
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
# S10  z-map written to NIfTI (Nilearn's default output is a z image)
# --------------------------------------------------------------------------


def fmrimod_zmap_to_nifti(inp: Inputs):
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
    path = inp.out_dir / "face_vs_house_z.nii.gz"
    return fit.contrast("face - house").to_nifti(path, kind="z")


def nilearn_zmap_to_nifti(inp: Inputs):
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
    path = inp.out_dir / "face_vs_house_z.nii.gz"
    model.compute_contrast("face - house").to_filename(path)
    return path


# --------------------------------------------------------------------------
# S11  a batch of named contrasts from one fit
# --------------------------------------------------------------------------

CONTRASTS = ("face", "house", "tool", "face - house", "face - tool", "house - tool")


def fmrimod_many_contrasts(inp: Inputs):
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
    return {name: fit.contrast(name) for name in CONTRASTS}


def nilearn_many_contrasts(inp: Inputs):
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
    return {
        name: model.compute_contrast(name, output_type="stat") for name in CONTRASTS
    }


# --------------------------------------------------------------------------
# S12  second level from effect maps on disk: two groups + age covariate
# --------------------------------------------------------------------------


def fmrimod_group_covariate(maps: GroupMapInputs):
    import fmrimod as fm

    result = fm.fmri_group_lm(
        maps.paths,
        formula="~ group + age",
        covariates=maps.participants,
        mask=maps.mask_img,  # parity-knob
    )
    return result.t_stat("patient")


def nilearn_group_covariate(maps: GroupMapInputs):
    from nilearn.glm.second_level import SecondLevelModel

    table = maps.participants
    design = pd.DataFrame(
        {
            "intercept": 1.0,
            "patient": (table["group"] == "patient").astype(float),
            "age": table["age"] - table["age"].mean(),
        }
    )
    model = SecondLevelModel(mask_img=maps.mask_img)  # parity-knob
    model.fit(maps.paths, design_matrix=design)
    return model.compute_contrast("patient", output_type="stat")


# --------------------------------------------------------------------------
# S13  first level for every subject of a BIDS + fMRIPrep dataset
# --------------------------------------------------------------------------


def fmrimod_bids_first_level(bids: BidsInputs):
    import fmrimod as fm
    from fmrimod.spec import confounds, drift, hrf

    datasets = fm.bids_datasets(
        bids.root,
        bids.task,
        space=bids.space,
        slice_timing_offset=0.0,  # parity-knob
    )
    spec = hrf("trial_type") + confounds(*MOTION) + drift()
    return {
        sub: fm.fmri_lm(spec, ds).contrast("face - house")
        for sub, ds in datasets.items()
    }


def nilearn_bids_first_level(bids: BidsInputs):
    from nilearn.glm.first_level import first_level_from_bids

    models, imgs, events, confounds = first_level_from_bids(
        bids.root,
        bids.task,
        space_label=bids.space,
        derivatives_folder=bids.derivatives,
        img_filters=[("desc", "preproc")],
        hrf_model="spm",
        mask_img=bids.mask_img,  # parity-knob
        noise_model="ols",  # parity-knob
        high_pass=1 / 128,  # parity-knob
        signal_scaling=False,  # parity-knob
    )
    maps = {}
    for model, img, ev, conf in zip(models, imgs, events, confounds):  # noqa: B905
        model.fit(img, events=ev, confounds=[c[list(MOTION)] for c in conf])
        maps[model.subject_label] = model.compute_contrast(
            "face - house", output_type="stat"
        )
    return maps


# --------------------------------------------------------------------------
# S14  BIDS dataset -> first level per subject -> group effect of age
# --------------------------------------------------------------------------


def fmrimod_bids_to_group(bids: BidsInputs):
    import fmrimod as fm
    from fmrimod.spec import confounds, drift, hrf

    datasets = fm.bids_datasets(
        bids.root,
        bids.task,
        space=bids.space,
        slice_timing_offset=0.0,  # parity-knob
    )
    spec = hrf("trial_type") + confounds(*MOTION) + drift()
    maps = {
        sub: fm.fmri_lm(spec, ds).contrast("face - house")
        for sub, ds in datasets.items()
    }
    group = fm.fmri_group_lm(
        maps, formula="~ age", covariates=bids.root / "participants.tsv"
    )
    return group.t_stat("age")


def nilearn_bids_to_group(bids: BidsInputs):
    from nilearn.glm.first_level import first_level_from_bids
    from nilearn.glm.second_level import SecondLevelModel

    models, imgs, events, confounds = first_level_from_bids(
        bids.root,
        bids.task,
        space_label=bids.space,
        derivatives_folder=bids.derivatives,
        img_filters=[("desc", "preproc")],
        hrf_model="spm",
        mask_img=bids.mask_img,  # parity-knob
        noise_model="ols",  # parity-knob
        high_pass=1 / 128,  # parity-knob
        signal_scaling=False,  # parity-knob
    )
    maps = []
    for model, img, ev, conf in zip(models, imgs, events, confounds):  # noqa: B905
        model.fit(img, events=ev, confounds=[c[list(MOTION)] for c in conf])
        maps.append(model.compute_contrast("face - house", output_type="effect_size"))
    participants = pd.read_csv(bids.root / "participants.tsv", sep="\t")
    subjects = [f"sub-{m.subject_label}" for m in models]
    ages = participants.set_index("participant_id").loc[subjects, "age"]
    design = pd.DataFrame({"intercept": 1.0, "age": ages.to_numpy()})
    group = SecondLevelModel(mask_img=bids.mask_img)  # parity-knob
    group.fit(maps, design_matrix=design)
    return group.compute_contrast("age", output_type="stat")


# --------------------------------------------------------------------------
# S15  FDR + cluster-extent thresholded z-map and a cluster table
# --------------------------------------------------------------------------

CLUSTER_MIN_VOXELS = 10


def fmrimod_fdr_clusters(inp: Inputs):
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
    contrast = fit.contrast("face - house")
    hits = contrast.threshold(0.05, cluster_size=CLUSTER_MIN_VOXELS)
    return hits, hits.clusters()


def nilearn_fdr_clusters(inp: Inputs):
    from nilearn.glm import threshold_stats_img
    from nilearn.glm.first_level import FirstLevelModel
    from nilearn.reporting import get_clusters_table

    model = FirstLevelModel(
        t_r=inp.tr,
        mask_img=inp.mask_img,
        hrf_model="spm",
        noise_model="ols",  # parity-knob
        high_pass=1 / 128,  # parity-knob
        signal_scaling=False,  # parity-knob
    )
    model.fit(inp.img, events=inp.events[0])
    z_map = model.compute_contrast("face - house")
    thresholded, threshold = threshold_stats_img(
        z_map,
        mask_img=inp.mask_img,
        alpha=0.05,
        height_control="fdr",
        cluster_threshold=CLUSTER_MIN_VOXELS,
    )
    table = get_clusters_table(
        z_map, stat_threshold=threshold, cluster_threshold=CLUSTER_MIN_VOXELS
    )
    return thresholded, table


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


def extract_fmrimod_zmap(result: Any, inp: Inputs) -> np.ndarray:
    import nibabel as nib

    return _masked(nib.load(result), inp)


extract_nilearn_zmap = extract_fmrimod_zmap


def extract_fmrimod_many(result: Any, inp: Inputs) -> np.ndarray:
    return np.concatenate([np.asarray(result[c].stat).ravel() for c in CONTRASTS])


def extract_nilearn_many(result: Any, inp: Inputs) -> np.ndarray:
    return np.concatenate([_masked(result[c], inp) for c in CONTRASTS])


def extract_fmrimod_group_map(result: Any, maps: GroupMapInputs) -> np.ndarray:
    if hasattr(result, "get_fdata"):
        return np.asarray(result.get_fdata(), dtype=np.float64)[maps.mask]
    return np.asarray(result, dtype=np.float64).ravel()


extract_nilearn_group_map = extract_fmrimod_group_map


def extract_fmrimod_bids(result: Any, bids: BidsInputs) -> np.ndarray:
    return np.concatenate(
        [np.asarray(result[s].stat, dtype=np.float64).ravel() for s in bids.subjects]
    )


def extract_nilearn_bids(result: Any, bids: BidsInputs) -> np.ndarray:
    return np.concatenate(
        [np.asarray(result[s].get_fdata())[bids.mask] for s in bids.subjects]
    )


def extract_bids_group(result: Any, bids: BidsInputs) -> np.ndarray:
    if hasattr(result, "get_fdata"):
        return np.asarray(result.get_fdata(), dtype=np.float64)[bids.mask]
    return np.asarray(result, dtype=np.float64).ravel()


def extract_thresholded(result: Any, inp: Inputs) -> np.ndarray:
    image = result[0]
    if hasattr(image, "get_fdata"):
        return _masked(image, inp)
    return np.asarray(
        image.z if hasattr(image, "z") else image, dtype=np.float64
    ).ravel()


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
    full_brain_inputs: Callable[[], Any] | None = None


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
        full_brain_inputs=lambda: make_inputs(seed=1, shape=(64, 64, 40), n_scans=300),
        notes=(
            "By design: the default noise model is OLS (R fmrireg's `iid`), where Nilearn's `FirstLevelModel` uses AR(1); pass `ar=\"ar1_voxelwise\"` for Nilearn's model.",
            "By design: the sampling grid starts at TR/2 (R fmrihrf #94); Nilearn/SPM/FitLins use frame start, so cross-tool comparisons pass `slice_timing_offset=0.0`.",
            "By design: drift is opt-in (`+ drift()`, SPM's 128 s); Nilearn high-passes at 100 s by default. Fitting drift implicitly would change every existing fit.",
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
            "By design: `hrf(..., basis='fir', nbasis=N)` keeps R fmrihrf's 24 s span; one bin per scan (Nilearn's `fir_delays`) is `span=N * tr`.",
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
        0.9999,
        full_brain_inputs=lambda: make_inputs(
            seed=6, ar=0.4, shape=(64, 64, 40), n_scans=300
        ),
        large_inputs=lambda: make_inputs(
            seed=6, ar=0.4, shape=(34, 34, 28), n_scans=300
        ),
        notes=(
            'Uses `ar="ar1_voxelwise"` (Nilearn\'s model, r>0.9999). Round 4 found that path 10-33x slower than Nilearn; vectorised AR estimation plus one batched GLS solve (identical to ~1e-12) closed the gap. The pooled default `ar="ar1"` agrees only to r~0.996.',
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
            'By design: canonical column names follow R fmridesign (`trial_type_trial_type.face`); `names="short"` gives display labels (`face`, `rt:face`, `drift1`).',
        ),
    ),
    Scenario(
        "s10_zmap_to_nifti",
        "z-map of a t contrast written to NIfTI",
        fmrimod_zmap_to_nifti,
        nilearn_zmap_to_nifti,
        lambda: make_inputs(seed=10),
        extract_fmrimod_zmap,
        extract_nilearn_zmap,
        0.999,
        notes=(
            'By design: a contrast returns a `ContrastResult` holding t, z, effect, SE and p; writing z is `to_nifti(path, kind="z")`, where Nilearn\'s default output is a z image.',
        ),
    ),
    Scenario(
        "s11_many_contrasts",
        "Six named contrasts from one fit",
        fmrimod_many_contrasts,
        nilearn_many_contrasts,
        lambda: make_inputs(seed=11, conditions=("face", "house", "tool"), spacing=9.0),
        extract_fmrimod_many,
        extract_nilearn_many,
        0.999,
    ),
    Scenario(
        "s12_group_covariate",
        "Second level from effect maps on disk: group + age",
        fmrimod_group_covariate,
        nilearn_group_covariate,
        make_group_map_inputs,
        extract_fmrimod_group_map,
        extract_nilearn_group_map,
        0.999,
    ),
    Scenario(
        "s13_bids_first_level",
        "First level for every subject of a BIDS + fMRIPrep dataset",
        fmrimod_bids_first_level,
        nilearn_bids_first_level,
        lambda: make_bids_inputs(n_subjects=3, shape=(8, 8, 6), n_scans=120),
        extract_fmrimod_bids,
        extract_nilearn_bids,
        0.999,
        notes=(
            "`fm.bids_datasets` covers the fMRIPrep first-level layout only (preproc BOLD, events, sidecar TR, brain mask, confounds); no BIDS Stats Model or pybids query language.",
        ),
    ),
    Scenario(
        "s14_bids_to_group",
        "BIDS dataset to group effect of age from participants.tsv",
        fmrimod_bids_to_group,
        nilearn_bids_to_group,
        lambda: make_bids_inputs(n_subjects=6, shape=(8, 8, 6), n_scans=100, seed=400),
        extract_bids_group,
        extract_bids_group,
        0.99,
        notes=(
            "Agreement gate is r>=0.99, not 0.999: first-level estimates agree at r>=0.9997 (with a constant 1.20 HRF-normalisation scale) and the group stage is exact, but six subjects' age t in null voxels amplifies that last 0.03%.",
        ),
    ),
    Scenario(
        "s15_fdr_clusters",
        "FDR + cluster-extent thresholded z-map with a cluster table",
        fmrimod_fdr_clusters,
        nilearn_fdr_clusters,
        lambda: make_inputs(seed=15, shape=(14, 14, 10), noise_sd=2.0),
        extract_thresholded,
        extract_thresholded,
        0.999,
        notes=(
            "`ThresholdedMap.clusters()` lists one peak per cluster; Nilearn's `get_clusters_table` also lists sub-peaks (8 mm apart).",
        ),
    ),
)


def scenario_by_id(scenario_id: str) -> Scenario:
    for scenario in SCENARIOS:
        if scenario.scenario_id == scenario_id:
            return scenario
    raise KeyError(scenario_id)
