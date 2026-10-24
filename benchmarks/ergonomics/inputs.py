"""Synthetic NIfTI inputs shared by the fmrimod and Nilearn scenario paths.

Every scenario starts from the same things a user has on disk after
preprocessing: 4-D BOLD images, a brain mask, a BIDS-style events table
and (where relevant) an fMRIPrep-style confounds table. Signal is generated
with a plain-numpy double-gamma HRF so that neither library's HRF code is
used to make the data it is then asked to recover.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from math import gamma as gamma_fn
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from numpy.typing import NDArray

Array = NDArray[np.float64]
TR = 2.0
MOTION = ("trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z")


@dataclass(frozen=True)
class Inputs:
    """What a user holds before writing any modeling code."""

    imgs: list[nib.Nifti1Image]
    mask_img: nib.Nifti1Image
    events: list[pd.DataFrame]
    confounds: list[pd.DataFrame] = field(default_factory=list)
    tr: float = TR
    out_dir: Path = field(
        default_factory=lambda: Path(tempfile.mkdtemp(prefix="fm-out-"))
    )

    @property
    def img(self) -> nib.Nifti1Image:
        return self.imgs[0]

    @property
    def events_df(self) -> pd.DataFrame:
        """All runs' events in one table with a 1-based ``run`` column."""
        return pd.concat(
            [ev.assign(run=i + 1) for i, ev in enumerate(self.events)],
            ignore_index=True,
        )

    @property
    def mask(self) -> NDArray[np.bool_]:
        return np.asarray(self.mask_img.dataobj).astype(bool)

    @property
    def n_voxels(self) -> int:
        return int(self.mask.sum())


def _hrf(tr: float, length: float = 32.0, oversampling: int = 16) -> Array:
    dt = tr / oversampling
    t = np.arange(0.0, length, dt)
    peak = t**5 * np.exp(-t) / gamma_fn(6)
    under = t**15 * np.exp(-t) / gamma_fn(16)
    h = peak - under / 6.0
    return h / h.sum()


def _regressor(
    onsets: Array, durations: Array, amps: Array, n_scans: int, tr: float
) -> Array:
    oversampling = 16
    dt = tr / oversampling
    grid = np.zeros(n_scans * oversampling)
    for onset, dur, amp in zip(onsets, durations, amps, strict=True):
        start = int(round(onset / dt))
        stop = max(start + 1, int(round((onset + dur) / dt)))
        grid[start:stop] += amp
    conv = np.convolve(grid, _hrf(tr, oversampling=oversampling))[: grid.size]
    return conv[::oversampling] * oversampling


def _events(
    rng: np.random.Generator, n_scans: int, conditions: tuple[str, ...], spacing: float
) -> pd.DataFrame:
    onsets = np.arange(10.0, n_scans * TR - 30.0, spacing)
    onsets = onsets + rng.uniform(-1.0, 1.0, onsets.size)
    labels = [conditions[i % len(conditions)] for i in range(onsets.size)]
    return pd.DataFrame(
        {
            "onset": onsets,
            "duration": 1.0,
            "trial_type": labels,
            "rt": rng.uniform(0.4, 1.6, onsets.size),
        }
    )


def _image(
    signal: Array,
    mask: NDArray[np.bool_],
    rng: np.random.Generator,
    noise_sd: float,
    ar: float,
) -> nib.Nifti1Image:
    n_scans, n_vox = signal.shape
    noise = rng.normal(scale=noise_sd, size=(n_scans, n_vox))
    if ar:
        for t in range(1, n_scans):
            noise[t] += ar * noise[t - 1]
    data = np.zeros(mask.shape + (n_scans,), dtype=np.float32)
    data[mask] = (100.0 + signal + noise).T
    return nib.Nifti1Image(data, np.diag([3.0, 3.0, 3.0, 1.0]))


def make_inputs(
    *,
    seed: int = 0,
    shape: tuple[int, int, int] = (10, 10, 8),
    n_scans: int = 160,
    n_runs: int = 1,
    conditions: tuple[str, ...] = ("face", "house"),
    spacing: float = 12.0,
    with_confounds: bool = False,
    rt_effect: float = 0.0,
    trial_effects: bool = False,
    noise_sd: float = 1.0,
    ar: float = 0.0,
) -> Inputs:
    """Build synthetic runs with a known per-condition effect pattern.

    Voxels in the first half of the mask carry ``face > house`` (effects 2.0
    vs 1.0 for the first two conditions; later conditions get 1.5); voxels in
    the second half are null. ``rt_effect`` adds a mean-centred RT modulation
    and ``trial_effects`` draws an independent amplitude per trial.
    """
    rng = np.random.default_rng(seed)
    mask = np.zeros(shape, dtype=bool)
    mask[1:-1, 1:-1, 1:-1] = True
    n_vox = int(mask.sum())
    active = np.arange(n_vox) < n_vox // 2
    base_amp = dict(zip(conditions, [2.0, 1.0] + [1.5] * 8, strict=False))

    imgs, events, confounds = [], [], []
    for _ in range(n_runs):
        ev = _events(rng, n_scans, conditions, spacing)
        signal = np.zeros((n_scans, n_vox))
        for cond in conditions:
            rows = ev[ev.trial_type == cond]
            amps = np.full(len(rows), base_amp[cond])
            if trial_effects:
                amps = amps * rng.uniform(0.2, 1.8, len(rows))
            reg = _regressor(
                rows.onset.to_numpy(), rows.duration.to_numpy(), amps, n_scans, TR
            )
            signal += np.outer(reg, active)
        if rt_effect:
            rt = ev.rt.to_numpy() - ev.rt.mean()
            reg = _regressor(
                ev.onset.to_numpy(), ev.duration.to_numpy(), rt, n_scans, TR
            )
            signal += rt_effect * np.outer(reg, active)
        if with_confounds:
            motion = np.cumsum(
                rng.normal(scale=0.05, size=(n_scans, len(MOTION))), axis=0
            )
            signal += motion @ rng.normal(scale=2.0, size=(len(MOTION), n_vox))
            confounds.append(pd.DataFrame(motion, columns=list(MOTION)))
        imgs.append(_image(signal, mask, rng, noise_sd, ar))
        events.append(ev)

    mask_img = nib.Nifti1Image(mask.astype(np.uint8), imgs[0].affine)
    return Inputs(imgs=imgs, mask_img=mask_img, events=events, confounds=confounds)


def make_group_inputs(
    n_subjects: int = 8, seed: int = 100, **kwargs: object
) -> list[Inputs]:
    """One single-run :class:`Inputs` per subject, with independent noise."""
    return [make_inputs(seed=seed + s, **kwargs) for s in range(n_subjects)]


@dataclass(frozen=True)
class BidsInputs:
    """A raw BIDS dataset with an fMRIPrep-style derivatives tree on disk."""

    root: Path
    task: str
    space: str
    subjects: tuple[str, ...]
    mask_img: nib.Nifti1Image
    derivatives: str = "derivatives/fmriprep"

    @property
    def mask(self) -> NDArray[np.bool_]:
        return np.asarray(self.mask_img.dataobj).astype(bool)


def make_bids_inputs(
    root: Path | None = None,
    *,
    n_subjects: int = 3,
    n_runs: int = 2,
    task: str = "faces",
    space: str = "MNI152NLin2009cAsym",
    seed: int = 300,
    **kwargs: object,
) -> BidsInputs:
    """Write ``n_subjects`` x ``n_runs`` synthetic runs as BIDS + fMRIPrep.

    Raw tree: ``sub-XX/func/sub-XX_task-<task>_run-N_events.tsv``.
    Derivatives: preprocessed BOLD, brain mask, a JSON sidecar with
    ``RepetitionTime`` and a confounds TSV whose motion columns sit among
    other fMRIPrep columns (with the ``n/a`` first row fMRIPrep writes for
    derivative columns).
    """
    root = Path(root) if root is not None else Path(tempfile.mkdtemp(prefix="bids-"))
    deriv = root / "derivatives" / "fmriprep"
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "synthetic", "BIDSVersion": "1.8.0"})
    )
    deriv.mkdir(parents=True, exist_ok=True)
    (deriv / "dataset_description.json").write_text(
        json.dumps(
            {
                "Name": "fMRIPrep",
                "BIDSVersion": "1.8.0",
                "DatasetType": "derivative",
                "GeneratedBy": [{"Name": "fMRIPrep"}],
            }
        )
    )
    subjects = []
    mask_img = None
    for s in range(n_subjects):
        sub = f"{s + 1:02d}"
        subjects.append(sub)
        inp = make_inputs(seed=seed + s, n_runs=n_runs, with_confounds=True, **kwargs)
        mask_img = inp.mask_img
        raw_func = root / f"sub-{sub}" / "func"
        der_func = deriv / f"sub-{sub}" / "func"
        raw_func.mkdir(parents=True, exist_ok=True)
        der_func.mkdir(parents=True, exist_ok=True)
        for r, (img, ev, conf) in enumerate(
            zip(inp.imgs, inp.events, inp.confounds, strict=True), start=1
        ):
            stem = f"sub-{sub}_task-{task}_run-{r}"
            ev.to_csv(raw_func / f"{stem}_events.tsv", sep="\t", index=False)
            prefix = der_func / f"{stem}_space-{space}"
            nib.save(img, f"{prefix}_desc-preproc_bold.nii.gz")
            nib.save(inp.mask_img, f"{prefix}_desc-brain_mask.nii.gz")
            (der_func / f"{stem}_space-{space}_desc-preproc_bold.json").write_text(
                json.dumps({"RepetitionTime": TR, "SkullStripped": True})
            )
            table = conf.assign(
                global_signal=np.linspace(0, 1, len(conf)),
                framewise_displacement=np.r_[np.nan, np.full(len(conf) - 1, 0.1)],
            )
            table.to_csv(
                der_func / f"{stem}_desc-confounds_timeseries.tsv",
                sep="\t",
                index=False,
                na_rep="n/a",
            )
    assert mask_img is not None
    return BidsInputs(
        root=root, task=task, space=space, subjects=tuple(subjects), mask_img=mask_img
    )


@dataclass(frozen=True)
class GroupMapInputs:
    """Per-subject first-level effect maps on disk plus a participants table."""

    paths: list[str]
    participants: pd.DataFrame
    mask_img: nib.Nifti1Image

    @property
    def mask(self) -> NDArray[np.bool_]:
        return np.asarray(self.mask_img.dataobj).astype(bool)


def make_group_map_inputs(
    n_per_group: int = 10,
    *,
    shape: tuple[int, int, int] = (10, 10, 8),
    seed: int = 500,
) -> GroupMapInputs:
    """Effect maps for two groups with an age effect, written as NIfTI files.

    Active voxels (first half of the mask) carry a patient > control
    difference of 0.8 and an age slope of 0.03 per year.
    """
    rng = np.random.default_rng(seed)
    mask = np.zeros(shape, dtype=bool)
    mask[1:-1, 1:-1, 1:-1] = True
    n_vox = int(mask.sum())
    active = np.arange(n_vox) < n_vox // 2
    n = 2 * n_per_group
    group = np.repeat(["control", "patient"], n_per_group)
    age = rng.uniform(20, 70, n)
    out = Path(tempfile.mkdtemp(prefix="group-maps-"))
    affine = np.diag([3.0, 3.0, 3.0, 1.0])
    paths = []
    for i in range(n):
        effect = 1.0 + 0.8 * (group[i] == "patient") + 0.03 * (age[i] - 45)
        values = effect * active + rng.normal(scale=0.5, size=n_vox)
        vol = np.zeros(shape, dtype=np.float32)
        vol[mask] = values
        path = out / f"sub-{i + 1:02d}_contrast-faceVsHouse_stat-effect.nii.gz"
        nib.save(nib.Nifti1Image(vol, affine), path)
        paths.append(str(path))
    participants = pd.DataFrame(
        {
            "participant_id": [f"sub-{i + 1:02d}" for i in range(n)],
            "group": group,
            "age": age,
        }
    )
    return GroupMapInputs(
        paths=paths,
        participants=participants,
        mask_img=nib.Nifti1Image(mask.astype(np.uint8), affine),
    )
