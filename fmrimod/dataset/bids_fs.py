"""Load fMRIPrep-style BIDS derivatives into :class:`FmriDataset` objects.

A dependency-free reader for the common first-level layout::

    <root>/sub-01/[ses-*/]func/sub-01_task-x_run-1_events.tsv
    <root>/<derivatives>/sub-01/[ses-*/]func/
        sub-01_task-x_run-1_space-S_desc-preproc_bold.nii.gz
        sub-01_task-x_run-1_space-S_desc-preproc_bold.json   (RepetitionTime)
        sub-01_task-x_run-1_space-S_desc-brain_mask.nii.gz
        sub-01_task-x_run-1_desc-confounds_timeseries.tsv

It covers what :func:`nilearn.glm.first_level.first_level_from_bids` covers
for a first-level fit: per-run preprocessed images, events, TR, brain mask
and confound tables. Runs from several sessions are concatenated in
``ses``/``run`` order. For anything more exotic, read the files yourself and
call :func:`fmrimod.fmri_dataset`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from .constructors import fmri_dataset
from .fmri_dataset import FmriDataset

_ENTITY = re.compile(r"(?P<key>[a-zA-Z]+)-(?P<value>[a-zA-Z0-9]+)")
_RAW_ENTITIES = ("sub", "ses", "task", "acq", "ce", "rec", "dir", "run", "echo")


def bids_datasets(
    root: str | Path,
    task: str,
    *,
    space: str | None = None,
    subjects: Sequence[str] | None = None,
    derivatives: str | Path = "derivatives/fmriprep",
    desc: str = "preproc",
    slice_timing_offset: float | None = None,
) -> dict[str, FmriDataset]:
    """One :class:`FmriDataset` per subject from a BIDS + fMRIPrep tree.

    Each dataset holds every run of ``task`` for that subject (sessions
    concatenated), the events tables (as ``run`` 1..n), the TR from the
    BOLD sidecar, the fMRIPrep brain mask of the first run, and the
    fMRIPrep confound tables, from which a spec term such as
    ``confounds("trans_x", "rot_z", ...)`` selects columns at fit time.

    Parameters
    ----------
    root
        BIDS dataset root (holds ``sub-*/func/*_events.tsv``).
    task
        Task label (``task-<label>``).
    space
        Output space label (``space-<label>``). Optional when the
        derivatives contain a single space for this task.
    subjects
        Subject labels with or without ``sub-``; default all subjects that
        have preprocessed runs for ``task``.
    derivatives
        Derivatives directory, relative to ``root`` or absolute.
    desc
        ``desc-`` label of the preprocessed BOLD files.
    slice_timing_offset
        Forwarded to :func:`fmrimod.fmri_dataset` (default ``TR/2``; pass
        ``0.0`` for Nilearn/SPM frame times).

    Returns
    -------
    dict[str, FmriDataset]
        Keyed by subject label without the ``sub-`` prefix, in sorted order.

    Examples
    --------
    ::

        from fmrimod.spec import confounds, drift, hrf

        spec = hrf("trial_type") + confounds("trans_x", "trans_y", "trans_z",
                                             "rot_x", "rot_y", "rot_z") + drift()
        maps = {
            sub: fm.fmri_lm(spec, ds).contrast("face - house")
            for sub, ds in fm.bids_datasets("/data/ds001", task="faces",
                                            space="MNI152NLin2009cAsym").items()
        }
        group = fm.fmri_group_lm(maps)
    """
    root = Path(root)
    deriv = Path(derivatives)
    deriv = deriv if deriv.is_absolute() else root / deriv
    if not deriv.is_dir():
        raise FileNotFoundError(f"derivatives directory not found: {deriv}")

    bolds = _preproc_bolds(deriv, task, space, desc)
    if subjects is not None:
        wanted = {str(s).removeprefix("sub-") for s in subjects}
        missing = wanted - set(bolds)
        if missing:
            raise FileNotFoundError(
                f"no preprocessed task-{task} runs for subjects {sorted(missing)}"
            )
        bolds = {sub: runs for sub, runs in bolds.items() if sub in wanted}
    if not bolds:
        raise FileNotFoundError(
            f"no *_task-{task}_*desc-{desc}_bold.nii[.gz] files under {deriv}"
        )

    datasets: dict[str, FmriDataset] = {}
    for sub in sorted(bolds):
        runs = bolds[sub]
        events = [_events_for(root, bold) for bold in runs]
        tables = [_confounds_for(bold) for bold in runs]
        mask = _sibling(runs[0], desc_suffix="desc-brain_mask")
        datasets[sub] = fmri_dataset(
            [str(p) for p in runs],
            mask=str(mask) if mask.exists() else None,
            tr=_repetition_time(root, runs[0]),
            events=events,
            confounds=None if any(t is None for t in tables) else tables,
            slice_timing_offset=slice_timing_offset,
        )
    return datasets


def _entities(path: Path) -> dict[str, str]:
    name = path.name.split(".")[0]
    return {m["key"]: m["value"] for m in _ENTITY.finditer(name)}


def _preproc_bolds(
    deriv: Path, task: str, space: str | None, desc: str
) -> dict[str, list[Path]]:
    found = [
        p
        for p in deriv.glob(f"sub-*/**/func/*_task-{task}_*desc-{desc}_bold.nii*")
        if p.name.endswith((".nii", ".nii.gz"))
    ]
    spaces = {_entities(p).get("space") for p in found}
    if space is None and len(spaces) > 1:
        raise ValueError(
            f"several spaces for task-{task}: {sorted(str(s) for s in spaces)}; "
            "pass space="
        )
    by_subject: dict[str, list[Path]] = {}
    for p in found:
        ent = _entities(p)
        if space is not None and ent.get("space") != space:
            continue
        by_subject.setdefault(ent["sub"], []).append(p)

    def order(p: Path) -> tuple[str, int, str]:
        ent = _entities(p)
        run = ent.get("run", "0")
        return (ent.get("ses", ""), int(run) if run.isdigit() else 0, p.name)

    return {sub: sorted(paths, key=order) for sub, paths in by_subject.items()}


def _raw_stem(bold: Path) -> str:
    """``sub-01_ses-a_task-x_run-1`` part of a derivative BOLD file name."""
    ent = _entities(bold)
    return "_".join(f"{k}-{ent[k]}" for k in _RAW_ENTITIES if k in ent)


def _sibling(bold: Path, *, desc_suffix: str) -> Path:
    name = bold.name
    head = name[: name.index("desc-")]
    return bold.with_name(f"{head}{desc_suffix}.nii.gz")


def _events_for(root: Path, bold: Path) -> pd.DataFrame:
    ent = _entities(bold)
    func = root / f"sub-{ent['sub']}"
    if "ses" in ent:
        func = func / f"ses-{ent['ses']}"
    path = func / "func" / f"{_raw_stem(bold)}_events.tsv"
    if not path.exists():
        raise FileNotFoundError(f"events file not found: {path}")
    return pd.read_csv(path, sep="\t", na_values="n/a")


def _confounds_for(bold: Path) -> pd.DataFrame | None:
    stem = _raw_stem(bold)
    for suffix in ("desc-confounds_timeseries", "desc-confounds_regressors"):
        path = bold.with_name(f"{stem}_{suffix}.tsv")
        if path.exists():
            return pd.read_csv(path, sep="\t", na_values="n/a")
    return None


def _repetition_time(root: Path, bold: Path) -> float:
    ent = _entities(bold)
    candidates = [
        bold.with_name(bold.name.split(".")[0] + ".json"),
        root
        / f"sub-{ent['sub']}"
        / (f"ses-{ent['ses']}" if "ses" in ent else "")
        / "func"
        / f"{_raw_stem(bold)}_bold.json",
        root / f"task-{ent['task']}_bold.json",
    ]
    for path in candidates:
        if path.exists():
            meta = json.loads(path.read_text())
            if "RepetitionTime" in meta:
                return float(meta["RepetitionTime"])
    raise ValueError(
        f"no RepetitionTime for {bold.name}; looked in "
        + ", ".join(str(c) for c in candidates)
    )
