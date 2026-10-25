"""Fan-in: collect per-subject BIDS maps into a :class:`GroupDataset`."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .dataset import GroupDataset
from .errors import AdapterContractError
from .space import VoxelSpace

_SUB_RE = re.compile(r"(?:^|/)sub-([A-Za-z0-9]+)")
_DESC_RE = re.compile(r"desc-([A-Za-z0-9]+)")
_SPACE_RE = re.compile(r"space-([A-Za-z0-9]+)")
_BOLD_RE = re.compile(r"desc-.+_bold\.nii(\.gz)?$")


def _subject_key(path: Path) -> str:
    match = _SUB_RE.search(path.as_posix())
    if match is None:
        raise AdapterContractError(f"could not parse BIDS subject from {path}")
    return f"sub-{match.group(1)}"


def _list_maps(directory: Path, desc: str, space: str | None) -> list[Path]:
    found: list[Path] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        name = path.name
        if not _BOLD_RE.search(name):
            continue
        desc_match = _DESC_RE.search(name)
        if desc_match is None or desc_match.group(1) != desc:
            continue
        if space is not None:
            space_match = _SPACE_RE.search(name)
            if space_match is None or space_match.group(1) != space:
                continue
        found.append(path)
    return found


def _load_map(
    path: Path,
) -> tuple[NDArray[np.float64], NDArray[np.float64], tuple[int, int, int]]:
    try:
        from neuroim import read_image  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover
        raise AdapterContractError(
            "collect_results requires the neuroim-python data substrate"
        ) from exc
    image = read_image(path, type="vol")
    data = np.asarray(image.as_array(), dtype=np.float64)
    if data.ndim != 3:
        raise AdapterContractError(f"{path} must be a 3-D map")
    shape = (int(data.shape[0]), int(data.shape[1]), int(data.shape[2]))
    affine = np.asarray(image.space.affine, dtype=np.float64)
    flat = data.reshape(-1)
    return flat, affine, shape


def collect_results(
    directory: str | Path,
    *,
    space: str | None = None,
    beta_desc: str = "beta",
    se_desc: str = "se",
) -> GroupDataset:
    """Collect per-subject ``desc-beta`` / ``desc-se`` maps into a GroupDataset.

    Pairing is by BIDS ``sub-`` key. When ``se`` maps are present and match
    the beta set one-for-one, a ``var`` assay is added as ``se²``. Beta-only
    collections omit variance (the Wave 1 synthetic-var guard still refuses
    inverse-variance reducers).
    """
    root = Path(directory)
    if not root.is_dir():
        raise AdapterContractError(f"'dir' must be an existing directory: {root}")
    beta_paths = _list_maps(root, beta_desc, space)
    if not beta_paths:
        raise AdapterContractError(f"no 'desc-{beta_desc}' maps found under {root}")
    se_paths = _list_maps(root, se_desc, space)
    beta_by_sub = {_subject_key(p): p for p in beta_paths}
    if len(beta_by_sub) != len(beta_paths):
        raise AdapterContractError("duplicate BIDS subject keys among beta maps")
    subjects = sorted(beta_by_sub)
    first_flat, affine, shape = _load_map(beta_by_sub[subjects[0]])
    n_sample = first_flat.size
    beta = np.empty((n_sample, len(subjects), 1), dtype=np.float64)
    beta[:, 0, 0] = first_flat
    for j, sid in enumerate(subjects[1:], start=1):
        flat, aff_j, shape_j = _load_map(beta_by_sub[sid])
        if flat.size != n_sample or shape_j != shape:
            raise AdapterContractError(f"map shape mismatch for {sid}")
        if not np.allclose(aff_j, affine):
            raise AdapterContractError(f"affine mismatch for {sid}")
        beta[:, j, 0] = flat
    assays: dict[str, NDArray[np.float64]] = {"beta": beta}
    if se_paths:
        se_by_sub = {_subject_key(p): p for p in se_paths}
        if set(se_by_sub) == set(beta_by_sub):
            se = np.empty_like(beta)
            for j, sid in enumerate(subjects):
                flat, _, _ = _load_map(se_by_sub[sid])
                if flat.size != n_sample:
                    raise AdapterContractError(f"se map shape mismatch for {sid}")
                se[:, j, 0] = flat
            assays["se"] = se
            assays["var"] = se * se
    return GroupDataset(
        assays=assays,
        space=VoxelSpace(shape=shape, affine=affine),
        subjects=subjects,
        contrasts=["collected"],
        metadata={
            "source": "collect_results",
            "directory": str(root),
            "space": space,
        },
    )
