"""BIDS-Stats-Model export for GLM results.

Writes fitted GLM results (betas, contrasts, statistics) as NIfTI
images with BIDS-compliant filenames and JSON sidecar metadata.

Requires ``nibabel`` (optional dependency).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass
class BidsEntities:
    """BIDS filename entities.

    Attributes
    ----------
    subject : str
        Subject label (without ``sub-`` prefix).
    task : str
        Task label.
    space : str
        Space label (e.g. ``"MNI152NLin2009cAsym"``).
    run : str or None
        Run label.
    desc : str or None
        Description label.
    """

    subject: str
    task: str
    space: str = "MNI152NLin2009cAsym"
    run: Optional[str] = None
    desc: Optional[str] = None


def _sanitize_label(label: str) -> str:
    """Sanitise a string to be a valid BIDS label (alphanumeric only)."""
    return re.sub(r"[^a-zA-Z0-9]", "", label)


def _bids_filename(
    entities: BidsEntities,
    suffix: str,
    extension: str = ".nii.gz",
    stat: Optional[str] = None,
    contrast: Optional[str] = None,
) -> str:
    """Generate a BIDS-compliant filename.

    Parameters
    ----------
    entities : BidsEntities
    suffix : str
        BIDS suffix (e.g. ``"bold"``, ``"statmap"``).
    extension : str
    stat : str, optional
        Statistic type (e.g. ``"tstat"``, ``"beta"``).
    contrast : str, optional
        Contrast name.
    """
    parts = [f"sub-{_sanitize_label(entities.subject)}"]
    parts.append(f"task-{_sanitize_label(entities.task)}")
    if entities.run:
        parts.append(f"run-{_sanitize_label(entities.run)}")
    parts.append(f"space-{_sanitize_label(entities.space)}")
    if entities.desc:
        parts.append(f"desc-{_sanitize_label(entities.desc)}")
    if contrast:
        parts.append(f"contrast-{_sanitize_label(contrast)}")
    if stat:
        parts.append(f"stat-{_sanitize_label(stat)}")
    parts_str = "_".join(parts)
    return f"{parts_str}_{suffix}{extension}"


def _write_json_sidecar(
    path: Path,
    content: dict[str, Any],
) -> None:
    """Write a BIDS JSON sidecar file."""
    with open(path, "w") as f:
        json.dump(content, f, indent=2, default=str)


def _make_nifti_image(
    data_3d: NDArray,
    mask: NDArray[np.bool_],
    affine: NDArray[np.float64],
) -> Any:
    """Create a NIfTI image from a 1-D voxel vector and mask.

    Parameters
    ----------
    data_3d : NDArray, shape ``(V,)`` or ``(nx, ny, nz)``
        Data vector (in-mask voxels) or pre-shaped volume.
    mask : NDArray[bool], shape ``(nx, ny, nz)``
        Brain mask.
    affine : NDArray, shape ``(4, 4)``
        Affine transformation matrix.

    Returns
    -------
    nib.Nifti1Image
    """
    try:
        import nibabel as nib
    except ImportError as e:
        raise ImportError(
            "nibabel is required for BIDS export. "
            "Install with: pip install fmrimod[nibabel]"
        ) from e

    if data_3d.ndim == 1:
        vol = np.zeros(mask.shape, dtype=np.float64)
        vol[mask] = data_3d
    else:
        vol = data_3d

    return nib.Nifti1Image(vol.astype(np.float32), affine)


def write_betas(
    betas: NDArray[np.float64],
    mask: NDArray[np.bool_],
    affine: NDArray[np.float64],
    output_dir: Path,
    entities: BidsEntities,
    column_names: Optional[Sequence[str]] = None,
) -> list[Path]:
    """Write beta coefficient maps as NIfTI files.

    Parameters
    ----------
    betas : NDArray, shape ``(p, V)``
        Coefficient matrix.
    mask : NDArray[bool], shape ``(nx, ny, nz)``
    affine : NDArray, shape ``(4, 4)``
    output_dir : Path
    entities : BidsEntities
    column_names : sequence of str, optional
        Names for each coefficient.

    Returns
    -------
    list of Path
        Paths to written files.
    """
    try:
        import nibabel as nib
    except ImportError as e:
        raise ImportError(
            "nibabel required for BIDS export: pip install fmrimod[nibabel]"
        ) from e

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    p, V = betas.shape
    if column_names is None:
        column_names = [f"reg{i:03d}" for i in range(p)]

    written: list[Path] = []
    for i, name in enumerate(column_names):
        fname = _bids_filename(
            entities, suffix="statmap", stat="beta", contrast=name,
        )
        img = _make_nifti_image(betas[i], mask, affine)
        out_path = output_dir / fname
        nib.save(img, str(out_path))
        written.append(out_path)

    # JSON sidecar
    meta = {
        "Description": "GLM beta coefficients",
        "Software": "fmrimod",
        "GeneratedBy": {"Name": "fmrimod", "CodeURL": "https://github.com/bbuchsbaum/fmrimod"},
        "Columns": list(column_names),
        "Timestamp": datetime.now(timezone.utc).isoformat(),
    }
    json_path = output_dir / _bids_filename(
        entities, suffix="statmap", stat="beta", extension=".json",
    )
    _write_json_sidecar(json_path, meta)
    written.append(json_path)

    return written


def write_contrasts(
    contrasts: dict,
    mask: NDArray[np.bool_],
    affine: NDArray[np.float64],
    output_dir: Path,
    entities: BidsEntities,
    stats: Sequence[str] = ("beta", "tstat", "pvalue", "se"),
) -> list[Path]:
    """Write contrast statistic maps as NIfTI files.

    Parameters
    ----------
    contrasts : dict
        ``{name: ContrastResult}`` from a fitted GLM.
    mask : NDArray[bool]
    affine : NDArray, shape ``(4, 4)``
    output_dir : Path
    entities : BidsEntities
    stats : sequence of str
        Which statistics to write.

    Returns
    -------
    list of Path
    """
    try:
        import nibabel as nib
    except ImportError as e:
        raise ImportError(
            "nibabel required for BIDS export: pip install fmrimod[nibabel]"
        ) from e

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    stat_extractors = {
        "beta": lambda c: c.estimate if c.estimate.ndim == 1 else c.estimate[0],
        "tstat": lambda c: c.stat,
        "pvalue": lambda c: c.p_value,
        "se": lambda c: c.se if c.se is not None else np.zeros_like(c.stat),
    }

    for con_name, cres in contrasts.items():
        for stat_name in stats:
            extractor = stat_extractors.get(stat_name)
            if extractor is None:
                continue
            data = extractor(cres)
            fname = _bids_filename(
                entities, suffix="statmap",
                stat=stat_name, contrast=con_name,
            )
            img = _make_nifti_image(data, mask, affine)
            out_path = output_dir / fname
            nib.save(img, str(out_path))
            written.append(out_path)

        # JSON sidecar per contrast
        meta = {
            "Description": f"Contrast: {con_name}",
            "ContrastType": cres.stat_type,
            "DegreesOfFreedom": cres.df,
            "Software": "fmrimod",
            "Timestamp": datetime.now(timezone.utc).isoformat(),
        }
        json_path = output_dir / _bids_filename(
            entities, suffix="statmap",
            contrast=con_name, extension=".json",
        )
        _write_json_sidecar(json_path, meta)
        written.append(json_path)

    return written
