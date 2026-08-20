"""BIDS-Stats-Model export for GLM results.

Writes fitted GLM results (betas, contrasts, statistics) as NIfTI images with
BIDS-compliant filenames and JSON sidecar metadata.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Protocol, Sequence, cast

if TYPE_CHECKING:
    import neuroim  # type: ignore[import-untyped]

import numpy as np
from numpy.typing import NDArray


class _ContrastResultLike(Protocol):
    """Duck-typed view of fmrimod.glm.ContrastResult for BIDS export.

    Names the attributes the export writer reads off each contrast result.
    Mirrors the protocols in fmrimod/io/results.py and fmrimod/dataset/
    study.py; intentionally kept local so the typed seam stays decoupled
    from the concrete result class.
    """

    @property
    def estimate(self) -> object:
        ...

    @property
    def stat(self) -> object:
        ...

    @property
    def p_value(self) -> object:
        ...

    @property
    def se(self) -> object | None:
        ...

    @property
    def stat_type(self) -> str:
        ...

    @property
    def df(self) -> object:
        ...


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
    content: dict[str, object],
) -> None:
    """Write a BIDS JSON sidecar file."""
    with open(path, "w") as f:
        json.dump(content, f, indent=2, default=str)


def _make_nifti_image(
    data_3d: NDArray[np.float64],
    mask: NDArray[np.bool_],
    affine: NDArray[np.float64],
) -> neuroim.DenseNeuroVol:
    """Create a neuroim volume from a 1-D voxel vector and mask.

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
    neuroim.DenseNeuroVol
    """
    from neuroim import DenseNeuroVol, NeuroSpace

    if data_3d.ndim == 1:
        vol = np.zeros(mask.shape, dtype=np.float64)
        vol[mask] = data_3d
    else:
        vol = data_3d

    space = NeuroSpace(dim=vol.shape, trans=np.asarray(affine, dtype=np.float64))
    return DenseNeuroVol(vol.astype(np.float32), space)


def _write_nifti_image(img: neuroim.DenseNeuroVol, out_path: Path) -> None:
    """Write a neuroim volume as NIfTI."""
    from neuroim import write_vol

    write_vol(img, out_path, data_type="FLOAT")
    ensure_nifti_scl_slope(out_path)


def _nifti_header_payload(path: Path) -> tuple[bytearray, bool, str, int]:
    """Return ``(payload, gzipped, endian, sizeof_hdr)`` for a NIfTI file."""
    import gzip
    import struct

    raw = Path(path).read_bytes()
    gzipped = raw[:2] == b"\x1f\x8b"
    payload = bytearray(gzip.decompress(raw) if gzipped else raw)
    if len(payload) < 120:
        raise ValueError(f"{path} is too small to be a NIfTI header")
    sizeof_le = struct.unpack_from("<i", payload, 0)[0]
    sizeof_be = struct.unpack_from(">i", payload, 0)[0]
    if sizeof_le in (348, 540):
        return payload, gzipped, "<", sizeof_le
    if sizeof_be in (348, 540):
        return payload, gzipped, ">", sizeof_be
    raise ValueError(f"{path} is not a NIfTI-1/2 file")


def _raw_nifti_scl_slope(path: Path) -> float:
    """Read the on-disk ``scl_slope`` without nibabel identity sanitizing."""
    import struct

    payload, _, endian, sizeof_hdr = _nifti_header_payload(path)
    if sizeof_hdr == 348:
        return float(struct.unpack_from(endian + "f", payload, 112)[0])
    return float(struct.unpack_from(endian + "d", payload, 176)[0])


def ensure_nifti_scl_slope(path: Path, slope: float = 1.0) -> None:
    """Force a usable NIfTI ``scl_slope`` (fmrigds #6).

    Some writers leave ``scl_slope=0`` or NaN, which is spec-valid
    "no scaling" but breaks readers that multiply by the header
    slope. nibabel also drops an identity ``(1, 0)`` pair on write,
    so this patches the on-disk header after the writer finishes.
    """
    import gzip
    import struct

    payload, gzipped, endian, sizeof_hdr = _nifti_header_payload(path)
    if sizeof_hdr == 348:
        struct.pack_into(endian + "f", payload, 112, float(slope))
        struct.pack_into(endian + "f", payload, 116, 0.0)
    else:
        struct.pack_into(endian + "d", payload, 176, float(slope))
        struct.pack_into(endian + "d", payload, 184, 0.0)
    Path(path).write_bytes(gzip.compress(payload) if gzipped else bytes(payload))


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
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    p, V = betas.shape
    if column_names is None:
        column_names = [f"reg{i:03d}" for i in range(p)]

    written: list[Path] = []
    for i, name in enumerate(column_names):
        fname = _bids_filename(
            entities,
            suffix="statmap",
            stat="beta",
            contrast=name,
        )
        img = _make_nifti_image(betas[i], mask, affine)
        out_path = output_dir / fname
        _write_nifti_image(img, out_path)
        written.append(out_path)

    # JSON sidecar
    meta: dict[str, object] = {
        "Description": "GLM beta coefficients",
        "Software": "fmrimod",
        "GeneratedBy": {
            "Name": "fmrimod",
            "CodeURL": "https://github.com/bbuchsbaum/fmrimod",
        },
        "Columns": list(column_names),
        "Timestamp": datetime.now(timezone.utc).isoformat(),
    }
    json_path = output_dir / _bids_filename(
        entities,
        suffix="statmap",
        stat="beta",
        extension=".json",
    )
    _write_json_sidecar(json_path, meta)
    written.append(json_path)

    return written


def write_contrasts(
    contrasts: dict[str, object],
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
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    stat_extractors = {
        "beta": lambda c: c.estimate if c.estimate.ndim == 1 else c.estimate[0],
        "tstat": lambda c: c.stat,
        "pvalue": lambda c: c.p_value,
        "se": lambda c: c.se if c.se is not None else np.zeros_like(c.stat),
    }

    for con_name, raw_cres in contrasts.items():
        cres = cast(_ContrastResultLike, raw_cres)
        for stat_name in stats:
            extractor = stat_extractors.get(stat_name)
            if extractor is None:
                continue
            data = extractor(cres)
            fname = _bids_filename(
                entities,
                suffix="statmap",
                stat=stat_name,
                contrast=con_name,
            )
            img = _make_nifti_image(data, mask, affine)
            out_path = output_dir / fname
            _write_nifti_image(img, out_path)
            written.append(out_path)

        # JSON sidecar per contrast
        meta: dict[str, object] = {
            "Description": f"Contrast: {con_name}",
            "ContrastType": cres.stat_type,
            "DegreesOfFreedom": cres.df,
            "Software": "fmrimod",
            "Timestamp": datetime.now(timezone.utc).isoformat(),
        }
        json_path = output_dir / _bids_filename(
            entities,
            suffix="statmap",
            contrast=con_name,
            extension=".json",
        )
        _write_json_sidecar(json_path, meta)
        written.append(json_path)

    return written
