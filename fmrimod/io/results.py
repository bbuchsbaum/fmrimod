"""High-level result export helpers."""

from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from ..bids import BidsEntities, write_betas, write_contrasts


def _infer_mask_and_affine(
    result: object,
    mask: Optional[NDArray[np.bool_]],
    affine: Optional[NDArray[np.float64]],
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    if mask is not None and affine is not None:
        return np.asarray(mask, dtype=bool), np.asarray(affine, dtype=np.float64)

    model = getattr(result, "model", None)
    dataset = getattr(model, "dataset", None)

    inferred_mask = mask
    if inferred_mask is None and dataset is not None and hasattr(dataset, "get_mask"):
        inferred_mask = dataset.get_mask()

    inferred_affine = affine
    if inferred_affine is None and dataset is not None:
        if hasattr(dataset, "get_affine"):
            inferred_affine = dataset.get_affine()
        else:
            source = getattr(dataset, "_source", None)
            if source is not None and hasattr(source, "get_affine"):
                inferred_affine = source.get_affine()

    if inferred_mask is None:
        raise ValueError(
            "Could not infer mask for write_results; pass mask explicitly or use a dataset adapter with get_mask()"
        )
    if inferred_affine is None:
        inferred_affine = np.eye(4, dtype=np.float64)

    return np.asarray(inferred_mask, dtype=bool), np.asarray(inferred_affine, dtype=np.float64)


def write_results(
    result: object,
    output_dir: str | Path,
    *,
    subject: str,
    task: str,
    space: str = "MNI152NLin2009cAsym",
    run: Optional[str] = None,
    desc: Optional[str] = "GLM",
    save_betas: bool = True,
    save_contrasts: bool = True,
    contrast_stats: Sequence[str] = ("beta", "tstat", "pvalue", "se"),
    mask: Optional[NDArray[np.bool_]] = None,
    affine: Optional[NDArray[np.float64]] = None,
    column_names: Optional[Sequence[str]] = None,
    overwrite: bool = False,
) -> list[Path]:
    """Write fitted model results using BIDS-style filenames.

    The function writes into a temporary directory first and then atomically
    moves files into ``output_dir``.
    """
    if not hasattr(result, "betas"):
        raise TypeError("result must expose a 'betas' attribute")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    mask_arr, affine_arr = _infer_mask_and_affine(result, mask=mask, affine=affine)
    entities = BidsEntities(subject=subject, task=task, space=space, run=run, desc=desc)

    tmp_dir = Path(tempfile.mkdtemp(prefix=".write_results-", dir=out))
    staged: list[Path] = []
    try:
        if save_betas:
            staged.extend(
                write_betas(
                    betas=np.asarray(result.betas, dtype=np.float64),
                    mask=mask_arr,
                    affine=affine_arr,
                    output_dir=tmp_dir,
                    entities=entities,
                    column_names=column_names,
                )
            )

        if save_contrasts:
            contrasts = getattr(result, "contrasts", None) or {}
            if contrasts:
                staged.extend(
                    write_contrasts(
                        contrasts=contrasts,
                        mask=mask_arr,
                        affine=affine_arr,
                        output_dir=tmp_dir,
                        entities=entities,
                        stats=contrast_stats,
                    )
                )

        final_paths: list[Path] = []
        for staged_path in staged:
            target = out / staged_path.name
            if target.exists() and not overwrite:
                raise FileExistsError(
                    f"Refusing to overwrite existing output: {target}"
                )
            if target.exists():
                target.unlink()
            staged_path.replace(target)
            final_paths.append(target)

        return final_paths
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

