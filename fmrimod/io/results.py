"""High-level result export helpers."""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Union

import numpy as np
from numpy.typing import NDArray

from fmrimod.bids import BidsEntities
from fmrimod.bids.export import _bids_filename, _make_nifti_image, _write_nifti_image

BetaSelection = Union[Literal["task", "all"], Sequence[str], bool]


@dataclass(frozen=True)
class ManifestVolume:
    """One volume inside a written 4-D result bundle."""

    index: int
    column: int
    label: str
    safe_label: str
    role: str
    term: str | None = None
    condition: str | None = None
    basis_index: int | None = None
    basis_name: str | None = None
    model_source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _drop_none(
            {
                "index": self.index,
                "column": self.column,
                "label": self.label,
                "safe_label": self.safe_label,
                "role": self.role,
                "term": self.term,
                "condition": self.condition,
                "basis_index": self.basis_index,
                "basis_name": self.basis_name,
                "model_source": self.model_source,
            }
        )


@dataclass(frozen=True)
class ManifestFile:
    """A result file recorded in the manifest."""

    path: Path
    kind: str
    layout: Literal["3d", "4d", "json"]
    stat: str | None = None
    contrast: str | None = None
    beta_group: str | None = None
    volumes: tuple[ManifestVolume, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self, *, root: Path) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "path": self.path.relative_to(root).as_posix(),
            "kind": self.kind,
            "layout": self.layout,
        }
        if self.stat is not None:
            payload["stat"] = self.stat
        if self.contrast is not None:
            payload["contrast"] = self.contrast
        if self.beta_group is not None:
            payload["beta_group"] = self.beta_group
        if self.volumes:
            payload["volumes"] = [volume.to_dict() for volume in self.volumes]
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class ResultsManifest:
    """Manifest returned by :func:`write_results` and written as JSON."""

    root: Path
    manifest_path: Path
    files: tuple[ManifestFile, ...]
    entities: BidsEntities
    schema_version: str = "fmrimod.results_manifest.v1"
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def paths(self) -> tuple[Path, ...]:
        """All written paths, including the manifest JSON."""
        return tuple(file.path for file in self.files) + (self.manifest_path,)

    def __iter__(self) -> Iterator[Path]:
        return iter(self.paths)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Path:
        return self.paths[index]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "generated_by": {"name": "fmrimod"},
            "entities": _drop_none(
                {
                    "subject": self.entities.subject,
                    "task": self.entities.task,
                    "space": self.entities.space,
                    "run": self.entities.run,
                    "desc": self.entities.desc,
                }
            ),
            "files": [file.to_dict(root=self.root) for file in self.files],
        }

    def write_json(self, path: str | Path | None = None) -> Path:
        """Write the manifest JSON and return its path."""
        out = Path(path) if path is not None else self.manifest_path
        with open(out, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
            f.write("\n")
        return out


@dataclass(frozen=True)
class _ColumnRecord:
    index: int
    name: str
    safe_label: str
    role: str
    term: str | None = None
    condition: str | None = None
    basis_index: int | None = None
    basis_name: str | None = None
    model_source: str | None = None

    def as_volume(self, volume_index: int) -> ManifestVolume:
        return ManifestVolume(
            index=volume_index,
            column=self.index,
            label=self.name,
            safe_label=self.safe_label,
            role=self.role,
            term=self.term,
            condition=self.condition,
            basis_index=self.basis_index,
            basis_name=self.basis_name,
            model_source=self.model_source,
        )


def _infer_mask_and_affine(
    result: object,
    mask: NDArray[np.bool_] | None,
    affine: NDArray[np.float64] | None,
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

    return (
        np.asarray(inferred_mask, dtype=bool),
        np.asarray(inferred_affine, dtype=np.float64),
    )


def write_results(
    result: object,
    output_dir: str | Path,
    *,
    subject: str,
    task: str,
    space: str = "MNI152NLin2009cAsym",
    run: str | None = None,
    desc: str | None = "firstlevel",
    betas: BetaSelection = "task",
    beta_layout: Literal["4d"] = "4d",
    beta_groups: Sequence[str] = ("task",),
    contrasts: Literal["all"] | Sequence[str] | bool = "all",
    stats: Sequence[str] = ("effect", "stat", "se", "pvalue"),
    save_betas: bool | None = None,
    save_contrasts: bool | None = None,
    contrast_stats: Sequence[str] | None = None,
    mask: NDArray[np.bool_] | None = None,
    affine: NDArray[np.float64] | None = None,
    column_names: Sequence[str] | None = None,
    overwrite: bool = False,
) -> ResultsManifest:
    """Write fitted model results as NIfTI files plus a manifest JSON.

    Betas are bundled as 4-D NIfTI files by semantic group. The default writes
    task betas only, leaving nuisance, drift, and intercept columns out of the
    main export unless explicitly requested with ``betas="all"`` and
    corresponding ``beta_groups``.
    """
    if not hasattr(result, "betas"):
        raise TypeError("result must expose a 'betas' attribute")
    if beta_layout != "4d":
        raise ValueError("write_results currently supports beta_layout='4d' only")

    if save_betas is not None:
        betas = "task" if save_betas else False
    if save_contrasts is not None:
        contrasts = "all" if save_contrasts else False
    if contrast_stats is not None:
        stats = tuple(_normalize_stat_name(stat) for stat in contrast_stats)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    mask_arr, affine_arr = _infer_mask_and_affine(result, mask=mask, affine=affine)
    entities = BidsEntities(subject=subject, task=task, space=space, run=run, desc=desc)
    beta_matrix = np.asarray(result.betas, dtype=np.float64)
    columns = _column_records(result, beta_matrix.shape[0], column_names=column_names)

    tmp_dir = Path(tempfile.mkdtemp(prefix=".write_results-", dir=out))
    staged: list[ManifestFile] = []
    try:
        if betas:
            selected = _select_beta_columns(columns, betas)
            staged.extend(
                _write_beta_bundles(
                    beta_matrix=beta_matrix,
                    columns=selected,
                    beta_groups=beta_groups,
                    mask=mask_arr,
                    affine=affine_arr,
                    output_dir=tmp_dir,
                    entities=entities,
                )
            )

        selected_contrasts = _select_contrasts(
            getattr(result, "contrasts", None) or {}, contrasts
        )
        if selected_contrasts:
            staged.extend(
                _write_contrast_statmaps(
                    selected_contrasts,
                    stats=stats,
                    mask=mask_arr,
                    affine=affine_arr,
                    output_dir=tmp_dir,
                    entities=entities,
                )
            )

        manifest_name = _bids_filename(
            entities,
            suffix="manifest",
            extension=".json",
        )
        manifest_tmp = tmp_dir / manifest_name
        manifest_final = out / manifest_name

        final_files = tuple(
            ManifestFile(
                path=out / file.path.name,
                kind=file.kind,
                layout=file.layout,
                stat=file.stat,
                contrast=file.contrast,
                beta_group=file.beta_group,
                volumes=file.volumes,
                metadata=file.metadata,
            )
            for file in staged
        )
        manifest = ResultsManifest(
            root=out,
            manifest_path=manifest_final,
            files=final_files,
            entities=entities,
        )
        with open(manifest_tmp, "w") as f:
            json.dump(manifest.to_dict(), f, indent=2)
            f.write("\n")

        targets = [out / file.path.name for file in staged] + [manifest_final]
        collisions = [target for target in targets if target.exists()]
        if collisions and not overwrite:
            raise FileExistsError(
                "Refusing to overwrite existing output: "
                + ", ".join(str(path) for path in collisions)
            )

        for target in collisions:
            target.unlink()
        for file in staged:
            file.path.replace(out / file.path.name)
        manifest_tmp.replace(manifest_final)

        return manifest
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _write_beta_bundles(
    *,
    beta_matrix: NDArray[np.float64],
    columns: Sequence[_ColumnRecord],
    beta_groups: Sequence[str],
    mask: NDArray[np.bool_],
    affine: NDArray[np.float64],
    output_dir: Path,
    entities: BidsEntities,
) -> list[ManifestFile]:
    grouped = _group_beta_columns(columns, beta_groups)
    files: list[ManifestFile] = []
    for group_name, group_columns in grouped:
        _require_unique_safe_labels(group_columns, context=f"beta group {group_name!r}")
        data = np.zeros(mask.shape + (len(group_columns),), dtype=np.float32)
        for volume_index, column in enumerate(group_columns):
            data[..., volume_index][mask] = beta_matrix[column.index]
        fname = _bids_filename(
            entities,
            suffix="statmap",
            stat="beta",
            contrast=f"{group_name}Betas",
        )
        path = output_dir / fname
        _write_4d_nifti(data, affine, path)
        files.append(
            ManifestFile(
                path=path,
                kind="beta_bundle",
                layout="4d",
                stat="beta",
                beta_group=group_name,
                volumes=tuple(
                    column.as_volume(volume_index)
                    for volume_index, column in enumerate(group_columns)
                ),
            )
        )
    return files


def _write_contrast_statmaps(
    contrast_results: Mapping[str, object],
    *,
    stats: Sequence[str],
    mask: NDArray[np.bool_],
    affine: NDArray[np.float64],
    output_dir: Path,
    entities: BidsEntities,
) -> list[ManifestFile]:
    files: list[ManifestFile] = []
    safe_contrasts = [_safe_label(name) for name in contrast_results]
    _require_unique_values(
        safe_contrasts,
        context="contrast names",
        labels=list(contrast_results),
    )

    for contrast_name, cres in contrast_results.items():
        for stat_name in stats:
            data = _contrast_stat_array(cres, stat_name)
            if data is None:
                continue
            fname = _bids_filename(
                entities,
                suffix="statmap",
                stat=_filename_stat(stat_name),
                contrast=contrast_name,
            )
            path = output_dir / fname
            img = _make_nifti_image(np.asarray(data, dtype=np.float64), mask, affine)
            _write_nifti_image(img, path)
            files.append(
                ManifestFile(
                    path=path,
                    kind="contrast_statmap",
                    layout="3d",
                    stat=stat_name,
                    contrast=contrast_name,
                    metadata={
                        "stat_type": getattr(cres, "stat_type", None),
                        "df": getattr(cres, "df", None),
                    },
                )
            )
    return files


def _write_4d_nifti(
    data: NDArray[np.float32],
    affine: NDArray[np.float64],
    path: Path,
) -> None:
    import nibabel as nib

    img = nib.Nifti1Image(data, affine=np.asarray(affine, dtype=np.float64))
    nib.save(img, str(path))


def _column_records(
    result: object,
    n_columns: int,
    *,
    column_names: Sequence[str] | None,
) -> tuple[_ColumnRecord, ...]:
    design_columns = None
    method = getattr(result, "design_columns", None)
    if callable(method):
        try:
            design_columns = method()
        except Exception:
            design_columns = None
    if design_columns is None:
        model = getattr(result, "model", None)
        method = getattr(model, "design_columns", None)
        if callable(method):
            try:
                design_columns = method()
            except Exception:
                design_columns = None
    default_role = "unknown" if design_columns is not None else "task"

    names = list(column_names) if column_names is not None else None
    if (
        names is None
        and design_columns is not None
        and hasattr(design_columns, "names")
    ):
        names = list(design_columns.names)
    if names is None:
        names = [f"reg{i:03d}" for i in range(n_columns)]
    if len(names) != n_columns:
        raise ValueError(
            f"column_names length {len(names)} does not match beta rows {n_columns}"
        )

    by_index: dict[int, Any] = {}
    if design_columns is not None:
        for fallback_index, column in enumerate(design_columns):
            by_index[int(getattr(column, "index", fallback_index))] = column

    records: list[_ColumnRecord] = []
    for index, name in enumerate(names):
        column = by_index.get(index)
        records.append(
            _ColumnRecord(
                index=index,
                name=str(getattr(column, "name", name)),
                safe_label=_safe_label(str(getattr(column, "name", name))),
                role=str(getattr(column, "role", default_role)),
                term=_attr_or_none(column, "term"),
                condition=_attr_or_none(column, "condition"),
                basis_index=_attr_or_none(column, "basis_ix"),
                basis_name=_attr_or_none(column, "basis_name"),
                model_source=_attr_or_none(column, "model_source"),
            )
        )
    return tuple(records)


def _select_beta_columns(
    columns: Sequence[_ColumnRecord],
    betas: BetaSelection,
) -> tuple[_ColumnRecord, ...]:
    if betas is True:
        return tuple(columns)
    if betas == "all":
        return tuple(columns)
    if betas == "task":
        return tuple(column for column in columns if _beta_group(column) == "task")
    if betas is False:
        return ()
    selected_names = {str(name) for name in betas}
    selected = tuple(column for column in columns if column.name in selected_names)
    missing = selected_names.difference(column.name for column in selected)
    if missing:
        raise ValueError(f"Unknown beta column(s): {', '.join(sorted(missing))}")
    return selected


def _group_beta_columns(
    columns: Sequence[_ColumnRecord],
    beta_groups: Sequence[str],
) -> list[tuple[str, tuple[_ColumnRecord, ...]]]:
    groups: list[tuple[str, tuple[_ColumnRecord, ...]]] = []
    for group in beta_groups:
        selected = tuple(column for column in columns if _beta_group(column) == group)
        if selected:
            groups.append((group, selected))
    if not groups and columns:
        groups.append(("selected", tuple(columns)))
    return groups


def _beta_group(column: _ColumnRecord) -> str:
    role = column.role.lower()
    if role in {"task", "event", "condition"}:
        return "task"
    if role in {"nuisance", "confound", "confounds"}:
        return "nuisance"
    if role in {"drift", "trend", "poly", "fourier"}:
        return "drift"
    if role in {"intercept", "constant"}:
        return "intercept"
    if role == "baseline":
        name = column.name.lower()
        if "intercept" in name or name in {"constant", "const"}:
            return "intercept"
        if "drift" in name or "trend" in name:
            return "drift"
        return "nuisance"
    return role or "unknown"


def _select_contrasts(
    contrast_results: Mapping[str, object],
    contrasts: Literal["all"] | Sequence[str] | bool,
) -> Mapping[str, object]:
    if contrasts is False:
        return {}
    if contrasts is True or contrasts == "all":
        return contrast_results
    selected_names = {str(name) for name in contrasts}
    missing = selected_names.difference(contrast_results)
    if missing:
        raise ValueError(f"Unknown contrast(s): {', '.join(sorted(missing))}")
    return {name: contrast_results[name] for name in contrasts}


def _contrast_stat_array(cres: object, stat_name: str) -> NDArray[np.float64] | None:
    stat = _normalize_stat_name(stat_name)
    if stat == "effect":
        estimate = np.asarray(cres.estimate, dtype=np.float64)
        return estimate if estimate.ndim == 1 else estimate[0]
    if stat == "stat":
        return np.asarray(cres.stat, dtype=np.float64)
    if stat == "pvalue":
        return np.asarray(cres.p_value, dtype=np.float64)
    if stat == "se":
        se = getattr(cres, "se", None)
        if se is None:
            return np.zeros_like(np.asarray(cres.stat, dtype=np.float64))
        return np.asarray(se, dtype=np.float64)
    return None


def _normalize_stat_name(stat_name: str) -> str:
    aliases = {"beta": "effect", "tstat": "stat", "t": "stat", "p": "pvalue"}
    return aliases.get(str(stat_name).lower(), str(stat_name).lower())


def _filename_stat(stat_name: str) -> str:
    names = {"effect": "effect", "stat": "stat", "pvalue": "pvalue", "se": "se"}
    return names.get(_normalize_stat_name(stat_name), stat_name)


def _safe_label(label: str) -> str:
    import re

    safe = re.sub(r"[^a-zA-Z0-9]", "", label)
    if not safe:
        raise ValueError(f"Label {label!r} cannot be used in a BIDS filename")
    return safe


def _require_unique_safe_labels(
    columns: Sequence[_ColumnRecord],
    *,
    context: str,
) -> None:
    _require_unique_values(
        [column.safe_label for column in columns],
        context=context,
        labels=[column.name for column in columns],
    )


def _require_unique_values(
    values: Sequence[str],
    *,
    context: str,
    labels: Sequence[str],
) -> None:
    seen: dict[str, str] = {}
    for value, label in zip(values, labels):
        previous = seen.get(value)
        if previous is not None:
            raise ValueError(
                f"Sanitized label collision in {context}: {previous!r} and {label!r} both map to {value!r}"
            )
        seen[value] = label


def _attr_or_none(obj: object, attr: str) -> Any:
    if obj is None:
        return None
    value = getattr(obj, attr, None)
    return value


def _drop_none(values: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}
