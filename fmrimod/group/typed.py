"""Typed-Spec entry point for group-level (second-level) analysis.

This module exposes :func:`fmri_group_lm`, a top-level counterpart to
:func:`fmrimod.fmri_lm` for the group-level / second-level stage of
the modeling pipeline. It:

- Accepts ``(n_subjects, n_voxels)`` per-subject contrast arrays
  directly, or a pre-built :class:`GroupDataset` for users who need
  the lower-level surface.
- Defaults to the one-sample t test
  (``intercept_only=True``) so the most common case is a one-line
  call.
- Returns a typed :class:`GroupLmResult` with ``.effect()``,
  ``.t_stat()``, ``.p_value()`` accessors so MVPA / group analyses
  pull stats by name instead of by ``"coef:Intercept"`` key strings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence, Union, cast

import numpy as np
from numpy.typing import NDArray

from .dataset import GroupDataset, group_dataset
from .reducers import ols_voxelwise
from .space import VoxelSpace

if TYPE_CHECKING:
    from ..glm.spatial import SpatialContext


Array = NDArray[np.float64]


@dataclass(frozen=True)
class GroupLmResult:
    """Typed result wrapper for :func:`fmri_group_lm`.

    Wraps the raw :class:`GroupDataset` returned by
    :func:`ols_voxelwise` and provides typed accessors so users pull
    stats by predictor name (``.effect("Intercept")``) rather than
    by ``"coef:Intercept"`` key strings.

    Attributes
    ----------
    dataset
        The underlying reduced ``GroupDataset`` — exposed for users
        who need the raw assays.
    predictor_names
        Tuple of predictor names from the second-level design (e.g.
        ``("Intercept",)`` for the one-sample default).
    residual_df
        Residual degrees of freedom (``n_subjects - n_predictors``).
    """

    dataset: GroupDataset
    predictor_names: tuple[str, ...]
    residual_df: float
    spatial: Optional["SpatialContext"] = None

    def _assay(self, prefix: str, predictor: str) -> Array:
        """Pull a per-predictor assay array, validating the predictor name.

        Squeezes the underlying ``(samples, subjects=1, contrasts=1)``
        layout into a flat ``(n_voxels,)`` array for the common case;
        when ``contrasts > 1`` the result keeps the contrast axis.
        """
        predictor = self.resolve_predictor(predictor)
        key = f"{prefix}:{predictor}"
        arr = np.asarray(self.dataset.assay(key), dtype=np.float64)
        # Default shape from ``ols_voxelwise`` is
        # ``(n_voxels, subjects=1, n_contrasts)``. Drop the singleton
        # subjects axis and squeeze a single-contrast trailing axis.
        if arr.ndim == 3:
            arr = arr[:, 0, :]
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        return arr

    def resolve_predictor(self, name: str) -> str:
        """Map ``name`` to a design column in :attr:`predictor_names`.

        Exact names win. Otherwise a factor level resolves to its
        treatment-coded column when exactly one column codes it, so
        ``"patient"`` finds ``"group[T.patient]"`` (the first-level rule
        that lets ``"face"`` mean ``trial_type.face``).
        """
        if name in self.predictor_names:
            return name
        matches = [p for p in self.predictor_names if p.endswith(f"[T.{name}]")]
        if len(matches) == 1:
            return matches[0]
        hint = f"; {name!r} matches {matches!r}" if matches else ""
        raise KeyError(
            f"GroupLmResult: predictor {name!r} not in "
            f"{self.predictor_names!r}{hint}"
        )

    def effect(self, predictor: str = "Intercept") -> Array:
        """Per-voxel coefficient estimate for ``predictor`` (default Intercept).

        Parameters
        ----------
        predictor
            Design column name: ``"Intercept"`` (default), a
            ``design_matrix=`` column, or a formula term such as
            ``"age"`` or ``"group[T.patient]"`` (see ``predictor_names``).
            An unambiguous factor level (``"patient"``) also works.

        Examples
        --------
        ::

            res = fm.fmri_group_lm(maps, formula="~ group + age",
                                   covariates=participants)
            res.effect("patient")  # same as "group[T.patient]"
        """
        return self._assay("coef", predictor)

    def t_stat(self, predictor: str = "Intercept") -> Array:
        """Per-voxel t statistic for ``predictor``.

        Parameters
        ----------
        predictor
            Design column name: ``"Intercept"`` (default), a
            ``design_matrix=`` column, or a formula term such as
            ``"age"`` or ``"group[T.patient]"`` (see ``predictor_names``).
            An unambiguous factor level (``"patient"``) also works.

        Examples
        --------
        ::

            res = fm.fmri_group_lm(maps, formula="~ group + age",
                                   covariates=participants)
            res.t_stat("patient")  # same as "group[T.patient]"
        """
        return self._assay("t_coef", predictor)

    def p_value(self, predictor: str = "Intercept") -> Array:
        """Per-voxel two-sided p-value for ``predictor``.

        Parameters
        ----------
        predictor
            Design column name: ``"Intercept"`` (default), a
            ``design_matrix=`` column, or a formula term such as
            ``"age"`` or ``"group[T.patient]"`` (see ``predictor_names``).
            An unambiguous factor level (``"patient"``) also works.

        Examples
        --------
        ::

            res = fm.fmri_group_lm(maps, formula="~ group + age",
                                   covariates=participants)
            res.p_value("patient")  # same as "group[T.patient]"
        """
        return self._assay("p_coef", predictor)

    def standard_error(self, predictor: str = "Intercept") -> Array:
        """Per-voxel SE for ``predictor``'s coefficient.

        Parameters
        ----------
        predictor
            Design column name: ``"Intercept"`` (default), a
            ``design_matrix=`` column, or a formula term such as
            ``"age"`` or ``"group[T.patient]"`` (see ``predictor_names``).
            An unambiguous factor level (``"patient"``) also works.

        Examples
        --------
        ::

            res = fm.fmri_group_lm(maps, formula="~ group + age",
                                   covariates=participants)
            res.standard_error("patient")  # same as "group[T.patient]"
        """
        return self._assay("se_coef", predictor)

    def z_score(self, predictor: str = "Intercept") -> Array:
        """Per-voxel z-score equivalent of :meth:`t_stat`.

        Parameters
        ----------
        predictor
            Design column name: ``"Intercept"`` (default), a
            ``design_matrix=`` column, or a formula term such as
            ``"age"`` or ``"group[T.patient]"`` (see ``predictor_names``).
            An unambiguous factor level (``"patient"``) also works.

        Examples
        --------
        ::

            res = fm.fmri_group_lm(maps, formula="~ group + age",
                                   covariates=participants)
            res.z_score("patient")  # same as "group[T.patient]"
        """
        from ..glm.contrasts import z_from_stat

        return z_from_stat(self.t_stat(predictor), self.residual_df, "t")

    def to_nifti(
        self,
        path: Union[str, Path],
        predictor: str = "Intercept",
        *,
        kind: str = "t",
        fill: float = 0.0,
    ) -> Path:
        """Write one group map to disk as NIfTI.

        Available when the inputs carried a spatial frame (NIfTI maps, or
        first-level ``ContrastResult``s fitted on images).

        Parameters
        ----------
        path
            Output ``.nii`` / ``.nii.gz`` path.
        predictor
            Design column, e.g. ``"Intercept"`` or ``"group[T.patient]"``.
        kind
            ``"t"`` (default), ``"z"``, ``"effect"``, ``"se"`` or ``"p"``.
        fill
            Value written outside the mask.

        Examples
        --------
        ::

            res = fm.fmri_group_lm(paths, formula="~ group + age",
                                   covariates=participants)
            res.to_nifti("patient_vs_control_z.nii.gz", "group[T.patient]",
                         kind="z")
        """
        if self.spatial is None:
            raise ValueError(
                "GroupLmResult has no spatial frame; fit it from NIfTI maps or "
                "image-based first-level contrasts to write NIfTI."
            )
        getters = {
            "t": self.t_stat,
            "z": self.z_score,
            "effect": self.effect,
            "se": self.standard_error,
            "p": self.p_value,
        }
        if kind not in getters:
            raise ValueError(f"kind must be one of {sorted(getters)}; got {kind!r}")
        vec = np.asarray(getters[kind](predictor), dtype=np.float64).ravel()
        return self.spatial.write_nifti(
            vec, path, label=f"{predictor}.{kind}", fill=fill
        )


def fmri_group_lm(
    data: Union[NDArray[np.floating], GroupDataset, Mapping[str, Any], Sequence[Any]],
    *,
    intercept_only: bool = True,
    formula: Optional[str] = None,
    design_matrix: Optional[Any] = None,
    subjects: Optional[Sequence[Any]] = None,
    contrasts: Optional[Sequence[Any]] = None,
    space: Optional[VoxelSpace] = None,
    n_jobs: int = 1,
    chunk_size: Optional[int] = None,
    covariates: Optional[Any] = None,
    mask: object = None,
) -> GroupLmResult:
    """Fit a group-level / second-level GLM.

    Closes the typed-spec gap between the first-level
    (:func:`fmrimod.fmri_lm`) and group-level stages of the
    modeling pipeline.

    Headline shape::

        # n_subjects x n_voxels per-subject contrast betas
        result = fm.fmri_group_lm(betas)
        result.t_stat()       # (n_voxels,) one-sample group t map
        result.effect()       # (n_voxels,) group-mean effect
        result.p_value()      # (n_voxels,) two-sided p map

    Parameters
    ----------
    data
        Either:

        - ``(n_subjects, n_voxels)`` numpy array of per-subject
          contrast values (the common case — wrap into a
          :class:`GroupDataset` internally).
        - A :class:`GroupDataset` for users who already have the
          full sample × subject × contrast layout.
        - A mapping ``{subject_id: ContrastResult}`` (or a list of
          ``ContrastResult``, named by ``subjects=``) straight from
          first-level ``fit.contrast(...)`` calls; lowered with
          :func:`fmrimod.group_dataset_from_contrasts`, which also
          carries each subject's standard errors.
        - A list of per-subject effect maps: NIfTI paths, nibabel images
          or neuroim volumes (as Nilearn's ``SecondLevelModel.fit``).
          Voxels are those in ``mask=`` (default: finite and non-zero in
          every map), and the result can be written back with
          :meth:`GroupLmResult.to_nifti`.
    intercept_only
        Default ``True`` — the canonical one-sample group t test
        (``formula="~ 1"``). Mutually exclusive with ``formula=``
        and ``design_matrix=``.
    formula
        Optional R-style design formula (e.g. ``"~ group + age"``).
        Mutually exclusive with ``intercept_only=True`` and
        ``design_matrix=``.
    design_matrix
        Optional explicit design matrix or DataFrame; predictor
        column names become the keys for :meth:`GroupLmResult.effect`.
    subjects
        Subject ids when wrapping a raw array. Defaults to
        ``"sub-{i:02d}"`` strings.
    contrasts
        First-level contrast names. Defaults to
        ``("contrast_0", "contrast_1", ...)``.
    space
        Optional explicit :class:`VoxelSpace` when wrapping a raw
        array. Defaults to a dense voxel space with shape
        ``(n_voxels, 1, 1)`` and identity affine.
    n_jobs, chunk_size
        Forwarded to :func:`ols_voxelwise`.
    covariates
        Subject-level table for ``formula=``: a DataFrame or the path of a
        TSV/CSV file such as a BIDS ``participants.tsv`` (``n/a`` read as
        missing). Rows are matched
        to subjects through a ``subject`` or ``participant_id`` column when
        present (a ``sub-`` prefix is ignored), otherwise taken in input
        order.
    mask
        Image-map input only: brain mask (path, image or boolean array).

    Returns
    -------
    GroupLmResult
        Typed wrapper exposing ``.effect()``, ``.t_stat()``,
        ``.p_value()``, ``.standard_error()`` per predictor.

    Examples
    --------
    One-sample group t in one line::

        result = fm.fmri_group_lm(per_subject_betas)
        t_map = result.t_stat()

    Straight from first-level fits (subject id -> ``ContrastResult``)::

        per_subject = {
            sub: fm.fmri_lm(spec, ds).contrast("face - house")
            for sub, ds in datasets.items()
        }
        t_map = fm.fmri_group_lm(per_subject).t_stat()

    Two-group comparison with covariate::

        import pandas as pd
        design = pd.DataFrame({"group": [0,0,0,1,1,1], "age": [...]})
        result = fm.fmri_group_lm(
            per_subject_betas,
            intercept_only=False,
            design_matrix=design,
        )
        group_effect = result.effect("group")
        group_t = result.t_stat("group")
    """
    # Mode resolution — at most one of (intercept_only, formula,
    # design_matrix) is meaningful.
    n_explicit = sum(
        1 for x in (formula, design_matrix) if x is not None
    )
    if n_explicit > 1:
        raise ValueError(
            "fmri_group_lm: pass at most one of formula= or "
            "design_matrix=; intercept_only=True is the default for "
            "the one-sample case"
        )

    # Coerce ``data`` into a GroupDataset.
    spatial = None
    contrast_map = _as_contrast_mapping(data, subjects)
    if contrast_map is None and _is_image_list(data):
        data, spatial = _load_masked_maps(cast(Sequence[Any], data), mask)
    elif mask is not None:
        raise ValueError("fmri_group_lm: mask= applies only to image-map input")
    if contrast_map is not None:
        from ..contrast.group_collection import group_dataset_from_contrasts

        names = list(contrast_map)
        dataset = group_dataset_from_contrasts(
            contrast_map, covariates=_subject_table(covariates, names)
        )
        spatial = getattr(next(iter(contrast_map.values())), "spatial", None)
    elif isinstance(data, GroupDataset):
        if covariates is not None:
            raise ValueError(
                "fmri_group_lm: a GroupDataset carries its own col_data; "
                "do not also pass covariates="
            )
        dataset = data
    else:
        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(
                f"fmri_group_lm: ``data`` must be 2-D "
                f"(n_subjects, n_voxels) or a GroupDataset; got shape "
                f"{arr.shape!r}"
            )
        n_subjects, n_voxels = arr.shape
        beta_3d = arr.T[:, :, np.newaxis]
        resolved_space = (
            space
            if space is not None
            else VoxelSpace(shape=(n_voxels, 1, 1))
        )
        if subjects is None:
            # Maps/rows carry no ids: rows of the covariate table describe
            # them in order (as a Nilearn second-level design does).
            subjects = _ids_from_covariates(covariates, n_subjects)
        resolved_subjects = (
            list(subjects)
            if subjects is not None
            else [f"sub-{i:02d}" for i in range(n_subjects)]
        )
        resolved_contrasts = (
            list(contrasts) if contrasts is not None else ["contrast_0"]
        )
        dataset = group_dataset(
            assays={"beta": beta_3d},
            space=resolved_space,
            subjects=resolved_subjects,
            contrasts=resolved_contrasts,
            col_data=_subject_table(covariates, resolved_subjects),
        )

    # Resolve the design. When ``design_matrix=`` is a DataFrame, the
    # column names become the typed predictor labels; otherwise the
    # reducer falls back to ``x0, x1, ...`` placeholders that we
    # don't want to leak to the user.
    user_predictor_names: Optional[tuple[str, ...]] = None
    if formula is not None:
        ols_kwargs: dict[str, Any] = {"formula": formula}
    elif design_matrix is not None:
        try:
            import pandas as _pd
        except ImportError:  # pragma: no cover - pandas is a hard dep
            _pd = None  # type: ignore[assignment]
        if _pd is not None and isinstance(design_matrix, _pd.DataFrame):
            user_predictor_names = tuple(str(c) for c in design_matrix.columns)
            ols_kwargs = {
                "X": np.asarray(design_matrix.to_numpy(), dtype=np.float64),
            }
        else:
            ols_kwargs = {
                "X": np.asarray(design_matrix, dtype=np.float64),
            }
    elif intercept_only:
        ols_kwargs = {"formula": "~ 1"}
    else:
        raise ValueError(
            "fmri_group_lm: with intercept_only=False, pass either "
            "formula= or design_matrix=."
        )

    result_ds = ols_voxelwise(
        dataset,
        **ols_kwargs,
        n_jobs=n_jobs,
        chunk_size=chunk_size,
    )

    # Predictor names live on the result's metadata. Prefer the
    # user-supplied DataFrame column names when available; otherwise
    # take the reducer's record.
    metadata = getattr(result_ds, "metadata", {}) or {}
    raw_names = tuple(
        metadata.get("predictor_names") or ("Intercept",)
    )
    if user_predictor_names is not None:
        # Rename the assay keys from ``x{i}`` to the user-visible
        # DataFrame column names so ``result.effect("group")``
        # resolves cleanly.
        predictor_names = user_predictor_names
        result_ds = _rename_predictor_assays(
            result_ds, raw_names, predictor_names
        )
    else:
        predictor_names = raw_names
    df_res_assay = np.asarray(result_ds.assay("df_res"))
    residual_df = float(df_res_assay.flat[0])

    return GroupLmResult(
        dataset=result_ds,
        predictor_names=predictor_names,
        residual_df=residual_df,
        spatial=spatial,
    )


def _rename_predictor_assays(
    result_ds: GroupDataset,
    raw_names: tuple[str, ...],
    new_names: tuple[str, ...],
) -> GroupDataset:
    """Rebuild a GroupDataset with predictor-suffix assay keys renamed.

    Used when the caller passes a DataFrame ``design_matrix=`` whose
    column names should appear on the typed accessors but
    ``ols_voxelwise`` only saw the ndarray and tagged the assays as
    ``coef:x0``, ``coef:x1``, ...
    """
    if len(raw_names) != len(new_names):
        return result_ds
    if raw_names == new_names:
        return result_ds
    raw_to_new = dict(zip(raw_names, new_names))
    new_assays: dict[str, Any] = {}
    for key, value in result_ds.assays.items():
        if ":" not in key:
            new_assays[key] = value
            continue
        prefix, predictor = key.split(":", 1)
        renamed = raw_to_new.get(predictor, predictor)
        new_assays[f"{prefix}:{renamed}"] = value

    # Preserve metadata but record the rename.
    new_meta = dict(result_ds.metadata)
    new_meta["predictor_names"] = tuple(new_names)
    return GroupDataset(
        assays=new_assays,
        space=result_ds.space,
        subjects=result_ds.subjects,
        contrasts=result_ds.contrasts,
        col_data=result_ds.col_data,
        row_data=result_ds.row_data,
        contrast_data=result_ds.contrast_data,
        metadata=new_meta,
    )


def _as_contrast_mapping(
    data: object, subjects: Optional[Sequence[Any]]
) -> Optional[dict[str, Any]]:
    """Return ``{subject: ContrastResult}`` when ``data`` holds contrast results."""
    from ..glm.contrasts import ContrastResult

    if isinstance(data, Mapping):
        values = list(data.values())
        if values and all(isinstance(v, ContrastResult) for v in values):
            if subjects is not None:
                raise ValueError(
                    "fmri_group_lm: subjects= comes from the mapping keys; "
                    "do not pass both"
                )
            return {str(k): v for k, v in data.items()}
        return None
    if isinstance(data, (list, tuple)) and data and all(
        isinstance(v, ContrastResult) for v in data
    ):
        names = (
            [str(s) for s in subjects]
            if subjects is not None
            else [f"sub-{i:02d}" for i in range(len(data))]
        )
        if len(names) != len(data):
            raise ValueError(
                f"fmri_group_lm: {len(names)} subjects for {len(data)} contrasts"
            )
        return dict(zip(names, data, strict=True))
    return None


def _subject_table(covariates: Any, subjects: Sequence[Any]) -> Any:
    """Return ``covariates`` ordered like ``subjects`` with a ``subject`` column."""
    if covariates is None:
        return None
    table = _read_covariates(covariates).copy()
    names = [str(s) for s in subjects]
    if "subject" not in table.columns and "participant_id" in table.columns:
        table = table.rename(columns={"participant_id": "subject"})
    if "subject" not in table.columns:
        if len(table) != len(names):
            raise ValueError(
                f"fmri_group_lm: {len(table)} covariate rows for {len(names)} "
                "subjects and no subject/participant_id column to match on"
            )
        return table.reset_index(drop=True).assign(subject=names)
    keys = table["subject"].astype(str).str.replace(r"^sub-", "", regex=True)
    wanted = [n.removeprefix("sub-") for n in names]
    missing = sorted(set(wanted) - set(keys))
    if missing:
        raise ValueError(
            f"fmri_group_lm: covariates have no row for subjects {missing[:5]}"
        )
    position = {key: i for i, key in enumerate(keys)}
    ordered = table.iloc[[position[w] for w in wanted]].reset_index(drop=True)
    return ordered.assign(subject=names)


def _read_covariates(covariates: Any) -> Any:
    """A DataFrame from a frame, mapping, or TSV/CSV path (BIDS ``n/a`` aware)."""
    import pandas as pd

    if isinstance(covariates, (str, Path)):
        path = Path(covariates)
        sep = "," if path.suffix.lower() == ".csv" else "\t"
        return pd.read_csv(path, sep=sep, na_values="n/a")
    return pd.DataFrame(covariates)


def _ids_from_covariates(covariates: Any, n_subjects: int) -> Optional[list[str]]:
    if covariates is None:
        return None
    table = _read_covariates(covariates)
    for column in ("subject", "participant_id"):
        if column in table.columns and len(table) == n_subjects:
            return [str(v) for v in table[column]]
    return None


def _is_image_list(data: object) -> bool:
    if not isinstance(data, (list, tuple)) or not data:
        return False
    return all(
        isinstance(item, (str, Path))
        or hasattr(item, "dataobj")
        or hasattr(item, "as_array")
        for item in data
    )


def _read_volume(item: Any) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(3-D data, 4x4 affine)`` for a path, nibabel image or NeuroVol."""
    if isinstance(item, (str, Path)):
        import neuroim

        item = neuroim.read_vol(str(item))
    if hasattr(item, "as_array"):
        return (
            np.asarray(item.as_array(), dtype=np.float64),
            np.asarray(item.trans, dtype=np.float64),
        )
    return (
        np.asarray(np.asanyarray(item.dataobj), dtype=np.float64),
        np.asarray(item.affine, dtype=np.float64),
    )


def _load_masked_maps(
    items: Sequence[Any], mask: object
) -> tuple[NDArray[np.float64], "SpatialContext"]:
    """Stack per-subject maps inside a mask; return ``(n_subjects, n_voxels)``."""
    from ..glm.spatial import SpatialContext

    volumes: list[NDArray[np.float64]] = []
    affine: Optional[NDArray[np.float64]] = None
    for item in items:
        data, item_affine = _read_volume(item)
        data = np.squeeze(data)
        if data.ndim != 3:
            raise ValueError(
                f"fmri_group_lm: expected 3-D maps; got shape {data.shape}"
            )
        if volumes and data.shape != volumes[0].shape:
            raise ValueError(
                f"fmri_group_lm: map shapes differ ({volumes[0].shape} vs "
                f"{data.shape})"
            )
        volumes.append(data)
        affine = item_affine if affine is None else affine
    stack = np.stack(volumes)
    if mask is None:
        mask_arr = np.all(np.isfinite(stack) & (stack != 0), axis=0)
    elif isinstance(mask, np.ndarray):
        mask_arr = mask.astype(bool)
    else:
        mask_arr = _read_volume(mask)[0].astype(bool)
    if mask_arr.shape != stack.shape[1:]:
        raise ValueError(
            f"fmri_group_lm: mask shape {mask_arr.shape} != map shape "
            f"{stack.shape[1:]}"
        )
    spatial = SpatialContext(
        mask=mask_arr,
        spatial_shape=cast("tuple[int, int, int]", tuple(mask_arr.shape)),
        affine=affine,
    )
    return stack[:, mask_arr], spatial
