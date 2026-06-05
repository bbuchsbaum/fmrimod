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
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray

from .dataset import GroupDataset, group_dataset
from .reducers import ols_voxelwise
from .space import VoxelSpace

if TYPE_CHECKING:
    pass


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

    def _assay(self, prefix: str, predictor: str) -> Array:
        """Pull a per-predictor assay array, validating the predictor name.

        Squeezes the underlying ``(samples, subjects=1, contrasts=1)``
        layout into a flat ``(n_voxels,)`` array for the common case;
        when ``contrasts > 1`` the result keeps the contrast axis.
        """
        if predictor not in self.predictor_names:
            raise KeyError(
                f"GroupLmResult: predictor {predictor!r} not in "
                f"{self.predictor_names!r}"
            )
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

    def effect(self, predictor: str = "Intercept") -> Array:
        """Per-voxel coefficient estimate for ``predictor`` (default Intercept)."""
        return self._assay("coef", predictor)

    def t_stat(self, predictor: str = "Intercept") -> Array:
        """Per-voxel t statistic for ``predictor``."""
        return self._assay("t_coef", predictor)

    def p_value(self, predictor: str = "Intercept") -> Array:
        """Per-voxel two-sided p-value for ``predictor``."""
        return self._assay("p_coef", predictor)

    def standard_error(self, predictor: str = "Intercept") -> Array:
        """Per-voxel SE for ``predictor``'s coefficient."""
        return self._assay("se_coef", predictor)


def fmri_group_lm(
    data: Union[NDArray[np.floating], GroupDataset],
    *,
    intercept_only: bool = True,
    formula: Optional[str] = None,
    design_matrix: Optional[Any] = None,
    subjects: Optional[Sequence[Any]] = None,
    contrasts: Optional[Sequence[Any]] = None,
    space: Optional[VoxelSpace] = None,
    n_jobs: int = 1,
    chunk_size: Optional[int] = None,
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
    if isinstance(data, GroupDataset):
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
