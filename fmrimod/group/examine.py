"""Native group examination.

``examine_group`` branches from subject-level :class:`GroupDataset` data at a
group reducer. It reports model-conditioned predictive surprise separately
from each subject's exact influence on requested group estimands. It never
deletes a subject or rewrites the source analysis.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ._examine_control import (
    ExaminationControl,
    NaAction,
    examination_control,
)
from ._examine_diagnostics import (
    BlockDiagnostic,
    diagnose_ivw,
    diagnose_ivw_refit_subject,
    diagnose_linear,
    dl_tau2,
)
from ._examine_geometry import geometry_projection
from ._examine_review import assign_review, contrast_rows, estimand_rows
from ._reducers_policy import (
    _beta_and_var,
    _design_matrix,
    _refuse_synthetic_unit_variance,
)
from .dataset import GroupDataset
from .errors import AdapterContractError, GroupConfigError
from .space import SampleLabelSpace

SupportedMethod = Literal[
    "meta:fe",
    "meta:re",
    "meta:fe_reg",
    "meta:re_reg",
    "ols:voxelwise",
]

_REG_METHODS = frozenset({"meta:fe_reg", "meta:re_reg", "ols:voxelwise"})
_RE_METHODS = frozenset({"meta:re", "meta:re_reg"})


@dataclass(frozen=True)
class ResidualEmbedding:
    """Low-rank residual geometry (feature-ID projection, not a scan artifact)."""

    coordinates: NDArray[np.float64]
    feature_ids: tuple[str, ...]
    captured_energy: float
    status: str


@dataclass(frozen=True)
class GroupExamination:
    """Result of :func:`examine_group`.

    Spatial maps stay on :class:`GroupDataset` objects. Subject, contrast,
    and estimand records live in tables.
    """

    dataset: GroupDataset
    group_maps: GroupDataset
    subject_data: pd.DataFrame
    contrast_data: pd.DataFrame
    estimand_data: pd.DataFrame
    embedding: ResidualEmbedding | None
    method: str
    formula: str | None
    control: ExaminationControl

    @property
    def review_queue(self) -> pd.DataFrame:
        flagged = self.subject_data[self.subject_data["review_status"] == "review"]
        return cast(
            pd.DataFrame,
            flagged.sort_values(
                ["review_priority", "subject"], ascending=[False, True]
            ).reset_index(drop=True),
        )


def _feature_ids(dataset: GroupDataset, contrast: str) -> tuple[str, ...]:
    if isinstance(dataset.space, SampleLabelSpace):
        labels = tuple(str(label) for label in dataset.space.labels)
    else:
        labels = tuple(f"feature-{i + 1}" for i in range(dataset.n_samples))
    return tuple(f"{contrast}|{label}" for label in labels)


def _resolve_estimands(
    predictor_names: Sequence[str],
    estimands: Sequence[str] | Mapping[str, Sequence[float]] | None,
) -> tuple[tuple[str, ...], NDArray[np.float64]]:
    names = list(predictor_names)
    if estimands is None:
        eye: NDArray[np.float64] = np.eye(len(names), dtype=np.float64)
        return tuple(names), eye
    if isinstance(estimands, Mapping):
        rows: list[NDArray[np.float64]] = []
        labels: list[str] = []
        for label, weights in estimands.items():
            vec = np.asarray(weights, dtype=np.float64).reshape(-1)
            if vec.size != len(names):
                raise GroupConfigError(
                    f"estimand {label!r} has {vec.size} weights; expected {len(names)}"
                )
            labels.append(str(label))
            rows.append(vec)
        return tuple(labels), np.vstack(rows)
    labels = [str(name) for name in estimands]
    rows = []
    lookup = {name: i for i, name in enumerate(names)}
    for label in labels:
        if label not in lookup:
            raise GroupConfigError(
                f"unknown estimand {label!r}; available: {', '.join(names)}"
            )
        e: NDArray[np.float64] = np.zeros(len(names), dtype=np.float64)
        e[lookup[label]] = 1.0
        rows.append(e)
    return tuple(labels), np.vstack(rows)


def _build_embedding(
    dataset: GroupDataset,
    contrast: str,
    diagnostic: BlockDiagnostic,
    control: ExaminationControl,
) -> ResidualEmbedding:
    resid = diagnostic.predictive_resid.copy()
    resid[~diagnostic.surprise_eligible] = 0.0
    cap = float(control.geometry.cap)
    resid = np.clip(resid, -cap, cap)
    if resid.size == 0 or not np.any(np.isfinite(resid)):
        return ResidualEmbedding(
            coordinates=np.zeros((0, 0), dtype=np.float64),
            feature_ids=(),
            captured_energy=float("nan"),
            status="unavailable",
        )
    e_subj = np.nan_to_num(resid, nan=0.0)
    n_feat = e_subj.shape[1]
    dim = min(int(control.geometry.rank), max(n_feat, 1))
    ids = _feature_ids(dataset, contrast)
    omega = geometry_projection(ids, dim, seed="digest")
    coords = e_subj @ omega if omega.size else np.zeros((e_subj.shape[0], 0))
    energy = float(np.sum(e_subj * e_subj))
    captured = float(np.sum(coords * coords)) / energy if energy > 0 else 0.0
    return ResidualEmbedding(
        coordinates=coords,
        feature_ids=ids,
        captured_energy=captured,
        status="available",
    )


def _exclude_missing_covariates(
    dataset: GroupDataset, formula: str, na_action: NaAction
) -> GroupDataset:
    if dataset.col_data is None or formula.replace(" ", "") in ("~1", "1"):
        return dataset
    frame = dataset.col_data.reset_index(drop=True)
    missing = frame.isna().any(axis=1).to_numpy()
    if not np.any(missing):
        return dataset
    if na_action == "fail":
        bad = [str(dataset.subjects[int(i)]) for i in np.flatnonzero(missing)]
        raise GroupConfigError("missing covariates for subjects: " + ", ".join(bad))
    keep: NDArray[np.bool_] = ~missing
    assays = {name: arr[:, keep, :] for name, arr in dataset.assays.items()}
    col = frame.loc[keep].reset_index(drop=True)
    subjects = [sid for sid, ok in zip(dataset.subjects, keep, strict=True) if ok]
    return GroupDataset(
        assays=assays,
        space=dataset.space,
        subjects=subjects,
        contrasts=dataset.contrasts,
        col_data=col,
        row_data=dataset.row_data,
        contrast_data=dataset.contrast_data,
        metadata=dict(dataset.metadata),
    )


def _diagnose_contrast(
    method: str,
    y: NDArray[np.float64],
    v: NDArray[np.float64] | None,
    *,
    X: NDArray[np.float64] | None,
    estimands: NDArray[np.float64] | None,
    estimand_names: tuple[str, ...],
    control: ExaminationControl,
) -> BlockDiagnostic:
    if method == "meta:fe":
        return diagnose_ivw(y, v, tolerance=control.tolerance, mode="exact")
    if method == "meta:re":
        tau2 = dl_tau2(y, v)
        return diagnose_ivw(
            y, v, tau2=tau2, tolerance=control.tolerance, mode="tau2_fixed_full"
        )
    if X is None or estimands is None:
        raise GroupConfigError(f"{method} requires a design matrix")
    tau2 = None
    mode: Literal["exact", "tau2_fixed_full"] = "exact"
    if method == "meta:re_reg":
        # DL tau2 from the FE residual of the same design, per sample.
        tau2 = np.full(y.shape[0], np.nan)
        for b in range(y.shape[0]):
            yb = y[b]
            vb = v[b] if v is not None else np.ones(y.shape[1])
            ok = (
                np.isfinite(yb)
                & np.isfinite(vb)
                & (vb > 0)
                & np.all(np.isfinite(X), axis=1)
            )
            if int(np.count_nonzero(ok)) < X.shape[1] + 1:
                continue
            Xok = X[ok]
            wok = 1.0 / np.maximum(vb[ok], 1e-12)
            yok = yb[ok]
            Xw = Xok * np.sqrt(wok)[:, np.newaxis]
            from ._reducers_kernels import _safe_inverse

            gram_inv = _safe_inverse(Xw.T @ Xw)
            if gram_inv is None:
                continue
            bhat = gram_inv @ (Xok.T @ (wok * yok))
            resid = yok - Xok @ bhat
            q_val = float(np.sum(wok * resid * resid))
            x_a_x = np.sum((Xok @ gram_inv) * Xok, axis=1)
            c_term = float(np.sum(wok) - np.sum((wok**2) * x_a_x))
            df_val = float(np.count_nonzero(ok) - X.shape[1])
            tau2[b] = max((q_val - df_val) / max(c_term, 1e-12), 0.0)
        mode = "tau2_fixed_full"
    return diagnose_linear(
        y,
        None if method == "ols:voxelwise" else v,
        X,
        estimands,
        estimand_names,
        tau2=tau2,
        tolerance=control.tolerance,
        mode=mode,
    )


def examine_group(
    dataset: GroupDataset,
    method: SupportedMethod | str = "meta:fe",
    *,
    formula: str | None = None,
    estimands: Sequence[str] | Mapping[str, Sequence[float]] | None = None,
    quality: Sequence[str] | None = None,
    retain: Sequence[str] | None = None,
    control: ExaminationControl | None = None,
    na_action: NaAction = "fail",
    X: NDArray[np.float64] | None = None,
) -> GroupExamination:
    """Examine subjects under an intended group model.

    Parameters
    ----------
    dataset
        Subject-level :class:`GroupDataset`. Must not already be a group
        reduce (``beta_g`` / singleton ``group`` subject).
    method
        Native reducer used for the *intended* model. The source ``dataset``
        is not rewritten.
    formula
        Patsy formula for regression methods. Defaults to ``~ 1``.
    estimands
        Design-column names or a mapping of name → weights. Defaults to every
        design column (or ``pooled_effect`` for intercept-only IVW).
    """
    if not isinstance(dataset, GroupDataset):
        raise TypeError("examine_group requires a GroupDataset")
    if "beta_g" in dataset.assays or dataset.subjects == ("group",):
        raise AdapterContractError(
            "examine_group branches from subject-level data; pass the "
            "unreduced GroupDataset, not a reduce() result"
        )
    cfg = control if control is not None else examination_control()
    method_s = str(method)
    if method_s not in {
        "meta:fe",
        "meta:re",
        "meta:fe_reg",
        "meta:re_reg",
        "ols:voxelwise",
    }:
        raise GroupConfigError(
            f"examine_group has no diagnostic kernel for {method_s!r}"
        )
    _refuse_synthetic_unit_variance(dataset, method=method_s)
    formula_s = "~ 1" if formula is None else str(formula)
    working = _exclude_missing_covariates(dataset, formula_s, na_action)
    retain_ids = tuple(str(x) for x in (retain or ()))
    unknown = set(retain_ids) - set(working.subjects)
    if unknown:
        raise GroupConfigError(
            "Unknown retained subjects: " + ", ".join(sorted(unknown))
        )

    extra_quality: dict[str, NDArray[np.float64]] = {}
    if quality:
        if working.col_data is None:
            raise GroupConfigError("quality columns require dataset.col_data")
        for name in quality:
            if name not in working.col_data.columns:
                raise GroupConfigError(f"quality column {name!r} is not in col_data")
            extra_quality[str(name)] = np.asarray(
                working.col_data[name], dtype=np.float64
            )

    beta, var = _beta_and_var(working, method=method_s)
    design: NDArray[np.float64] | None = None
    predictor_names: list[str] = ["pooled_effect"]
    if method_s in _REG_METHODS:
        design, predictor_names = _design_matrix(working, X=X, formula=formula_s)
    est_names, est_mat = (
        (("pooled_effect",), np.ones((1, 1), dtype=np.float64))
        if method_s not in _REG_METHODS
        else _resolve_estimands(predictor_names, estimands)
    )

    contrast_frames: list[pd.DataFrame] = []
    estimand_frames: list[pd.DataFrame] = []
    embeddings: list[ResidualEmbedding] = []
    map_assays: dict[str, NDArray[np.float64]] = {}
    subjects = [str(subject) for subject in working.subjects]

    for c_idx, contrast in enumerate(working.contrasts):
        y = np.asarray(beta[:, :, c_idx], dtype=np.float64)
        v = np.asarray(var[:, :, c_idx], dtype=np.float64)
        diagnostic = _diagnose_contrast(
            method_s,
            y,
            v,
            X=design,
            estimands=est_mat,
            estimand_names=est_names,
            control=cfg,
        )
        contrast_frames.append(
            contrast_rows(
                subjects=subjects,
                contrast=str(contrast),
                diagnostic=diagnostic,
                observed=y,
                control=cfg,
            )
        )
        estimand_frames.append(
            estimand_rows(
                subjects=subjects,
                contrast=str(contrast),
                diagnostic=diagnostic,
                control=cfg,
            )
        )
        embeddings.append(_build_embedding(working, str(contrast), diagnostic, cfg))
        for e, name in enumerate(diagnostic.estimand_names):
            key = f"effect:{name}" if diagnostic.full_effect.shape[0] else name
            map_assays.setdefault(
                key, np.full((working.n_samples, 1, working.n_contrasts), np.nan)
            )
            map_assays[key][:, 0, c_idx] = diagnostic.full_effect[e]
            se_key = f"se:{name}"
            map_assays.setdefault(
                se_key, np.full((working.n_samples, 1, working.n_contrasts), np.nan)
            )
            map_assays[se_key][:, 0, c_idx] = diagnostic.full_se[e]

    contrast_data = pd.concat(contrast_frames, ignore_index=True)
    estimand_data = pd.concat(estimand_frames, ignore_index=True)
    subject_data = assign_review(
        subjects,
        contrast_data,
        estimand_data,
        control=cfg,
        quality_values=extra_quality or None,
        retain=retain_ids,
    )

    if method_s in _RE_METHODS and cfg.exact_refit_n > 0:
        ranked = subject_data.sort_values(
            ["review_priority", "subject"], ascending=[False, True]
        )
        refit_ids = [str(sid) for sid in ranked["subject"].head(cfg.exact_refit_n)]
        extra_refit: list[pd.DataFrame] = []
        for sid in refit_ids:
            i = subjects.index(sid)
            for c_idx, contrast in enumerate(working.contrasts):
                y = np.asarray(beta[:, :, c_idx], dtype=np.float64)
                v = np.asarray(var[:, :, c_idx], dtype=np.float64)
                keep: NDArray[np.bool_] = np.ones(
                    working.n_subjects, dtype=bool
                )
                keep[i] = False
                pairs: list[tuple[str, float, float, int]] = []
                if method_s == "meta:re":
                    delta, eligible = diagnose_ivw_refit_subject(
                        y, v, i, tolerance=cfg.tolerance
                    )
                    pairs.append(
                        (
                            "pooled_effect",
                            float(np.sqrt(np.mean(delta[eligible] ** 2)))
                            if np.any(eligible)
                            else float("nan"),
                            float(np.max(np.abs(delta[eligible])))
                            if np.any(eligible)
                            else float("nan"),
                            int(np.count_nonzero(eligible)),
                        )
                    )
                else:
                    full = _diagnose_contrast(
                        method_s,
                        y,
                        v,
                        X=design,
                        estimands=est_mat,
                        estimand_names=est_names,
                        control=cfg,
                    )
                    reduced_diag = _diagnose_contrast(
                        method_s,
                        y[:, keep],
                        v[:, keep],
                        X=None if design is None else design[keep],
                        estimands=est_mat,
                        estimand_names=est_names,
                        control=cfg,
                    )
                    for e, name in enumerate(est_names):
                        ok = np.isfinite(full.full_stat[e]) & np.isfinite(
                            reduced_diag.full_stat[e]
                        )
                        delta = full.full_stat[e] - reduced_diag.full_stat[e]
                        pairs.append(
                            (
                                name,
                                float(np.sqrt(np.mean(delta[ok] ** 2)))
                                if np.any(ok)
                                else float("nan"),
                                float(np.max(np.abs(delta[ok])))
                                if np.any(ok)
                                else float("nan"),
                                int(np.count_nonzero(ok)),
                            )
                        )
                for name, energy, max_abs, n_ok in pairs:
                    extra_refit.append(
                        pd.DataFrame(
                            [
                                {
                                    "subject": sid,
                                    "contrast": str(contrast),
                                    "estimand": name,
                                    "mode": "tau2_refit_exact",
                                    "ranking_stage": "exact_refit",
                                    "influence_energy": energy,
                                    "max_abs_delta_stat": max_abs,
                                    "eligible_n": n_ok,
                                    "status": "available" if n_ok else "nonestimable",
                                    "stability": np.nan,
                                    "stable": False,
                                }
                            ]
                        )
                    )
        if extra_refit:
            estimand_data = pd.concat([estimand_data, *extra_refit], ignore_index=True)

    group_maps = GroupDataset(
        assays=map_assays
        or {
            "effect:pooled_effect": np.full(
                (working.n_samples, 1, working.n_contrasts), np.nan
            )
        },
        space=working.space,
        subjects=["group"],
        contrasts=working.contrasts,
        metadata={"source": "examine_group", "method": method_s},
    )
    return GroupExamination(
        dataset=dataset,
        group_maps=group_maps,
        subject_data=subject_data,
        contrast_data=contrast_data,
        estimand_data=estimand_data,
        embedding=embeddings[0] if embeddings else None,
        method=method_s,
        formula=formula_s if method_s in _REG_METHODS else None,
        control=cfg,
    )


def examine_reduced(
    reduced: GroupDataset,
    source: GroupDataset,
    *,
    estimands: Sequence[str] | Mapping[str, Sequence[float]] | None = None,
    quality: Sequence[str] | None = None,
    retain: Sequence[str] | None = None,
    control: ExaminationControl | None = None,
    na_action: NaAction = "fail",
) -> GroupExamination:
    """Examine ``source`` using the reducer recorded on ``reduced``.

    Convenience for callers who already ran :func:`fmrimod.group.reduce`.
    Method and formula are inherited from the reduce metadata.
    """
    method = str(
        reduced.metadata.get("reduce_method") or reduced.metadata.get("method") or ""
    )
    if not method:
        raise GroupConfigError("reduced dataset has no recorded reduce method")
    formula = reduced.metadata.get("formula")
    return examine_group(
        source,
        method=method,
        formula=formula if isinstance(formula, str) else None,
        estimands=estimands,
        quality=quality,
        retain=retain,
        control=control,
        na_action=na_action,
    )
