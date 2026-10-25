"""Subject / contrast / estimand tables and the review queue."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ._examine_control import ExaminationControl, QualityRule
from ._examine_diagnostics import BlockDiagnostic


def _percentile(values: NDArray[np.float64]) -> NDArray[np.float64]:
    out = np.full(values.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        return out
    ranks = np.argsort(np.argsort(values[finite]))
    n = int(np.count_nonzero(finite))
    if n == 1:
        out[finite] = 0.5
    else:
        out[finite] = ranks / (n - 1)
    return out


def _safe_max(values: NDArray[np.float64]) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.max(finite))


def _rms(values: NDArray[np.float64]) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(finite * finite)))


def _fold_stability(
    values: NDArray[np.float64],
    eligible: NDArray[np.bool_],
    n_split: int,
    threshold: float,
) -> float:
    """Fraction of feature-index folds whose RMS exceeds ``threshold``."""
    if n_split < 2:
        energy = _rms(values[eligible])
        return 1.0 if np.isfinite(energy) else float("nan")
    triggers: list[float] = []
    n_sample = values.size
    folds: NDArray[np.intp] = np.arange(n_sample, dtype=np.intp) % n_split
    for r in range(n_split):
        mask = eligible & (folds == r)
        if not np.any(mask):
            continue
        energy = _rms(values[mask])
        if np.isfinite(energy):
            triggers.append(float(energy >= threshold))
    if not triggers:
        return float("nan")
    return float(np.mean(triggers))


def _zero_intercept_gain(
    expected: NDArray[np.float64],
    observed: NDArray[np.float64],
    weight: NDArray[np.float64],
    degeneracy: float,
) -> tuple[float, str]:
    ok = (
        np.isfinite(expected)
        & np.isfinite(observed)
        & np.isfinite(weight)
        & (weight > 0)
    )
    if not np.any(ok):
        return float("nan"), "insufficient_samples"
    w = weight[ok]
    x = expected[ok]
    y = observed[ok]
    sxx = float(np.sum(w * x * x))
    sxy = float(np.sum(w * x * y))
    if sxx <= degeneracy:
        return float("nan"), "degenerate_expected"
    return sxy / sxx, "available"


def contrast_rows(
    *,
    subjects: Sequence[str],
    contrast: str,
    diagnostic: BlockDiagnostic,
    observed: NDArray[np.float64],
    control: ExaminationControl,
) -> pd.DataFrame:
    """Build one-contrast subject rows from a diagnostic block.

    ``observed`` is sample × subject (the raw beta slice).
    """
    n_subject, n_sample = diagnostic.predictive_resid.shape
    residual_thr = float(control.surprise.residual_threshold)
    n_split = int(control.geometry.stability_replicates)
    rows: list[dict[str, object]] = []
    for i, subject in enumerate(subjects):
        eligible = diagnostic.surprise_eligible[i]
        resid = diagnostic.predictive_resid[i]
        count = int(np.count_nonzero(eligible))
        energy = _rms(resid[eligible]) if count else float("nan")
        tail = (
            float(np.mean(np.abs(resid[eligible]) >= residual_thr))
            if count
            else float("nan")
        )
        stability = _fold_stability(
            resid, eligible, n_split, control.surprise.energy_threshold
        )
        gain, gain_status = _zero_intercept_gain(
            diagnostic.expected[i],
            observed[:, i],
            diagnostic.predictive_weight[i],
            control.tolerance.degeneracy,
        )
        finite_beta = (
            float(np.mean(np.isfinite(observed[:, i]))) if n_sample else float("nan")
        )
        rows.append(
            {
                "subject": subject,
                "contrast": contrast,
                "coverage_fraction": finite_beta,
                "surprise_energy": energy,
                "tail_extent": tail,
                "surprise_eligible_n": count,
                "surprise_status": "available" if count else "insufficient_samples",
                "surprise_stability": stability,
                "zero_intercept_gain": gain,
                "gain_status": gain_status,
                "correlation_status": gain_status,
            }
        )
    return pd.DataFrame(rows)


def estimand_rows(
    *,
    subjects: Sequence[str],
    contrast: str,
    diagnostic: BlockDiagnostic,
    control: ExaminationControl,
    mode: str | None = None,
) -> pd.DataFrame:
    n_subject, n_estimand, n_sample = diagnostic.delta_stat.shape
    n_split = int(control.geometry.stability_replicates)
    label = mode if mode is not None else diagnostic.mode
    rows: list[dict[str, object]] = []
    for e, estimand in enumerate(diagnostic.estimand_names):
        for i, subject in enumerate(subjects):
            eligible = diagnostic.influence_eligible[i, e]
            delta = diagnostic.delta_stat[i, e]
            count = int(np.count_nonzero(eligible))
            energy = _rms(delta[eligible]) if count else float("nan")
            max_abs = float(np.max(np.abs(delta[eligible]))) if count else float("nan")
            stability = _fold_stability(
                delta, eligible, n_split, control.influence.energy_threshold
            )
            rows.append(
                {
                    "subject": subject,
                    "contrast": contrast,
                    "estimand": estimand,
                    "mode": label,
                    "ranking_stage": (
                        "screening" if label != "tau2_refit_exact" else "exact_refit"
                    ),
                    "influence_energy": energy,
                    "max_abs_delta_stat": max_abs,
                    "eligible_n": count,
                    "status": "available" if count else "nonestimable",
                    "stability": stability,
                    "stable": bool(
                        np.isfinite(stability) and stability >= control.min_stability
                    ),
                }
            )
    return pd.DataFrame(rows)


def assign_review(
    subject_ids: Sequence[str],
    contrast_data: pd.DataFrame,
    estimand_data: pd.DataFrame,
    *,
    control: ExaminationControl,
    quality_values: Mapping[str, NDArray[np.float64]] | None,
    retain: Sequence[str],
) -> pd.DataFrame:
    """Build ``subject_data`` and apply absolute review criteria."""
    subjects = list(subject_ids)
    coverage = []
    surprise = []
    influence = []
    for subject in subjects:
        crows = contrast_data[contrast_data["subject"] == subject]
        erows = estimand_data[
            (estimand_data["subject"] == subject)
            & (estimand_data["mode"] != "tau2_refit_exact")
        ]
        coverage.append(
            float(crows["coverage_fraction"].max()) if len(crows) else float("nan")
        )
        surprise.append(
            _safe_max(np.asarray(crows["surprise_energy"], dtype=np.float64))
            if len(crows)
            else float("nan")
        )
        influence.append(
            _safe_max(np.asarray(erows["influence_energy"], dtype=np.float64))
            if len(erows)
            else float("nan")
        )
    coverage_arr = np.asarray(coverage, dtype=np.float64)
    surprise_arr = np.asarray(surprise, dtype=np.float64)
    influence_arr = np.asarray(influence, dtype=np.float64)
    quality_score = 1.0 - coverage_arr
    quality_pct = _percentile(quality_score)
    surprise_pct = _percentile(surprise_arr)
    influence_pct = _percentile(influence_arr)
    extra_quality: dict[str, NDArray[np.float64]] = {}
    if quality_values:
        extra_quality = {
            str(k): np.asarray(v, dtype=np.float64) for k, v in quality_values.items()
        }

    review_priority = np.nanmax(
        np.vstack([quality_pct, surprise_pct, influence_pct]), axis=0
    )
    status = ["none"] * len(subjects)
    source: list[str | float] = [np.nan] * len(subjects)
    reason = ["No absolute review criterion met."] * len(subjects)

    for i, subject in enumerate(subjects):
        crows = contrast_data[contrast_data["subject"] == subject]
        erows = estimand_data[
            (estimand_data["subject"] == subject)
            & (estimand_data["mode"] != "tau2_refit_exact")
        ]
        surprise_trigger = (
            np.isfinite(crows["surprise_energy"].to_numpy(dtype=float))
            & (
                crows["surprise_energy"].to_numpy(dtype=float)
                >= control.surprise.energy_threshold
            )
            & (
                crows["tail_extent"].to_numpy(dtype=float)
                >= control.surprise.tail_threshold
            )
            & (
                crows["surprise_stability"].to_numpy(dtype=float)
                >= control.min_stability
            )
        )
        influence_trigger = (
            np.isfinite(erows["influence_energy"].to_numpy(dtype=float))
            & (
                erows["influence_energy"].to_numpy(dtype=float)
                >= control.influence.energy_threshold
            )
            & (
                erows["max_abs_delta_stat"].to_numpy(dtype=float)
                >= control.influence.max_abs_threshold
            )
            & (erows["stability"].to_numpy(dtype=float) >= control.min_stability)
        )
        quality_trigger = False
        quality_reason = None
        for name, spec in control.quality.items():
            rule = spec if isinstance(spec, QualityRule) else QualityRule(**spec)
            if name == "coverage_fraction":
                value = coverage_arr[i]
            elif name in extra_quality:
                value = float(extra_quality[name][i])
            else:
                value = float("nan")
            if not np.isfinite(value):
                continue
            hit = (
                value >= rule.threshold
                if rule.direction == "high"
                else value <= rule.threshold
            )
            if hit:
                quality_trigger = True
                quality_reason = f"Data-validity criterion met for {name}."
                break
        available = (
            bool(np.any(crows["surprise_status"].to_numpy() == "available"))
            or bool(np.any(erows["status"].to_numpy() == "available"))
            or bool(np.isfinite(coverage_arr[i]))
        )
        if not available:
            status[i] = "insufficient"
            reason[i] = "Insufficient eligible data for examination."
            continue
        triggers = {
            "quality": quality_trigger,
            "surprise": bool(np.any(surprise_trigger)),
            "influence": bool(np.any(influence_trigger)),
        }
        if not any(triggers.values()):
            continue
        scores = {
            "quality": quality_pct[i],
            "surprise": surprise_pct[i],
            "influence": influence_pct[i],
        }
        chosen = max(
            (key for key, on in triggers.items() if on),
            key=lambda key: scores[key] if np.isfinite(scores[key]) else -np.inf,
        )
        status[i] = "review"
        source[i] = chosen
        if chosen == "quality":
            reason[i] = quality_reason or "Data-validity criterion met."
        elif chosen == "surprise":
            row = crows.loc[crows.index[np.flatnonzero(surprise_trigger)[0]]]
            gain = row["zero_intercept_gain"]
            if np.isfinite(gain) and float(gain) < 0:
                reason[
                    i
                ] = f"High unexpectedness with negative map gain in contrast {row['contrast']}."
            else:
                reason[i] = f"High model surprise in contrast {row['contrast']}."
        else:
            row = erows.loc[erows.index[np.flatnonzero(influence_trigger)[0]]]
            reason[i] = (
                f"High group-statistic influence for estimand {row['estimand']} "
                f"in contrast {row['contrast']}."
            )

    out = pd.DataFrame(
        {
            "subject": subjects,
            "coverage_fraction": coverage_arr,
            "surprise_score": surprise_arr,
            "influence_score": influence_arr,
            "quality_percentile": quality_pct,
            "surprise_percentile": surprise_pct,
            "influence_percentile": influence_pct,
            "review_priority": review_priority,
            "review_status": status,
            "review_source": source,
            "review_reason": reason,
        }
    )
    order = np.argsort(-out["review_priority"].to_numpy(dtype=float), kind="stable")
    automatic = [subjects[int(j)] for j in order[: control.retain_n]]
    flagged = set(out.loc[out["review_status"] == "review", "subject"])
    keep = set(retain) | flagged | set(automatic)
    out["retained"] = [sid in keep for sid in subjects]
    return out
