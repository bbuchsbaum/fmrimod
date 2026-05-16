"""Group-level t-test wrapper with parity-oriented interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional, Sequence, cast

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy import stats as sp_stats

from ..dataset.group_data import GroupData
from .meta import _coerce_group_data, fmri_meta

TTestEngine = Literal["auto", "meta", "classic", "welch"]


@dataclass
class FmriTTestResult:
    """Result container for one-sample group t-tests."""

    estimate: NDArray[np.float64]
    se: NDArray[np.float64]
    statistic: NDArray[np.float64]
    p: NDArray[np.float64]
    feature_names: list[str]
    engine: str
    meta_result: Optional[object] = None


def _extract_csv_feature_effects(
    data: GroupData,
    *,
    effect_name: str = "beta",
) -> tuple[list[str], list[NDArray[np.float64]]]:
    if data.format != "csv":
        raise NotImplementedError(
            "fmri_ttest currently supports GroupData format='csv' only"
        )

    payload = data.data
    df = cast(Any, payload.get("data"))
    effect_cols = cast(Any, payload.get("effect_cols") or {})
    subject_col = payload.get("subject_col")
    roi_col = payload.get("roi_col")
    contrast_col = payload.get("contrast_col")
    subjects = list(data.subjects)

    effect_col = effect_cols.get(effect_name)
    if effect_col is None:
        raise ValueError(f"CSV GroupData is missing effect column mapping for '{effect_name}'")

    group_cols = [c for c in (roi_col, contrast_col) if c is not None]
    if group_cols:
        grouped = list(df.groupby(group_cols, sort=False))
    else:
        grouped = [("__all__", df)]

    feature_names: list[str] = []
    vectors: list[NDArray[np.float64]] = []
    for key, block in grouped:
        key_block = block.copy()
        if key_block[subject_col].duplicated().any():
            raise ValueError(
                "Each feature must have at most one row per subject; found duplicates"
            )
        by_subj = key_block.set_index(subject_col)
        missing = [s for s in subjects if s not in by_subj.index]
        if missing:
            raise ValueError(
                "Missing feature rows for subjects: " + ", ".join(str(s) for s in missing)
            )
        aligned = by_subj.loc[subjects]
        y = np.asarray(aligned[effect_col], dtype=np.float64)
        if not np.all(np.isfinite(y)):
            raise ValueError("Effect values must be finite")

        if group_cols:
            if isinstance(key, tuple):
                pieces = [f"{col}={val}" for col, val in zip(group_cols, key)]
                fname = "|".join(pieces)
            else:
                fname = f"{group_cols[0]}={key}"
        else:
            fname = "__all__"

        feature_names.append(fname)
        vectors.append(y)

    return feature_names, vectors


def _classic_one_sample(
    y: NDArray[np.float64],
) -> tuple[float, float, float, float]:
    n = y.shape[0]
    if n < 2:
        raise ValueError("Classic t-test requires at least 2 subjects")
    estimate = float(np.mean(y))
    sd = float(np.std(y, ddof=1))
    se = sd / np.sqrt(n)
    if se == 0.0:
        tval = 0.0
        pval = 1.0
    else:
        tval = estimate / se
        pval = float(2.0 * sp_stats.t.sf(np.abs(tval), df=n - 1))
    return estimate, se, tval, pval


def fmri_ttest(
    data: "GroupData | pd.DataFrame",
    *,
    engine: TTestEngine = "auto",
    formula: str = "~ 1",
    method: str = "pm",
    weights: str = "ivw",
    weights_custom: Optional[NDArray[np.float64]] = None,
    effect_cols: Optional[Mapping[str, str] | Sequence[str]] = None,
) -> FmriTTestResult:
    """Run a parity-oriented one-sample group t-test.

    ``data`` may be a frozen :class:`GroupData` or a pandas DataFrame
    (validated and converted to a frozen ``GroupData`` at entry via the
    typed :func:`group_data_from_csv`; pass its ``effect_cols`` schema).

    Current slice:
    - CSV-backed GroupData only
    - one-sample inference around zero (``formula='~ 1'``)
    """
    data = _coerce_group_data(data, effect_cols)
    if formula.replace(" ", "") not in ("~1", "1"):
        raise NotImplementedError("fmri_ttest currently supports formula='~ 1' only")

    if engine == "auto":
        effect_cols = cast(Any, data.data.get("effect_cols") or {})
        if "se" in effect_cols or "var" in effect_cols:
            resolved_engine = "meta"
        else:
            resolved_engine = "classic"
    else:
        resolved_engine = engine

    if resolved_engine == "meta":
        if method not in ("fe", "pm", "dl", "reml"):
            raise ValueError(
                f"method must be one of fe/pm/dl/reml, got {method!r}"
            )
        if weights not in ("ivw", "equal", "custom"):
            raise ValueError(
                f"weights must be one of ivw/equal/custom, got {weights!r}"
            )
        meta_result = fmri_meta(
            data,
            formula=formula,
            method=cast('Literal["fe", "pm", "dl", "reml"]', method),
            weights=cast('Literal["ivw", "equal", "custom"]', weights),
            weights_custom=weights_custom,
        )
        if meta_result.coefficients.shape[1] != 1:
            raise NotImplementedError(
                "fmri_ttest meta engine currently expects an intercept-only predictor"
            )
        return FmriTTestResult(
            estimate=meta_result.coefficients[:, 0],
            se=meta_result.se[:, 0],
            statistic=meta_result.z[:, 0],
            p=meta_result.p[:, 0],
            feature_names=meta_result.feature_names,
            engine="meta",
            meta_result=meta_result,
        )

    if resolved_engine in ("classic", "welch"):
        if resolved_engine == "welch":
            raise NotImplementedError(
                "Welch engine is not implemented yet in this parity slice"
            )
        feature_names, vectors = _extract_csv_feature_effects(data, effect_name="beta")
        estimates = np.zeros(len(vectors), dtype=np.float64)
        ses = np.zeros(len(vectors), dtype=np.float64)
        stats = np.zeros(len(vectors), dtype=np.float64)
        pvals = np.zeros(len(vectors), dtype=np.float64)
        for i, y in enumerate(vectors):
            est, se, tval, pval = _classic_one_sample(y)
            estimates[i] = est
            ses[i] = se
            stats[i] = tval
            pvals[i] = pval
        return FmriTTestResult(
            estimate=estimates,
            se=ses,
            statistic=stats,
            p=pvals,
            feature_names=feature_names,
            engine="classic",
            meta_result=None,
        )

    raise ValueError("engine must be one of: auto, meta, classic, welch")

