"""BIDS parametric first-level to native group follow-through scenario.

This workflow is intentionally small, but it crosses the user-facing seam that
was missing from the ergonomics inventory: a BIDS Stats Model JSON creates a
typed first-level model, BIDS-authored task contrasts lower semantically, an
authored parametric slope contrast uses condition-level sugar, and the resulting
subject maps feed the native group-model API without formula strings.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

import fmrimod as fm
from fmrimod.bids.stats_model import translate_run_node
from fmrimod.contrast import group_dataset_from_contrasts, modulator
from fmrimod.group import group_model, ols_voxelwise
from fmrimod.sampling import SamplingFrame

Array = NDArray[np.float64]

SCHEMA_VERSION = "bids-parametric-group-followthrough/v1"
TR = 2.0
N_SCANS = 120
N_TRIALS = 32
DEFAULT_SUBJECTS = 9
DEFAULT_VOXELS = 24
REPORT_NAME = "followthrough_report.json"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def stats_model() -> dict[str, Any]:
    """Return the BIDS Stats Model fixture used by the scenario."""

    return {
        "Name": "bids_parametric_group_followthrough",
        "BIDSModelVersion": "1.0.0",
        "Nodes": [
            {
                "Level": "run",
                "Transformations": [
                    {"Name": "Factor", "Input": ["trial_type"]},
                    {"Name": "Scale", "Input": ["rt"], "Output": "rt_z"},
                    {
                        "Name": "Product",
                        "Input": [
                            "trial_type.word",
                            "trial_type.pseudoword",
                            "rt_z",
                        ],
                        "Output": "trial_type_rt_z",
                    },
                    {
                        "Name": "Convolve",
                        "Input": [
                            "trial_type.word",
                            "trial_type.pseudoword",
                        ],
                        "Model": "spm",
                    },
                ],
                "Model": {
                    "X": ["trial_type.word", "trial_type.pseudoword", "rt_z", 1]
                },
                "Contrasts": [
                    {
                        "Name": "word_gt_pseudoword",
                        "ConditionList": [
                            "trial_type.word",
                            "trial_type.pseudoword",
                        ],
                        "Weights": [1, -1],
                        "Test": "t",
                    },
                    {
                        "Name": "task_omnibus",
                        "ConditionList": [
                            "trial_type.word",
                            "trial_type.pseudoword",
                        ],
                        "Weights": [[1, 0], [0, 1]],
                        "Test": "F",
                    },
                ],
            }
        ],
    }


def make_events(seed: int = 20260514) -> pd.DataFrame:
    """Create a balanced event table with an RT modulator."""

    rng = np.random.default_rng(seed)
    labels = np.array(["word", "pseudoword"] * (N_TRIALS // 2), dtype=object)
    onsets = np.linspace(8.0, N_SCANS * TR - 20.0, N_TRIALS, dtype=np.float64)
    rt = np.empty(N_TRIALS, dtype=np.float64)
    rt[labels == "word"] = rng.normal(0.78, 0.08, int(np.sum(labels == "word")))
    rt[labels == "pseudoword"] = rng.normal(
        0.92,
        0.09,
        int(np.sum(labels == "pseudoword")),
    )
    return pd.DataFrame(
        {
            "onset": onsets,
            "duration": np.zeros(N_TRIALS, dtype=np.float64),
            "trial_type": labels,
            "rt": rt,
            "run": 1,
        }
    )


def translate_model(events: pd.DataFrame):
    """Translate the BIDS node through fmrimod's typed BIDS seam."""

    return translate_run_node(
        stats_model(),
        events=events,
        sampling_frame=SamplingFrame(blocklens=[N_SCANS], tr=TR),
    )


def _column_lookup(fit: Any) -> dict[tuple[str | None, str | None], int]:
    lookup: dict[tuple[str | None, str | None], int] = {}
    for column in fit.design_columns():
        lookup[(column.term, column.level)] = int(column.index)
    return lookup


def _subject_bold(
    translated: Any,
    events: pd.DataFrame,
    *,
    behavior: float,
    subject_index: int,
    n_voxels: int,
) -> Array:
    """Generate one subject from the translated design and known beta truth."""

    zero = np.zeros((N_SCANS, 1), dtype=np.float64)
    fit = translated.fit(fm.fmri_dataset(zero, tr=TR, events=events))
    design = np.asarray(fit.model.design_matrix_array(run=0), dtype=np.float64)
    lookup = _column_lookup(fit)

    betas = np.zeros((design.shape[1], n_voxels), dtype=np.float64)
    ramp = np.linspace(0.85, 1.15, n_voxels, dtype=np.float64)
    word_main = lookup[("trial_type", "word")]
    pseudo_main = lookup[("trial_type", "pseudoword")]
    word_slope = lookup[("trial_type:rt_z", "word")]
    pseudo_slope = lookup[("trial_type:rt_z", "pseudoword")]
    intercept = design.shape[1] - 1

    betas[word_main] = 0.45 * ramp
    betas[pseudo_main] = 0.10 * ramp
    betas[word_slope] = (0.55 + 0.28 * behavior) * ramp
    betas[pseudo_slope] = (-0.05 - 0.12 * behavior) * ramp
    betas[intercept] = 95.0 + 0.2 * subject_index

    rng = np.random.default_rng(20260514 + subject_index)
    noise = rng.normal(scale=2e-4, size=(N_SCANS, n_voxels))
    return (design @ betas + noise).astype(np.float64)


def fit_subject(
    translated: Any,
    events: pd.DataFrame,
    bold: Array,
) -> dict[str, Any]:
    """Fit one subject using BIDS and semantic parametric contrast sugar."""

    dataset = fm.fmri_dataset(bold, tr=TR, events=events)
    fit = translated.fit(dataset)
    main = translated.contrast(fit, "word_gt_pseudoword")
    omnibus = translated.contrast(fit, "task_omnibus")
    rt = modulator("rt_z").within("trial_type")
    slope_spec = rt.slope("word") - rt.slope("pseudoword")
    slope = fit.contrast(
        slope_spec,
        name="word_rt_slope_gt_pseudoword_rt_slope",
    )
    return {
        "fit": fit,
        "main": main,
        "omnibus": omnibus,
        "slope": slope,
    }


def group_followthrough(contrasts: dict[str, Any], covariates: pd.DataFrame) -> Any:
    """Run the native group model on first-level semantic slope estimates."""

    return ols_voxelwise(
        group_dataset_from_contrasts(contrasts, covariates=covariates),
        model=group_model("behavior"),
    )


def _contrast_kind(result: Any) -> str:
    intent = getattr(result, "intent", {}) or {}
    if isinstance(intent, dict):
        return str(intent.get("kind", "missing"))
    return str(getattr(intent, "kind", "missing"))


def run_benchmark(
    *,
    n_subjects: int = DEFAULT_SUBJECTS,
    n_voxels: int = DEFAULT_VOXELS,
    seed: int = 20260514,
) -> dict[str, Any]:
    """Run the complete BIDS -> first-level -> group scenario."""

    timings: dict[str, float] = {}
    load_start = time.perf_counter()
    events = make_events(seed)
    translated = translate_model(events)
    timings["translate_bids_model_seconds"] = time.perf_counter() - load_start

    behavior = np.linspace(-1.0, 1.0, n_subjects, dtype=np.float64)
    subjects = [f"sub-{i + 1:02d}" for i in range(n_subjects)]
    subject_summaries: list[dict[str, Any]] = []
    slope_contrasts: dict[str, Any] = {}
    first_level_start = time.perf_counter()
    for subject_index, (subject, value) in enumerate(zip(subjects, behavior)):
        bold = _subject_bold(
            translated,
            events,
            behavior=float(value),
            subject_index=subject_index,
            n_voxels=n_voxels,
        )
        outputs = fit_subject(translated, events, bold)
        slope = outputs["slope"]
        slope_contrasts[subject] = slope
        subject_summaries.append(
            {
                "subject": subject,
                "behavior": float(value),
                "main_kind": _contrast_kind(outputs["main"]),
                "omnibus_kind": _contrast_kind(outputs["omnibus"]),
                "slope_kind": _contrast_kind(slope),
                "slope_touched_columns": list(slope.touched_columns),
                "slope_estimate_median": float(np.median(slope.estimate)),
            }
        )
    timings["first_level_subjects_seconds"] = time.perf_counter() - first_level_start

    group_start = time.perf_counter()
    covariates = pd.DataFrame({"subject": subjects, "behavior": behavior})
    group_result = group_followthrough(slope_contrasts, covariates)
    timings["native_group_model_seconds"] = time.perf_counter() - group_start

    behavior_coef = np.asarray(group_result.assay("coef:behavior"))[:, 0, 0]
    behavior_t = np.asarray(group_result.assay("t_coef:behavior"))[:, 0, 0]
    slope_kinds = {item["slope_kind"] for item in subject_summaries}
    main_kinds = {item["main_kind"] for item in subject_summaries}
    group_positive = bool(np.nanmedian(behavior_coef) > 0.2)
    group_detected = bool(np.nanmedian(behavior_t) > 5.0)
    status = (
        "pass"
        if main_kinds == {"semantic_linear_contrast"}
        and slope_kinds == {"semantic_contrast"}
        and group_positive
        and group_detected
        else "fail"
    )
    return _json_safe(
        {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "caveats": [],
            "timings": {
                "status": "recorded",
                "seconds": float(sum(timings.values())),
                "stages": timings,
            },
            "bids": {
                "model_name": stats_model()["Name"],
                "translated_columns": translated.column_names,
                "caveats": list(translated.caveats),
            },
            "ergonomics": {
                "ux_status": "typed_contrast_full_group",
                "fmrimod_path": (
                    "BIDS JSON -> translate_run_node -> fmri_dataset -> "
                    "StatsModelContrast.apply -> modulator('rt_z').within('trial_type') "
                    "-> group_model('behavior') -> ols_voxelwise"
                ),
                "raw_vector_user_code": False,
                "typed_contrast_authoring_status": "full",
                "e2e_ux_status": "full",
                "parametric_contrast_callsite": (
                    "modulator('rt_z').within('trial_type').slope('word') - "
                    "modulator('rt_z').within('trial_type').slope('pseudoword')"
                ),
                "ux_blockers": [],
            },
            "subject_summary": subject_summaries,
            "group": {
                "model": "GroupLinearModel",
                "assays": sorted(group_result.assay_names()),
                "behavior_coef_median": float(np.nanmedian(behavior_coef)),
                "behavior_t_median": float(np.nanmedian(behavior_t)),
                "behavior_coef_min": float(np.nanmin(behavior_coef)),
            },
            "checks": {
                "bids_main_semantic": main_kinds == {"semantic_linear_contrast"},
                "parametric_slope_semantic": slope_kinds == {"semantic_contrast"},
                "group_behavior_positive": group_positive,
                "group_behavior_detected": group_detected,
            },
        }
    )


def write_report(report: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / REPORT_NAME
    markdown_path = out_dir / "REPORT.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(
        "\n".join(
            [
                "# BIDS Parametric Group Follow-through",
                "",
                f"Status: `{report['status']}`",
                "",
                "## Ergonomics",
                "",
                str(report["ergonomics"]["fmrimod_path"]),
                "",
                (
                    "UX status: "
                    f"`{report['ergonomics']['ux_status']}`; "
                    "E2E UX status: "
                    f"`{report['ergonomics']['e2e_ux_status']}`."
                ),
                "",
                "## Group Check",
                "",
                (
                    "Median behavior coefficient: "
                    f"`{report['group']['behavior_coef_median']:.6g}`; "
                    "median t: "
                    f"`{report['group']['behavior_t_median']:.6g}`."
                ),
            ]
        )
        + "\n"
    )
    return json_path, markdown_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "reports",
    )
    parser.add_argument("--n-subjects", type=int, default=DEFAULT_SUBJECTS)
    parser.add_argument("--n-voxels", type=int, default=DEFAULT_VOXELS)
    args = parser.parse_args(argv)
    report = run_benchmark(n_subjects=args.n_subjects, n_voxels=args.n_voxels)
    write_report(report, args.out_dir)
    if report["status"] == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
