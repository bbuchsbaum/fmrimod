"""Parity-adjacent showcases for fmrimod-specific capabilities."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fmrimod.ar import estimate_ar
from fmrimod.glm.solver import fast_lm_matrix, fast_preproject
from fmrimod.lowrank import LowRankConfig, fit_sketched
from fmrimod.robust import huber_weights, mad_scale
from fmrimod.single import estimate_single_trial

Array = NDArray[np.float64]


@dataclass(frozen=True)
class ShowcaseRow:
    """One executable showcase result."""

    case_id: str
    capability: str
    status: str
    metric: str
    value: float
    threshold: float | None
    details: dict[str, Any]


@dataclass(frozen=True)
class ShowcaseProofScorecard:
    """JSON-ready evidence that Tier D is a public fmrimod artifact."""

    public_seam: bool
    fmrimod_path: str
    reference_path: str
    typed_objects: tuple[str, ...]
    public_rows: tuple[str, ...]
    semantic_survival: dict[str, Any]
    win_axes: dict[str, str]


def _elapsed(func):
    start = time.perf_counter()
    value = func()
    return value, time.perf_counter() - start


def _simulate_ar(phi: Array, n: int, n_voxels: int, rng: np.random.Generator) -> Array:
    noise = rng.normal(size=(n, n_voxels))
    y = np.zeros_like(noise)
    for t in range(n):
        y[t] = noise[t]
        for lag, coef in enumerate(phi, start=1):
            if t >= lag:
                y[t] += coef * y[t - lag]
    return y


def run_ar_robust_showcase(seed: int = 2026) -> ShowcaseRow:
    rng = np.random.default_rng(seed)
    true_phi = np.array([0.45, -0.22])
    residuals = _simulate_ar(true_phi, n=220, n_voxels=64, rng=rng)
    phi1 = estimate_ar(residuals, order=1, voxelwise=False)
    phi2 = estimate_ar(residuals, order=2, voxelwise=False)

    x = np.linspace(-1.0, 1.0, 80)
    X = np.column_stack([np.ones_like(x), x])
    y = X @ np.array([0.5, 1.25]) + rng.normal(scale=0.05, size=x.size)
    y[40] += 5.0
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    robust_residuals = (y - X @ beta)[:, np.newaxis]
    scale = mad_scale(robust_residuals, axis=0)
    weights = huber_weights(robust_residuals, scale)
    outlier_weight = float(weights[40, 0])
    typical_weight = float(np.median(np.delete(weights[:, 0], 40)))

    ar2_error = float(np.linalg.norm(phi2 - true_phi))
    status = "pass" if ar2_error < 0.12 and outlier_weight < 0.25 else "fail"
    return ShowcaseRow(
        case_id="tier_d_ar2_robust",
        capability="AR(2) estimation plus Huber downweighting",
        status=status,
        metric="ar2_error",
        value=ar2_error,
        threshold=0.12,
        details={
            "true_phi": true_phi.tolist(),
            "estimated_ar1": phi1.tolist(),
            "estimated_ar2": phi2.tolist(),
            "outlier_huber_weight": outlier_weight,
            "typical_huber_weight": typical_weight,
        },
    )


def run_sketched_glm_showcase(seed: int = 2027) -> ShowcaseRow:
    rng = np.random.default_rng(seed)
    n_time = 180
    n_predictors = 8
    n_voxels = 900
    X = rng.normal(size=(n_time, n_predictors))
    X[:, 0] = 1.0
    beta_true = rng.normal(scale=0.4, size=(n_predictors, n_voxels))
    Y = X @ beta_true + rng.normal(scale=0.15, size=(n_time, n_voxels))

    exact, exact_seconds = _elapsed(lambda: fast_lm_matrix(X, Y, fast_preproject(X)))
    sketch_config = LowRankConfig(sketch_kind="gaussian", sketch_ratio=0.75, seed=seed)
    sketched, sketch_seconds = _elapsed(lambda: fit_sketched(X, Y, sketch_config))

    beta_corr = float(np.corrcoef(exact.betas.ravel(), sketched.betas.ravel())[0, 1])
    rel_error = float(
        np.linalg.norm(exact.betas - sketched.betas) / np.linalg.norm(exact.betas)
    )
    status = "pass" if beta_corr > 0.98 and rel_error < 0.15 else "fail"
    return ShowcaseRow(
        case_id="tier_d_sketched_glm",
        capability="sketched high-voxel GLM",
        status=status,
        metric="beta_corr",
        value=beta_corr,
        threshold=0.98,
        details={
            "relative_beta_error": rel_error,
            "exact_seconds": exact_seconds,
            "sketched_seconds": sketch_seconds,
            "time_ratio_sketched_over_exact": sketch_seconds / exact_seconds
            if exact_seconds > 0
            else None,
            "n_time": n_time,
            "n_predictors": n_predictors,
            "n_voxels": n_voxels,
        },
    )


def _lss_reference(X: Array, Y: Array) -> Array:
    betas = np.zeros((X.shape[1], Y.shape[1]), dtype=np.float64)
    intercept = np.ones((X.shape[0], 1), dtype=np.float64)
    for trial in range(X.shape[1]):
        others = np.sum(np.delete(X, trial, axis=1), axis=1, keepdims=True)
        design = np.column_stack([X[:, trial], others[:, 0], intercept[:, 0]])
        betas[trial] = np.linalg.lstsq(design, Y, rcond=None)[0][0]
    return betas


def run_lss_showcase(seed: int = 2028) -> ShowcaseRow:
    rng = np.random.default_rng(seed)
    n_time = 90
    n_trials = 12
    n_voxels = 32
    X = np.zeros((n_time, n_trials), dtype=np.float64)
    onsets = np.linspace(4, n_time - 8, n_trials, dtype=int)
    kernel = np.array([0.2, 0.8, 1.0, 0.5, 0.15])
    for trial, onset in enumerate(onsets):
        X[onset : onset + kernel.size, trial] = kernel
    true_betas = rng.normal(scale=0.7, size=(n_trials, n_voxels))
    Y = X @ true_betas + rng.normal(scale=0.03, size=(n_time, n_voxels))

    reference, reference_seconds = _elapsed(lambda: _lss_reference(X, Y))
    fmrimod_result, fmrimod_seconds = _elapsed(
        lambda: estimate_single_trial(Y, X, method="lss", include_intercept=True)
    )
    max_abs = float(np.max(np.abs(fmrimod_result.betas - reference)))
    status = "pass" if max_abs < 1e-10 else "fail"
    return ShowcaseRow(
        case_id="tier_d_lss_trialwise",
        capability="vectorized LSS trial-wise betas",
        status=status,
        metric="max_abs_delta",
        value=max_abs,
        threshold=1e-10,
        details={
            "reference_seconds": reference_seconds,
            "fmrimod_seconds": fmrimod_seconds,
            "time_ratio_fmrimod_over_reference": fmrimod_seconds / reference_seconds
            if reference_seconds > 0
            else None,
            "n_time": n_time,
            "n_trials": n_trials,
            "n_voxels": n_voxels,
        },
    )


def run_lss_public_seam_showcase(seed: int = 2029) -> ShowcaseRow:
    """Exercise the public-seam single-trial path end-to-end.

    Builds a synthetic ``FmriDataset`` plus events table, calls
    ``estimate_single_trial_from_dataset(ds, "trialwise()", method="lss",
    include_intercept=True)`` (the Slice 1 public-seam wrapper from
    ``bd-01KRGQCT34QWSYKQ38BVFHD51E``), and verifies the recovered betas
    match the ground-truth ``true_betas`` used to synthesize ``Y``.

    The claim is "the wrapper, called with only the dataset and a string
    spec, recovers per-trial activation amplitudes to within noise
    tolerance." The wrapper does the design build, the label extraction,
    and the solver dispatch; the showcase row asserts the end-to-end
    behavior, not internal numpy equivalence (a wrapper-vs-matrix-first
    comparison on the same realized design would be tautological because
    both paths reduce to identical numpy ops; see review feedback at
    work-requests/post-01KRGZF070Q9Q7CSYA9X7ZEYJW).

    The matrix-first ``run_lss_showcase`` row still owns the third-party
    reference parity canary.
    """
    import pandas as pd

    from fmrimod import fmri_dataset
    from fmrimod.design.event_model import event_model as _build_event_model
    from fmrimod.single import estimate_single_trial_from_dataset

    rng = np.random.default_rng(seed)
    n_time = 90
    n_trials = 12
    n_voxels = 32
    tr = 2.0
    noise_scale = 0.03
    beta_scale = 0.7

    onsets = np.linspace(4.0, n_time * tr - 16.0, n_trials, dtype=np.float64)
    events = pd.DataFrame({
        "onset": onsets,
        "duration": np.zeros_like(onsets),
        "run": np.ones(n_trials, dtype=int),
    })

    # Realize the design matrix once to synthesize Y from known true_betas.
    # The wrapper rebuilds X internally from the spec; this branch only
    # constructs ground truth.
    em = _build_event_model(
        formula="trialwise()",
        data=events,
        block="run",
        tr=tr,
        n_scans=n_time,
    )
    X = np.asarray(em.design_matrix, dtype=np.float64)
    true_betas = rng.normal(scale=beta_scale, size=(n_trials, n_voxels))
    bold = X @ true_betas + rng.normal(scale=noise_scale, size=(n_time, n_voxels))

    ds = fmri_dataset(bold.astype(np.float64), tr=tr, events=events)

    wrapper_result, wrapper_seconds = _elapsed(
        lambda: estimate_single_trial_from_dataset(
            ds, "trialwise()", method="lss", include_intercept=True,
        )
    )

    recovery_error = float(np.max(np.abs(wrapper_result.betas - true_betas)))
    # Threshold is generous: with noise_scale=0.03 and ~5 effective HRF
    # samples per trial, per-beta LSS error is bounded around 0.05 voxel-wise.
    # 0.5 is the meaningful order-of-magnitude check (true betas live near
    # ±beta_scale=0.7).
    threshold = 0.5
    n_recovered_labels = (
        len(wrapper_result.trial_labels)
        if wrapper_result.trial_labels is not None
        else 0
    )
    label_check_ok = (
        wrapper_result.trial_labels is not None
        and n_recovered_labels == n_trials
    )
    status = "pass" if recovery_error < threshold and label_check_ok else "fail"
    return ShowcaseRow(
        case_id="tier_d_lss_public_seam",
        capability="public-seam single-trial wrapper (dataset + spec)",
        status=status,
        metric="max_abs_recovery_error",
        value=recovery_error,
        threshold=threshold,
        details={
            "wrapper_seconds": wrapper_seconds,
            "wrapper_trial_labels_present": wrapper_result.trial_labels is not None,
            "wrapper_n_trial_labels": n_recovered_labels,
            "noise_scale": noise_scale,
            "beta_scale": beta_scale,
            "n_time": n_time,
            "n_trials": n_trials,
            "n_voxels": n_voxels,
        },
    )


def run_lss_public_seam_independent_generative(seed: int = 2030) -> ShowcaseRow:
    """Public-seam LSS with a generator independent of EventModel lowering."""
    import pandas as pd

    from fmrimod import fmri_dataset
    from fmrimod.hrf.functions import gamma_hrf
    from fmrimod.sampling import SamplingFrame
    from fmrimod.single import estimate_single_trial_from_dataset

    rng = np.random.default_rng(seed)
    n_time = 120
    n_trials = 10
    n_voxels = 24
    tr = 2.0
    noise_scale = 0.001
    beta_scale = 0.7

    onsets = np.linspace(8.0, n_time * tr - 32.0, n_trials, dtype=np.float64)
    events = pd.DataFrame({
        "onset": onsets,
        "duration": np.zeros_like(onsets),
        "run": np.ones(n_trials, dtype=int),
    })
    scan_times = SamplingFrame(blocklens=(n_time,), tr=tr).samples
    X = np.zeros((n_time, n_trials), dtype=np.float64)
    for col, onset in enumerate(onsets):
        delta = scan_times - float(onset)
        X[:, col] = np.where(delta >= 0.0, gamma_hrf(delta), 0.0)

    true_betas = rng.normal(scale=beta_scale, size=(n_trials, n_voxels))
    bold = X @ true_betas + rng.normal(scale=noise_scale, size=(n_time, n_voxels))
    ds = fmri_dataset(bold.astype(np.float64), tr=tr, events=events)

    result, seconds = _elapsed(
        lambda: estimate_single_trial_from_dataset(
            ds, "trialwise(basis='gamma')", method="lss", include_intercept=True,
        )
    )
    recovery_error = float(np.max(np.abs(result.betas - true_betas)))
    threshold = 0.05
    n_recovered_labels = len(result.trial_labels) if result.trial_labels else 0
    status = (
        "pass"
        if recovery_error < threshold and n_recovered_labels == n_trials
        else "fail"
    )
    return ShowcaseRow(
        case_id="tier_d_lss_public_seam_independent_generative",
        capability="public-seam LSS with independently generated HRF signal",
        status=status,
        metric="max_abs_recovery_error",
        value=recovery_error,
        threshold=threshold,
        details={
            "wrapper_seconds": seconds,
            "wrapper_trial_labels_present": result.trial_labels is not None,
            "wrapper_n_trial_labels": n_recovered_labels,
            "noise_scale": noise_scale,
            "beta_scale": beta_scale,
            "n_time": n_time,
            "n_trials": n_trials,
            "n_voxels": n_voxels,
            "generator_uses_event_model": False,
            "hrf_basis": "gamma",
            "generative_design": "direct_gamma_hrf",
        },
    )


def run_showcases() -> list[ShowcaseRow]:
    """Run all Tier D showcase cases."""

    return [
        run_ar_robust_showcase(),
        run_sketched_glm_showcase(),
        run_lss_showcase(),
        run_lss_public_seam_showcase(),
        run_lss_public_seam_independent_generative(),
    ]


def _public_seam_rows(rows: list[ShowcaseRow]) -> list[ShowcaseRow]:
    return [row for row in rows if "public-seam" in row.capability]


def _low_level_canary_rows(rows: list[ShowcaseRow]) -> list[ShowcaseRow]:
    public_ids = {row.case_id for row in _public_seam_rows(rows)}
    return [row for row in rows if row.case_id not in public_ids]


def build_proof_scorecard(rows: list[ShowcaseRow]) -> ShowcaseProofScorecard:
    """Build the Tier D public-seam proof scorecard.

    The flagship receipt must not mix low-level canaries into the public proof.
    Low-level Tier D capabilities are rendered to a separate canary report and
    classified as numerical canaries in ``proof_artifacts.json``.
    """
    from benchmarks.parity.tier_group_semantic_survival.workflow import run_demo

    semantic = run_demo(n_subjects=3, max_voxels=12).report
    checks = semantic["checks"]
    public_rows = tuple(row.case_id for row in _public_seam_rows(rows))
    semantic_survival = {
        "source": "tier_group_semantic_survival",
        "status": semantic["status"],
        "contrast_name_survives": checks["contrast_name_survives"],
        "typed_intent_kind": checks["typed_intent_kind"],
        "typed_intent_term": checks["typed_intent_term"],
        "typed_intent_levels": checks["typed_intent_levels"],
        "statistic_family": checks["statistic_family"],
        "group_assays": checks["group_assays"],
        "group_result_assays": checks["group_result_assays"],
        "timings": semantic["timings"],
        "fit_provenance": semantic.get("fit_provenance"),
    }
    return ShowcaseProofScorecard(
        public_seam=True,
        fmrimod_path=(
            "fmri_dataset -> fmri_lm -> OmnibusContrast -> "
            "ContrastResult.explain -> GroupDataset -> ols_voxelwise"
        ),
        reference_path=(
            "Nilearn-compatible numerical oracles plus low-level exact "
            "matrix canaries for capabilities not yet promoted to public seams"
        ),
        typed_objects=(
            "fmrimod.dataset.FmriDataset",
            "fmrimod.spec.Spec",
            "fmrimod.contrast.OmnibusContrast",
            "fmrimod.glm.FmriLm",
            "fmrimod.glm.ContrastResult",
            "fmrimod.group.GroupDataset",
        ),
        public_rows=public_rows,
        semantic_survival=semantic_survival,
        win_axes={
            "design": "statistical intent is represented as typed objects",
            "elegance": "the public path reads as the analysis sequence",
            "power": "single-trial and group evidence live in one artifact",
            "trust": "the report carries timings, caveats, and explanations",
        },
    )


def render(rows: list[ShowcaseRow], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "showcase_report.json"
    md_path = out_dir / "SHOWCASE.md"
    canary_json_path = out_dir / "showcase_canaries.json"
    canary_md_path = out_dir / "SHOWCASE_CANARIES.md"
    scorecard = build_proof_scorecard(rows)
    public_rows = _public_seam_rows(rows)
    low_level_canaries = _low_level_canary_rows(rows)
    payload = {
        "name": "tier_d_showcase",
        "status": "pass"
        if all(row.status == "pass" for row in public_rows)
        else "fail",
        "caveats": [],
        "proof_scorecard": asdict(scorecard),
        "fit_provenance": scorecard.semantic_survival.get("fit_provenance"),
        "rows": [asdict(row) for row in public_rows],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    canary_payload = {
        "name": "tier_d_showcase_canaries",
        "status": "pass"
        if all(row.status == "pass" for row in low_level_canaries)
        else "fail",
        "caveats": [],
        "rows": [asdict(row) for row in low_level_canaries],
    }
    canary_json_path.write_text(
        json.dumps(canary_payload, indent=2, sort_keys=True) + "\n"
    )

    lines = [
        "# Tier D fmrimod Showcase",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Proof Scorecard",
        "",
        f"- public seam: `{scorecard.public_seam}`",
        f"- path: `{scorecard.fmrimod_path}`",
        f"- semantic source: `{scorecard.semantic_survival['source']}`",
        f"- typed intent: `{scorecard.semantic_survival['typed_intent_term']}`",
        f"- public rows: `{', '.join(scorecard.public_rows)}`",
        "",
        "| Case | Capability | Metric | Value | Threshold | Status |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in public_rows:
        threshold = "" if row.threshold is None else f"{row.threshold:.6g}"
        lines.append(
            "| "
            + " | ".join(
                [
                    row.case_id,
                    row.capability,
                    row.metric,
                    f"{row.value:.6g}",
                    threshold,
                    row.status,
                ]
            )
            + " |"
        )
    md_path.write_text("\n".join(lines) + "\n")

    canary_lines = [
        "# Tier D Numerical Canaries",
        "",
        f"Status: `{canary_payload['status']}`",
        "",
        "| Case | Capability | Metric | Value | Threshold | Status |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in low_level_canaries:
        threshold = "" if row.threshold is None else f"{row.threshold:.6g}"
        canary_lines.append(
            "| "
            + " | ".join(
                [
                    row.case_id,
                    row.capability,
                    row.metric,
                    f"{row.value:.6g}",
                    threshold,
                    row.status,
                ]
            )
            + " |"
        )
    canary_md_path.write_text("\n".join(canary_lines) + "\n")
    return json_path, md_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "reports",
        help="Directory where the Tier D showcase report should be written.",
    )
    args = parser.parse_args(argv)
    rows = run_showcases()
    render(rows, args.out_dir)
    if any(row.status != "pass" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
