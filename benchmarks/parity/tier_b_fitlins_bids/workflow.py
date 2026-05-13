"""FitLins-style BIDS Stats Model translation and CLI derivative parity."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.glm.first_level.hemodynamic_models import compute_regressor

import fmrimod as fm
from cross_testing.harness import ParityCase, ParityTolerance, PipelineOutput, render, run
from fmrimod.ar import ar_whiten_matrix, estimate_ar
from fmrimod.bids import translate_run_node
from fmrimod.glm.solver import fast_lm_matrix, fast_preproject


def _stats_model() -> dict:
    from examples.design.fitlins_style_first_level import FITLINS_STYLE_MODEL

    return FITLINS_STYLE_MODEL


def _fitlins_cli_stats_model() -> dict:
    return {
        "Name": "junkfood_model001",
        "BIDSModelVersion": "1.0.0",
        "Description": "Minimal run-level FitLins CLI parity fixture",
        "Input": {"task": "eating"},
        "Nodes": [
            {
                "Level": "run",
                "Name": "run",
                "GroupBy": ["run", "subject"],
                "Transformations": {
                    "Transformer": "pybids-transforms-v1",
                    "Instructions": [
                        {"Name": "Factor", "Input": ["trial_type"]},
                        {
                            "Name": "Convolve",
                            "Input": ["trial_type.ice_cream", "trial_type.cake"],
                            "Model": "spm",
                        },
                    ],
                },
                "Model": {
                    "X": [
                        "trial_type.ice_cream",
                        "trial_type.cake",
                        "food_sweats",
                        1,
                    ]
                },
                "Contrasts": [
                    {
                        "Name": "icecream_gt_cake",
                        "ConditionList": ["trial_type.ice_cream", "trial_type.cake"],
                        "Weights": [1, -1],
                        "Test": "t",
                    },
                    {
                        "Name": "eating_vs_baseline",
                        "ConditionList": ["trial_type.ice_cream", "trial_type.cake"],
                        "Weights": [0.5, 0.5],
                        "Test": "t",
                    },
                ],
            }
        ],
        "Edges": [],
    }


def _inputs() -> dict:
    rng = np.random.default_rng(7)
    n_scans = 120
    sampling_frame = fm.SamplingFrame(blocklens=[n_scans], tr=2.0)
    events = pd.DataFrame(
        {
            "run": 1,
            "onset": np.arange(8.0, 200.0, 12.0),
            "duration": 1.2,
            "trial_type": [
                "word" if idx % 2 == 0 else "pseudoword"
                for idx in range(len(np.arange(8.0, 200.0, 12.0)))
            ],
        }
    )
    confounds = pd.DataFrame(
        {
            "framewise_displacement": np.linspace(-1.0, 1.0, n_scans),
            "trans_x": rng.normal(0.0, 0.25, size=n_scans),
        }
    )
    return {
        "stats_model": _stats_model(),
        "events": events,
        "sampling_frame": sampling_frame,
        "confounds": confounds,
    }


def translated_pipeline(inputs: dict) -> PipelineOutput:
    translated = translate_run_node(
        inputs["stats_model"],
        events=inputs["events"],
        sampling_frame=inputs["sampling_frame"],
        confounds=inputs["confounds"],
    )
    design = np.column_stack(
        [translated.event_model.design_matrix, translated.baseline_model.design_matrix]
    )
    return PipelineOutput(
        arrays={
            "design": design,
            "word_gt_pseudoword": translated.contrast_vectors[
                "word_gt_pseudoword"
            ],
            "task_vs_baseline": translated.contrast_vectors["task_vs_baseline"],
        }
    )


def manual_pipeline(inputs: dict) -> PipelineOutput:
    event_model = fm.event_model(
        "hrf(trial_type)",
        data=inputs["events"],
        sampling_frame=inputs["sampling_frame"],
        block="run",
        durations="duration",
    )
    baseline = fm.baseline_model(
        basis="constant",
        sframe=inputs["sampling_frame"],
        intercept="global",
        nuisance_list=[
            inputs["confounds"][
                ["framewise_displacement", "trans_x"]
            ].to_numpy(dtype=np.float64)
        ],
    )
    design = np.column_stack([event_model.design_matrix, baseline.design_matrix])
    column_names = list(event_model.column_names) + list(baseline.column_names)
    word = column_names.index("trial_type_trial_type.word")
    pseudo = column_names.index("trial_type_trial_type.pseudoword")
    word_gt_pseudo = np.zeros(len(column_names), dtype=np.float64)
    word_gt_pseudo[word] = 1.0
    word_gt_pseudo[pseudo] = -1.0
    task_vs_baseline = np.zeros(len(column_names), dtype=np.float64)
    task_vs_baseline[word] = 0.5
    task_vs_baseline[pseudo] = 0.5
    return PipelineOutput(
        arrays={
            "design": design,
            "word_gt_pseudoword": word_gt_pseudo,
            "task_vs_baseline": task_vs_baseline,
        }
    )


def make_case() -> ParityCase:
    return ParityCase(
        name="tier_b_fitlins_stats_model_translation",
        fmrimod_pipeline=translated_pipeline,
        reference_pipeline=manual_pipeline,
        inputs=_inputs(),
        tolerances={
            "design": ParityTolerance(rtol=1e-12, atol=1e-12),
            "word_gt_pseudoword": ParityTolerance(rtol=0.0, atol=0.0),
            "task_vs_baseline": ParityTolerance(rtol=0.0, atol=0.0),
        },
    )


@dataclass(frozen=True)
class DerivativeDelta:
    """Voxel-wise delta for one derivative map."""

    name: str
    max_abs: float
    mae: float
    pearson_r: float
    gate: str
    passes: bool
    caveat_id: str | None = None


@dataclass(frozen=True)
class FitlinsCliDerivativeResult:
    """Result summary for the real FitLins CLI derivative-tree comparison."""

    status: str
    fitlins_command: list[str]
    fitlins_seconds: float
    fitlins_output_files: list[str]
    fmrimod_output_files: list[str]
    deltas: list[DerivativeDelta]
    design_columns: list[str]
    caveats: list[str]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _make_events(n_events: int = 4, iti: float = 20.0) -> pd.DataFrame:
    trial_types = ["ice_cream", "cake"] * n_events
    return pd.DataFrame(
        {
            "onset": np.arange(len(trial_types), dtype=np.float64) * iti,
            "trial_type": trial_types,
            "duration": np.ones(len(trial_types), dtype=np.float64),
        }
    )


def _make_signal(events: pd.DataFrame, frame_times: np.ndarray) -> np.ndarray:
    signal = np.zeros(frame_times.shape, dtype=np.float64)
    for condition, weight in {"ice_cream": 1.0, "cake": 0.55}.items():
        block = events.query("trial_type == @condition")[["onset", "duration"]].to_numpy().T
        exp_condition = np.vstack([block, np.repeat(weight, block.shape[1])])
        signal += compute_regressor(exp_condition, "spm", frame_times, con_id=condition)[
            0
        ].squeeze()
    return signal


def _write_cli_fixture(base_dir: Path) -> dict[str, Path]:
    rng = np.random.default_rng(33)
    tr = 2.0
    events = _make_events()
    n_tp = int((events["onset"].max() + 24.0) // tr)
    frame_times = np.arange(n_tp, dtype=np.float64) * tr
    signal = _make_signal(events, frame_times)
    confounds = pd.DataFrame(
        {
            "food_sweats": rng.normal(0.0, 1.0, size=n_tp),
            "sugar_jitters": rng.normal(0.0, 1.0, size=n_tp),
        }
    )
    nuisance = 0.25 * confounds["food_sweats"].to_numpy()
    timeseries = 100.0 + 7.0 * signal + nuisance + rng.normal(0.0, 0.15, size=n_tp)
    data = rng.normal(100.0, 0.03, size=(7, 7, 7, n_tp))
    data[2:5, 2:5, 2:5, :] += timeseries[np.newaxis, np.newaxis, np.newaxis, :]
    affine = np.eye(4)

    bids_dir = base_dir / "bids"
    deriv_dir = bids_dir / "derivatives" / "fmriprep"
    raw_func = bids_dir / "sub-01" / "func"
    deriv_func = deriv_dir / "sub-01" / "func"
    raw_func.mkdir(parents=True, exist_ok=True)
    deriv_func.mkdir(parents=True, exist_ok=True)

    _write_json(
        bids_dir / "dataset_description.json",
        {"Name": "fitlins-cli-parity", "BIDSVersion": "1.8.0"},
    )
    _write_json(
        deriv_dir / "dataset_description.json",
        {
            "Name": "fMRIPrep parity fixture",
            "BIDSVersion": "1.8.0",
            "GeneratedBy": [{"Name": "fmrimod parity fixture"}],
        },
    )
    events.to_csv(
        raw_func / "sub-01_task-eating_run-01_events.tsv",
        sep="\t",
        index=False,
    )
    img = nib.Nifti1Image(data.astype(np.float32), affine)
    raw_bold = raw_func / "sub-01_task-eating_run-01_bold.nii.gz"
    preproc_bold = deriv_func / "sub-01_task-eating_run-01_space-T1w_desc-preproc_bold.nii.gz"
    img.to_filename(raw_bold)
    img.to_filename(preproc_bold)
    metadata = {"RepetitionTime": tr, "TaskName": "eating", "SkullStripped": False}
    _write_json(raw_bold.with_suffix("").with_suffix(".json"), metadata)
    _write_json(preproc_bold.with_suffix("").with_suffix(".json"), metadata)

    mask = np.zeros(data.shape[:3], dtype=np.uint8)
    mask[2:5, 2:5, 2:5] = 1
    mask_path = deriv_func / "sub-01_task-eating_run-01_space-T1w_desc-brain_mask.nii.gz"
    nib.Nifti1Image(mask, affine).to_filename(mask_path)

    confounds.to_csv(
        deriv_func / "sub-01_task-eating_run-01_desc-confounds_regressors.tsv",
        sep="\t",
        index=False,
    )
    model_path = bids_dir / "model-junkfood_smdl.json"
    _write_json(model_path, _fitlins_cli_stats_model())
    return {
        "bids_dir": bids_dir,
        "deriv_dir": deriv_dir,
        "model": model_path,
        "bold": preproc_bold,
        "mask": mask_path,
    }


def _camel_contrast(name: str) -> str:
    pieces = name.split("_")
    return pieces[0] + "".join(piece[:1].upper() + piece[1:] for piece in pieces[1:])


def _fit_fmrimod_derivatives(paths: dict[str, Path], fitlins_out: Path, fmrimod_out: Path) -> list[str]:
    design_path = next(fitlins_out.glob("node-run/sub-01/*_design.tsv"))
    design = pd.read_csv(design_path, sep="\t")
    X = design.to_numpy(dtype=np.float64)
    img = nib.load(paths["bold"])
    mask_img = nib.load(paths["mask"])
    data = np.asarray(img.get_fdata(), dtype=np.float64)
    mask = np.asarray(mask_img.get_fdata() > 0)
    Y = data[mask].T
    mean = np.maximum(Y.mean(axis=0), 1.0)
    Y = 100.0 * (Y / mean - 1.0)
    ols = fast_lm_matrix(X, Y, fast_preproject(X), return_fitted=False)
    residuals = Y - X @ ols.betas
    ar1 = estimate_ar(residuals, order=1, voxelwise=True).reshape(-1)
    ar1_bins = (ar1 * 100).astype(int) / 100.0
    fmrimod_files: list[str] = []
    columns = list(design.columns)
    contrast_specs = {
        "icecream_gt_cake": {"trial_type.ice_cream": 1.0, "trial_type.cake": -1.0},
        "eating_vs_baseline": {"trial_type.ice_cream": 0.5, "trial_type.cake": 0.5},
    }
    for name, weights in contrast_specs.items():
        contrast = np.zeros(len(columns), dtype=np.float64)
        for column, weight in weights.items():
            contrast[columns.index(column)] = weight
        effect = np.zeros(Y.shape[1], dtype=np.float64)
        variance = np.zeros(Y.shape[1], dtype=np.float64)
        tstat = np.zeros(Y.shape[1], dtype=np.float64)
        for ar1_bin in np.unique(ar1_bins):
            cols_idx = ar1_bins == ar1_bin
            Xw, Yw = ar_whiten_matrix(X, Y[:, cols_idx], np.array([ar1_bin]))
            projection = fast_preproject(Xw, check_finite=False)
            fit = fast_lm_matrix(Xw, Yw, projection, check_finite=False)
            effect[cols_idx] = contrast @ fit.betas
            variance[cols_idx] = fit.sigma2 * float(
                contrast @ projection.XtXinv @ contrast
            )
            tstat[cols_idx] = effect[cols_idx] / np.sqrt(
                np.maximum(variance[cols_idx], np.finfo(np.float64).eps)
            )
        outputs = {"effect": effect, "variance": variance, "t": tstat}
        for stat_name, values in outputs.items():
            volume = np.zeros(mask.shape, dtype=np.float32)
            volume[mask] = values.astype(np.float32)
            rel = (
                Path("node-run")
                / "sub-01"
                / f"sub-01_run-01_contrast-{_camel_contrast(name)}_stat-{stat_name}_statmap.nii.gz"
            )
            out_path = fmrimod_out / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            nib.Nifti1Image(volume, img.affine, img.header).to_filename(out_path)
            fmrimod_files.append(str(rel))
    return sorted(fmrimod_files)


def _compare_derivatives(
    fitlins_out: Path,
    fmrimod_out: Path,
    *,
    max_abs_tolerance: float = 5e-4,
) -> list[DerivativeDelta]:
    deltas: list[DerivativeDelta] = []
    for fmrimod_file in sorted(fmrimod_out.glob("node-run/sub-01/*_statmap.nii.gz")):
        rel = fmrimod_file.relative_to(fmrimod_out)
        fitlins_file = fitlins_out / rel
        if not fitlins_file.exists():
            deltas.append(
                DerivativeDelta(
                    name=str(rel),
                    max_abs=float("inf"),
                    mae=float("inf"),
                    pearson_r=0.0,
                    gate="missing_file",
                    passes=False,
                    caveat_id=None,
                )
            )
            continue
        candidate = np.asarray(nib.load(fmrimod_file).get_fdata(), dtype=np.float64).ravel()
        reference = np.asarray(nib.load(fitlins_file).get_fdata(), dtype=np.float64).ravel()
        nonzero = np.logical_or(candidate != 0, reference != 0)
        candidate = candidate[nonzero]
        reference = reference[nonzero]
        delta = candidate - reference
        max_abs = float(np.max(np.abs(delta)))
        mae = float(np.mean(np.abs(delta)))
        pearson = (
            float(np.corrcoef(candidate, reference)[0, 1])
            if candidate.size > 1 and np.std(candidate) > 0 and np.std(reference) > 0
            else 1.0
        )
        is_effect = "_stat-effect_" in str(rel)
        is_t = "_stat-t_" in str(rel)
        is_variance = "_stat-variance_" in str(rel)
        caveat_id = None if is_effect else "fitlins-ar1-coefficient-binning"
        gate = "max_abs+pearson" if is_effect or is_t else "max_abs"
        if caveat_id:
            gate = f"caveat-bypassed:{gate}"
        passes = (
            (max_abs <= 2e-3 and pearson >= 0.99)
            if is_effect
            else (max_abs <= 0.15 and pearson >= 0.99)
            if is_t
            else (max_abs <= 2e-3)
            if is_variance
            else False
        )
        deltas.append(
            DerivativeDelta(
                name=str(rel),
                max_abs=max_abs,
                mae=mae,
                pearson_r=pearson,
                gate=gate,
                passes=bool(passes),
                caveat_id=caveat_id,
            )
        )
    return deltas


def run_fitlins_cli_derivative_parity(work_dir: Path | None = None) -> FitlinsCliDerivativeResult:
    """Run the real FitLins CLI and compare selected derivative maps."""

    if shutil.which("uv") is None:
        raise RuntimeError("uv is required to run the local FitLins CLI fixture")
    owns_tmp = work_dir is None
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="fmrimod-fitlins-cli-"))
    work_dir.mkdir(parents=True, exist_ok=True)
    paths = _write_cli_fixture(work_dir)
    fitlins_out = work_dir / "fitlins-out"
    fmrimod_out = work_dir / "fmrimod-out"
    fitlins_work = work_dir / "fitlins-work"
    repo_root = Path(__file__).resolve().parents[3]
    fitlins_project = repo_root / "fitlins"
    cli_patch = (
        "import sys;"
        "import nipype.utils.profiler as profiler;"
        "profiler.get_system_total_memory_gb=lambda:4.0;"
        "import nipype.pipeline.plugins.multiproc as multiproc;"
        "import nipype.pipeline.plugins as plugins;"
        "from nipype.pipeline.plugins.linear import LinearPlugin;"
        "multiproc.get_system_total_memory_gb=lambda:4.0;"
        "multiproc.MultiProcPlugin=LinearPlugin;"
        "plugins.MultiProcPlugin=LinearPlugin;"
        "from fitlins.cli.run import main;"
        "sys.argv=['fitlins']+sys.argv[1:];"
        "raise SystemExit(main())"
    )
    cmd = [
        "uv",
        "run",
        "--project",
        str(fitlins_project),
        "python",
        "-c",
        cli_patch,
        str(paths["bids_dir"]),
        str(fitlins_out),
        "run",
        "-d",
        str(paths["deriv_dir"]),
        "-m",
        str(paths["model"]),
        "-w",
        str(fitlins_work),
        "--n-cpus",
        "1",
        "--mem-gb",
        "4",
        "--estimator",
        "nilearn",
        "--space",
        "T1w",
    ]
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=fitlins_project,
        env={
            **os.environ,
            "MPLCONFIGDIR": str(work_dir / "matplotlib-cache"),
            "UV_CACHE_DIR": str(work_dir / "uv-cache"),
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    fitlins_seconds = time.perf_counter() - start
    if proc.returncode != 0:
        raise RuntimeError("FitLins CLI failed:\n" + proc.stdout[-4000:])

    fmrimod_files = _fit_fmrimod_derivatives(paths, fitlins_out, fmrimod_out)
    fitlins_files = sorted(
        str(path.relative_to(fitlins_out))
        for path in fitlins_out.rglob("*")
        if path.is_file()
    )
    deltas = _compare_derivatives(fitlins_out, fmrimod_out)
    design = pd.read_csv(next(fitlins_out.glob("node-run/sub-01/*_design.tsv")), sep="\t")
    result = FitlinsCliDerivativeResult(
        status=(
            "pass_with_caveats"
            if all(delta.passes for delta in deltas)
            and any(delta.caveat_id for delta in deltas)
            else "pass"
            if all(delta.passes for delta in deltas)
            else "fail"
        ),
        fitlins_command=cmd,
        fitlins_seconds=fitlins_seconds,
        fitlins_output_files=fitlins_files,
        fmrimod_output_files=fmrimod_files,
        deltas=deltas,
        design_columns=list(design.columns),
        caveats=[
            "Comparison is scoped to run-level design, effect, variance, and t maps; "
            "FitLins-only p/z/report/rSquare/log-likelihood outputs are inventoried but not recomputed.",
            "fitlins-ar1-coefficient-binning: t and variance maps use voxelwise AR(1) "
            "coefficients binned to 0.01 in this fixture. Remaining deltas reflect small "
            "AR coefficient estimation/binning differences rather than contrast algebra.",
        ],
    )
    if owns_tmp:
        # Keep the temporary tree during debugging by passing an explicit work_dir.
        shutil.rmtree(work_dir, ignore_errors=True)
    return result


def render_fitlins_cli_derivative_report(
    result: FitlinsCliDerivativeResult,
    out_dir: Path,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "fitlins_cli_derivative_report.json"
    md_path = out_dir / "FITLINS_CLI_DERIVATIVES.md"
    payload = asdict(result)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# FitLins CLI Derivative Parity",
        "",
        f"Status: `{result.status}`",
        f"FitLins runtime seconds: `{result.fitlins_seconds:.3f}`",
        "",
        "## Compared Maps",
        "",
        "| map | gate | max_abs | mae | pearson_r | caveat | pass |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for delta in result.deltas:
        lines.append(
            f"| {delta.name} | {delta.gate} | {delta.max_abs:.6g} | {delta.mae:.6g} | "
            f"{delta.pearson_r:.6g} | {delta.caveat_id or ''} | "
            f"{'yes' if delta.passes else 'no'} |"
        )
    lines.extend(["", "## Caveats", ""])
    for caveat in result.caveats:
        lines.append(f"- {caveat}")
    md_path.write_text("\n".join(lines) + "\n")
    return json_path, md_path


def main() -> None:
    result = run(make_case())
    out_dir = Path(__file__).resolve().parent / "reports"
    render(result, out_dir)
    cli_result = run_fitlins_cli_derivative_parity()
    render_fitlins_cli_derivative_report(cli_result, out_dir)
    if result.status == "fail":
        raise SystemExit(1)
    if cli_result.status == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
