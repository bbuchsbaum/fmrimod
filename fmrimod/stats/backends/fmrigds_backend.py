"""Optional fmrigds-backed second-level backend."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from fmrimod.stats.interfaces import GroupFitRequest, GroupFitResult

_DEFAULT_RSCRIPT = "Rscript"
_BRIDGE_SCRIPT = Path(__file__).with_name("fmrigds_bridge.R")


def _as_2d_float(x: object) -> NDArray[np.float64]:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        return arr[:, np.newaxis]
    if arr.ndim == 2:
        return arr
    raise ValueError("Bridge returned invalid array rank; expected 1-D or 2-D")


def _fmrigds_dev_source() -> str | None:
    env = os.environ.get("FMRIGDS_SOURCE_DIR", "").strip()
    if env and Path(env).exists():
        return env
    home_candidate = Path.home() / "code" / "fmrigds"
    if home_candidate.exists():
        return str(home_candidate)
    return None


def fmrigds_backend_available(
    *,
    rscript: str = _DEFAULT_RSCRIPT,
    fmrigds_source: str | None = None,
) -> tuple[bool, str]:
    """Check whether fmrigds bridge prerequisites are available."""
    if shutil.which(rscript) is None:
        return False, "Rscript not found on PATH"
    source = fmrigds_source or _fmrigds_dev_source()
    check_code = """
args <- commandArgs(trailingOnly = TRUE)
source_dir <- if (length(args) >= 1L) args[[1L]] else ""
ok <- FALSE
if (nzchar(source_dir)) {
  if (!requireNamespace("pkgload", quietly = TRUE)) {
    cat("missing_pkgload")
    quit(status = 0)
  }
  suppressWarnings(pkgload::load_all(source_dir, quiet = TRUE, export_all = FALSE, helpers = FALSE))
  ok <- "fmrigds" %in% loadedNamespaces()
}
if (!ok) ok <- requireNamespace("fmrigds", quietly = TRUE)
cat(if (ok) "ok" else "missing_fmrigds")
"""
    args = [rscript, "-e", check_code]
    if source is not None:
        args.append(source)
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        return False, proc.stderr.strip() or "R check failed"
    out = (proc.stdout or "").strip()
    if out == "ok":
        return True, "ok"
    if out == "missing_pkgload":
        return False, "pkgload not installed (required for fmrigds source-dir loading)"
    return False, "fmrigds package not available"


def _write_csv_payload_inputs(
    request: GroupFitRequest,
    workdir: Path,
) -> Mapping[str, object]:
    data = request.data.data
    csv_path = workdir / "group_data.csv"
    df = cast(Any, data["data"]).copy()
    sample_col = data.get("roi_col") or "sample"
    contrast_col = data.get("contrast_col") or "contrast"

    if sample_col not in df.columns:
        df[sample_col] = "sample1"
    if contrast_col not in df.columns:
        df[contrast_col] = "c1"

    df.to_csv(csv_path, index=False)

    payload: dict[str, object] = {
        "format": "csv",
        "path": str(csv_path),
        "effect_cols": dict(cast(Any, data["effect_cols"])),
        "subject_col": data["subject_col"],
        "sample_col": sample_col,
        "contrast_col": contrast_col,
        "subjects": list(request.data.subjects),
    }

    if request.data.covariates is not None:
        cov_path = workdir / "covariates.csv"
        cov = request.data.covariates.copy()
        cov.insert(0, "subject", request.data.subjects)
        cov.to_csv(cov_path, index=False)
        payload["covariates_path"] = str(cov_path)
        payload["covariates_id_col"] = "subject"

    return payload


def _build_bridge_payload(
    request: GroupFitRequest,
    workdir: Path,
    *,
    backend_options: Mapping[str, object],
) -> dict[str, object]:
    """Build JSON-serializable payload for the R bridge."""
    gd = request.data
    if gd.format == "csv":
        input_payload = _write_csv_payload_inputs(request, workdir)
    elif gd.format == "h5":
        input_payload = {
            "format": "h5",
            "paths": list(cast(Any, gd.data.get("paths") or [])),
            "mask": gd.data.get("mask"),
            "contrast": gd.data.get("contrast"),
            "stat": list(cast(Any, gd.data.get("stat") or [])),
            "subjects": list(gd.subjects),
        }
    elif gd.format == "nifti":
        input_payload = {
            "format": "nifti",
            "beta_paths": gd.data.get("beta_paths"),
            "se_paths": gd.data.get("se_paths"),
            "var_paths": gd.data.get("var_paths"),
            "t_paths": gd.data.get("t_paths"),
            "df": gd.data.get("df"),
            "mask": gd.data.get("mask"),
            "target_space": gd.data.get("target_space"),
            "subjects": list(gd.subjects),
        }
    else:
        raise NotImplementedError(
            f"fmrigds backend does not yet support GroupData format='{gd.format}'"
        )

    return {
        "model": request.model,
        "formula": request.formula,
        "method": request.method,
        "weights": request.weights,
        "robust": request.robust,
        "combine": request.combine,
        "backend_options": dict(backend_options),
        "input": input_payload,
    }


def _decode_bridge_result(result: Mapping[str, object], request: GroupFitRequest) -> GroupFitResult:
    estimate = _as_2d_float(result["estimate"])
    se = _as_2d_float(result["se"])
    statistic = _as_2d_float(result["statistic"])
    p = _as_2d_float(result["p"])
    tau2_raw = result.get("tau2")
    tau2 = None if tau2_raw is None else np.asarray(tau2_raw, dtype=np.float64)

    md = dict(cast("dict[str, Any]", result.get("metadata") or {}))
    md["source"] = "fmrigds_bridge"
    return GroupFitResult(
        estimate=estimate,
        se=se,
        statistic=statistic,
        p=p,
        q=None,
        tau2=tau2,
        predictor_names=list(cast("list[str]", result.get("predictor_names") or ["Intercept"])),
        feature_names=list(cast("list[str]", result.get("feature_names") or [f"f{i+1}" for i in range(estimate.shape[0])])),
        model=str(result.get("model") or request.model),
        method=str(result.get("method") or request.method),
        formula=str(result.get("formula") or request.formula),
        backend="fmrigds-r",
        metadata=md,
    )


class FmrigdsBackend:
    """Backend shim for fmrigds delegation via an R bridge script."""

    name = "fmrigds-r"

    def fit(self, request: GroupFitRequest) -> GroupFitResult:
        method = str(request.method)
        if method in ("pm", "reml"):
            raise NotImplementedError(
                "fmrigds backend currently supports method='fe' or 'dl' in this bridge phase"
            )
        if str(request.weights) == "custom":
            raise NotImplementedError("fmrigds backend does not yet support custom weights")
        if request.robust != "none":
            raise NotImplementedError("fmrigds backend does not yet support robust != 'none'")
        if request.combine is not None:
            raise NotImplementedError("fmrigds backend does not yet support combine modes")

        backend_options = dict(request.backend_options)
        rscript = str(backend_options.get("rscript") or _DEFAULT_RSCRIPT)
        fmrigds_source = backend_options.get("fmrigds_source") or _fmrigds_dev_source()
        timeout_sec = int(backend_options.get("timeout_sec") or 60)

        ok, reason = fmrigds_backend_available(rscript=rscript, fmrigds_source=fmrigds_source)
        if not ok:
            raise RuntimeError(f"fmrigds backend unavailable: {reason}")

        with tempfile.TemporaryDirectory(prefix="fmrimod-fmrigds-") as td:
            tdir = Path(td)
            payload = _build_bridge_payload(
                request,
                tdir,
                backend_options={"fmrigds_source": fmrigds_source},
            )
            in_path = tdir / "bridge_request.json"
            out_path = tdir / "bridge_result.json"
            in_path.write_text(json.dumps(payload), encoding="utf-8")

            cmd = [rscript, str(_BRIDGE_SCRIPT), str(in_path), str(out_path)]
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            if proc.returncode != 0:
                msg = (proc.stderr or proc.stdout or "").strip()
                raise RuntimeError(f"fmrigds bridge failed: {msg}")
            if not out_path.exists():
                raise RuntimeError("fmrigds bridge failed: missing result file")
            result_payload = json.loads(out_path.read_text(encoding="utf-8"))

        return _decode_bridge_result(result_payload, request)


__all__ = ["FmrigdsBackend", "fmrigds_backend_available", "_build_bridge_payload"]
