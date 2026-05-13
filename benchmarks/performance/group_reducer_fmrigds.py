"""Benchmark native group reducers against compiled fmrigds kernels.

This script is intentionally outside the test suite. It is a local performance
gate for the fmrigds port and requires an R installation with fmrigds available
when ``--skip-r`` is not used.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fmrimod.group import SampleLabelSpace, group_dataset, meta_fe, perm_onesample


def _make_data(
    *,
    n_features: int,
    n_subjects: int,
    n_perm: int,
    seed: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int32]]:
    rng = np.random.default_rng(seed)
    subject_offsets = rng.normal(scale=0.25, size=(n_subjects, 1))
    feature_offsets = rng.normal(scale=0.60, size=(1, n_features))
    beta_2d = subject_offsets + feature_offsets + rng.normal(
        scale=0.80,
        size=(n_subjects, n_features),
    )
    var_2d = rng.uniform(0.04, 0.16, size=(n_subjects, n_features))
    signs = rng.choice(
        np.array([-1, 1], dtype=np.int32),
        size=(n_perm, n_subjects),
        replace=True,
    )
    signs[0, :] = 1
    return beta_2d.astype(np.float64), var_2d.astype(np.float64), signs


def _min_elapsed(fn: Any, *, repeat: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    elapsed: list[float] = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        elapsed.append(time.perf_counter() - start)
    return min(elapsed)


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _python_benchmarks(
    beta_2d: NDArray[np.float64],
    var_2d: NDArray[np.float64],
    signs: NDArray[np.int32],
    *,
    repeat: int,
    warmup: int,
    n_jobs: int,
    chunk_size: int | None,
    blas_threads: int | None,
) -> dict[str, float]:
    n_subjects, n_features = beta_2d.shape
    beta = beta_2d.T.reshape(n_features, n_subjects, 1)
    var = var_2d.T.reshape(n_features, n_subjects, 1)
    dataset = group_dataset(
        {"beta": beta, "var": var},
        space=SampleLabelSpace([f"feature_{idx}" for idx in range(n_features)]),
        subjects=[f"sub-{idx:03d}" for idx in range(n_subjects)],
        contrasts=["c1"],
    )
    sign_arr = signs.astype(np.int8, copy=False)
    return {
        "python_meta_fe_s": _min_elapsed(
            lambda: meta_fe(dataset),
            repeat=repeat,
            warmup=warmup,
        ),
        "python_perm_onesample_serial_s": _min_elapsed(
            lambda: perm_onesample(dataset, signs=sign_arr, n_jobs=1),
            repeat=repeat,
            warmup=warmup,
        ),
        "python_perm_onesample_parallel_s": _min_elapsed(
            lambda: perm_onesample(
                dataset,
                signs=sign_arr,
                n_jobs=n_jobs,
                chunk_size=chunk_size,
                blas_threads=blas_threads,
            ),
            repeat=repeat,
            warmup=warmup,
        ),
    }


def _r_benchmarks(
    beta_2d: NDArray[np.float64],
    var_2d: NDArray[np.float64],
    signs: NDArray[np.int32],
    *,
    repeat: int,
    warmup: int,
    r_inner: int,
    fmrigds_source: str | None,
) -> dict[str, Any]:
    rscript = shutil.which("Rscript")
    if rscript is None:
        return {"r_available": 0.0}

    with tempfile.TemporaryDirectory(prefix="fmrimod-group-bench-") as tmp:
        tmp_path = Path(tmp)
        beta_path = tmp_path / "beta.csv"
        var_path = tmp_path / "var.csv"
        signs_path = tmp_path / "signs.csv"
        script_path = tmp_path / "bench.R"
        np.savetxt(beta_path, beta_2d, delimiter=",")
        np.savetxt(var_path, var_2d, delimiter=",")
        np.savetxt(signs_path, signs.astype(np.int32), fmt="%d", delimiter=",")
        source_arg = "" if fmrigds_source is None else fmrigds_source
        script_path.write_text(
            f"""
args <- commandArgs(trailingOnly = TRUE)
source_path <- "{source_arg}"
if (nzchar(source_path)) {{
  pkgload::load_all(source_path, quiet = TRUE)
}} else {{
  suppressPackageStartupMessages(library(fmrigds))
}}
beta <- as.matrix(read.csv("{beta_path}", header = FALSE))
var <- as.matrix(read.csv("{var_path}", header = FALSE))
signs <- as.matrix(read.csv("{signs_path}", header = FALSE))
storage.mode(signs) <- "integer"
repeat_n <- {int(repeat)}
warmup_n <- {int(warmup)}
inner_n <- {int(r_inner)}
bench <- function(expr) {{
  for (i in seq_len(warmup_n)) {{
    for (j in seq_len(inner_n)) force(eval(expr))
  }}
  vals <- numeric(repeat_n)
  for (i in seq_len(repeat_n)) {{
    vals[[i]] <- system.time(for (j in seq_len(inner_n)) force(eval(expr)))[["elapsed"]] / inner_n
  }}
  min(vals)
}}
meta_fe_s <- bench(quote(fmrigds:::meta_fe_cpp(beta, var, min_subj = 2L, eps = 1e-12, tail = 0L)))
perm_s <- bench(quote(fmrigds:::perm_onesample_t_cpp(beta, signs, tail = 0L, min_subj = 2L)))
cat(sprintf("r_meta_fe_s %.9f\\n", meta_fe_s))
cat(sprintf("r_perm_onesample_s %.9f\\n", perm_s))
""",
            encoding="utf-8",
        )
        proc = subprocess.run(
            [rscript, str(script_path)],
            check=False,
            capture_output=True,
            text=True,
        )
    if proc.returncode != 0:
        return {
            "r_available": 0.0,
            "r_error": proc.stderr.strip() or proc.stdout.strip(),
        }
    out: dict[str, float] = {"r_available": 1.0}
    for line in proc.stdout.splitlines():
        name, _, value = line.partition(" ")
        if name and value:
            out[name] = float(value)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=int, default=1000)
    parser.add_argument("--subjects", type=int, default=24)
    parser.add_argument("--permutations", type=int, default=512)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--r-inner", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260513)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--fmrigds-source", default="/Users/bbuchsbaum/code/fmrigds")
    parser.add_argument("--skip-r", action="store_true")
    args = parser.parse_args()

    beta_2d, var_2d, signs = _make_data(
        n_features=args.features,
        n_subjects=args.subjects,
        n_perm=args.permutations,
        seed=args.seed,
    )
    results: dict[str, Any] = {
        "schema_version": "group-reducer-fmrigds-benchmark/v1",
        "features": args.features,
        "subjects": args.subjects,
        "permutations": args.permutations,
        "repeat": args.repeat,
        "warmup": args.warmup,
        "r_inner": args.r_inner,
        "n_jobs": args.n_jobs,
        "chunk_size": args.chunk_size,
        "blas_threads": args.blas_threads,
    }
    results.update(
        _python_benchmarks(
            beta_2d,
            var_2d,
            signs,
            repeat=args.repeat,
            warmup=args.warmup,
            n_jobs=args.n_jobs,
            chunk_size=args.chunk_size,
            blas_threads=args.blas_threads,
        )
    )
    if not args.skip_r:
        results.update(
            _r_benchmarks(
                beta_2d,
                var_2d,
                signs,
                repeat=args.repeat,
                warmup=args.warmup,
                r_inner=args.r_inner,
                fmrigds_source=args.fmrigds_source,
            )
        )
    if results.get("r_available") == 1.0:
        results["meta_fe_slowdown_vs_fmrigds"] = _ratio(
            results["python_meta_fe_s"],
            results["r_meta_fe_s"],
        )
        results["perm_serial_slowdown_vs_fmrigds"] = _ratio(
            results["python_perm_onesample_serial_s"],
            results["r_perm_onesample_s"],
        )
        results["perm_parallel_slowdown_vs_fmrigds"] = _ratio(
            results["python_perm_onesample_parallel_s"],
            results["r_perm_onesample_s"],
        )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
