"""R-oracle parity tests for native :mod:`fmrimod.group` reducers.

These tests compare the native eager reducers against the explicit
``fmrigds-r`` oracle backend. They auto-skip when R/fmrigds is unavailable.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import pytest

from fmrimod.dataset import group_data_from_csv
from fmrimod.group import (
    SampleLabelSpace,
    group_dataset,
    group_dataset_from_group_data,
    reduce,
)
from fmrimod.stats import GroupFitRequest, fmrigds_backend_available, group_fit

_R_KERNEL_ORACLE = r"""
args <- commandArgs(trailingOnly = TRUE)
request_path <- args[[1L]]
result_path <- args[[2L]]
source_dir <- if (length(args) >= 3L) args[[3L]] else ""

`%||%` <- function(x, y) if (is.null(x)) y else x

if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("jsonlite is required for fmrigds kernel oracle", call. = FALSE)
}

load_fmrigds <- function(source_dir = "") {
  if (nzchar(source_dir)) {
    if (!requireNamespace("pkgload", quietly = TRUE)) {
      stop("pkgload is required to load fmrigds from source directory", call. = FALSE)
    }
    suppressWarnings(pkgload::load_all(source_dir, quiet = TRUE, export_all = FALSE, helpers = FALSE))
  }
  if (!("fmrigds" %in% loadedNamespaces()) && !requireNamespace("fmrigds", quietly = TRUE)) {
    stop("fmrigds package is not available", call. = FALSE)
  }
  asNamespace("fmrigds")
}

payload <- jsonlite::fromJSON(request_path, simplifyVector = FALSE)
ns <- load_fmrigds(source_dir)
get_fn <- function(name) get(name, envir = ns)
has_fn <- function(name) exists(name, envir = ns, mode = "function", inherits = FALSE)

compiled_names <- c(
  "meta_fe_reg_cpp",
  "meta_re_reg_dl_cpp",
  "stouffer_combine_cpp",
  "fisher_combine_cpp",
  "lancaster_combine_cpp",
  "perm_onesample_t_cpp",
  "perm_twosample_t_cpp"
)
compiled <- as.list(stats::setNames(
  vapply(compiled_names, has_fn, logical(1)),
  compiled_names
))

write_payload <- function(result = NULL, missing = NULL) {
  encode_values <- function(x) {
    lapply(as.numeric(x), function(v) {
      if (is.na(v)) return("__NA__")
      if (is.infinite(v) && v > 0) return("__INF__")
      if (is.infinite(v) && v < 0) return("__NEGINF__")
      v
    })
  }
  pack <- function(x) {
    d <- dim(x)
    if (is.null(d)) d <- length(x)
    list(data = encode_values(x), dim = as.integer(d))
  }
  packed <- if (is.null(result)) NULL else lapply(result, pack)
  out <- list(result = packed, compiled = compiled, missing = missing)
  writeLines(jsonlite::toJSON(out, auto_unbox = TRUE, na = "null", digits = 16), result_path)
}

flatten_values <- function(x) {
  v <- unlist(x, use.names = FALSE)
  if (is.character(v)) v[v == "__NA__"] <- NA_character_
  v
}

as_matrix <- function(obj) {
  matrix(
    as.numeric(flatten_values(obj$data)),
    nrow = as.integer(obj$nrow),
    ncol = as.integer(obj$ncol),
    byrow = TRUE
  )
}

as_int_matrix <- function(obj) {
  matrix(
    as.integer(flatten_values(obj$data)),
    nrow = as.integer(obj$nrow),
    ncol = as.integer(obj$ncol),
    byrow = TRUE
  )
}

kernel <- payload$kernel
required <- switch(
  kernel,
  "combine:stouffer" = "stouffer_combine_cpp",
  "combine:fisher" = "fisher_combine_cpp",
  "combine:lancaster" = "lancaster_combine_cpp",
  "meta:fe_reg" = "meta_fe_reg_cpp",
  "meta:re_reg" = "meta_re_reg_dl_cpp",
  "perm:onesample" = "perm_onesample_t_cpp",
  "perm:twosample" = "perm_twosample_t_cpp",
  NULL
)
if (!is.null(required) && !isTRUE(compiled[[required]])) {
  write_payload(missing = required)
  quit(status = 0L)
}

result <- switch(
  kernel,
  "combine:stouffer" = {
    opts <- list(min_subjects = as.integer(payload$min_subjects %||% 1L))
    if (!is.null(payload$weights)) opts$weights <- as.numeric(unlist(payload$weights))
    get_fn("core_stouffer_kernel")(z = as_matrix(payload$z), opts = opts)
  },
  "combine:fisher" = {
    opts <- list(min_subjects = as.integer(payload$min_subjects %||% 1L))
    get_fn("core_fisher_kernel")(p = as_matrix(payload$p), opts = opts)
  },
  "combine:lancaster" = {
    opts <- list(min_subjects = as.integer(payload$min_subjects %||% 1L))
    get_fn("core_lancaster_kernel")(
      p = as_matrix(payload$p),
      dfw = as.numeric(unlist(payload$dfw)),
      opts = opts
    )
  },
  "meta:fe_reg" = get_fn("meta_fe_reg_cpp")(
    beta = as_matrix(payload$beta),
    var = as_matrix(payload$var),
    X = as_matrix(payload$X),
    min_subj = as.integer(payload$min_subjects %||% 2L),
    eps = payload$eps %||% 1e-12
  ),
  "meta:re_reg" = get_fn("meta_re_reg_dl_cpp")(
    beta = as_matrix(payload$beta),
    var = as_matrix(payload$var),
    X = as_matrix(payload$X),
    min_subj = as.integer(payload$min_subjects %||% 2L),
    eps = payload$eps %||% 1e-12
  ),
  "perm:onesample" = get_fn("core_perm_onesample_kernel")(
    beta = as_matrix(payload$beta),
    opts = list(
      signs = as_int_matrix(payload$signs),
      alternative = "two.sided",
      min_subjects = as.integer(payload$min_subjects %||% 2L)
    )
  ),
  "perm:twosample" = get_fn("core_perm_twosample_kernel")(
    beta = as_matrix(payload$beta),
    opts = list(
      group = as.integer(unlist(payload$group)),
      group_mat = as_int_matrix(payload$group_mat),
      alternative = "two.sided",
      variance = payload$variance %||% "welch",
      min_group = as.integer(payload$min_group %||% 2L)
    )
  ),
  stop("unknown kernel: ", kernel, call. = FALSE)
)
write_payload(result = result)
"""


def _candidate_fmrigds_source() -> str | None:
    env = os.environ.get("FMRIGDS_SOURCE_DIR", "").strip()
    if env and Path(env).exists():
        return env
    candidate = Path.home() / "code" / "fmrigds"
    if candidate.exists():
        return str(candidate)
    return None


def _skip_if_no_fmrigds() -> dict[str, str]:
    source = _candidate_fmrigds_source()
    ok, reason = fmrigds_backend_available(fmrigds_source=source)
    if not ok:
        pytest.skip(f"fmrigds unavailable: {reason}")
    opts: dict[str, str] = {}
    if source is not None:
        opts["fmrigds_source"] = source
    return opts


def _matrix_payload(matrix: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(matrix)
    if arr.ndim != 2:
        raise AssertionError("oracle matrices must be 2-D")
    finite = np.isfinite(arr.astype(np.float64))
    data = arr.astype(object)
    data[~finite] = "__NA__"
    return {"data": data.tolist(), "nrow": arr.shape[0], "ncol": arr.shape[1]}


def _r_array(value: dict[str, Any]) -> np.ndarray:
    dim_raw = value["dim"]
    dim = (
        (int(dim_raw),) if isinstance(dim_raw, int) else tuple(int(x) for x in dim_raw)
    )
    data_raw = value["data"]
    data = data_raw if isinstance(data_raw, list) else [data_raw]

    def decode(item: Any) -> float:
        if item == "__NA__" or item is None:
            return np.nan
        if item == "__INF__":
            return np.inf
        if item == "__NEGINF__":
            return -np.inf
        return float(item)

    return np.asarray([decode(item) for item in data], dtype=np.float64).reshape(
        dim, order="F"
    )


def _run_fmrigds_kernel_oracle(kernel: str, **payload: Any) -> dict[str, Any]:
    backend_opts = _skip_if_no_fmrigds()
    request = {"kernel": kernel, **payload}
    with tempfile.TemporaryDirectory(prefix="fmrimod-fmrigds-kernel-") as td:
        tdir = Path(td)
        request_path = tdir / "request.json"
        result_path = tdir / "result.json"
        request_path.write_text(json.dumps(request), encoding="utf-8")
        cmd = [
            "Rscript",
            "-e",
            _R_KERNEL_ORACLE,
            str(request_path),
            str(result_path),
            backend_opts.get("fmrigds_source", ""),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if proc.returncode != 0:
            pytest.fail(
                proc.stderr.strip()
                or proc.stdout.strip()
                or "fmrigds kernel oracle failed"
            )
        result = json.loads(result_path.read_text(encoding="utf-8"))
    missing = result.get("missing")
    if missing:
        pytest.skip(f"fmrigds compiled kernel unavailable: {missing}")
    return result


def _matrix_group_dataset(assays: dict[str, np.ndarray]):
    first = next(iter(assays.values()))
    n_subjects, n_features = np.asarray(first).shape
    shaped = {
        name: np.asarray(values, dtype=np.float64).T[:, :, np.newaxis]
        for name, values in assays.items()
    }
    return group_dataset(
        shaped,
        space=SampleLabelSpace([f"f{i + 1}" for i in range(n_features)]),
        subjects=[f"s{i + 1}" for i in range(n_subjects)],
        contrasts=["c1"],
    )


def _assay_vector(dataset: Any, name: str) -> np.ndarray:
    return dataset.assay(name)[:, 0, 0]


def _coef_matrix(dataset: Any, names: list[str], prefix: str) -> np.ndarray:
    return np.vstack([_assay_vector(dataset, f"{prefix}:{name}") for name in names])


def _assert_r_close(actual: np.ndarray, oracle: dict[str, Any], name: str) -> None:
    np.testing.assert_allclose(
        actual,
        _r_array(oracle["result"][name]),
        rtol=1e-6,
        atol=1e-8,
        equal_nan=True,
    )


def _make_csv_group_data() -> Any:
    rows: list[dict[str, Any]] = []
    for roi, offset in (("r1", 0.0), ("r2", 0.08)):
        for subject, beta, se in (
            ("s1", 0.20, 0.10),
            ("s2", 0.10, 0.20),
            ("s3", 0.30, 0.10),
            ("s4", 0.25, 0.10),
            ("s5", 0.15, 0.15),
        ):
            rows.append(
                {
                    "subject": subject,
                    "roi": roi,
                    "beta": beta + offset,
                    "se": se,
                }
            )
    df = pd.DataFrame(rows)
    return group_data_from_csv(
        df,
        effect_cols={"beta": "beta", "se": "se"},
        subject_col="subject",
        roi_col="roi",
    )


@pytest.mark.cross_test
@pytest.mark.parametrize(
    ("native_method", "effects", "tau2"),
    [
        ("meta:fe", "fixed", "pm"),
        ("meta:re", "random", "dl"),
    ],
)
def test_native_meta_reducers_match_fmrigds_oracle(
    native_method: str,
    effects: Literal["fixed", "random"],
    tau2: Literal["pm", "dl"],
) -> None:
    backend_opts = _skip_if_no_fmrigds()
    gd = _make_csv_group_data()
    ds = group_dataset_from_group_data(gd)

    native = reduce(ds, method=native_method)
    oracle = group_fit(
        GroupFitRequest(
            data=gd,
            model="meta",
            effects=effects,
            tau2=tau2,
            backend="fmrigds-r",
            backend_options=backend_opts,
        )
    )

    np.testing.assert_allclose(
        native.assay("beta_g")[:, 0, 0], oracle.estimate[:, 0], rtol=1e-6, atol=1e-8
    )
    np.testing.assert_allclose(
        native.assay("se_g")[:, 0, 0], oracle.se[:, 0], rtol=1e-6, atol=1e-8
    )
    np.testing.assert_allclose(
        native.assay("z_g")[:, 0, 0], oracle.statistic[:, 0], rtol=1e-6, atol=1e-8
    )
    np.testing.assert_allclose(
        native.assay("p_g")[:, 0, 0], oracle.p[:, 0], rtol=1e-6, atol=1e-8
    )
    if native_method == "meta:re":
        assert oracle.tau2 is not None
        np.testing.assert_allclose(
            native.assay("tau2")[:, 0, 0],
            np.asarray(oracle.tau2, dtype=np.float64).ravel(),
            rtol=1e-6,
            atol=1e-8,
        )


@pytest.mark.cross_test
def test_native_combiners_match_fmrigds_compiled_kernels() -> None:
    z = np.array(
        [
            [1.0, -0.5, np.nan],
            [2.0, 0.2, 1.1],
            [np.nan, 0.7, -0.4],
            [-1.0, 1.5, 0.3],
        ],
        dtype=np.float64,
    )
    p = np.array(
        [
            [0.01, 0.0, 0.20],
            [0.20, 0.50, 0.80],
            [np.nan, 0.90, 1.0],
            [0.07, 0.03, 0.40],
        ],
        dtype=np.float64,
    )

    stouffer = reduce(
        _matrix_group_dataset({"z": z}), method="combine:stouffer", weights=[1, 2, 1, 2]
    )
    stouffer_oracle = _run_fmrigds_kernel_oracle(
        "combine:stouffer",
        z=_matrix_payload(z),
        weights=[1, 2, 1, 2],
    )
    _assert_r_close(_assay_vector(stouffer, "z_g"), stouffer_oracle, "z_g")
    _assert_r_close(_assay_vector(stouffer, "p_g"), stouffer_oracle, "p_g")

    fisher = reduce(_matrix_group_dataset({"p": p}), method="combine:fisher")
    fisher_oracle = _run_fmrigds_kernel_oracle("combine:fisher", p=_matrix_payload(p))
    for assay in ("chi2", "df", "p_g"):
        _assert_r_close(_assay_vector(fisher, assay), fisher_oracle, assay)

    lancaster = reduce(
        _matrix_group_dataset({"p": p}), method="combine:lancaster", dfw=[1, 2, 0, 3]
    )
    lancaster_oracle = _run_fmrigds_kernel_oracle(
        "combine:lancaster",
        p=_matrix_payload(p),
        dfw=[1, 2, 0, 3],
    )
    for assay in ("chi2", "df", "p_g"):
        _assert_r_close(_assay_vector(lancaster, assay), lancaster_oracle, assay)


@pytest.mark.cross_test
@pytest.mark.parametrize("method", ["meta:fe_reg", "meta:re_reg"])
def test_native_meta_regressions_match_fmrigds_compiled_kernels(method: str) -> None:
    beta = np.array(
        [
            [1.0, 2.0, 0.5],
            [2.0, 3.0, 1.5],
            [3.0, 3.5, 2.0],
            [4.0, 5.0, 2.5],
            [5.0, 5.3, 3.0],
        ],
        dtype=np.float64,
    )
    var = np.array(
        [
            [0.20, 0.30, 0.25],
            [0.25, 0.20, 0.30],
            [0.30, 0.25, 0.20],
            [0.22, 0.28, 0.26],
            [0.27, 0.24, 0.23],
        ],
        dtype=np.float64,
    )
    X = np.column_stack([np.ones(beta.shape[0]), np.linspace(-1.0, 1.0, beta.shape[0])])
    native = reduce(
        _matrix_group_dataset({"beta": beta, "var": var}), method=method, X=X
    )
    oracle = _run_fmrigds_kernel_oracle(
        method,
        beta=_matrix_payload(beta),
        var=_matrix_payload(var),
        X=_matrix_payload(X),
    )

    _assert_r_close(_coef_matrix(native, ["x0", "x1"], "coef"), oracle, "coef")
    _assert_r_close(_coef_matrix(native, ["x0", "x1"], "se_coef"), oracle, "se_coef")
    _assert_r_close(_assay_vector(native, "Q"), oracle, "Q")
    _assert_r_close(_assay_vector(native, "df_res"), oracle, "df_res")
    if method == "meta:re_reg":
        _assert_r_close(_assay_vector(native, "tau2"), oracle, "tau2")


@pytest.mark.cross_test
def test_native_permutation_reducers_match_fmrigds_compiled_kernels() -> None:
    beta = np.array(
        [
            [1.0, 2.0, 1.5],
            [2.0, 2.5, 1.2],
            [3.0, 3.5, 2.4],
            [4.0, 4.5, 2.2],
            [5.0, 5.2, 3.3],
            [6.0, 6.1, 3.1],
        ],
        dtype=np.float64,
    )
    signs = np.array(
        [
            [1, 1, 1, 1, 1, 1],
            [-1, 1, 1, 1, 1, 1],
            [1, -1, 1, -1, 1, 1],
            [1, 1, -1, 1, -1, 1],
            [-1, -1, 1, 1, -1, 1],
        ],
        dtype=np.int8,
    )
    onesample = reduce(
        _matrix_group_dataset({"beta": beta}), method="perm:onesample", signs=signs
    )
    onesample_oracle = _run_fmrigds_kernel_oracle(
        "perm:onesample",
        beta=_matrix_payload(beta),
        signs=_matrix_payload(signs),
    )
    for assay in ("beta_g", "se_g", "t_g", "df", "p_g", "p_perm", "p_fwer"):
        _assert_r_close(_assay_vector(onesample, assay), onesample_oracle, assay)

    group = [0, 0, 0, 1, 1, 1]
    group_mat = np.array(
        [
            [0, 0, 0, 1, 1, 1],
            [0, 0, 1, 0, 1, 1],
            [1, 0, 0, 1, 0, 1],
            [1, 1, 0, 0, 0, 1],
            [0, 1, 1, 0, 1, 0],
        ],
        dtype=np.int8,
    )
    twosample = reduce(
        _matrix_group_dataset({"beta": beta}),
        method="perm:twosample",
        group=group,
        group_mat=group_mat,
        variance="welch",
    )
    twosample_oracle = _run_fmrigds_kernel_oracle(
        "perm:twosample",
        beta=_matrix_payload(beta),
        group=group,
        group_mat=_matrix_payload(group_mat),
        variance="welch",
    )
    for assay in ("beta_g", "se_g", "t_g", "df", "p_g", "p_perm", "p_fwer"):
        _assert_r_close(_assay_vector(twosample, assay), twosample_oracle, assay)
