#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2L) {
  stop("Usage: fmrigds_bridge.R <request.json> <result.json>", call. = FALSE)
}

request_path <- args[[1L]]
result_path <- args[[2L]]

`%||%` <- function(x, y) if (is.null(x)) y else x

if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("jsonlite is required for fmrigds bridge", call. = FALSE)
}

req <- jsonlite::fromJSON(request_path, simplifyVector = FALSE)

load_fmrigds <- function(source_dir = NULL) {
  if (!is.null(source_dir) && nzchar(source_dir)) {
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

ns <- load_fmrigds(req$backend_options$fmrigds_source %||% "")

gds_fn <- get("gds", envir = ns)
with_col_data_fn <- get("with_col_data", envir = ns)
reduce_fn <- get("reduce", envir = ns)
compute_fn <- get("compute", envir = ns)
assays_fn <- get("assays", envir = ns)
space_fn <- get("space", envir = ns)

input <- req$input
format <- input$format %||% ""

build_plan <- function() {
  if (identical(format, "csv")) {
    args <- list(
      source = input$path,
      format = "tabular",
      effect_cols = input$effect_cols,
      subject_col = input$subject_col %||% "subject",
      sample_col = input$sample_col %||% "sample",
      contrast_col = input$contrast_col %||% "contrast"
    )
    plan <- do.call(gds_fn, args)
    if (!is.null(input$covariates_path) && nzchar(input$covariates_path)) {
      cov <- utils::read.csv(input$covariates_path, stringsAsFactors = FALSE, check.names = FALSE)
      id_col <- input$covariates_id_col %||% "subject"
      if (!id_col %in% names(cov)) stop("Covariates id column not found: ", id_col, call. = FALSE)
      ids <- as.character(cov[[id_col]])
      rownames(cov) <- ids
      cov[[id_col]] <- NULL
      plan <- with_col_data_fn(plan, cov)
    }
    return(plan)
  }
  stop("Bridge currently executes only csv input format", call. = FALSE)
}

method <- req$method %||% "fe"
formula <- req$formula %||% "~ 1"
weights <- req$weights %||% "ivw"
if (identical(weights, "ivw")) weights <- "1/var"

if (identical(method, "fe")) {
  reduce_method <- if (identical(formula, "~ 1")) "meta:fe" else "meta:fe_reg"
} else if (identical(method, "dl")) {
  reduce_method <- if (identical(formula, "~ 1")) "meta:re" else "meta:re_reg"
} else {
  stop("Bridge supports only methods fe/dl", call. = FALSE)
}

plan <- build_plan()
plan <- reduce_fn(plan, method = reduce_method, weights = weights, formula = formula)
g <- compute_fn(plan)
ass <- assays_fn(g)

get_assay <- function(name) {
  if (name %in% names(ass)) ass[[name]] else NULL
}

as_vec <- function(arr) {
  if (is.null(arr)) return(NULL)
  as.numeric(arr[, 1, 1])
}

estimate <- NULL
se <- NULL
stat <- NULL
p <- NULL
predictor_names <- character()

coef_names <- grep("^coef:", names(ass), value = TRUE)
if (length(coef_names)) {
  predictor_names <- sub("^coef:", "", coef_names)
  n_feat <- length(as_vec(ass[[coef_names[[1L]]]]))
  estimate <- vapply(coef_names, function(nm) as_vec(ass[[nm]]), FUN.VALUE = numeric(n_feat))
  se_names <- paste0("se_coef:", predictor_names)
  if (!all(se_names %in% names(ass))) stop("Missing se_coef assays for regression result", call. = FALSE)
  se <- vapply(se_names, function(nm) as_vec(ass[[nm]]), FUN.VALUE = numeric(n_feat))
  z_names <- paste0("z_coef:", predictor_names)
  t_names <- paste0("t_coef:", predictor_names)
  p_names <- paste0("p_coef:", predictor_names)
  if (all(z_names %in% names(ass))) {
    stat <- vapply(z_names, function(nm) as_vec(ass[[nm]]), FUN.VALUE = numeric(n_feat))
  } else if (all(t_names %in% names(ass))) {
    stat <- vapply(t_names, function(nm) as_vec(ass[[nm]]), FUN.VALUE = numeric(n_feat))
  } else {
    stat <- estimate / se
  }
  if (all(p_names %in% names(ass))) {
    p <- vapply(p_names, function(nm) as_vec(ass[[nm]]), FUN.VALUE = numeric(n_feat))
  } else {
    p <- 2 * stats::pnorm(-abs(stat))
  }
} else {
  estimate <- as_vec(get_assay("beta")) %||% as_vec(get_assay("beta_g"))
  se <- as_vec(get_assay("se")) %||% as_vec(get_assay("se_g"))
  stat <- as_vec(get_assay("z")) %||% as_vec(get_assay("z_g")) %||% as_vec(get_assay("t"))
  p <- as_vec(get_assay("p")) %||% as_vec(get_assay("p_g"))
  if (is.null(estimate) || is.null(se) || is.null(stat) || is.null(p)) {
    stop("Could not extract required assays from fmrigds result", call. = FALSE)
  }
  predictor_names <- "(Intercept)"
  estimate <- matrix(estimate, ncol = 1L)
  se <- matrix(se, ncol = 1L)
  stat <- matrix(stat, ncol = 1L)
  p <- matrix(p, ncol = 1L)
}

tau2 <- as_vec(get_assay("tau2"))

space_obj <- tryCatch(space_fn(g), error = function(e) NULL)
feature_names <- NULL
if (!is.null(space_obj) && !is.null(space_obj$labels)) {
  feature_names <- as.character(space_obj$labels)
}
if (is.null(feature_names)) {
  feature_names <- paste0("f", seq_len(nrow(estimate)))
}

out <- list(
  estimate = unname(estimate),
  se = unname(se),
  statistic = unname(stat),
  p = unname(p),
  tau2 = if (is.null(tau2)) NULL else unname(tau2),
  predictor_names = as.character(predictor_names),
  feature_names = as.character(feature_names),
  model = req$model %||% "meta",
  method = method,
  formula = formula,
  metadata = list(
    reduce_method = reduce_method,
    bridge = "fmrigds_bridge_r"
  )
)

jsonlite::write_json(out, result_path, auto_unbox = TRUE, digits = NA, pretty = FALSE, null = "null")
