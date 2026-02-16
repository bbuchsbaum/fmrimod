#!/usr/bin/env Rscript

# Generate comprehensive test fixtures for pyfmridesign numerical validation
# This script creates R golden reference data using fmridesign and saves as JSON
# for comparison with the Python pyfmridesign implementation.

library(fmridesign)
library(fmrihrf)
library(jsonlite)

# Set seed for reproducibility
set.seed(42)

# Output directory
fixture_dir <- "fixtures"
if (!dir.exists(fixture_dir)) {
  dir.create(fixture_dir, recursive = TRUE)
}

# Helper function to save fixtures as JSON with high precision
save_json <- function(obj, name) {
  filepath <- file.path(fixture_dir, paste0(name, ".json"))
  write_json(obj, filepath, auto_unbox = TRUE, pretty = TRUE, digits = 10)
  cat("Saved:", name, ".json\n")
}

# Helper to extract design matrix as list
dm_to_list <- function(dm) {
  list(
    matrix = as.matrix(dm),
    colnames = colnames(dm),
    nrow = nrow(dm),
    ncol = ncol(dm)
  )
}

cat("=== Generating R fixtures for numerical validation ===\n\n")

# ============================================================================
# Fixture 1: Simple HRF convolution (spmg1) - 2 conditions
# ============================================================================
cat("Fixture 1: Simple HRF convolution (spmg1)\n")

events1 <- data.frame(
  onset = c(2, 6, 12, 18, 24, 30),
  condition = factor(c("A", "B", "A", "B", "A", "B")),
  run = 1
)

sf1 <- sampling_frame(blocklens = 100, TR = 2.0)

model1 <- event_model(
  onset ~ hrf(condition, basis = "spmg1"),
  data = events1,
  block = ~run,
  sampling_frame = sf1,
  durations = 1,
  precision = 0.3
)

dm1 <- design_matrix(model1)

fixture1 <- list(
  description = "Simple 2-condition model with SPM canonical HRF (spmg1)",
  events = events1,
  sampling_frame = list(
    blocklens = as.integer(blocklens(sf1)),
    TR = 2.0
  ),
  design_matrix = dm_to_list(dm1),
  model_info = list(
    n_events = nrow(events1),
    n_conditions = length(unique(events1$condition)),
    hrf_basis = "spmg1",
    durations = 1,
    precision = 0.3
  )
)

save_json(fixture1, "fixture_simple_hrf")

# ============================================================================
# Fixture 2: HRF convolution with derivatives (spmg2)
# ============================================================================
cat("Fixture 2: HRF with temporal derivative (spmg2)\n")

events2 <- data.frame(
  onset = c(2, 6, 12, 18, 24, 30),
  condition = factor(c("A", "B", "A", "B", "A", "B")),
  run = 1
)

sf2 <- sampling_frame(blocklens = 100, TR = 2.0)

model2 <- event_model(
  onset ~ hrf(condition, basis = "spmg2"),
  data = events2,
  block = ~run,
  sampling_frame = sf2,
  durations = 1,
  precision = 0.3
)

dm2 <- design_matrix(model2)

fixture2 <- list(
  description = "2-condition model with SPM canonical + temporal derivative (spmg2)",
  events = events2,
  sampling_frame = list(
    blocklens = as.integer(blocklens(sf2)),
    TR = 2.0
  ),
  design_matrix = dm_to_list(dm2),
  model_info = list(
    n_events = nrow(events2),
    n_conditions = length(unique(events2$condition)),
    hrf_basis = "spmg2",
    nbasis = 2,
    durations = 1,
    precision = 0.3
  )
)

save_json(fixture2, "fixture_spmg2")

# ============================================================================
# Fixture 3: Multi-block design
# ============================================================================
cat("Fixture 3: Multi-block design\n")

events3 <- data.frame(
  onset = c(5, 15, 25, 35, 45,    # Block 1
            5, 15, 25, 35, 45),    # Block 2 (same relative onsets)
  condition = factor(rep(c("A", "B", "A", "B", "A"), 2)),
  run = rep(1:2, each = 5)
)

sf3 <- sampling_frame(blocklens = c(50, 50), TR = 2.0)

model3 <- event_model(
  onset ~ hrf(condition, basis = "spmg1"),
  data = events3,
  block = ~run,
  sampling_frame = sf3,
  precision = 0.3
)

dm3 <- design_matrix(model3)
dm3_block1 <- design_matrix(model3, blockid = 1)
dm3_block2 <- design_matrix(model3, blockid = 2)

fixture3 <- list(
  description = "Multi-block design to verify HRF isolation across blocks",
  events = events3,
  sampling_frame = list(
    blocklens = as.integer(blocklens(sf3)),
    TR = 2.0
  ),
  design_matrix = dm_to_list(dm3),
  block1_matrix = dm_to_list(dm3_block1),
  block2_matrix = dm_to_list(dm3_block2),
  model_info = list(
    n_blocks = 2,
    n_events_per_block = 5
  )
)

save_json(fixture3, "fixture_multiblock")

# ============================================================================
# Fixture 4: Contrast weights (unit and pair)
# ============================================================================
cat("Fixture 4: Contrast weights\n")

events4 <- data.frame(
  onset = c(2, 6, 12, 18, 24, 30),
  condition = factor(c("A", "B", "A", "B", "A", "B")),
  run = 1
)

sf4 <- sampling_frame(blocklens = 100, TR = 2.0)

cset4 <- contrast_set(
  unit_A = unit_contrast(~ condition == "A", name = "unit_A"),
  pair_AB = pair_contrast(~ condition == "A", ~ condition == "B", name = "pair_AB")
)

model4 <- event_model(
  onset ~ hrf(condition, basis = "spmg1", contrasts = cset4),
  data = events4,
  block = ~run,
  sampling_frame = sf4,
  durations = 1,
  precision = 0.3
)

cweights4 <- contrast_weights(model4)

fixture4 <- list(
  description = "Contrast weights for unit and pair contrasts",
  events = events4,
  sampling_frame = list(
    blocklens = as.integer(blocklens(sf4)),
    TR = 2.0
  ),
  design_matrix = dm_to_list(design_matrix(model4)),
  contrasts = lapply(cweights4, function(cw) {
    list(
      name = cw$name,
      weights = as.numeric(cw$weights),
      condnames = rownames(cw$weights)
    )
  })
)

save_json(fixture4, "fixture_contrasts")

# ============================================================================
# Fixture 5: Poly contrast
# ============================================================================
cat("Fixture 5: Poly contrast\n")

events5 <- data.frame(
  onset = c(5, 15, 25, 35, 45, 55, 65, 75),
  condition = factor(rep(c("A", "B", "C", "D"), 2)),
  run = 1
)

sf5 <- sampling_frame(blocklens = 100, TR = 2.0)

vmap <- list(A = 1, B = 2, C = 3, D = 4)
cset5 <- contrast_set(
  poly_lin = poly_contrast(~ condition, name = "poly_linear", degree = 1, value_map = vmap),
  poly_quad = poly_contrast(~ condition, name = "poly_quadratic", degree = 2, value_map = vmap)
)

model5 <- event_model(
  onset ~ hrf(condition, basis = "spmg1", contrasts = cset5),
  data = events5,
  block = ~run,
  sampling_frame = sf5,
  precision = 0.3
)

cweights5 <- contrast_weights(model5)

fixture5 <- list(
  description = "Polynomial contrasts (linear and quadratic) for 4-condition model",
  events = events5,
  sampling_frame = list(
    blocklens = as.integer(blocklens(sf5)),
    TR = 2.0
  ),
  design_matrix = dm_to_list(design_matrix(model5)),
  contrasts = lapply(cweights5, function(cw) {
    list(
      name = cw$name,
      weights = as.numeric(cw$weights),
      condnames = rownames(cw$weights)
    )
  })
)

save_json(fixture5, "fixture_poly_contrast")

# ============================================================================
# Fixture 6: Oneway and interaction contrasts (2x2 factorial)
# ============================================================================
cat("Fixture 6: Oneway and interaction contrasts\n")

events6 <- data.frame(
  onset = c(5, 15, 25, 35, 45, 55, 65, 75),
  factor1 = factor(rep(c("a1", "a1", "a2", "a2"), 2)),
  factor2 = factor(rep(c("b1", "b2", "b1", "b2"), 2)),
  run = 1
)

sf6 <- sampling_frame(blocklens = 100, TR = 2.0)

cset6 <- contrast_set(
  main_f1 = oneway_contrast(~ factor1, name = "main_factor1"),
  interact = interaction_contrast(~ factor1 * factor2, name = "f1_x_f2")
)

model6 <- event_model(
  onset ~ hrf(factor1, factor2, contrasts = cset6),
  data = events6,
  block = ~run,
  sampling_frame = sf6,
  precision = 0.3
)

dm6 <- design_matrix(model6)
cweights6 <- contrast_weights(model6)

fixture6 <- list(
  description = "Oneway and interaction contrasts for 2x2 factorial design",
  events = events6,
  sampling_frame = list(
    blocklens = as.integer(blocklens(sf6)),
    TR = 2.0
  ),
  design_matrix = dm_to_list(dm6),
  contrasts = lapply(cweights6, function(cw) {
    list(
      name = cw$name,
      weights = as.numeric(cw$weights),
      condnames = rownames(cw$weights),
      ncol = ncol(cw$weights)
    )
  })
)

save_json(fixture6, "fixture_factorial_contrasts")

# ============================================================================
# Fixture 7: F-contrasts
# ============================================================================
cat("Fixture 7: F-contrasts\n")

# Use the 4-condition model from fixture 5
fcons7 <- Fcontrasts(model5)

fixture7 <- list(
  description = "F-contrasts for 4-condition factor",
  events = events5,
  sampling_frame = list(
    blocklens = as.integer(blocklens(sf5)),
    TR = 2.0
  ),
  design_matrix = dm_to_list(design_matrix(model5)),
  fcontrasts = lapply(fcons7, function(fc) {
    list(
      matrix = as.matrix(fc),
      rownames = rownames(fc),
      colnames = colnames(fc),
      nrow = nrow(fc),
      ncol = ncol(fc)
    )
  })
)

save_json(fixture7, "fixture_fcontrasts")

# ============================================================================
# Fixture 8: Basis functions (Poly, BSpline, Scale)
# ============================================================================
cat("Fixture 8: Basis functions\n")

x_basis <- seq(0, 10, length.out = 100)

# Polynomial basis
poly_basis <- Poly(x_basis, degree = 3)
poly_mat <- poly_basis$y

# BSpline basis
bspline_basis <- BSpline(x_basis, degree = 3)
bspline_mat <- bspline_basis$y

# Scale
scale_basis <- Scale(x_basis)
scale_vals <- scale_basis$y

# RobustScale
robust_basis <- RobustScale(x_basis)
robust_vals <- robust_basis$y

fixture8 <- list(
  description = "Basis functions: Poly, BSpline, Scale, RobustScale",
  x = x_basis,
  poly = list(
    matrix = as.matrix(poly_mat),
    degree = 3,
    ncol = ncol(poly_mat)
  ),
  bspline = list(
    matrix = as.matrix(bspline_mat),
    degree = 3,
    ncol = ncol(bspline_mat)
  ),
  scale = list(
    values = as.numeric(scale_vals),
    center = attr(scale_vals, "scaled:center") %||% mean(x_basis),
    scale_factor = attr(scale_vals, "scaled:scale") %||% sd(x_basis)
  ),
  robust_scale = list(
    values = as.numeric(robust_vals),
    center = robust_basis$center,
    scale_factor = robust_basis$scale
  )
)

save_json(fixture8, "fixture_basis_functions")

# ============================================================================
# Fixture 9: Baseline model (polynomial drift + DCT basis)
# ============================================================================
cat("Fixture 9: Baseline model\n")

sf9 <- sampling_frame(blocklens = 200, TR = 2.0)

# Polynomial drift (degree 2)
bmod_poly <- baseline_model(basis = "poly", degree = 2, sframe = sf9)
dm_poly <- design_matrix(bmod_poly)

# Constant baseline
bmod_const <- baseline_model(basis = "constant", sframe = sf9)
dm_const <- design_matrix(bmod_const)

# DCT basis from R (internal function, use namespace)
n_timepoints <- 200
dct_mat <- fmridesign:::dctbasis(n_timepoints, p = 5)

fixture9 <- list(
  description = "Baseline model: polynomial drift and DCT basis",
  sampling_frame = list(
    blocklens = as.integer(blocklens(sf9)),
    TR = 2.0
  ),
  poly_drift = list(
    matrix = as.matrix(dm_poly),
    colnames = colnames(dm_poly),
    nrow = nrow(dm_poly),
    ncol = ncol(dm_poly),
    degree = 2
  ),
  constant_baseline = list(
    matrix = as.matrix(dm_const),
    colnames = colnames(dm_const),
    nrow = nrow(dm_const),
    ncol = ncol(dm_const)
  ),
  dct_basis = list(
    matrix = as.matrix(dct_mat),
    n = n_timepoints,
    p = 5,
    nrow = nrow(dct_mat),
    ncol = ncol(dct_mat)
  )
)

save_json(fixture9, "fixture_baseline")

# ============================================================================
# Fixture 10: Parametric modulation
# ============================================================================
cat("Fixture 10: Parametric modulation\n")

set.seed(42)  # Reset seed for RT values
events10 <- data.frame(
  onset = c(5, 15, 25, 35, 45, 55),
  condition = factor(rep("stim", 6)),
  RT = c(0.5, 1.2, 0.8, 1.5, 0.9, 1.1),
  run = 1
)

sf10 <- sampling_frame(blocklens = 100, TR = 2.0)

model10 <- event_model(
  onset ~ hrf(condition) + hrf(RT, basis = "spmg1"),
  data = events10,
  block = ~run,
  sampling_frame = sf10,
  precision = 0.3
)

dm10 <- design_matrix(model10)

fixture10 <- list(
  description = "Parametric modulation with continuous RT variable",
  events = events10,
  sampling_frame = list(
    blocklens = as.integer(blocklens(sf10)),
    TR = 2.0
  ),
  design_matrix = dm_to_list(dm10),
  model_info = list(
    categorical_var = "condition",
    continuous_var = "RT"
  )
)

save_json(fixture10, "fixture_parametric")

# ============================================================================
# Fixture 11: Variable durations
# ============================================================================
cat("Fixture 11: Variable durations\n")

events11 <- data.frame(
  onset = c(10, 30, 50, 70),
  condition = factor(c("short", "long", "short", "long")),
  duration = c(1, 5, 1, 5),
  run = 1
)

sf11 <- sampling_frame(blocklens = 100, TR = 2.0)

model11 <- event_model(
  onset ~ hrf(condition, basis = "spmg1"),
  data = events11,
  block = ~run,
  sampling_frame = sf11,
  durations = events11$duration,
  precision = 0.3
)

dm11 <- design_matrix(model11)

fixture11 <- list(
  description = "Variable event durations",
  events = events11,
  sampling_frame = list(
    blocklens = as.integer(blocklens(sf11)),
    TR = 2.0
  ),
  design_matrix = dm_to_list(dm11),
  model_info = list(
    durations = events11$duration
  )
)

save_json(fixture11, "fixture_durations")

# ============================================================================
# Fixture 12: Basis-expanded HRF (spmg3)
# ============================================================================
cat("Fixture 12: Basis-expanded HRF (spmg3)\n")

events12 <- data.frame(
  onset = c(10, 30, 50, 70),
  condition = factor(c("stim", "stim", "stim", "stim")),
  run = 1
)

sf12 <- sampling_frame(blocklens = 100, TR = 2.0)

model12 <- event_model(
  onset ~ hrf(condition, basis = "spmg3"),
  data = events12,
  block = ~run,
  sampling_frame = sf12,
  precision = 0.3
)

dm12 <- design_matrix(model12)

fixture12 <- list(
  description = "Basis-expanded HRF with SPMG3 (canonical + derivatives)",
  events = events12,
  sampling_frame = list(
    blocklens = as.integer(blocklens(sf12)),
    TR = 2.0
  ),
  design_matrix = dm_to_list(dm12),
  model_info = list(
    hrf_basis = "spmg3",
    nbasis = 3
  )
)

save_json(fixture12, "fixture_basis_expanded")

# ============================================================================
# Summary
# ============================================================================
cat("\n=== Fixture generation complete! ===\n")
fixture_files <- list.files(fixture_dir, pattern = "\\.json$")
cat("Generated", length(fixture_files), "JSON fixtures:\n")
for (f in sort(fixture_files)) {
  cat("  -", f, "\n")
}
cat("Fixtures saved in:", normalizePath(fixture_dir), "\n")
