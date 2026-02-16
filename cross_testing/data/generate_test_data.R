# Generate test cases for cross-testing
library(fmrihrf)
library(jsonlite)

# Generate test cases
generate_test_cases <- function() {
  test_cases <- list()
  
  # HRF evaluations
  t <- seq(0, 30, length.out = 100)
  test_cases$hrf_evaluations <- list(
    spmg1 = list(
      t = t,
      result = hrf_spmg1(t)
    ),
    spmg2 = list(
      t = t,
      result = hrf_spmg2(t)
    ),
    spmg3 = list(
      t = t,
      result = hrf_spmg3(t)
    ),
    gamma = list(
      t = t,
      params = list(shape = 6, rate = 1),
      result = hrf_gamma(t, shape = 6, rate = 1)
    ),
    gaussian = list(
      t = t,
      params = list(mean = 6, sd = 2),
      result = hrf_gaussian(t, mean = 6, sd = 2)
    )
  )
  
  # Regressor examples
  sf <- sampling_frame(100, TR = 2.0)
  reg <- regressor(
    onsets = c(10, 30, 50),
    hrf = HRF_SPMG1,
    duration = 2
  )
  
  test_cases$regressor_example <- list(
    sampling_frame = list(
      blocklens = 100,
      TR = 2.0
    ),
    regressor = list(
      onsets = c(10, 30, 50),
      duration = 2
    ),
    result = evaluate(reg, samples(sf))
  )
  
  # Regressor set example
  rset <- regressor_set(
    onsets = c(10, 20, 30, 40),
    fac = as.factor(c("A", "B", "A", "B")),
    hrf = HRF_SPMG1
  )
  design <- regressor_matrix(rset, sf)
  
  test_cases$regressor_set_example <- list(
    onsets = c(10, 20, 30, 40),
    conditions = c("A", "B", "A", "B"),
    design_matrix = design,
    n_conditions = length(levels(attr(rset, 'fac')))
  )
  
  # Multi-block example
  sf_multi <- sampling_frame(
    blocklens = c(50, 50, 50),
    TR = c(2.0, 2.0, 1.5)
  )
  
  test_cases$multi_block <- list(
    blocklens = c(50, 50, 50),
    TR = c(2.0, 2.0, 1.5),
    samples = samples(sf_multi),
    n_blocks = 3,
    n_scans = 150
  )
  
  # Penalty matrix example
  hrf_bspline <- get_hrf(HRF_BSPLINE)
  penalty <- penalty_matrix(hrf_bspline, order = 2)
  
  test_cases$penalty_matrix <- list(
    hrf_type = "bspline",
    order = 2,
    matrix = penalty,
    dimension = dim(penalty)
  )
  
  # Save as JSON
  write_json(test_cases, "test_cases.json", pretty = TRUE, auto_unbox = TRUE)
  
  # Also save as RDS for exact numerical preservation
  saveRDS(test_cases, "test_cases.rds")
  
  cat("Test cases generated successfully!\n")
  cat("- JSON file: test_cases.json\n")
  cat("- RDS file: test_cases.rds\n")
}

# Generate the test cases
generate_test_cases()