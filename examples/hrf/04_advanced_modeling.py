"""
Advanced HRF Modeling and Design - Python Tutorial
=================================================

This tutorial explores advanced features of fmrimod for systematic HRF modeling,
regularization, and experimental design, including:

- HRF libraries for parameter exploration
- Reconstruction matrices for basis coefficients
- Regressor sets for multi-condition designs
- Penalty matrices for regularization
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse import issparse
from fmrimod import (
    get_hrf, gen_hrf, hrf_library, reconstruction_matrix,
    penalty_matrix, regressor, regressor_set, SamplingFrame
)
from fmrimod.hrf.generators import gamma_generator
from fmrimod.hrf.functions import hrf_gamma, lag_hrf

# Set up plotting style
plt.style.use('seaborn-v0_8-darkgrid')

# %% HRF Libraries: Systematic Parameter Exploration

print("=== HRF Libraries ===")
print("\nCreating systematic collections of HRF variants...\n")

# Define parameter grid for gamma HRFs
gamma_params = pd.DataFrame({
    'shape': [4, 6, 8] * 3,
    'rate': [0.8] * 3 + [1.0] * 3 + [1.2] * 3
})
print("Gamma HRF parameter grid:")
print(gamma_params)

# Create HRF library
gamma_lib = hrf_library(gamma_generator, gamma_params)
print(f"\nCreated HRF library with {gamma_lib.nbasis} variants")
print(f"Library span: {gamma_lib.span} seconds")

# Evaluate and visualize
time_points = np.linspace(0, 20, 200)
gamma_responses = gamma_lib(time_points)

# Plot all HRF variants
plt.figure(figsize=(12, 8))
for i in range(gamma_responses.shape[1]):
    shape = gamma_params.iloc[i]['shape']
    rate = gamma_params.iloc[i]['rate']
    plt.plot(time_points, gamma_responses[:, i], 
             linewidth=2, label=f'Shape={shape}, Rate={rate}')

plt.xlabel('Time (seconds)')
plt.ylabel('HRF Response')
plt.title('Library of Gamma HRFs\nSystematic variation of shape and rate parameters')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% Library of Lagged SPM HRFs

# Create library with temporal lags
lag_values = np.arange(-2, 5, 1)
print(f"\nCreating library of lagged SPM HRFs with lags: {lag_values}")

# Create lagged HRFs manually (since we need to use lag_hrf)
hrf_spm = get_hrf("spmg1")
lagged_hrfs = []
for lag in lag_values:
    lagged_hrf = lag_hrf(hrf_spm, lag=lag)
    lagged_hrf.name = f"SPM_lag_{lag:+d}"
    lagged_hrfs.append(lagged_hrf)

# Evaluate all lagged HRFs
plt.figure(figsize=(12, 8))
for i, (hrf, lag) in enumerate(zip(lagged_hrfs, lag_values)):
    response = hrf(time_points)
    color = plt.cm.RdBu(i / (len(lag_values) - 1))
    plt.plot(time_points, response, linewidth=2, 
             color=color, label=f'Lag = {lag:+d}s')

plt.xlabel('Time (seconds)')
plt.ylabel('HRF Response')
plt.title('Library of Temporally Shifted SPM HRFs')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.show()

# %% Reconstruction Matrix

print("\n=== Reconstruction Matrix ===")
print("Converting basis coefficients back to HRF shapes...\n")

# Use SPMG3 (canonical + derivatives) as example
hrf_spmg3 = get_hrf("spmg3")
print(f"SPMG3 has {hrf_spmg3.nbasis} basis functions:")
print("  1. Canonical HRF")
print("  2. Temporal derivative")
print("  3. Dispersion derivative")

# Create reconstruction matrix
times_recon = np.linspace(0, 30, 150)
R = reconstruction_matrix(hrf_spmg3, times_recon)
print(f"\nReconstruction matrix shape: {R.shape}")
print(f"  Rows: {R.shape[0]} time points")
print(f"  Cols: {R.shape[1]} basis functions")

# Example: Different coefficient combinations
coef_sets = [
    ([1, 0, 0], "Canonical only"),
    ([1, 0.5, 0], "Canonical + temporal shift"),
    ([1, 0, 0.5], "Canonical + dispersion"),
    ([1, -0.3, 0.2], "Mixed adjustments")
]

plt.figure(figsize=(12, 10))
for i, (coeffs, label) in enumerate(coef_sets):
    plt.subplot(2, 2, i+1)
    
    # Reconstruct HRF shape
    hrf_shape = R @ coeffs
    
    plt.plot(times_recon, hrf_shape, 'b-', linewidth=2)
    plt.xlabel('Time (seconds)')
    plt.ylabel('HRF Response')
    plt.title(f'{label}\nCoeffs = {coeffs}')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

plt.tight_layout()
plt.show()

# %% Penalty Matrix for Regularization

print("\n=== Penalty Matrices ===")
print("Regularization for smooth HRF estimation...\n")

# B-spline basis for flexible HRF modeling
from fmrimod.hrf.functions import hrf_bspline
hrf_bs = gen_hrf(hrf_bspline, n_basis=10, span=25)

# Get penalty matrix (2nd order = penalize 2nd derivative)
P = penalty_matrix(hrf_bs, order=2)
print(f"Penalty matrix shape: {P.shape}")
print(f"Symmetric: {np.allclose(P, P.T)}")
print(f"Positive semi-definite: {np.all(np.linalg.eigvals(P) >= -1e-10)}")

# Visualize penalty matrix
plt.figure(figsize=(8, 6))
plt.imshow(P, cmap='RdBu_r', aspect='auto')
plt.colorbar(label='Penalty weight')
plt.title('B-spline Penalty Matrix (2nd order)')
plt.xlabel('Basis function index')
plt.ylabel('Basis function index')
plt.show()

# Example: Effect of regularization
# Simulate noisy data
true_hrf = get_hrf("spmg1")
times_fit = np.linspace(0, 25, 50)
y_true = true_hrf(times_fit)
y_noisy = y_true + 0.1 * np.random.randn(len(times_fit))

# B-spline basis matrix
B = hrf_bs(times_fit)

# Fit with different regularization strengths
lambdas = [0, 0.1, 1, 10]
plt.figure(figsize=(12, 8))

for i, lam in enumerate(lambdas):
    plt.subplot(2, 2, i+1)
    
    # Regularized least squares: minimize ||y - Bc||² + λ*c'Pc
    if lam == 0:
        coeffs = np.linalg.lstsq(B, y_noisy, rcond=None)[0]
    else:
        # Solve (B'B + λP)c = B'y
        coeffs = np.linalg.solve(B.T @ B + lam * P, B.T @ y_noisy)
    
    # Reconstruct
    y_fit = B @ coeffs
    
    # Plot
    plt.scatter(times_fit, y_noisy, alpha=0.5, s=20, label='Noisy data')
    plt.plot(times_fit, y_true, 'k--', linewidth=2, alpha=0.7, label='True HRF')
    plt.plot(times_fit, y_fit, 'r-', linewidth=2, label=f'Fit (λ={lam})')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Response')
    plt.title(f'Regularization: λ = {lam}')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% Regressor Sets for Complex Designs

print("\n=== Complex Experimental Design ===")
print("Managing multiple conditions with RegressorSet...\n")

# Simulate a 2x2 factorial design
np.random.seed(42)
n_trials = 40
trial_times = np.sort(np.random.uniform(0, 300, n_trials))

# Factors: Task (A/B) x Difficulty (Easy/Hard)
tasks = np.random.choice(['A', 'B'], n_trials)
difficulties = np.random.choice(['Easy', 'Hard'], n_trials)

# Create interaction labels
conditions = [f"{task}_{diff}" for task, diff in zip(tasks, difficulties)]

print(f"2x2 Factorial design:")
print(f"  Total trials: {n_trials}")
print(f"  Conditions: {np.unique(conditions)}")

# Create regressor set
reg_set = regressor_set(
    onsets=trial_times,
    fac=conditions,
    hrf="spmg1",
    duration=2.0  # 2-second trials
)

# Create sampling frame and design matrix
sf = SamplingFrame(blocklens=180, tr=2.0)  # 6-minute run
design = reg_set.evaluate(sf.samples)

# Plot design matrix
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, (level, ax) in enumerate(zip(sorted(reg_set.levels), axes)):
    ax.plot(sf.samples, design[:, i], linewidth=2)
    ax.set_title(f'Condition: {level}')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('BOLD Response')
    ax.grid(True, alpha=0.3)
    
    # Add event markers
    cond_times = trial_times[np.array(conditions) == level]
    for t in cond_times:
        ax.axvline(x=t, color='red', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# %% Multi-run Design with Different HRFs

print("\n=== Multi-run Experiment ===")
print("Different runs with different acquisition parameters...\n")

# Three runs with different parameters
run_configs = [
    {'length': 150, 'TR': 2.0, 'n_events': 15},
    {'length': 120, 'TR': 1.5, 'n_events': 20},
    {'length': 180, 'TR': 2.5, 'n_events': 12}
]

all_designs = []
cumulative_time = 0

plt.figure(figsize=(14, 8))

for run_idx, config in enumerate(run_configs):
    # Create sampling frame for this run
    sf_run = SamplingFrame(
        blocklens=config['length'],
        tr=config['TR'],
        start_time=cumulative_time
    )
    
    # Generate events for this run
    run_start = cumulative_time
    run_end = run_start + config['length'] * config['TR']
    event_times = np.sort(np.random.uniform(run_start + 10, run_end - 20, 
                                          config['n_events']))
    
    # Create regressor
    reg_run = regressor(event_times, hrf="spmg1")
    design_run = reg_run.evaluate(sf_run.samples)
    
    # Plot
    plt.plot(sf_run.samples, design_run, linewidth=2, 
             label=f'Run {run_idx+1} (TR={config["TR"]}s)')
    
    # Mark run boundaries
    if run_idx < len(run_configs) - 1:
        plt.axvline(x=run_end, color='gray', linestyle='-', 
                   linewidth=2, alpha=0.7)
    
    cumulative_time = run_end

plt.xlabel('Time (seconds)')
plt.ylabel('BOLD Response')
plt.title('Multi-run Experiment with Different TRs')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %% Advanced: Custom HRF Library with Constraints

print("\n=== Custom HRF Library ===")
print("Creating constrained HRF libraries...\n")

# Create library with physiologically plausible constraints
def constrained_gamma_generator(peak_time, width):
    """Generate gamma HRF with specified peak time and width."""
    # Convert peak_time and width to gamma parameters
    # For gamma distribution: peak = (shape-1)/rate
    # Standard deviation ≈ sqrt(shape)/rate
    rate = 1.0  # Fix rate
    shape = peak_time * rate + 1
    
    return gen_hrf(hrf_gamma, shape=shape, rate=rate, 
                  name=f"Gamma_peak{peak_time:.1f}")

# Parameter grid with physiological constraints
param_grid = pd.DataFrame({
    'peak_time': np.linspace(4, 8, 5),  # Peak between 4-8 seconds
    'width': [2.0] * 5  # Fixed width for now
})

# Create library
physio_lib = hrf_library(constrained_gamma_generator, param_grid)

# Evaluate and plot
responses = physio_lib(time_points)

plt.figure(figsize=(10, 6))
for i in range(responses.shape[1]):
    peak = param_grid.iloc[i]['peak_time']
    plt.plot(time_points, responses[:, i], linewidth=2, 
             label=f'Peak @ {peak:.1f}s')

plt.xlabel('Time (seconds)')
plt.ylabel('HRF Response')
plt.title('Physiologically Constrained HRF Library')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("\nTutorial complete! You've learned how to:")
print("1. Create HRF libraries for parameter exploration")
print("2. Use reconstruction matrices for basis coefficients")
print("3. Apply penalty matrices for regularization")
print("4. Build complex multi-condition designs")
print("5. Handle multi-run experiments")
print("6. Create custom constrained HRF libraries")