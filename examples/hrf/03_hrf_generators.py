"""
HRF Generators - Python Tutorial
================================

This tutorial demonstrates HRF generators in fmrimod, which are functions
that create flexible HRF basis sets like B-splines and FIR models.

Generators allow you to specify the number of basis functions and time span
at creation time, providing flexibility for advanced modeling.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from fmrimod import get_hrf, gen_hrf, list_available_hrfs
from fmrimod.hrf.functions import hrf_bspline, hrf_fir

# Set up plotting style
plt.style.use('seaborn-v0_8-darkgrid')

# %% Why Generators?

print("=== HRF Generators ===")
print("\nGenerators are functions that create HRF objects with customizable parameters.")
print("They're useful for flexible basis sets like B-splines and FIR models.\n")

# List available HRFs and identify generators
available_hrfs = list_available_hrfs()
print("Available HRFs:")
for hrf_name in available_hrfs:
    try:
        hrf = get_hrf(hrf_name)
        hrf_type = "basis" if hrf.nbasis > 1 else "single"
        print(f"  {hrf_name}: {hrf_type} (nbasis={hrf.nbasis})")
    except:
        print(f"  {hrf_name}: generator function")

# %% Creating a B-spline Basis

# Create B-spline bases with different numbers of functions
bs4 = gen_hrf(hrf_bspline, n_basis=4, span=20, name="B-spline (N=4)")
bs8 = gen_hrf(hrf_bspline, n_basis=8, span=20, name="B-spline (N=8)")
bs12 = gen_hrf(hrf_bspline, n_basis=12, span=20, name="B-spline (N=12)")

print(f"\nB-spline bases created:")
print(f"  {bs4.name}: {bs4.nbasis} basis functions, span={bs4.span}s")
print(f"  {bs8.name}: {bs8.nbasis} basis functions, span={bs8.span}s")
print(f"  {bs12.name}: {bs12.nbasis} basis functions, span={bs12.span}s")

# %% Visualizing B-spline Basis Functions

times = np.linspace(0, 20, 200)

# Evaluate B-spline bases
mat4 = bs4(times)
mat8 = bs8(times)
mat12 = bs12(times)

# Plot all three B-spline bases
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# 4 basis functions
for i in range(mat4.shape[1]):
    axes[0].plot(times, mat4[:, i], linewidth=2, label=f'Basis {i+1}')
axes[0].set_title('B-spline Basis with 4 Functions')
axes[0].set_ylabel('Response')
axes[0].grid(True, alpha=0.3)
axes[0].legend(loc='upper right')

# 8 basis functions
for i in range(mat8.shape[1]):
    axes[1].plot(times, mat8[:, i], linewidth=2, label=f'B{i+1}')
axes[1].set_title('B-spline Basis with 8 Functions')
axes[1].set_ylabel('Response')
axes[1].grid(True, alpha=0.3)

# 12 basis functions
for i in range(mat12.shape[1]):
    axes[2].plot(times, mat12[:, i], linewidth=2)
axes[2].set_title('B-spline Basis with 12 Functions')
axes[2].set_xlabel('Time (seconds)')
axes[2].set_ylabel('Response')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% Creating and Visualizing FIR Basis

# Create FIR bases with different resolutions
fir6 = gen_hrf(hrf_fir, n_basis=6, span=18, name="FIR (6 bins)")
fir10 = gen_hrf(hrf_fir, n_basis=10, span=20, name="FIR (10 bins)")
fir15 = gen_hrf(hrf_fir, n_basis=15, span=30, name="FIR (15 bins)")

# Evaluate FIR basis
times_fir = np.linspace(0, 30, 300)
mat_fir10 = fir10(times_fir[:201])  # Only up to 20s for the 10-bin FIR

# Plot FIR basis functions
plt.figure(figsize=(12, 6))
for i in range(mat_fir10.shape[1]):
    plt.plot(times_fir[:201], mat_fir10[:, i], linewidth=2, label=f'Bin {i+1}')

plt.xlabel('Time (seconds)')
plt.ylabel('Response')
plt.title('Finite Impulse Response (FIR) Basis\n10 bins over 20 seconds')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% Comparing Basis Representations

# Use a canonical HRF as "ground truth"
hrf_true = get_hrf("spmg1")
y_true = hrf_true(times)

# Fit the true HRF using different bases
from scipy.linalg import lstsq

# Function to fit and reconstruct HRF
def fit_and_reconstruct(basis_matrix, target):
    """Fit basis coefficients and reconstruct signal."""
    coeffs, _, _, _ = lstsq(basis_matrix, target)
    reconstruction = basis_matrix @ coeffs
    return coeffs, reconstruction

# Fit with different B-spline resolutions
coeffs4, recon4 = fit_and_reconstruct(mat4, y_true)
coeffs8, recon8 = fit_and_reconstruct(mat8, y_true)
coeffs12, recon12 = fit_and_reconstruct(mat12, y_true)

# Plot reconstructions
plt.figure(figsize=(12, 8))

# Original
plt.subplot(2, 2, 1)
plt.plot(times, y_true, 'k-', linewidth=3, label='True HRF')
plt.xlabel('Time (seconds)')
plt.ylabel('Response')
plt.title('Original SPM Canonical HRF')
plt.grid(True, alpha=0.3)
plt.legend()

# Reconstructions
plt.subplot(2, 2, 2)
plt.plot(times, y_true, 'k-', linewidth=2, alpha=0.5, label='True')
plt.plot(times, recon4, 'b--', linewidth=2, label='4 basis')
plt.xlabel('Time (seconds)')
plt.ylabel('Response')
plt.title('B-spline Reconstruction (N=4)')
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(times, y_true, 'k-', linewidth=2, alpha=0.5, label='True')
plt.plot(times, recon8, 'g--', linewidth=2, label='8 basis')
plt.xlabel('Time (seconds)')
plt.ylabel('Response')
plt.title('B-spline Reconstruction (N=8)')
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(times, y_true, 'k-', linewidth=2, alpha=0.5, label='True')
plt.plot(times, recon12, 'r--', linewidth=2, label='12 basis')
plt.xlabel('Time (seconds)')
plt.ylabel('Response')
plt.title('B-spline Reconstruction (N=12)')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()

# Calculate reconstruction errors
mse4 = np.mean((y_true - recon4)**2)
mse8 = np.mean((y_true - recon8)**2)
mse12 = np.mean((y_true - recon12)**2)

print(f"\nReconstruction Mean Squared Errors:")
print(f"  4 basis functions: {mse4:.6f}")
print(f"  8 basis functions: {mse8:.6f}")
print(f"  12 basis functions: {mse12:.6f}")

# %% Using Basis Sets in Regressors

from fmrimod import regressor, SamplingFrame

# Create event sequence
onsets = np.array([10, 30, 50, 70, 90])

# Create regressors with different basis sets
reg_canonical = regressor(onsets, hrf="spmg1")
reg_bspline = regressor(onsets, hrf=bs8)
reg_fir = regressor(onsets, hrf=fir10)

# Evaluate on sampling grid
sf = SamplingFrame(blocklens=60, tr=2.0)
times_eval = sf.samples

# Get design matrices
design_canonical = reg_canonical.evaluate(times_eval)  # Shape: (60,)
design_bspline = reg_bspline.evaluate(times_eval)    # Shape: (60, 8)
design_fir = reg_fir.evaluate(times_eval)            # Shape: (60, 10)

print(f"\nDesign matrix shapes:")
print(f"  Canonical HRF: {design_canonical.shape}")
print(f"  B-spline basis: {design_bspline.shape}")
print(f"  FIR basis: {design_fir.shape}")

# Plot design matrices
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Canonical
axes[0].plot(times_eval, design_canonical, 'b-', linewidth=2)
axes[0].set_title('Design Matrix: Canonical HRF')
axes[0].set_ylabel('Response')
for onset in onsets:
    axes[0].axvline(x=onset, color='red', linestyle='--', alpha=0.5)
axes[0].grid(True, alpha=0.3)

# B-spline
for i in range(design_bspline.shape[1]):
    axes[1].plot(times_eval, design_bspline[:, i], linewidth=2, 
                 label=f'B{i+1}' if i < 4 else None)
axes[1].set_title('Design Matrix: B-spline Basis (8 functions)')
axes[1].set_ylabel('Response')
for onset in onsets:
    axes[1].axvline(x=onset, color='red', linestyle='--', alpha=0.5)
axes[1].grid(True, alpha=0.3)
axes[1].legend(loc='upper right')

# FIR
for i in range(design_fir.shape[1]):
    axes[2].plot(times_eval, design_fir[:, i], linewidth=2)
axes[2].set_title('Design Matrix: FIR Basis (10 bins)')
axes[2].set_xlabel('Time (seconds)')
axes[2].set_ylabel('Response')
for onset in onsets:
    axes[2].axvline(x=onset, color='red', linestyle='--', alpha=0.5)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% Summary of Basis Functions

print("\n=== Summary: When to Use Each Basis Type ===")
print("\n1. Canonical HRFs (SPMG1, Gamma, etc.):")
print("   - Use when HRF shape is well-known")
print("   - Fewer parameters (1 per condition)")
print("   - Strong assumptions about shape")

print("\n2. B-spline Basis:")
print("   - Use for flexible HRF estimation")
print("   - Smooth basis functions")
print("   - Good for capturing shape variations")
print("   - Typical: 4-12 basis functions")

print("\n3. FIR Basis:")
print("   - Use for model-free HRF estimation")
print("   - No shape assumptions")
print("   - One parameter per time bin")
print("   - Can capture any shape but needs more data")

print("\n4. Other Bases:")
print("   - Fourier: For periodic/oscillatory responses")
print("   - SPM derivatives: For timing/dispersion variations")

print("\nTutorial complete! You've learned how to:")
print("1. Create basis functions using generators")
print("2. Visualize different basis types")
print("3. Compare basis representations")
print("4. Use basis sets in regression models")