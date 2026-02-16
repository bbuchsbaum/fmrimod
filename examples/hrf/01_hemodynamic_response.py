"""
Hemodynamic Response Functions - Python Tutorial
===============================================

This tutorial demonstrates how to use hemodynamic response functions (HRFs) 
in fmrimod, replicating the functionality shown in the R vignette.

A hemodynamic response function (HRF) models the temporal evolution of the 
fMRI BOLD signal in response to a brief neural event.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from fmrimod import get_hrf, gen_hrf, block_hrf, list_available_hrfs
from fmrimod.hrf.functions import hrf_gaussian

# Set up plotting style
plt.style.use('seaborn-v0_8-darkgrid')

# %% Introduction to HRFs
print("Available HRFs in fmrimod:")
print(list_available_hrfs())

# %% Pre-defined HRF Objects

# Get SPM canonical HRF and Gaussian HRF
hrf_spm = get_hrf("spmg1")
hrf_gauss = get_hrf("gaussian")

print(f"\nSPM Canonical HRF: {hrf_spm}")
print(f"Gaussian HRF: {hrf_gauss}")

# %% Evaluate and Plot Basic HRFs

time_points = np.linspace(0, 25, 250)

# Evaluate the HRFs
y_spm = hrf_spm(time_points)
y_gauss = hrf_gauss(time_points)

# Manually scale each to peak at 1.0 for easier shape comparison
y_spm_scaled = y_spm / np.max(y_spm)
y_gauss_scaled = y_gauss / np.max(y_gauss)

# Plot comparison
plt.figure(figsize=(10, 6))
plt.plot(time_points, y_spm_scaled, label='SPM Canonical', linewidth=2)
plt.plot(time_points, y_gauss_scaled, label='Gaussian', linewidth=2)
plt.xlabel('Time (seconds)')
plt.ylabel('BOLD Response (normalized)')
plt.title('Comparison of SPM Canonical and Gaussian HRFs\nHRFs manually scaled to peak at 1.0')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %% Modifying HRF Parameters with gen_hrf

# Create Gaussian HRFs with different parameters
hrf_gauss_7_3 = gen_hrf(hrf_gaussian, mean=7, sd=3, name="Gaussian (Mean=7, SD=3)")
hrf_gauss_5_2 = gen_hrf(hrf_gaussian, mean=5, sd=2, name="Gaussian (Mean=5, SD=2)")
hrf_gauss_4_1 = gen_hrf(hrf_gaussian, mean=4, sd=1, name="Gaussian (Mean=4, SD=1)")

# Evaluate the new HRFs
vals1 = hrf_gauss_7_3(time_points)
vals2 = hrf_gauss_5_2(time_points)
vals3 = hrf_gauss_4_1(time_points)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(time_points, vals1, label='Mean=7, SD=3', linewidth=2)
plt.plot(time_points, vals2, label='Mean=5, SD=2', linewidth=2)
plt.plot(time_points, vals3, label='Mean=4, SD=1', linewidth=2)
plt.xlabel('Time (seconds)')
plt.ylabel('BOLD Response')
plt.title('Gaussian HRFs with Different Parameters')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %% Modeling Event Duration with block_hrf

# Create blocked HRFs using the SPM canonical HRF with different durations
hrf_spm_w1 = block_hrf(hrf_spm, width=1)
hrf_spm_w2 = block_hrf(hrf_spm, width=2)
hrf_spm_w4 = block_hrf(hrf_spm, width=4)

# Evaluate
resp_w1 = hrf_spm_w1(time_points)
resp_w2 = hrf_spm_w2(time_points)
resp_w4 = hrf_spm_w4(time_points)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(time_points, resp_w1, label='Width=1s', linewidth=2)
plt.plot(time_points, resp_w2, label='Width=2s', linewidth=2)
plt.plot(time_points, resp_w4, label='Width=4s', linewidth=2)
plt.xlabel('Time (seconds)')
plt.ylabel('BOLD Response')
plt.title('SPM Canonical HRF for Different Event Durations\nUsing block_hrf()')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %% Normalization

# Create normalized blocked HRFs
hrf_spm_w1_norm = block_hrf(hrf_spm, width=1, normalize=True)
hrf_spm_w2_norm = block_hrf(hrf_spm, width=2, normalize=True)
hrf_spm_w4_norm = block_hrf(hrf_spm, width=4, normalize=True)

# Evaluate
resp_w1_norm = hrf_spm_w1_norm(time_points)
resp_w2_norm = hrf_spm_w2_norm(time_points)
resp_w4_norm = hrf_spm_w4_norm(time_points)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(time_points, resp_w1_norm, label='Width=1s', linewidth=2)
plt.plot(time_points, resp_w2_norm, label='Width=2s', linewidth=2)
plt.plot(time_points, resp_w4_norm, label='Width=4s', linewidth=2)
plt.xlabel('Time (seconds)')
plt.ylabel('BOLD Response')
plt.title('Normalized SPM Canonical HRF for Different Durations\nUsing block_hrf(normalize=True)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, None)  # Ensure y-axis starts at 0
plt.show()

# %% Modeling Saturation with summate

# Create non-summating blocked HRFs
hrf_spm_w2_sum = block_hrf(hrf_spm, width=2, summate=True, normalize=True)
hrf_spm_w2_nosum = block_hrf(hrf_spm, width=2, summate=False, normalize=True)
hrf_spm_w8_sum = block_hrf(hrf_spm, width=8, summate=True, normalize=True)
hrf_spm_w8_nosum = block_hrf(hrf_spm, width=8, summate=False, normalize=True)

# Evaluate
resp_w2_sum = hrf_spm_w2_sum(time_points)
resp_w2_nosum = hrf_spm_w2_nosum(time_points)
resp_w8_sum = hrf_spm_w8_sum(time_points)
resp_w8_nosum = hrf_spm_w8_nosum(time_points)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 2-second duration
ax1.plot(time_points, resp_w2_sum, label='Summate=True', linewidth=2)
ax1.plot(time_points, resp_w2_nosum, label='Summate=False', linewidth=2, linestyle='--')
ax1.set_xlabel('Time (seconds)')
ax1.set_ylabel('BOLD Response')
ax1.set_title('2-Second Duration')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 8-second duration
ax2.plot(time_points, resp_w8_sum, label='Summate=True', linewidth=2)
ax2.plot(time_points, resp_w8_nosum, label='Summate=False', linewidth=2, linestyle='--')
ax2.set_xlabel('Time (seconds)')
ax2.set_ylabel('BOLD Response')
ax2.set_title('8-Second Duration')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.suptitle('Effect of Summation on Blocked HRFs', fontsize=14)
plt.tight_layout()
plt.show()

# %% Summary Statistics

print("\n=== HRF Summary Statistics ===")
print(f"SPM HRF peak time: {time_points[np.argmax(y_spm)]:.2f} seconds")
print(f"SPM HRF peak value: {np.max(y_spm):.4f}")
print(f"Gaussian HRF peak time: {time_points[np.argmax(y_gauss)]:.2f} seconds")
print(f"Gaussian HRF peak value: {np.max(y_gauss):.4f}")

# %% Advanced: Creating Custom HRFs

def custom_double_gamma(t, a1=6, b1=1, a2=16, b2=1, c=1/6):
    """Custom double gamma HRF function."""
    from scipy.stats import gamma as gamma_dist
    t = np.asarray(t)
    # First gamma
    g1 = gamma_dist.pdf(t, a=a1, scale=1/b1)
    # Second gamma (undershoot)
    g2 = gamma_dist.pdf(t, a=a2, scale=1/b2)
    # Combine with weighting
    return g1 - c * g2

# Create custom HRF
hrf_custom = gen_hrf(custom_double_gamma, name="Custom Double Gamma", span=32)

# Evaluate and plot against SPM
y_custom = hrf_custom(time_points)
y_custom = y_custom / np.max(y_custom)  # Normalize

plt.figure(figsize=(10, 6))
plt.plot(time_points, y_spm_scaled, label='SPM Canonical', linewidth=2)
plt.plot(time_points, y_custom, label='Custom Double Gamma', linewidth=2, linestyle='--')
plt.xlabel('Time (seconds)')
plt.ylabel('BOLD Response (normalized)')
plt.title('Comparison of SPM and Custom HRFs')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("\nTutorial complete! You've learned how to:")
print("1. Use pre-defined HRFs")
print("2. Modify HRF parameters")
print("3. Model event durations with block_hrf")
print("4. Control normalization and summation")
print("5. Create custom HRF functions")