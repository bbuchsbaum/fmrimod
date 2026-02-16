"""
Building fMRI Regressors - Python Tutorial
==========================================

This tutorial demonstrates how to create and manipulate fMRI regressors
using fmrimod, replicating the functionality from the R vignette.

A regressor represents the expected BOLD signal timecourse for a specific
experimental condition, created by convolving event onsets with an HRF.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from fmrimod import regressor, get_hrf, SamplingFrame, regressor_set

# Set up plotting style
plt.style.use('seaborn-v0_8-darkgrid')

# %% Basic Regressor from Event Onsets

# Define event onsets - stimuli every 12 seconds
onsets = np.arange(0, 10 * 12 + 1, 12)
print(f"Event onsets: {onsets}")

# Create the regressor object using SPM canonical HRF
reg1 = regressor(onsets=onsets, hrf="spmg1")  # Can also use get_hrf("spmg1")

# Print regressor properties
print(f"\nRegressor properties:")
print(f"  Number of events: {len(reg1.onsets)}")
print(f"  HRF name: {reg1.hrf.name}")
print(f"  HRF span: {reg1.hrf.span} seconds")
print(f"  Number of basis functions: {reg1.hrf.nbasis}")

# %% Evaluating and Plotting a Regressor

# Define a time grid corresponding to scan times (TR=2s)
TR = 2
scan_times = np.arange(0, 141, TR)

# Evaluate the regressor at scan times
predicted_bold = reg1.evaluate(scan_times)

# Plot the predicted timecourse
plt.figure(figsize=(12, 6))
plt.plot(scan_times, predicted_bold, 'b-', linewidth=2, label='Predicted BOLD')

# Add vertical lines for event onsets
for onset in onsets:
    plt.axvline(x=onset, color='red', linestyle='--', alpha=0.7)

plt.xlabel('Time (seconds)')
plt.ylabel('Predicted Response')
plt.title('Predicted BOLD Response (SPM HRF)')
plt.legend(['Predicted BOLD', 'Event Onsets'])
plt.grid(True, alpha=0.3)
plt.show()

# %% Varying Event Durations

# Example onsets and durations
onsets_var_dur = np.linspace(0, 5 * 12, 6)
durations_var = np.arange(1, 7)  # Durations increase from 1s to 6s

print(f"\nVariable duration events:")
print(f"  Onsets: {onsets_var_dur}")
print(f"  Durations: {durations_var}")

# Create regressor with varying durations
reg_var_dur = regressor(onsets_var_dur, hrf="spmg1", duration=durations_var)

# Evaluate and plot
scan_times_dur = np.arange(0, max(onsets_var_dur) + 30 + 1, TR)
pred_var_dur = reg_var_dur.evaluate(scan_times_dur)

plt.figure(figsize=(12, 6))
plt.plot(scan_times_dur, pred_var_dur, 'b-', linewidth=2)

# Add event markers with duration info
for onset, dur in zip(onsets_var_dur, durations_var):
    plt.axvline(x=onset, color='red', linestyle='--', alpha=0.7)
    plt.text(onset, plt.ylim()[1] * 0.9, f'{dur}s', ha='center', fontsize=8)

plt.xlabel('Time (seconds)')
plt.ylabel('Predicted Response')
plt.title('Regressor with Varying Event Durations\nDuration increases over time')
plt.grid(True, alpha=0.3)
plt.show()

# %% Event Amplitudes

# Create events with different amplitudes
onsets_amp = np.array([0, 20, 40, 60, 80])
amplitudes = np.array([0.5, 1.0, 2.0, 1.5, 0.8])

print(f"\nVariable amplitude events:")
print(f"  Onsets: {onsets_amp}")
print(f"  Amplitudes: {amplitudes}")

# Create regressor with varying amplitudes
reg_amp = regressor(onsets_amp, hrf="spmg1", amplitude=amplitudes)

# Evaluate and plot
scan_times_amp = np.arange(0, max(onsets_amp) + 30 + 1, TR)
pred_amp = reg_amp.evaluate(scan_times_amp)

plt.figure(figsize=(12, 6))
plt.plot(scan_times_amp, pred_amp, 'b-', linewidth=2)

# Add event markers with amplitude info
for onset, amp in zip(onsets_amp, amplitudes):
    plt.axvline(x=onset, color='red', linestyle='--', alpha=0.7)
    plt.text(onset, plt.ylim()[1] * 0.9, f'{amp}x', ha='center', fontsize=8)

plt.xlabel('Time (seconds)')
plt.ylabel('Predicted Response')
plt.title('Regressor with Varying Event Amplitudes')
plt.grid(True, alpha=0.3)
plt.show()

# %% Using Different HRFs

# Compare regressors with different HRFs
hrf_types = ['spmg1', 'gamma', 'gaussian']
colors = ['blue', 'green', 'orange']

plt.figure(figsize=(12, 6))

for hrf_type, color in zip(hrf_types, colors):
    reg_hrf = regressor(onsets[:5], hrf=hrf_type)  # Use first 5 events
    pred_hrf = reg_hrf.evaluate(scan_times)
    plt.plot(scan_times, pred_hrf, color=color, linewidth=2, label=hrf_type.upper())

# Add event markers
for onset in onsets[:5]:
    plt.axvline(x=onset, color='red', linestyle='--', alpha=0.3)

plt.xlabel('Time (seconds)')
plt.ylabel('Predicted Response')
plt.title('Comparison of Different HRF Types')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %% Multiple Conditions with RegressorSet

# Create a multi-condition experiment
n_trials = 20
trial_onsets = np.sort(np.random.uniform(0, 200, n_trials))
conditions = np.random.choice(['visual', 'motor', 'auditory'], n_trials)

print(f"\nMulti-condition experiment:")
print(f"  Total trials: {n_trials}")
print(f"  Conditions: {np.unique(conditions)}")
print(f"  Trials per condition:")
for cond in np.unique(conditions):
    print(f"    {cond}: {np.sum(conditions == cond)}")

# Create regressor set
reg_set = regressor_set(
    onsets=trial_onsets,
    fac=conditions,
    hrf="spmg1"
)

# Create sampling frame and evaluate
sf = SamplingFrame(blocklens=120, tr=2.0)
design_matrix = reg_set.evaluate(sf.samples)

# Plot design matrix
fig, axes = plt.subplots(len(reg_set.levels), 1, figsize=(12, 8), sharex=True)
if len(reg_set.levels) == 1:
    axes = [axes]

for i, (level, ax) in enumerate(zip(reg_set.levels, axes)):
    ax.plot(sf.samples, design_matrix[:, i], linewidth=2)
    ax.set_ylabel(f'{level}\nResponse')
    ax.grid(True, alpha=0.3)
    
    # Add event markers for this condition
    cond_onsets = trial_onsets[conditions == level]
    for onset in cond_onsets:
        ax.axvline(x=onset, color='red', linestyle='--', alpha=0.5)

axes[-1].set_xlabel('Time (seconds)')
axes[0].set_title('Design Matrix for Multi-Condition Experiment')
plt.tight_layout()
plt.show()

# %% Working with SamplingFrame

# Create multi-block experiment
blocklens = [100, 100, 80]
TRs = [2.0, 2.0, 1.5]

sf_multi = SamplingFrame(blocklens=blocklens, tr=TRs)

print(f"\nMulti-block sampling frame:")
print(f"  Number of blocks: {sf_multi.n_blocks}")
print(f"  Total scans: {sf_multi.n_scans}")
print(f"  Block lengths: {sf_multi.blocklens}")
print(f"  TRs: {sf_multi.tr}")

# Create events across blocks
all_onsets = [10, 30, 50,  # Block 1
              110, 130, 150,  # Block 2
              250, 270]  # Block 3

reg_multi = regressor(all_onsets, hrf="spmg1")
pred_multi = reg_multi.evaluate(sf_multi.samples)

# Plot with block boundaries
plt.figure(figsize=(12, 6))
plt.plot(sf_multi.samples, pred_multi, 'b-', linewidth=2)

# Add block boundaries
block_ends = np.cumsum([0] + [b * tr for b, tr in zip(blocklens, TRs)])
for i, end in enumerate(block_ends[1:-1]):
    plt.axvline(x=sf_multi.samples[np.sum(blocklens[:i+1])-1], 
                color='gray', linestyle='-', alpha=0.5, linewidth=2)
    plt.text(sf_multi.samples[np.sum(blocklens[:i+1])-1], 
             plt.ylim()[1] * 0.95, f'Block {i+1}|{i+2}', 
             ha='center', fontsize=10)

# Add event markers
for onset in all_onsets:
    plt.axvline(x=onset, color='red', linestyle='--', alpha=0.7)

plt.xlabel('Time (seconds)')
plt.ylabel('Predicted Response')
plt.title('Multi-Block Experiment with Different TRs')
plt.grid(True, alpha=0.3)
plt.show()

# %% Advanced: Sparse Matrix Output

# For large experiments, use sparse matrices
large_onsets = np.sort(np.random.uniform(0, 1000, 50))
reg_large = regressor(large_onsets, hrf="spmg1")

# Dense evaluation
sf_large = SamplingFrame(blocklens=600, tr=2.0)
dense_result = reg_large.evaluate(sf_large.samples, sparse=False)

# Sparse evaluation
sparse_result = reg_large.evaluate(sf_large.samples, sparse=True)

print(f"\nLarge experiment matrix comparison:")
print(f"  Dense matrix shape: {dense_result.shape}")
print(f"  Dense matrix memory: {dense_result.nbytes / 1024:.1f} KB")
if hasattr(sparse_result, 'data'):
    print(f"  Sparse matrix non-zero elements: {sparse_result.nnz}")
    print(f"  Sparse matrix memory: {(sparse_result.data.nbytes + sparse_result.indices.nbytes + sparse_result.indptr.nbytes) / 1024:.1f} KB")
    print(f"  Sparsity: {1 - sparse_result.nnz / (sparse_result.shape[0] * sparse_result.shape[1]):.1%}")

print("\nTutorial complete! You've learned how to:")
print("1. Create basic regressors from event onsets")
print("2. Use varying event durations and amplitudes")
print("3. Compare different HRF types")
print("4. Build multi-condition designs with RegressorSet")
print("5. Work with multi-block experiments")
print("6. Use sparse matrices for efficiency")