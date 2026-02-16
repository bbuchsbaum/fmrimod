"""
Parametric Modulation Example
=============================

This example shows how to include parametric modulation in your design matrix.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fmrimod import event_model

# Create event data with parametric modulator
np.random.seed(42)

# Simulate a task with varying difficulty
n_trials = 20
trial_onsets = np.arange(10, 10 + n_trials * 15, 15)
conditions = ['easy', 'hard'] * (n_trials // 2)
difficulty_ratings = np.random.uniform(1, 10, n_trials)  # 1-10 difficulty scale

# Create DataFrame
data = pd.DataFrame({
    'onset': trial_onsets,
    'condition': conditions,
    'difficulty': difficulty_ratings,
    'duration': 2.0  # 2 second stimulus duration
})

print("First 10 events:")
print(data.head(10))

# Model 1: Basic model without parametric modulation
model_basic = event_model(
    "condition:hrf('spm')",
    data=data,
    tr=2.0,
    n_scans=200
)

# Model 2: With parametric modulation by difficulty
model_param = event_model(
    "condition:hrf('spm') + condition:difficulty:hrf('spm')",
    data=data,
    tr=2.0,
    n_scans=200
)

# Model 3: With orthogonalized parametric modulation
# Note: In practice, you might want to demean the parametric modulator
data['difficulty_centered'] = data['difficulty'] - data['difficulty'].mean()

model_ortho = event_model(
    "condition:hrf('spm') + condition:difficulty_centered:hrf('spm')",
    data=data,
    tr=2.0,
    n_scans=200
)

# Visualize all three models
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

models = [model_basic, model_param, model_ortho]
titles = ['Basic Model', 'Parametric Modulation', 'Centered Parametric Modulation']

for ax, model, title in zip(axes, models, titles):
    im = ax.imshow(model.design_matrix, aspect='auto', cmap='RdBu_r')
    ax.set_title(f'{title} - {model.design_matrix.shape[1]} regressors')
    ax.set_xlabel('Regressors')
    ax.set_ylabel('Time (scans)')
    
    # Add regressor labels
    ax.set_xticks(range(len(model.column_names)))
    ax.set_xticklabels(model.column_names, rotation=45, ha='right')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

# Plot correlation matrix for the parametric model
X_param = model_param.design_matrix
corr_matrix = np.corrcoef(X_param.T)

plt.figure(figsize=(8, 6))
im = plt.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
plt.colorbar(im)
plt.title('Correlation Matrix - Parametric Model')
plt.xticks(range(len(model_param.column_names)), model_param.column_names, rotation=45, ha='right')
plt.yticks(range(len(model_param.column_names)), model_param.column_names)
plt.tight_layout()
plt.show()

# Compare variance explained
print("\nDesign matrix properties:")
print(f"Basic model: {model_basic.design_matrix.shape[1]} columns")
print(f"Parametric model: {model_param.design_matrix.shape[1]} columns")
print(f"Orthogonalized model: {model_ortho.design_matrix.shape[1]} columns")

# Check orthogonality in the centered model
X_ortho = model_ortho.design_matrix
within_condition_corr = []
for i in range(0, X_ortho.shape[1], 2):  # Pairs of main effect and parametric
    if i+1 < X_ortho.shape[1]:
        corr = np.corrcoef(X_ortho[:, i], X_ortho[:, i+1])[0, 1]
        within_condition_corr.append(corr)
        
print(f"\nWithin-condition correlations (centered model): {within_condition_corr}")