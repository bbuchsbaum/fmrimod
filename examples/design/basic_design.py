"""
Basic Design Matrix Example
===========================

This example demonstrates how to create a simple design matrix for an fMRI experiment.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fmrimod import event_model

# Create experimental design data
# Let's simulate a simple block design with two conditions
np.random.seed(42)

# Create events for condition A and B
onsets_A = [10, 50, 90, 130, 170]
onsets_B = [30, 70, 110, 150, 190]

# Combine into a DataFrame
events = []
for onset in onsets_A:
    events.append({'onset': onset, 'condition': 'A', 'duration': 15})
for onset in onsets_B:
    events.append({'onset': onset, 'condition': 'B', 'duration': 15})

data = pd.DataFrame(events)
print("Event data:")
print(data)

# Create the event model
model = event_model(
    "condition",  # Model formula
    data=data,
    tr=2.0,       # TR = 2 seconds
    n_scans=120   # 120 volumes = 240 seconds
)

# Print model summary
print("\nModel summary:")
print(model.summary())

# Get the design matrix
X = model.design_matrix
print(f"\nDesign matrix shape: {X.shape}")
print(f"Column names: {model.column_names}")

# Visualize the design matrix
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot the design matrix
im = ax1.imshow(X, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
ax1.set_xlabel('Regressors')
ax1.set_ylabel('Time (scans)')
ax1.set_title('Design Matrix')
ax1.set_xticks(range(len(model.column_names)))
ax1.set_xticklabels(model.column_names, rotation=45)
plt.colorbar(im, ax=ax1)

# Plot the time courses
time = np.arange(X.shape[0]) * model.tr
for i, col_name in enumerate(model.column_names):
    ax2.plot(time, X[:, i], label=col_name, linewidth=2)

ax2.set_xlabel('Time (seconds)')
ax2.set_ylabel('Signal')
ax2.set_title('Regressor Time Courses')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Create a contrast between conditions
from fmrimod.contrast import pair_contrast, contrast_weights
from fmrimod.contrast.contrast_spec import Formula

# Define contrast A > B
contrast_spec = pair_contrast(
    A=Formula("condition == 'A'"),
    B=Formula("condition == 'B'"),
    name="A_vs_B"
)

# Compute contrast weights
# Note: In a real scenario, you would use the model's term object
print("\nContrast specification created:")
print(f"Contrast name: {contrast_spec.name}")
print(f"Condition A: {contrast_spec.A.expr}")
print(f"Condition B: {contrast_spec.B.expr}")