Basic Event Model Tutorial
==========================

This tutorial covers the fundamentals of creating design matrices with fmrimod.

What is a Design Matrix?
------------------------

A design matrix (also called a model matrix) represents your experimental design in a format suitable for statistical analysis. Each column represents a predictor (regressor), and each row represents a time point in your fMRI scan.

Creating Your First Model
-------------------------

Let's start with a simple example:

.. code-block:: python

    import pandas as pd
    from fmrimod import event_model
    
    # Define your experimental events
    events = pd.DataFrame({
        'onset': [10, 25, 40, 55, 70],  # When events occurred (in seconds)
        'condition': ['A', 'B', 'A', 'B', 'A'],  # Event types
        'duration': [2, 2, 2, 2, 2]  # How long each event lasted
    })
    
    # Create the model
    model = event_model(
        formula="condition",  # What to model
        data=events,
        tr=2.0,              # Repetition time (TR) in seconds
        n_scans=50           # Number of brain volumes collected
    )
    
    # Access the design matrix
    X = model.design_matrix
    print(f"Design matrix shape: {X.shape}")
    print(f"Columns: {model.column_names}")

Understanding the Formula Syntax
--------------------------------

The formula syntax is inspired by R's formula notation:

Basic Formulas
^^^^^^^^^^^^^^

.. code-block:: python

    # Single factor
    "condition"
    
    # Multiple factors (additive)
    "condition + task"
    
    # Interaction
    "condition * task"
    
    # Main effects plus interaction
    "condition + task + condition:task"

Adding HRF Convolution
----------------------

By default, events are convolved with the canonical HRF. You can specify different HRF models:

.. code-block:: python

    # SPM's canonical HRF (default)
    model1 = event_model("condition", data=events, tr=2.0, n_scans=50)
    
    # Explicitly specify SPM HRF
    model2 = event_model("condition:hrf('spm')", data=events, tr=2.0, n_scans=50)
    
    # AFNI's GAM HRF
    model3 = event_model("condition:hrf('gam')", data=events, tr=2.0, n_scans=50)
    
    # No HRF (raw boxcar)
    model4 = event_model("condition:hrf('none')", data=events, tr=2.0, n_scans=50)

Working with Different Event Types
----------------------------------

Categorical Events
^^^^^^^^^^^^^^^^^^

For conditions or discrete states:

.. code-block:: python

    # Events with categorical conditions
    cat_events = pd.DataFrame({
        'onset': [5, 15, 25, 35],
        'stimulus': ['face', 'house', 'face', 'house'],
        'duration': 1.0
    })
    
    model = event_model("stimulus", data=cat_events, tr=2.0, n_scans=30)

Continuous Variables
^^^^^^^^^^^^^^^^^^^^

For parametric designs:

.. code-block:: python

    # Events with continuous values
    cont_events = pd.DataFrame({
        'onset': [5, 15, 25, 35],
        'stimulus': ['face', 'face', 'face', 'face'],
        'emotional_intensity': [0.2, 0.8, 0.5, 0.9],
        'duration': 1.0
    })
    
    # Model the continuous variable
    model = event_model(
        "stimulus + emotional_intensity",
        data=cont_events,
        tr=2.0,
        n_scans=30
    )

Mixed Designs
^^^^^^^^^^^^^

Combining categorical and continuous:

.. code-block:: python

    mixed_events = pd.DataFrame({
        'onset': [5, 15, 25, 35, 45, 55],
        'condition': ['easy', 'hard', 'easy', 'hard', 'easy', 'hard'],
        'reaction_time': [0.5, 0.8, 0.4, 0.9, 0.6, 0.7],
        'duration': 2.0
    })
    
    # Parametric modulation within conditions
    model = event_model(
        "condition + condition:reaction_time",
        data=mixed_events,
        tr=2.0,
        n_scans=40
    )

Visualizing Your Design
-----------------------

Always visualize your design matrix to verify it looks correct:

.. code-block:: python

    import matplotlib.pyplot as plt
    
    # Plot the design matrix
    plt.figure(figsize=(10, 6))
    plt.imshow(model.design_matrix, aspect='auto', cmap='RdBu_r')
    plt.colorbar(label='Amplitude')
    plt.xlabel('Regressors')
    plt.ylabel('Time (scans)')
    plt.title('Design Matrix')
    
    # Add column labels
    ax = plt.gca()
    ax.set_xticks(range(len(model.column_names)))
    ax.set_xticklabels(model.column_names, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()

Accessing Model Information
---------------------------

The event model provides various ways to inspect your design:

.. code-block:: python

    # Summary information
    print(model.summary())
    
    # Column names
    print("Regressors:", model.column_names)
    
    # Number of events
    print("Number of events:", model.n_events)
    
    # Event names
    print("Event types:", model.event_names)
    
    # Convert to pandas DataFrame for easier inspection
    df = model.to_dataframe()
    print(df.head())

Common Patterns
---------------

Block Design
^^^^^^^^^^^^

.. code-block:: python

    # Long blocks of stimulation
    block_events = pd.DataFrame({
        'onset': [0, 30, 60, 90],
        'condition': ['rest', 'task', 'rest', 'task'],
        'duration': [30, 30, 30, 30]
    })
    
    model = event_model("condition", data=block_events, tr=2.0, n_scans=75)

Event-Related Design
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Brief, randomly spaced events
    import numpy as np
    np.random.seed(42)
    
    n_events = 20
    isis = np.random.exponential(scale=8, size=n_events)  # Inter-stimulus intervals
    onsets = np.cumsum(isis)
    
    er_events = pd.DataFrame({
        'onset': onsets,
        'condition': np.random.choice(['A', 'B'], size=n_events),
        'duration': 0  # Impulse events
    })
    
    model = event_model("condition", data=er_events, tr=2.0, n_scans=150)

Mixed Block/Event Design
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Blocks with embedded events
    mixed_design = pd.DataFrame({
        'onset': [0, 5, 10, 15, 30, 35, 40, 45],
        'block': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
        'event': ['go', 'nogo', 'go', 'go', 'nogo', 'go', 'go', 'nogo'],
        'duration': [1, 1, 1, 1, 1, 1, 1, 1]
    })
    
    # Model both block and event
    model = event_model(
        "block + event + block:event",
        data=mixed_design,
        tr=2.0,
        n_scans=30
    )

Next Steps
----------

- Learn about :doc:`working_with_contrasts` to test hypotheses
- Explore :doc:`parametric_modulation` for more complex designs
- See :doc:`baseline_modeling` to handle scanner drift