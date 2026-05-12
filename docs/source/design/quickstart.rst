Quick Start Guide
=================

This guide will help you get started with fmrimod through practical examples.

Basic Event Model
-----------------

The most common use case is creating a design matrix from experimental events:

.. code-block:: python

    import pandas as pd
    from fmrimod import event_model
    
    # Create event data
    data = pd.DataFrame({
        'onset': [10, 30, 50, 70, 90],
        'condition': ['A', 'B', 'A', 'B', 'A'],
        'duration': [1, 1, 1, 1, 1]
    })
    
    # Create event model
    model = event_model(
        "condition",  # Formula specifying the model
        data=data,
        tr=2.0,       # TR in seconds
        n_scans=100   # Number of volumes
    )
    
    # Access the design matrix
    X = model.design_matrix
    print(f"Design matrix shape: {X.shape}")
    print(f"Columns: {model.column_names}")

Adding HRF Convolution
----------------------

You can specify different HRF models using the formula syntax:

.. code-block:: python

    # SPM canonical HRF (default)
    model_spm = event_model(
        "condition:hrf('spm')",
        data=data,
        tr=2.0,
        n_scans=100
    )
    
    # AFNI GAM HRF
    model_afni = event_model(
        "condition:hrf('gam')",
        data=data,
        tr=2.0,
        n_scans=100
    )

Including Covariates
--------------------

Continuous covariates can be included in the model:

.. code-block:: python

    # Add reaction time as a covariate
    data['rt'] = [0.5, 0.6, 0.4, 0.7, 0.5]
    
    model = event_model(
        "condition + rt",
        data=data,
        tr=2.0,
        n_scans=100
    )

Parametric Modulation
---------------------

Model parametric modulation of conditions by continuous variables:

.. code-block:: python

    # Condition modulated by reaction time
    model = event_model(
        "condition + condition:rt",
        data=data,
        tr=2.0,
        n_scans=100
    )

Adding Baseline Terms
---------------------

Include polynomial drift terms or spline bases:

.. code-block:: python

    from fmrimod import baseline_model
    
    # Polynomial drift (3rd order)
    baseline = baseline_model(
        baseline_formula="poly(3)",
        n_scans=100,
        tr=2.0
    )
    
    # Or include directly in event model
    model = event_model(
        "condition + poly(time, 3)",
        data=data,
        tr=2.0,
        n_scans=100
    )

Defining Contrasts
------------------

Create statistical contrasts for hypothesis testing:

.. code-block:: python

    from fmrimod.contrast import pair_contrast, contrast_weights
    
    # Simple pairwise contrast
    c = pair_contrast(
        A=Formula("condition == 'A'"),
        B=Formula("condition == 'B'"),
        name="A_vs_B"
    )
    
    # Compute contrast weights
    weights = contrast_weights(c, model)
    print(f"Contrast weights: {weights.weights}")

Visualization
-------------

Visualize your design matrix:

.. code-block:: python

    import matplotlib.pyplot as plt
    
    # Plot design matrix
    plt.figure(figsize=(10, 6))
    plt.imshow(model.design_matrix, aspect='auto', cmap='RdBu_r')
    plt.colorbar()
    plt.xlabel('Regressors')
    plt.ylabel('Time (scans)')
    plt.title('Design Matrix')
    plt.show()

Working with Multiple Runs
--------------------------

For multi-run experiments:

.. code-block:: python

    # Create data for multiple runs
    run1_data = pd.DataFrame({
        'onset': [10, 30, 50],
        'condition': ['A', 'B', 'A'],
        'run': [1, 1, 1]
    })
    
    run2_data = pd.DataFrame({
        'onset': [10, 30, 50],
        'condition': ['B', 'A', 'B'],
        'run': [2, 2, 2]
    })
    
    # Combine data
    all_data = pd.concat([run1_data, run2_data])
    
    # Model with run-specific intercepts
    model = event_model(
        "condition + run",
        data=all_data,
        tr=2.0,
        n_scans=200  # Total across both runs
    )

Next Steps
----------

* See the :doc:`tutorials/index` for more detailed examples
* Check the :doc:`api/index` for detailed function documentation
* Read the :doc:`migration_guide` for differences from the R package
