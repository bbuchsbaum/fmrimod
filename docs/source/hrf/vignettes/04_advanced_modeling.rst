Advanced HRF Modeling and Design
================================

Introduction
------------

This vignette explores advanced features of ``fmrimod`` for systematic HRF modeling, regularization, and experimental design. We'll cover five key functions that extend the basic HRF framework:

- **``hrf_library()``**: Creating systematic collections of HRF variants
- **``reconstruction_matrix()``**: Converting basis coefficients back to HRF shapes
- **``regressor_set()``**: Managing multi-condition experimental designs
- **``SamplingFrame``**: Defining temporal sampling for experiments

These tools are essential for advanced fMRI modeling where you need flexibility in HRF specification, robust estimation with limited data, or complex experimental designs.

HRF Libraries: Systematic Parameter Exploration
-----------------------------------------------

The ``hrf_library()`` function creates collections of HRF variants by systematically varying parameters. This is useful for exploring how different HRF assumptions affect your model or for building data-driven HRF basis sets.

Example 1: Library of Gamma HRFs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's create a library of gamma HRFs with different shape and rate parameters:

.. code-block:: python

    from fmrimod import hrf_library, gen_hrf
    from fmrimod.hrf.functions import gamma_hrf
    import numpy as np
    import pandas as pd

    # Define parameter grid for gamma HRFs
    gamma_params = pd.DataFrame({
        'shape': [4, 6, 8] * 3,
        'rate': [0.8] * 3 + [1.0] * 3 + [1.2] * 3
    })
    print(gamma_params)

    # Create a generator function for gamma HRFs
    def make_gamma_hrf(shape, rate):
        return gen_hrf(gamma_hrf, shape=shape, rate=rate, 
                      name=f"Gamma_{shape}_{rate}")

    # Create HRF library
    gamma_lib = hrf_library(make_gamma_hrf, gamma_params)
    print(f"Library contains {gamma_lib.nbasis} HRFs")

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from fmrimod import hrf_library, gen_hrf
    from fmrimod.hrf.functions import gamma_hrf
    import pandas as pd

    # Define parameter grid
    gamma_params = pd.DataFrame({
        'shape': [4, 6, 8] * 3,
        'rate': [0.8] * 3 + [1.0] * 3 + [1.2] * 3
    })

    # Create generator function
    def make_gamma_hrf(shape, rate):
        return gen_hrf(gamma_hrf, shape=shape, rate=rate, 
                      name=f"Gamma_{shape}_{rate}")

    # Create HRF library
    gamma_lib = hrf_library(make_gamma_hrf, gamma_params)

    # Evaluate and visualize
    time_points = np.linspace(0, 20, 200)
    gamma_responses = gamma_lib(time_points)

    # Plot
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, gamma_responses.shape[1]))
    
    for i in range(gamma_responses.shape[1]):
        shape = gamma_params.iloc[i]['shape']
        rate = gamma_params.iloc[i]['rate']
        plt.plot(time_points, gamma_responses[:, i], 
                 linewidth=2, color=colors[i],
                 label=f'Shape={shape}, Rate={rate}')
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('HRF Response')
    plt.title('Library of Gamma HRFs\nSystematic variation of shape and rate parameters')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

Example 2: Library of Lagged SPM HRFs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here's how to create a library of the SPM canonical HRF with different temporal lags:

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from fmrimod import hrf_library, get_hrf, lag_hrf
    import pandas as pd

    # Parameter grid for temporal lags
    lag_params = pd.DataFrame({'lag': np.arange(-2, 5)})

    # Create library using a helper function
    def create_lagged_spm(lag):
        return lag_hrf(get_hrf("spmg1"), lag=lag)

    spm_lag_lib = hrf_library(create_lagged_spm, lag_params)

    # Evaluate and plot
    time_points = np.linspace(0, 20, 200)
    spm_lag_responses = spm_lag_lib(time_points)

    plt.figure(figsize=(10, 6))
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(lag_params)))
    
    for i, lag in enumerate(lag_params['lag']):
        plt.plot(time_points, spm_lag_responses[:, i], 
                 linewidth=2, color=colors[i],
                 label=f'Lag = {lag}s')
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('HRF Response')
    plt.title('Library of Lagged SPM Canonical HRFs\nTemporal lags from -2 to +4 seconds')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

Reconstruction Matrices: From Coefficients to HRF Shapes
--------------------------------------------------------

The reconstruction process converts a set of basis coefficients into a continuous HRF shape. Understanding this transformation is key to interpreting estimated HRFs from fMRI analyses.

How Reconstruction Works
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from fmrimod import gen_hrf, reconstruction_matrix
    from fmrimod.hrf.functions import bspline_hrf
    import numpy as np

    # Use a small basis for clear visualization
    basis_set = gen_hrf(bspline_hrf, n_basis=5, degree=3, span=30)
    eval_times = np.linspace(0, 30, 301)

    # The reconstruction matrix
    recon_matrix = reconstruction_matrix(basis_set, eval_times)
    print(f"Reconstruction matrix: {recon_matrix.shape[0]} time points × "
          f"{recon_matrix.shape[1]} basis functions")

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from fmrimod import gen_hrf
    from fmrimod.hrf.functions import bspline_hrf

    # Create basis set
    basis_set = gen_hrf(bspline_hrf, n_basis=5, degree=3, span=30)
    eval_times = np.linspace(0, 30, 301)

    # Get basis functions
    basis_matrix = basis_set(eval_times)

    # Plot basis functions
    plt.figure(figsize=(10, 6))
    colors = plt.cm.turbo(np.linspace(0, 1, 5))
    
    for i in range(5):
        plt.plot(eval_times, basis_matrix[:, i], 
                 linewidth=2, color=colors[i],
                 label=f'B{i+1}')
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Basis Function Value')
    plt.title('B-spline Basis Functions\nEach basis function covers a different time window')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

Different HRF Shapes from Same Basis Set
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from fmrimod import gen_hrf
    from fmrimod.hrf.functions import bspline_hrf

    # Create basis set
    basis_set = gen_hrf(bspline_hrf, n_basis=5, degree=3, span=30)
    eval_times = np.linspace(0, 30, 301)
    recon_matrix = basis_set(eval_times)

    # Different coefficient patterns
    coefficient_sets = {
        "Early Peak": [0.2, 1.0, 0.3, 0.0, 0.0],
        "Canonical": [0.0, 0.3, 1.0, 0.4, -0.1],
        "Late Peak": [0.0, 0.0, 0.3, 1.0, 0.2],
        "Double Peak": [0.0, 0.8, 0.2, 0.9, 0.0]
    }

    # Reconstruct HRFs
    plt.figure(figsize=(10, 6))
    colors = {"Early Peak": "#E69F00", "Canonical": "#009E73",
              "Late Peak": "#0072B2", "Double Peak": "#D55E00"}
    
    for name, coefs in coefficient_sets.items():
        hrf_values = recon_matrix @ coefs
        plt.plot(eval_times, hrf_values, linewidth=2, 
                 color=colors[name], label=name)
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('HRF Response')
    plt.title('Different HRF Shapes from Same Basis Set\nVarying coefficients produces diverse HRF patterns')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

Building an HRF Step by Step
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from fmrimod import gen_hrf
    from fmrimod.hrf.functions import bspline_hrf

    # Create basis set
    basis_set = gen_hrf(bspline_hrf, n_basis=5, degree=3, span=30)
    eval_times = np.linspace(0, 30, 301)
    recon_matrix = basis_set(eval_times)

    # Canonical coefficients
    canonical_coefs = [0.0, 0.3, 1.0, 0.4, -0.1]

    # Create subplot for each step
    fig, axes = plt.subplots(1, 5, figsize=(15, 3), sharey=True)

    for i in range(5):
        ax = axes[i]
        
        # Individual contribution
        individual_coefs = [0] * 5
        individual_coefs[i] = canonical_coefs[i]
        individual_contrib = recon_matrix @ individual_coefs
        
        # Cumulative up to this point
        cumulative_coefs = canonical_coefs.copy()
        if i < 4:
            for j in range(i+1, 5):
                cumulative_coefs[j] = 0
        cumulative_hrf = recon_matrix @ cumulative_coefs
        
        ax.plot(eval_times, individual_contrib, 'r-', linewidth=2, 
                label='Individual', alpha=0.7)
        ax.plot(eval_times, cumulative_hrf, 'k-', linewidth=2, 
                label='Cumulative')
        
        ax.set_xlabel('Time (s)')
        ax.set_title(f'Adding B{i+1}\n(coef={canonical_coefs[i]:.1f})')
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.set_ylabel('Value')
            ax.legend(fontsize=8)
    
    plt.suptitle('Building an HRF: Sequential Addition of Weighted Basis Functions', 
                 fontsize=14)
    plt.tight_layout()
    plt.show()

Regressor Sets: Multi-Condition Experimental Designs
----------------------------------------------------

The ``regressor_set()`` function simplifies creating regressors for multi-condition experiments where each condition shares the same HRF but has different event timings.

.. code-block:: python

    from fmrimod import regressor_set, get_hrf
    import numpy as np

    # Simulate a 3-condition experiment
    np.random.seed(123)
    n_events_per_condition = 8
    total_duration = 240  # 4 minutes

    # Generate random onsets for each condition
    condition_A_onsets = np.sort(np.random.uniform(0, total_duration, n_events_per_condition))
    condition_B_onsets = np.sort(np.random.uniform(0, total_duration, n_events_per_condition))
    condition_C_onsets = np.sort(np.random.uniform(0, total_duration, n_events_per_condition))

    # Combine all onsets and create labels
    all_onsets = np.concatenate([condition_A_onsets, condition_B_onsets, condition_C_onsets])
    conditions = ['TaskA'] * n_events_per_condition + \
                ['TaskB'] * n_events_per_condition + \
                ['TaskC'] * n_events_per_condition

    # Create regressor set
    reg_set = regressor_set(onsets=all_onsets, fac=conditions,
                           hrf=get_hrf("spmg1"))
    print(f"Created {len(reg_set)} regressors")

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from fmrimod import regressor_set, get_hrf

    # Simulate experiment
    np.random.seed(123)
    n_events_per_condition = 8
    total_duration = 240

    # Generate onsets
    condition_A_onsets = np.sort(np.random.uniform(0, total_duration, n_events_per_condition))
    condition_B_onsets = np.sort(np.random.uniform(0, total_duration, n_events_per_condition))
    condition_C_onsets = np.sort(np.random.uniform(0, total_duration, n_events_per_condition))

    all_onsets = np.concatenate([condition_A_onsets, condition_B_onsets, condition_C_onsets])
    conditions = ['TaskA'] * n_events_per_condition + \
                ['TaskB'] * n_events_per_condition + \
                ['TaskC'] * n_events_per_condition

    # Create regressor set
    reg_set = regressor_set(onsets=all_onsets, fac=conditions,
                           hrf=get_hrf("spmg1"))

    # Evaluate at scan times (TR = 2s)
    TR = 2
    scan_times = np.arange(0, total_duration + 1, TR)
    
    # Plot design matrix
    plt.figure(figsize=(12, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, (name, reg) in enumerate(zip(reg_set.levels, reg_set.regressors)):
        response = reg.evaluate(scan_times)
        plt.plot(scan_times, response, linewidth=2,
                 color=colors[i], label=name)

        # Add event markers
        task_onsets = all_onsets[np.array(conditions) == name]
        plt.scatter(task_onsets, [-0.1] * len(task_onsets),
                    color=colors[i], s=50, alpha=0.7, marker='|')
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Predicted BOLD Response')
    plt.title('Multi-Condition fMRI Design Matrix\nThree experimental conditions with shared HRF')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

Complex Block Designs with SamplingFrame
----------------------------------------

For more complex experimental designs with multiple blocks or runs, ``SamplingFrame`` provides a way to handle block-relative timing and create design matrices directly.

.. code-block:: python

    from fmrimod import SamplingFrame, regressor_set, get_hrf
    import numpy as np

    # Create a sampling frame for 2 blocks of 120 seconds each
    sframe = SamplingFrame(
        blocklens=[120, 120],  # Two 2-minute blocks
        tr=2                   # 2-second TR
    )
    print(f"Total duration: {sframe.total_duration} seconds")
    print(f"Total samples: {sframe.total_samples}")

    # Block-relative event onsets
    # Block 1: Faces at 10, 50, 90; Houses at 30, 70 seconds
    # Block 2: Faces at 15, 55, 95; Houses at 35, 75 seconds
    
    block1_onsets = [10, 30, 50, 70, 90]
    block1_conditions = ['Faces', 'Houses', 'Faces', 'Houses', 'Faces']
    
    block2_onsets = [15, 35, 55, 75, 95]
    block2_conditions = ['Faces', 'Houses', 'Faces', 'Houses', 'Faces']

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from fmrimod import SamplingFrame, regressor, get_hrf

    # Create sampling frame
    sframe = SamplingFrame(
        blocklens=[120, 120],  # Two 2-minute blocks
        tr=2
    )

    # Convert block-relative to global onsets
    block1_faces = np.array([10, 50, 90])
    block1_houses = np.array([30, 70])
    block2_faces = np.array([15, 55, 95]) + 120  # Add block offset
    block2_houses = np.array([35, 75]) + 120

    # Create regressors
    faces_onsets = np.concatenate([block1_faces, block2_faces])
    houses_onsets = np.concatenate([block1_houses, block2_houses])
    
    faces_reg = regressor(onsets=faces_onsets, hrf=get_hrf("spmg1"))
    houses_reg = regressor(onsets=houses_onsets, hrf=get_hrf("spmg1"))

    # Evaluate
    time_points = np.arange(0, 241, 2)  # Full experiment duration
    faces_response = faces_reg.evaluate(time_points)
    houses_response = houses_reg.evaluate(time_points)

    # Plot with block separation
    plt.figure(figsize=(12, 6))
    plt.plot(time_points, faces_response, linewidth=2, 
             color='#1f77b4', label='Faces')
    plt.plot(time_points, houses_response, linewidth=2, 
             color='#ff7f0e', label='Houses')
    
    # Add block boundary
    plt.axvline(x=120, linestyle='--', color='gray', alpha=0.7, 
                label='Block boundary')
    
    # Add event markers
    plt.scatter(faces_onsets, [0] * len(faces_onsets), 
                color='#1f77b4', s=50, alpha=0.7, marker='o')
    plt.scatter(houses_onsets, [0] * len(houses_onsets), 
                color='#ff7f0e', s=50, alpha=0.7, marker='s')
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Predicted BOLD Response')
    plt.title('Multi-Block Experimental Design\nTwo blocks with different event schedules')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

Summary
-------

In this vignette, we've explored advanced features for fMRI modeling:

- **HRF Libraries** allow systematic exploration of parameter spaces
- **Reconstruction matrices** show how basis coefficients map to HRF shapes
- **Regressor sets** simplify multi-condition experimental designs
- **Sampling frames** handle complex multi-block experiments

These tools provide the flexibility needed for sophisticated fMRI analyses while maintaining a clean, intuitive interface. They're particularly valuable when:

- You need to test different HRF assumptions systematically
- Working with limited data requires regularization approaches
- Your experimental design has complex timing or blocking structure
- You want to understand how basis functions contribute to estimated HRFs