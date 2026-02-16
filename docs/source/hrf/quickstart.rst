Quick Start Guide
=================

This guide will get you started with fmrimod in just a few minutes.

Basic Usage
-----------

Creating and Using HRFs
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from fmrimod import get_hrf
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Get a pre-defined HRF
    hrf = get_hrf("spmg1")  # SPM canonical HRF
    
    # Evaluate at time points
    t = np.linspace(0, 30, 300)
    response = hrf(t)
    
    # Plot
    plt.plot(t, response)
    plt.xlabel('Time (s)')
    plt.ylabel('HRF Response')
    plt.title('SPM Canonical HRF')
    plt.show()

Creating Regressors
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from fmrimod import regressor, SamplingFrame
    
    # Define event onsets (in seconds)
    event_times = [10, 30, 50, 70, 90]
    
    # Create a regressor
    reg = regressor(
        onsets=event_times,
        hrf="spmg1",
        duration=2.0  # 2-second events
    )
    
    # Create sampling frame (fMRI acquisition timing)
    sf = SamplingFrame(blocklens=60, tr=2.0)  # 60 scans, TR=2s
    
    # Evaluate regressor at scan times
    timeseries = reg.evaluate(sf.samples)

Multi-Condition Designs
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from fmrimod import regressor_set
    
    # Events with conditions
    onsets = [5, 15, 25, 35, 45, 55]
    conditions = ['A', 'B', 'A', 'B', 'A', 'B']
    
    # Create regressor set
    reg_set = regressor_set(
        onsets=onsets,
        fac=conditions,
        hrf="spmg1"
    )
    
    # Get design matrix
    design_matrix = reg_set.evaluate(sf.samples)
    print(f"Design matrix shape: {design_matrix.shape}")
    # Output: (60, 2) - one column per condition

Using Different HRF Types
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from fmrimod import list_available_hrfs
    
    # See all available HRFs
    print(list_available_hrfs())
    
    # Use different HRF types
    hrf_gamma = get_hrf("gamma")
    hrf_gaussian = get_hrf("gaussian")
    hrf_spmg3 = get_hrf("spmg3")  # SPM + derivatives
    
    # SPMG3 has multiple basis functions
    print(f"SPMG3 basis functions: {hrf_spmg3.nbasis}")  # Output: 3

Custom HRF Parameters
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from fmrimod import gen_hrf
    from fmrimod.hrf.functions import hrf_gamma
    
    # Create custom gamma HRF
    custom_hrf = gen_hrf(
        hrf_gamma,
        shape=6,
        rate=1,
        name="Custom Gamma"
    )
    
    # Use in regressor
    reg_custom = regressor(
        onsets=[10, 30, 50],
        hrf=custom_hrf
    )

Next Steps
----------

- Read the :doc:`vignettes/01_hemodynamic_response` for a comprehensive HRF tutorial
- Explore :doc:`vignettes/02_building_regressors` for advanced regressor usage
- Check the :doc:`api/index` for detailed function documentation