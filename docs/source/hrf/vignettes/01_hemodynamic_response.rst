Hemodynamic Response Functions
==============================

A hemodynamic response function (HRF) models the temporal evolution of the fMRI 
BOLD (Blood-Oxygen-Level-Dependent) signal in response to a brief neural event. 
Typically, the BOLD signal peaks 4-6 seconds after the event onset and then 
returns to baseline, often with a slight undershoot.

``fmrimod`` provides tools to define, manipulate, and visualize various HRFs 
commonly used in fMRI analysis.

Pre-defined HRF Objects
-----------------------

``fmrimod`` includes several pre-defined HRF objects, which are essentially 
functions with specific attributes defining their type, number of basis 
functions (``nbasis``), and effective duration (``span``).

Let's look at two common examples: the SPM canonical HRF and a Gaussian HRF.

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from fmrimod import get_hrf

    # Get pre-defined HRFs
    hrf_spm = get_hrf("spmg1")      # SPM canonical HRF
    hrf_gauss = get_hrf("gaussian")  # Gaussian HRF

    print(f"SPM HRF: nbasis={hrf_spm.nbasis}, span={hrf_spm.span}s")
    print(f"Gaussian HRF: nbasis={hrf_gauss.nbasis}, span={hrf_gauss.span}s")

These objects are functions themselves, so you can evaluate them at specific time points:

.. code-block:: python

    time_points = np.linspace(0, 25, 250)
    
    # Evaluate the HRFs
    y_spm = hrf_spm(time_points)
    y_gauss = hrf_gauss(time_points)
    
    # Scale to peak at 1.0 for comparison
    y_spm_scaled = y_spm / np.max(y_spm)
    y_gauss_scaled = y_gauss / np.max(y_gauss)

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from fmrimod import get_hrf
    
    # Get HRFs and evaluate
    hrf_spm = get_hrf("spmg1")
    hrf_gauss = get_hrf("gaussian")
    time_points = np.linspace(0, 25, 250)
    y_spm = hrf_spm(time_points) / np.max(hrf_spm(time_points))
    y_gauss = hrf_gauss(time_points) / np.max(hrf_gauss(time_points))
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(time_points, y_spm, label='SPM Canonical', linewidth=2)
    plt.plot(time_points, y_gauss, label='Gaussian', linewidth=2)
    plt.xlabel('Time (seconds)')
    plt.ylabel('BOLD Response (normalized)')
    plt.title('Comparison of SPM Canonical and Gaussian HRFs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

Note that the ``span`` attribute (e.g., 32 seconds) indicates the approximate 
time window over which the HRF is non-zero.

Modifying HRF Parameters with gen_hrf
-------------------------------------

The ``gen_hrf`` function is a flexible way to create new HRF functions, often 
by modifying the parameters of existing ones.

For example, the ``hrf_gaussian`` function takes ``mean`` and ``sd`` arguments. 
We can use ``gen_hrf`` to create Gaussian HRFs with different peak times and widths.

.. code-block:: python

    from fmrimod import gen_hrf
    from fmrimod.hrf.functions import gaussian_hrf
    
    # Create Gaussian HRFs with different parameters
    hrf_early = gen_hrf(gaussian_hrf, mean=4, sd=1, name="Early Peak")
    hrf_normal = gen_hrf(gaussian_hrf, mean=5, sd=2, name="Normal Peak")
    hrf_late = gen_hrf(gaussian_hrf, mean=7, sd=3, name="Late Peak")

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from fmrimod import gen_hrf
    from fmrimod.hrf.functions import gaussian_hrf
    
    time_points = np.linspace(0, 25, 250)
    
    hrf_early = gen_hrf(gaussian_hrf, mean=4, sd=1)
    hrf_normal = gen_hrf(gaussian_hrf, mean=5, sd=2)
    hrf_late = gen_hrf(gaussian_hrf, mean=7, sd=3)
    
    plt.figure(figsize=(10, 6))
    plt.plot(time_points, hrf_early(time_points), label='Mean=4, SD=1', linewidth=2)
    plt.plot(time_points, hrf_normal(time_points), label='Mean=5, SD=2', linewidth=2)
    plt.plot(time_points, hrf_late(time_points), label='Mean=7, SD=3', linewidth=2)
    plt.xlabel('Time (seconds)')
    plt.ylabel('BOLD Response')
    plt.title('Gaussian HRFs with Different Parameters')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

Modeling Event Duration with block_hrf
--------------------------------------

fMRI events often have a duration (e.g., a stimulus presented for several seconds). 
The ``block_hrf`` function modifies an HRF to model the response to a sustained 
event of a specific ``width`` (duration). Internally, it convolves the original 
HRF with a boxcar function of the specified width.

.. code-block:: python

    from fmrimod import block_hrf
    
    # Create blocked HRFs with different durations
    hrf_spm = get_hrf("spmg1")
    hrf_w1 = block_hrf(hrf_spm, width=1)
    hrf_w2 = block_hrf(hrf_spm, width=2)
    hrf_w4 = block_hrf(hrf_spm, width=4)

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from fmrimod import get_hrf, block_hrf
    
    time_points = np.linspace(0, 30, 300)
    hrf_spm = get_hrf("spmg1")
    
    plt.figure(figsize=(10, 6))
    plt.plot(time_points, hrf_spm(time_points), label='Instantaneous', linewidth=2)
    plt.plot(time_points, block_hrf(hrf_spm, width=1)(time_points), label='Width=1s', linewidth=2)
    plt.plot(time_points, block_hrf(hrf_spm, width=2)(time_points), label='Width=2s', linewidth=2)
    plt.plot(time_points, block_hrf(hrf_spm, width=4)(time_points), label='Width=4s', linewidth=2)
    plt.xlabel('Time (seconds)')
    plt.ylabel('BOLD Response')
    plt.title('SPM Canonical HRF for Different Event Durations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

Normalization
^^^^^^^^^^^^^

By default, longer durations lead to higher peak responses (assuming summation). 
Setting ``normalize=True`` in ``block_hrf`` rescales the response so the peak 
amplitude is approximately 1, regardless of duration.

.. code-block:: python

    # Create normalized blocked HRFs
    hrf_w1_norm = block_hrf(hrf_spm, width=1, normalize=True)
    hrf_w2_norm = block_hrf(hrf_spm, width=2, normalize=True)
    hrf_w4_norm = block_hrf(hrf_spm, width=4, normalize=True)

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from fmrimod import get_hrf, block_hrf
    
    time_points = np.linspace(0, 30, 300)
    hrf_spm = get_hrf("spmg1")
    
    plt.figure(figsize=(10, 6))
    for width in [1, 2, 4, 8]:
        hrf = block_hrf(hrf_spm, width=width, normalize=True)
        plt.plot(time_points, hrf(time_points), label=f'Width={width}s', linewidth=2)
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('BOLD Response')
    plt.title('Normalized SPM Canonical HRF for Different Durations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, None)
    plt.show()

Modeling Saturation with summate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``summate`` argument in ``block_hrf`` controls whether the response
accumulates over the duration (``summate=True``, default) or stays constant
(``summate=False``). When ``summate=False``, the temporal profile is
identical to the summated version but the peak amplitude does not grow
with block duration.

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from fmrimod import get_hrf, block_hrf
    
    time_points = np.linspace(0, 30, 300)
    hrf_spm = get_hrf("spmg1")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 2-second duration
    ax1.plot(time_points, block_hrf(hrf_spm, width=2, summate=True, normalize=True)(time_points), 
             label='Summate=True', linewidth=2)
    ax1.plot(time_points, block_hrf(hrf_spm, width=2, summate=False, normalize=True)(time_points), 
             label='Summate=False', linewidth=2, linestyle='--')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('BOLD Response')
    ax1.set_title('2-Second Duration')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 8-second duration
    ax2.plot(time_points, block_hrf(hrf_spm, width=8, summate=True, normalize=True)(time_points), 
             label='Summate=True', linewidth=2)
    ax2.plot(time_points, block_hrf(hrf_spm, width=8, summate=False, normalize=True)(time_points), 
             label='Summate=False', linewidth=2, linestyle='--')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('BOLD Response')
    ax2.set_title('8-Second Duration')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Effect of Summation on Blocked HRFs', fontsize=14)
    plt.tight_layout()
    plt.show()

Summary
-------

In this vignette, you've learned how to:

1. Use pre-defined HRFs (SPM canonical, Gaussian, etc.)
2. Modify HRF parameters using ``gen_hrf``
3. Model event durations with ``block_hrf``
4. Control normalization and summation behavior
5. Create custom HRF functions for specific needs

These HRF tools form the foundation for building fMRI regression models, 
which we'll explore in the next vignette.