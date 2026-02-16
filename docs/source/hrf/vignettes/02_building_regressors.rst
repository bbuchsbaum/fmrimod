Building fMRI Regressors
========================

Introduction: What is a Regressor?
----------------------------------

In fMRI analysis, a **regressor** (or predictor) represents the expected BOLD signal timecourse associated with a specific experimental condition or event type. It's typically created by convolving a series of event onsets (often represented as delta functions or "sticks") with a hemodynamic response function (HRF).

``fmrimod`` provides the ``regressor()`` function to easily create these objects from event timings and an HRF. While these regressor objects are often constructed automatically by modeling functions in other packages, this vignette explores how to create and manipulate them directly, offering finer control over the model components.

Basic Regressor from Event Onsets
---------------------------------

Suppose we have a simple event-related fMRI design with stimuli presented every 12 seconds. We want to model these events using the SPM canonical HRF (``spmg1``). The events are brief, so we model them with a duration of 0 seconds (instantaneous).

.. code-block:: python

    from fmrimod import regressor, get_hrf
    import numpy as np

    # Define event onsets
    onsets = np.arange(0, 10 * 12 + 1, 12)

    # Create the regressor object
    # Uses spmg1 HRF
    # Duration is 0 by default
    reg1 = regressor(onsets=onsets, hrf=get_hrf("spmg1"))

    # Access components
    print(f"Number of events: {len(reg1.onsets)}")
    print(f"First few onsets: {reg1.onsets[:5]}")
    print(f"Number of basis functions: {reg1.nbasis}")

Evaluating and Plotting a Regressor
-----------------------------------

A ``regressor`` object stores the event information but doesn't automatically compute the timecourse. To get the predicted BOLD signal at specific time points (e.g., corresponding to scan acquisition times), we use the ``evaluate()`` method.

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from fmrimod import regressor, get_hrf

    # Define event onsets
    onsets = np.arange(0, 10 * 12 + 1, 12)
    reg1 = regressor(onsets=onsets, hrf=get_hrf("spmg1"))

    # Define a time grid corresponding to scan times (e.g., TR=2s)
    TR = 2
    scan_times = np.arange(0, 141, TR)

    # Evaluate the regressor at scan times
    predicted_bold = reg1.evaluate(scan_times)

    # Plot the predicted timecourse
    plt.figure(figsize=(10, 6))
    plt.plot(scan_times, predicted_bold, linewidth=2, label='Predicted BOLD')
    
    # Add vertical lines for event onsets
    for onset in reg1.onsets:
        plt.axvline(x=onset, linestyle='--', color='red', alpha=0.7)
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Predicted Response')
    plt.title('Predicted BOLD Response (SPM HRF)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

Varying Event Durations
-----------------------

Sometimes events have different durations. The ``duration`` argument in ``regressor()`` can take a vector matching the length of ``onsets``.

.. code-block:: python

    # Example onsets and durations
    onsets_var_dur = np.linspace(0, 5 * 12, 6)
    durations_var = np.arange(1, 7)  # Durations increase from 1s to 6s

    # Create regressor with varying durations
    reg_var_dur = regressor(
        onsets=onsets_var_dur, 
        hrf=get_hrf("spmg1"), 
        duration=durations_var
    )

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from fmrimod import regressor, get_hrf

    # Example onsets and durations
    onsets_var_dur = np.linspace(0, 5 * 12, 6)
    durations_var = np.arange(1, 7)  # Durations increase from 1s to 6s

    # Create regressor with varying durations
    reg_var_dur = regressor(
        onsets=onsets_var_dur, 
        hrf=get_hrf("spmg1"), 
        duration=durations_var
    )

    # Evaluate and plot
    TR = 2
    scan_times_dur = np.arange(0, max(onsets_var_dur) + 31, TR)
    pred_var_dur = reg_var_dur.evaluate(scan_times_dur)

    plt.figure(figsize=(10, 6))
    plt.plot(scan_times_dur, pred_var_dur, linewidth=2)
    
    # Add vertical lines for event onsets
    for onset in reg_var_dur.onsets:
        plt.axvline(x=onset, linestyle='--', color='red', alpha=0.7)
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Predicted Response')
    plt.title('Regressor with Varying Event Durations\nDuration increases over time')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

Duration and Summation
~~~~~~~~~~~~~~~~~~~~~~

By default (``summate=True``), the predicted response accumulates if events overlap or have extended duration. Setting ``summate=False`` preserves the same temporal profile but the peak amplitude does not grow with duration.

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from fmrimod import regressor, get_hrf

    # Example onsets and durations
    onsets_var_dur = np.linspace(0, 5 * 12, 6)
    durations_var = np.arange(1, 7)

    # Create regressor with varying durations, summate=False
    reg_var_dur_nosum = regressor(
        onsets=onsets_var_dur, 
        hrf=get_hrf("spmg1"), 
        duration=durations_var,
        summate=False
    )

    # Evaluate and plot
    TR = 2
    scan_times_dur = np.arange(0, max(onsets_var_dur) + 31, TR)
    pred_var_dur_nosum = reg_var_dur_nosum.evaluate(scan_times_dur)

    plt.figure(figsize=(10, 6))
    plt.plot(scan_times_dur, pred_var_dur_nosum, linewidth=2)
    
    # Add vertical lines for event onsets
    for onset in reg_var_dur_nosum.onsets:
        plt.axvline(x=onset, linestyle='--', color='red', alpha=0.7)
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Predicted Response')
    plt.title('Regressor with Varying Durations (summate=False)\nModels saturation, peak height may not increase with duration')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

Varying Event Amplitudes (Parametric Modulation)
-------------------------------------------------

We can model variations in event intensity or some associated parameter by providing an ``amplitude`` vector. This creates a *parametric regressor* where the height of the HRF for each event is scaled by the corresponding amplitude value.

.. code-block:: python

    from scipy.stats import zscore

    # Example onsets and amplitudes (e.g., representing task difficulty)
    onsets_amp = np.linspace(0, 10 * 12, 11)
    amplitudes_raw = np.arange(1, len(onsets_amp) + 1)

    # It's common practice to center the modulator
    amplitudes_scaled = zscore(amplitudes_raw, ddof=1)

    # Create the parametric regressor
    reg_amp = regressor(
        onsets=onsets_amp, 
        hrf=get_hrf("spmg1"), 
        amplitude=amplitudes_scaled
    )

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from fmrimod import regressor, get_hrf
    from scipy.stats import zscore

    # Example onsets and amplitudes
    onsets_amp = np.linspace(0, 10 * 12, 11)
    amplitudes_raw = np.arange(1, len(onsets_amp) + 1)

    # Center the modulator
    amplitudes_scaled = amplitudes_raw - amplitudes_raw.mean()

    # Create the parametric regressor
    reg_amp = regressor(
        onsets=onsets_amp, 
        hrf=get_hrf("spmg1"), 
        amplitude=amplitudes_scaled
    )

    # Evaluate and plot
    TR = 2
    scan_times_amp = np.arange(0, max(onsets_amp) + 31, TR)
    pred_amp = reg_amp.evaluate(scan_times_amp)

    plt.figure(figsize=(10, 6))
    plt.plot(scan_times_amp, pred_amp, linewidth=2, label='Predicted BOLD')
    
    # Add vertical lines for event onsets
    for onset in reg_amp.onsets:
        plt.axvline(x=onset, linestyle='--', color='red', alpha=0.7)
    
    # Add points showing amplitude (scaled for visibility)
    plt.scatter(reg_amp.onsets, reg_amp.amplitude * 0.2,
                color='blue', s=50, zorder=5, label='Amplitude (scaled)')
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Predicted Response')
    plt.title('Parametric Regressor with Varying Amplitude\nAmplitude (centered) increases over time')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

Combining Duration and Amplitude Modulation
-------------------------------------------

You can provide both ``duration`` and ``amplitude`` vectors to model events that vary in both aspects.

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from fmrimod import regressor, get_hrf

    np.random.seed(123)
    onsets_comb = np.linspace(0, 10 * 12, 11)
    amps_comb = np.arange(1, len(onsets_comb) + 1) - 6  # Centered
    durs_comb = np.random.choice(range(1, 6), len(onsets_comb))

    reg_comb = regressor(
        onsets=onsets_comb, 
        hrf=get_hrf("spmg1"), 
        amplitude=amps_comb, 
        duration=durs_comb
    )

    # Evaluate and plot
    TR = 2
    scan_times_comb = np.arange(0, max(onsets_comb) + 31, TR)
    pred_comb = reg_comb.evaluate(scan_times_comb)

    plt.figure(figsize=(10, 6))
    plt.plot(scan_times_comb, pred_comb, linewidth=2)
    
    # Add vertical lines for event onsets
    for onset in reg_comb.onsets:
        plt.axvline(x=onset, linestyle='--', color='red', alpha=0.7)
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Predicted Response')
    plt.title('Regressor with Varying Duration and Amplitude')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

Regressors with HRF Basis Sets
------------------------------

If you use an HRF object with multiple basis functions (e.g., ``spmg3``, ``bspline``), the ``regressor`` object will represent multiple timecourses, one for each basis function. ``evaluate()`` will return a matrix.

.. code-block:: python

    # Use a B-spline basis set
    onsets_basis = np.linspace(0, 10 * 12, 11)
    hrf_basis = get_hrf("bspline")  # Uses n_basis=5 basis functions by default

    reg_basis = regressor(onsets_basis, hrf_basis)
    print(f"Number of basis functions: {reg_basis.nbasis}")  # Should be 5

    # Evaluate - this returns a matrix
    scan_times_basis = np.arange(0, max(onsets_basis) + 31, TR)
    pred_basis_matrix = reg_basis.evaluate(scan_times_basis)
    print(f"Shape: {pred_basis_matrix.shape}")  # (time_points, basis_functions)

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from fmrimod import regressor, get_hrf
    import pandas as pd

    # Use a B-spline basis set
    onsets_basis = np.linspace(0, 10 * 12, 11)
    hrf_basis = get_hrf("bspline")  # Uses n_basis=5 basis functions by default

    reg_basis = regressor(onsets_basis, hrf_basis)

    # Evaluate - this returns a matrix
    TR = 2
    scan_times_basis = np.arange(0, max(onsets_basis) + 31, TR)
    pred_basis_matrix = reg_basis.evaluate(scan_times_basis)

    # Plot each basis function
    plt.figure(figsize=(10, 6))
    for i in range(pred_basis_matrix.shape[1]):
        plt.plot(scan_times_basis, pred_basis_matrix[:, i], 
                 linewidth=2, label=f'Basis {i+1}')
    
    # Add vertical lines for event onsets
    for onset in reg_basis.onsets:
        plt.axvline(x=onset, linestyle='--', color='red', alpha=0.3)
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Predicted Response')
    plt.title('Regressor using B-Spline Basis Set (n_basis=5)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

Shifting Regressors
-------------------

You can temporally shift all onsets within a regressor using the ``shift()`` method.

.. code-block:: python

    # Original regressor
    reg_orig = regressor(onsets=[10, 30, 50], hrf=get_hrf("spmg1"))

    # Shifted regressor (delay by 5 seconds)
    reg_shifted = reg_orig.shift(shift_amount=5)

    print(f"Original onsets: {reg_orig.onsets}")
    print(f"Shifted onsets: {reg_shifted.onsets}")  # Now 15, 35, 55

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from fmrimod import regressor, get_hrf

    # Original regressor
    reg_orig = regressor(onsets=[10, 30, 50], hrf=get_hrf("spmg1"))

    # Shifted regressor (delay by 5 seconds)
    reg_shifted = reg_orig.shift(shift_amount=5)

    # Plot both
    TR = 2
    scan_times_shift = np.arange(0, 81, TR)
    pred_orig = reg_orig.evaluate(scan_times_shift)
    pred_shifted = reg_shifted.evaluate(scan_times_shift)

    plt.figure(figsize=(10, 6))
    plt.plot(scan_times_shift, pred_orig, linewidth=2, label='Original', color='red')
    plt.plot(scan_times_shift, pred_shifted, linewidth=2, label='Shifted +5s', color='blue')
    
    # Add vertical lines for event onsets
    for onset in reg_orig.onsets:
        plt.axvline(x=onset, linestyle='--', color='red', alpha=0.5)
    for onset in reg_shifted.onsets:
        plt.axvline(x=onset, linestyle='--', color='blue', alpha=0.5)
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Predicted Response')
    plt.title('Shifting a Regressor\nOriginal (red) vs. Shifted +5s (blue)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

Summary
-------

In this vignette, we've explored:

- Creating basic regressors from event onsets
- Evaluating regressors at specific time points
- Modeling events with varying durations
- Using parametric modulation with varying amplitudes
- Combining duration and amplitude modulation
- Working with HRF basis sets
- Temporally shifting regressors

These tools provide fine-grained control over fMRI model specification, allowing you to create custom designs and test specific hypotheses about the hemodynamic response.