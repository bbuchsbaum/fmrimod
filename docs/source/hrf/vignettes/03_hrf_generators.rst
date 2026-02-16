HRF Generators
==============

Why Generators?
---------------

Most pre-defined HRFs in ``fmrimod`` (like ``spmg1`` or ``gaussian``) are ready-to-use objects. However, some HRFs are actually *generators*. A generator is a function that creates a new HRF object when you call it. This allows you to specify the number of basis functions (``N``) and the time span (``span``) at creation time.

The library provides generators for flexible basis sets such as B-splines and finite impulse response (FIR) models. They are available through ``list_available_hrfs()`` and can be created using ``gen_hrf()``.

.. code-block:: python

    from fmrimod import list_available_hrfs
    import pandas as pd

    # List all available HRFs with details
    hrfs = list_available_hrfs(details=True)
    
    # Filter to show only generators
    generators = hrfs[hrfs['type'] == 'generator']
    print(generators[['name', 'description', 'nbasis', 'type']])

Creating a Basis with a Generator
---------------------------------

To obtain an actual HRF object from a generator, use the ``gen_hrf()`` function with your desired parameters. For example, to create a B-spline basis with 8 functions spanning 32 seconds:

.. code-block:: python

    from fmrimod import gen_hrf
    from fmrimod.hrf.functions import bspline_hrf
    import numpy as np

    # Create a B-spline basis using gen_hrf
    bs8 = gen_hrf(bspline_hrf, n_basis=8, span=32, name="B-spline N=8")
    print(f"HRF name: {bs8.name}")
    print(f"Number of basis functions: {bs8.nbasis}")

    # The returned value is a standard HRF object
    times = np.linspace(0, 32, 65)
    mat = bs8(times)
    print(f"Shape of basis matrix: {mat.shape}")

Evaluating Basis Functions
--------------------------

Once created, basis functions can be evaluated at any time points:

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from fmrimod import gen_hrf
    from fmrimod.hrf.functions import bspline_hrf

    # Create a B-spline basis with 8 functions
    bs8 = gen_hrf(bspline_hrf, n_basis=8, span=32)
    
    # Evaluate at time points
    times = np.linspace(0, 32, 200)
    basis_matrix = bs8(times)
    
    # Plot all basis functions
    plt.figure(figsize=(10, 6))
    for i in range(basis_matrix.shape[1]):
        plt.plot(times, basis_matrix[:, i], linewidth=2, label=f'Basis {i+1}')
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Response')
    plt.title('B-spline Basis Functions (N=8, span=32s)')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

Visualizing FIR Basis Functions
-------------------------------

The Finite Impulse Response (FIR) basis is another common choice for flexible HRF modeling:

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from fmrimod import get_hrf, gen_hrf
    from fmrimod.hrf.functions import fir_basis

    # Use the pre-defined FIR basis or create a custom one
    fir_default = get_hrf("fir")  # Pre-defined FIR
    fir_custom = gen_hrf(fir_basis, n_basis=10, span=20)  # Custom FIR
    
    times = np.linspace(0, 20, 200)
    
    # Evaluate both
    resp_default = fir_default(times)
    resp_custom = fir_custom(times)
    
    # Plot custom FIR basis
    plt.figure(figsize=(10, 6))
    for i in range(resp_custom.shape[1]):
        plt.plot(times, resp_custom[:, i], linewidth=2, label=f'Bin {i+1}')
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Response')
    plt.title('Finite Impulse Response Basis (N=10, span=20s)')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

Comparing Different Basis Configurations
----------------------------------------

You can create multiple configurations of the same basis type to compare their properties:

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from fmrimod import gen_hrf
    from fmrimod.hrf.functions import bspline_hrf

    # Create B-spline bases with different numbers of functions
    bs4 = gen_hrf(bspline_hrf, n_basis=4, span=25)
    bs6 = gen_hrf(bspline_hrf, n_basis=6, span=25)
    bs10 = gen_hrf(bspline_hrf, n_basis=10, span=25)
    
    times = np.linspace(0, 25, 200)
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    
    # Plot each basis set
    for ax, (basis, name) in zip(axes, [(bs4, 'N=4'), (bs6, 'N=6'), (bs10, 'N=10')]):
        mat = basis(times)
        for i in range(mat.shape[1]):
            ax.plot(times, mat[:, i], linewidth=2)
        ax.set_ylabel('Response')
        ax.set_title(f'B-spline Basis {name}')
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time (seconds)')
    plt.tight_layout()
    plt.show()

Using Basis Sets in Regression Models
-------------------------------------

Generated basis sets can be used with regressors just like any other HRF:

.. code-block:: python

    from fmrimod import regressor, gen_hrf
    from fmrimod.hrf.functions import bspline_hrf
    import numpy as np

    # Create a custom B-spline basis
    custom_basis = gen_hrf(bspline_hrf, n_basis=6, span=24)
    
    # Use it in a regressor
    onsets = [5, 15, 25, 35, 45]
    reg = regressor(onsets=onsets, hrf=custom_basis)
    
    # Evaluate the regressor
    times = np.arange(0, 60, 0.1)
    design_matrix = reg.evaluate(times)
    
    print(f"Design matrix shape: {design_matrix.shape}")
    print(f"Columns represent {custom_basis.nbasis} basis functions")

Custom Generator Functions
--------------------------

You can also create your own generator functions that produce HRF objects:

.. code-block:: python

    from fmrimod import HRF
    import numpy as np
    from scipy import stats

    def gaussian_basis_generator(n_basis=5, span=20, min_mean=2, max_mean=15):
        """Generate a basis set of Gaussian functions with different peaks."""
        means = np.linspace(min_mean, max_mean, n_basis)
        sd = span / (2 * n_basis)  # Adjust SD based on span and number of bases

        def basis_fun(t):
            t = np.asarray(t)
            result = np.zeros((len(t), n_basis))
            for i, mean in enumerate(means):
                result[:, i] = stats.norm.pdf(t, loc=mean, scale=sd)
            return result

        return HRF(
            basis_fun,
            nbasis=n_basis,
            name=f"Gaussian Basis (N={n_basis})"
        )

    # Use the custom generator
    gauss_basis = gaussian_basis_generator(n_basis=6, span=25)
    print(f"Created: {gauss_basis.name}")

Summary
-------

Generator functions are simple factories that let you customize flexible HRF bases. Key points:

- Generators create HRF objects with customizable parameters (N, span)
- Common generators include B-splines and FIR bases
- Generated HRFs work exactly like pre-defined HRFs
- You can create custom generators for specialized applications
- Basis sets are particularly useful for data-driven HRF estimation

The flexibility of generators allows you to adapt the temporal resolution and complexity of your HRF model to match your experimental design and hypotheses.