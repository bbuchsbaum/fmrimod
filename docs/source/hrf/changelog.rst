Changelog
=========

v0.1.0 (2024)
--------------

Initial release.

* 20+ HRF types: SPM canonical family, Gamma, Gaussian, B-spline, FIR,
  Fourier, Laguerre, Mexican hat, inverse logit, half-cosine, sine,
  Lindquist-Wager-Ungerleider, boxcar, weighted, and time/identity.
* Pluggable HRF registry with ``get_hrf``, ``register_hrf``, ``remove_hrf``.
* HRF decorators: ``lag_hrf``, ``block_hrf``, ``normalize_hrf``.
* Basis-set generators: ``bspline_generator``, ``fir_generator``,
  ``fourier_generator``, ``daguerre_generator``, ``tent_generator``.
* ``Regressor`` and ``RegressorSet`` for event-related and multi-condition designs.
* ``SamplingFrame`` for multi-block acquisition timing.
* Penalty matrices, reconstruction matrices, HRF libraries, and empirical HRFs.
* Trial-varying HRFs (different HRF per event within a single regressor).
* Built-in plotting via Matplotlib.
* Full R ``fmrihrf`` parity with cross-language golden tests.
