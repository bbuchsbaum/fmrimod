Migrating from fmrireg
======================

This guide maps the common R ``fmrireg`` workflows to the current Python
``fmrimod`` surface. It is written as a migration aid, not as a claim that every
R helper has a one-to-one Python function.

The Python API keeps the same modeling stages but makes the data contracts more
explicit:

1. Build an event design with ``event_model`` and a nuisance/drift design with
   ``baseline_model``.
2. Combine design and data in ``FmriModel``.
3. Fit with ``fmri_lm`` and an ``FmriLmConfig``.
4. Compute contrasts from the fitted ``FmriLm`` result.
5. Export images with ``write_results`` when a mask and affine are available.

First-Level Model
-----------------

R ``fmrireg`` code often starts with ``fmri_model()`` or
``create_fmri_model()`` and then calls ``fmri_lm()``. In Python, build the
components explicitly and pass an ``FmriModel`` to ``fmri_lm``.

.. code-block:: python

   import numpy as np
   import pandas as pd

   import fmrimod as fm
   from fmrimod.dataset.adapters import NumpyAdapter
   from fmrimod.dataset import FmriDataset
   from fmrimod.model import FmriModel, FmriLmConfig

   events = pd.DataFrame({
       "onset": [8.0, 24.0, 40.0, 56.0],
       "condition": ["face", "house", "face", "house"],
       "run": [0, 0, 0, 0],
   })

   n_scans = 80
   tr = 2.0
   data = np.random.default_rng(1).standard_normal((n_scans, 200))

   sframe = fm.SamplingFrame(blocklens=n_scans, tr=tr)
   event = fm.event_model(
       "onset ~ hrf(condition, basis='spmg1')",
       data=events,
       sampling_frame=sframe,
       durations=1.0,
   )
   baseline = fm.baseline_model("poly", degree=2, sframe=sframe)

   dataset = FmriDataset(NumpyAdapter(data, sframe), event_table=events)
   model = FmriModel(event, baseline, dataset)

   fit = fm.fmri_lm(model, FmriLmConfig())
   betas = fit.coef()       # shape: n_columns x n_voxels
   tstats = fit.tstat()     # coefficient-wise t statistics

Result Accessors
----------------

The fitted result still exposes Python methods such as ``fit.coef()`` and
``fit.tstat()``, but R-style accessor names are available for migration code and
interactive summaries.

.. code-block:: python

   names = fm.coef_names(fit)
   se = fm.standard_error(fit)
   p = fm.p_values(fit)
   z = fm.zscores(fit)
   table = fm.tidy(fit)

   # Reconstruct a coefficient vector into a mask-shaped array.
   slope_img = fm.coef_image(fit, coef=names[0], mask=mask_bool_array)

   # Inspect stored contrast results after fit.contrast(...) calls.
   contrast_names = fm.get_contrasts(fit)

For models with HRF basis terms, ``fitted_hrf`` and ``tidy_fitted_hrf`` provide
best-effort fitted response reconstructions from the fitted coefficients.

.. code-block:: python

   hrf_curves = fm.fitted_hrf(fit, sample_at=np.arange(0.0, 24.0, 0.5))
   hrf_table = fm.tidy_fitted_hrf(fit, sample_at=np.arange(0.0, 24.0, 0.5))

Fit Provenance
--------------

New ``fmri_lm`` results carry ``fit.provenance`` as a typed ``FitProvenance``
object. Slice A records the fmrimod version, resolved solver path, and HRF
normalization modes at fit time. Seed, AR configuration, and mask mode are
present as value slots with explicit status fields while their wiring lands in
follow-up slices. Existing serialized or in-memory ``FmriLm`` objects from older
versions may not have this attribute populated.

Replay Comparisons
------------------

Use ``fm.replay_fits`` when two ``FmriLm`` objects already exist and should be
compared without recomputing a large first-level model. By default it compares
the intersection of compatible named contrasts and reports names dropped from
either side. Pass ``named_contrasts=...`` to require exact named contrast
availability; missing names, empty intersections, statistic-shape mismatches,
or contrast degrees-of-freedom mismatches raise ``ReplayContractError``.

``fm.replay`` is the convenience facade for fitting two specs on the same
dataset and then delegating to ``replay_fits``.

.. code-block:: python

   result = fm.replay_fits(fit_a, fit_b)
   for delta in result.contrast_deltas:
       print(delta.name, delta.max_abs_delta, delta.median_abs_delta)

Configuration: OLS, AR, Robust, Weights
---------------------------------------

R ``fmri_lm_control()`` maps to ``fmrimod.model.config.FmriLmConfig`` and the
``fmri_lm_control`` convenience factory.

.. code-block:: python

   from fmrimod.model.config import fmri_lm_control

   ols_cfg = fmri_lm_control()
   ar_cfg = fmri_lm_control(
       ar_options={"enabled": True, "order": 1, "method": "yule_walker"}
   )
   robust_cfg = fmri_lm_control(
       robust_options={"enabled": True, "type": "huber"}
   )

   fit = fm.fmri_lm(model, ar_cfg)

The default engine is ``"runwise"``. Use ``engine="chunkwise"`` for
voxel-chunked fitting and ``engine="sketch"`` for randomized low-rank fitting.
AR and robust settings are configuration options, not separate top-level
fitting functions.

Low-level fitting helpers from R ``fmrireg`` are available for migration and
engine integration, but they return Python ``FmriLm`` results or pandas tables
instead of R lists/tibbles.

.. code-block:: python

   # External engines can fit a transformed response matrix with a model design.
   fit2 = fm.fit_glm_with_config(model, transformed_Y, cfg=ols_cfg)

   # Streaming engines can build a result from sufficient statistics.
   fit3 = fm.fit_glm_from_suffstats(model, XtX, XtS, StS, df=n_scans - n_cols)

   # Matrix-level contrast helpers are also available.
   contrast_table = fm.compute_lm_contrasts(
       fit.betas,
       fit.XtXinv,
       fit.residual_df,
       sigma=fit.sigma,
       t_contrasts={"face_gt_house": contrast_vector},
   )

``fmri_rlm`` is represented as a Pythonized robust wrapper around the explicit
model/config contract. For new code, prefer ``fm.fmri_lm(model, robust_cfg)``.

Soft Subspace Projection
------------------------

R ``soft_projection`` and ``apply_soft_projection`` map to a small Python
projection object.

.. code-block:: python

   proj = fm.soft_projection(nuisance_matrix, lam="auto")
   cleaned = fm.apply_soft_projection(proj, Y, X)
   Y_clean = cleaned["Y"]
   X_clean = cleaned["X"]

The integrated ``fmri_lm`` path can also apply soft subspace projection through
``FmriLmConfig.soft_subspace`` or the ``soft_subspace_options`` factory.

Contrasts
---------

R workflows that call ``fit_contrasts()`` or extract named contrasts should use
methods on the fitted ``FmriLm`` object.

.. code-block:: python

   # Explicit t contrast over design columns.
   c = np.zeros(fit.n_coefficients)
   c[0] = 1.0
   t_result = fit.contrast(c, name="column_0")

   # Batch several t/F contrasts.
   out = fit.compute_contrasts({
       "column_0": c,
       "omnibus": np.eye(fit.n_coefficients),
   })

If the event model defines named contrast weights, ``fit.contrast("name")`` can
look them up through ``model.contrast_weights()``. ``fit_contrasts`` remains
available for migration code, but new code should prefer result methods.

Writing Results
---------------

R ``write_results()`` maps to ``fmrimod.write_results``. The Python function
writes BIDS-style beta and contrast images. It needs a mask and affine either
from the dataset adapter or from explicit arguments.

.. code-block:: python

   paths = fm.write_results(
       fit,
       "derivatives/fmrimod",
       subject="01",
       task="localizer",
       mask=mask_bool_array,
       affine=np.eye(4),
       overwrite=True,
   )

The writer refuses to overwrite existing files unless ``overwrite=True``.

LSA and LSS
-----------

R ``fmrireg`` single-trial workflows map to ``fmrimod.betas`` and
``fmrimod.single``. The compatibility dispatcher is ``estimate_betas``.

.. code-block:: python

   from fmrimod import estimate_betas

   # trial_regressors: n_time x n_trials
   # Y: n_time x n_voxels
   lss = estimate_betas(trial_regressors, Y, method="lss")
   lsa = estimate_betas(trial_regressors, Y, method="ols")

Use ``method="lss"`` for least-squares-separate estimates and ``method="ols"``
for the LSA-style all-trials-at-once model.

Simulation
----------

Simulation helpers are exposed at the top level for R migration:

.. code-block:: python

   sim = fm.simulate_bold_signal(ncond=2, nreps=8, tr=2.0, seed=4)
   noise = fm.simulate_noise_vector(n=200, tr=2.0, seed=4)
   matrix = fm.simulate_fmri_matrix(ncond=2, nreps=8, nvoxels=100, seed=4)

For lower-level control, use ``fmrimod.simulate.bold.simulate_bold`` and the
noise utilities in ``fmrimod.simulate.noise``.

Group and Meta Workflows
------------------------

Group-level R workflows map to ``group_data`` constructors plus ``fmri_meta``.
The strongest supported path today is effect-size data in tabular or file-backed
form.

.. code-block:: python

   group = fm.group_data_from_csv("subject_effects.csv")
   meta = fm.fmri_meta(group, formula="~ 1", method="pm")

Supported constructors include ``group_data_from_csv``, ``group_data_from_h5``,
``group_data_from_nifti``, and ``group_data_from_fmrilm``. Treat the current
meta API as a scoped first slice: inverse-variance and Paule-Mandel style
workflows are represented, while robust meta estimation and t-only combine
modes are intentionally not implemented yet.

Group-data accessors mirror the common R helper names:

.. code-block:: python

   subjects = fm.get_subjects(group)
   n = fm.n_subjects(group)
   covariates = fm.get_covariates(group)
   rois = fm.get_rois(group)
   contrasts = fm.get_contrasts(group)

Dataset, IO, and Benchmark Helpers
----------------------------------

R helper names from ``fmridataset`` are represented by explicit Python
constructors and readers.

.. code-block:: python

   dataset = fm.fmri_mem_dataset(data_matrix, tr=2.0)
   chunks = fm.data_chunks(dataset, nchunks=4)

   csv_payload = fm.extract_csv_data(group, roi="V1", contrast="faces")
   h5_array = fm.read_h5_full(group_h5, stat=["beta", "se"])
   nifti_payload = fm.read_nifti_full(group_nifti)

``latent_dataset`` and ``fmri_latent_lm`` are Pythonized around score/loadings
matrices. They are useful for migration and tests, but they do not attempt to
replicate R's ``LatentNeuroVec`` object model.

Benchmark helpers provide deterministic Python fixtures:

.. code-block:: python

   fixtures = fm.list_benchmark_datasets()
   bench = fm.load_benchmark_dataset("BM_Canonical_HighSNR")
   X = fm.create_design_matrix_from_benchmark("BM_Canonical_HighSNR")
   perf = fm.evaluate_method_performance(
       "BM_Canonical_HighSNR",
       bench["true_betas_condition"],
       method_name="oracle",
   )

The R ``design_plot`` Shiny helper is represented by ``fm.design_plot()``, which
returns a long-form pandas table suitable for matplotlib, seaborn, plotly, or
another plotting system.

Low-Level Meta Helpers
----------------------

The preferred second-level entry points are ``group_data`` plus ``fmri_meta`` or
``group_fit``. Low-level R helper names are also available for migration code
that already works with effect-size matrices.

.. code-block:: python

   out = fm.fmri_meta_fit(Y, V, X, method="fe")
   out_cov = fm.fmri_meta_fit_cov(Y, V, X, method="fe")
   out_con = fm.fmri_meta_fit_contrasts(Y, V, X, Cmat, method="fe")
   out_ext = fm.fmri_meta_fit_extended(Y, V, X, voxelwise=voxel_covariates)
   neff = fm.meta_effective_n(v=np.repeat(0.05, 4), tau2=0.01)

These helpers return dictionaries of NumPy arrays, keeping the low-level shape
contract explicit while leaving richer result objects to ``fmri_meta`` and
``group_fit``.

Deliberate Differences
----------------------

The migration is not a literal R S3 port.

* R control lists become dataclasses and option factories such as
  ``FmriLmConfig`` and ``fmri_lm_control``.
* Fitted values, standard errors, t statistics, and contrasts are methods or
  attributes on ``FmriLm``. R-style accessor functions are provided for common
  result and group-data summaries, but they delegate to the Python objects.
* Contrast fitting is result-owned: use ``fit.contrast(...)`` and
  ``fit.compute_contrasts(...)``.
* Low-level Rcpp exports are not stable public Python APIs. Dataset IO,
  benchmark, and suffstat helpers that are useful for migration are exposed as
  explicit Python wrappers.
* Dataset I/O is adapter-based. Prefer ``FmriDataset`` plus explicit adapters
  over implicit file inspection.
* Second-level support is scoped to the current ``group_data`` and
  ``fmri_meta`` surface until the remaining fmrireg inventory rows are resolved.

Current Tracking
----------------

The detailed export-by-export status lives in
``docs/contracts/trio_api_inventory_v1.md``. Historical work groupings
are archived in ``docs/contracts/archive/trio_pending_api_workplan_v1.md``;
the active picker is now flagship-workflow demand, tracked through the
governance loop in ``GOVERNANCE.md`` rather than a pending-row queue.
