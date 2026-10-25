"""LSS-03: run labels for prewhitening inferred from the sampling frame.

R lss_design() (already at f851fb0) and, since 6a2588b, sbhm_prepass()/
lss_sbhm() fill prewhiten$runs from the sampling-frame blocks when the caller
does not supply them, so AR estimation and filtering never span a run
boundary. fmrimod's estimate_single_trial_from_dataset forwards the
PrewhitenConfig unchanged (fmrimod/single/__init__.py:430-445), so with
runs=None the two runs are treated as one continuous series.
"""
import numpy as np
import pandas as pd

import fmrimod as fm
from fmrimod.single import estimate_single_trial_from_dataset
from fmrimod.single._prewhiten import PrewhitenConfig


def ar1(e, phi):
    out = np.empty_like(e); acc = 0.0
    for t, v in enumerate(e):
        acc = v + phi * acc; out[t] = acc
    return out


def main(seed=7):
    rng = np.random.default_rng(seed)
    n_run, V, tr = 100, 5, 2.0
    on = np.array([10.0, 50.0, 90.0, 130.0])
    events = pd.DataFrame({"onset": np.r_[on, on], "duration": 0.0,
                           "run": np.repeat([1, 2], len(on))})
    # Each run has its own AR(1) noise and a large run-specific offset.
    Y = np.vstack([np.column_stack([ar1(rng.standard_normal(n_run), 0.5) for _ in range(V)]) + off
                   for off in (0.0, 25.0)])
    ds = fm.fmri_dataset(Y, tr=tr, events=events, run_length=[n_run, n_run])
    runs = np.repeat([1, 2], n_run)

    # Run intercepts supplied explicitly as confounds (R lss_design adds them as Z).
    run_dummies = np.column_stack([(runs == r).astype(float) for r in (1, 2)])
    no_runs = estimate_single_trial_from_dataset(ds, "trialwise()", method="lss",
                                                 confounds=run_dummies,
                                                 prewhiten=PrewhitenConfig(p=1))
    with_runs = estimate_single_trial_from_dataset(ds, "trialwise()", method="lss",
                                                   confounds=run_dummies,
                                                   prewhiten=PrewhitenConfig(p=1, runs=runs))
    p0 = no_runs.extra.whitening_plan
    p1 = with_runs.extra.whitening_plan
    print("plan.runs (no runs supplied):", None if p0.runs is None else np.unique(p0.runs))
    print("phi without runs:", np.round(np.asarray(p0.phi[0]), 4))
    print("phi with frame runs (what R fills in):", np.round(np.asarray(p1.phi[0]), 4))
    print("max |beta diff|:", float(np.max(np.abs(no_runs.betas - with_runs.betas))))


if __name__ == "__main__":
    main()
