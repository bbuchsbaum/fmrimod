"""DS start_time / TR checks against R 99d19fe + cce6a44 (temporal_schema / as_sampling_frame).

R (fmridataset HEAD): per-run start_time (non-negative, finite, constant within a run,
may differ across runs) is passed through to fmrihrf::sampling_frame(start_time=);
TR must be positive finite; absent start_time -> TR/2.
"""
import neuroim as ni
import numpy as np

import fmrimod as fm
import fmrimod.dataset as fds

rng = np.random.default_rng(1)
TR = 2.0
st = [1.0, 0.25]          # per-run slice-timing reference, as R would honour

def r_samples(blocklens, tr, start):
    """Transcription of fmrihrf::sampling_frame samples: start_time + (k)*TR per run (relative)."""
    return np.concatenate([s + np.arange(n) * tr for n, s in zip(blocklens, start)])

# 1. matrix_dataset honours per-run start_time (control)
mat = rng.normal(size=(20, 3))
d1 = fds.matrix_dataset(mat, tr=TR, run_length=[10, 10], start_time=st)
print("matrix_dataset start_time:", d1.sampling_frame.start_time.tolist())

# 2. fmri_dataset, 2-D matrix + run_length + per-run start_time
d2 = fds.fmri_dataset(mat, tr=TR, run_length=[10, 10], start_time=st)
print("fmri_dataset(2-D) start_time:", d2.sampling_frame.start_time.tolist(), "(R/expected", st, ")")

# 3. fmri_dataset, two NeuroVec runs + per-run start_time
space = ni.NeuroSpace(dim=(2, 2, 2, 10), spacing=(1, 1, 1, 1), origin=(0, 0, 0, 0))
vecs = [ni.DenseNeuroVec(rng.normal(size=(2, 2, 2, 10)) + 5, space) for _ in range(2)]
d3 = fds.fmri_dataset(vecs, tr=TR, start_time=st)
print("fmri_dataset(NeuroVec x2) start_time:", d3.sampling_frame.start_time.tolist(), "(R/expected", st, ")")
exp = r_samples([10, 10], TR, st)
got = d3.sampling_frame.sample_times(global_time=False)
print("  max |block-relative samples - R| (s):", float(np.max(np.abs(np.asarray(got) - exp))))

# 4. validation the R schema now applies, fed straight to the fmrimod constructors
for label, kw in (("negative start_time", dict(start_time=-1.0)),
                  ("NaN start_time", dict(start_time=float("nan"))),
                  ("NaN tr", dict(tr=float("nan")))):
    args = dict(tr=TR, run_length=[10, 10]); args.update(kw)
    try:
        d = fds.matrix_dataset(mat, **args)
        print(f"{label}: ACCEPTED, start_time={d.sampling_frame.start_time.tolist()} tr={d.sampling_frame.tr.tolist()}")
    except Exception as e:  # noqa: BLE001
        print(f"{label}: refused ({type(e).__name__}: {e})")

# 5. top-level facade forwards start_time?
try:
    fm.matrix_dataset(mat, tr=TR, run_length=[10, 10], start_time=st)
    print("fmrimod.matrix_dataset(start_time=...): accepted")
except TypeError as e:
    print("fmrimod.matrix_dataset(start_time=...): TypeError:", e)

# 6. per-run TR with the default offset: documented default is TR/2 *per run*
d6 = fds.fmri_dataset(mat, tr=[1.5, 2.0], run_length=[10, 10])
d6m = fds.matrix_dataset(mat, tr=[1.5, 2.0], run_length=[10, 10])
print("fmri_dataset(tr=[1.5,2.0]) default start_time:", d6.sampling_frame.start_time.tolist(),
      "| matrix_dataset:", d6m.sampling_frame.start_time.tolist(), "| R/fmrihrf TR/2 per run: [0.75, 1.0]")
