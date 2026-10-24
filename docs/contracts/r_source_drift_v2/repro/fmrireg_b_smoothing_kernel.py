"""REG-B: hrf_smoothing_kernel normalisation (R review C6, 7c2a985).

Fixed R: K = X X'; normalised (or method="cosine") -> K[i,j]/sqrt(K[i,i]K[j,j])
(symmetric, unit diagonal; zero-signal scans -> identity row/col).
fmrimod HEAD: K / diag(K)[:, None] (row scaling, asymmetric); cosine normalises
the *columns* of X (trials), not the scans.
"""
import numpy as np

from fmrimod.glm.compat import hrf_smoothing_kernel

form = "onset ~ trialwise(basis='gaussian')"
for method in ("gram", "cosine"):
    for buf in (0, 3):
        try:
            K = hrf_smoothing_kernel(30, tr=1.5, form=form, method=method,
                                     buffer_scans=buf, normalize=True)
        except Exception as exc:  # report and continue
            print(method, buf, "error:", type(exc).__name__, exc)
            continue
        print(f"{method:6s} buf={buf}: finite={np.all(np.isfinite(K))} "
              f"max|K-K'|={np.max(np.abs(K - K.T)):.4f} diag==1={np.allclose(np.diag(K), 1)} "
              f"max|K|={np.max(np.abs(K)):.3f}")

# Reference: R's fixed formula on the same design.
import pandas as pd

from fmrimod import SamplingFrame, event_model

len_, buf, tr = 25, 2, 2.0
sf = SamplingFrame(blocklens=len_ + 2 * buf, tr=tr)
data = pd.DataFrame({"onset": sf.sample_times(), "block": 1})
em = event_model(form, data=data, sampling_frame=sf, durations=tr)
d = em.design_matrix
X = np.asarray(d() if callable(d) else d, dtype=float)
G = X @ X.T
s = np.where(np.diag(G) > 0, 1 / np.sqrt(np.diag(G)), 0)
ref = G * np.outer(s, s)
np.fill_diagonal(ref, 1)
keep = slice(buf, buf + len_)
Kn = hrf_smoothing_kernel(len_, tr=tr, form=form, buffer_scans=buf, normalize=True, method="gram")
Kc = hrf_smoothing_kernel(len_, tr=tr, form=form, buffer_scans=buf, normalize=False, method="cosine")
print("gram/normalised vs R:", float(np.max(np.abs(Kn - ref[keep, keep]))))
print("cosine/unnormalised vs R:", float(np.max(np.abs(Kc - ref[keep, keep]))))
