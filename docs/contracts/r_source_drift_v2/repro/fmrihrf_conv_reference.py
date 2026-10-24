"""Transcription of R test 'conv matches a full convolution of the fine-grid impulse train'
(tests/testthat/test_regressor.R @ fmrihrf 5b63d6a) run against fmrimod's conv engine."""
import numpy as np

from fmrimod.hrf import HRF_SPMG1, HRF_SPMG3
from fmrimod.regressor import regressor


def conv_reference(grid, onsets, amp, hrf, dt):
    span = hrf.span
    hgrid = np.arange(0, span + dt / 1e6, dt)  # seq(0, span, by = dt)
    hmat = np.asarray(hrf(hgrid), float)
    if hmat.ndim == 1:
        hmat = hmat[:, None]
    start = min(grid) - span
    end = max(max(grid), max(onsets)) + span
    n = int(np.floor((end - start) / dt)) + 1
    neural = np.zeros(n + 1)
    pos = (np.asarray(onsets) - start) / dt
    a = np.floor(pos).astype(int)
    for i in np.where(pos >= 0)[0]:
        f = pos[i] - a[i]
        neural[a[i]] += amp[i] * (1 - f)
        neural[a[i] + 1] += amp[i] * f
    neural = neural[:n]
    fine = start + np.arange(n) * dt
    out = []
    for b in range(hmat.shape[1]):
        full = np.convolve(neural, hmat[:, b])[:n]
        out.append(np.interp(grid, fine, full))
    return np.column_stack(out)


onsets = np.array([3.17, 20, 41.9, 42.4, 75])
amp = np.array([1, -0.5, 2, 1.25, 0.8])
grids = {
    "scans": 1 + 2 * np.arange(45.0),
    "denser_than_fine": np.arange(0, 30.0001, 0.1),
    "irregular": np.array([0.5, 0.52, 7, 7, 33.3, 61, 88.8, 120]),
    "single": np.array([44.4]),
}
worst = 0.0
for hrf in (HRF_SPMG1, HRF_SPMG3):
    reg = regressor(onsets, hrf=hrf, amplitude=amp)
    for dt in (0.1, 0.33, 1.0):
        for g, grid in grids.items():
            exp = conv_reference(grid, onsets, amp, hrf, dt)
            got = np.asarray(reg.evaluate(grid, precision=dt))
            if got.ndim == 1:
                got = got[:, None]
            d = float(np.max(np.abs(got - exp)))
            worst = max(worst, d)
            if d > 1e-10:
                print(f"{hrf.name} dt={dt} {g}: max|diff|={d:.3e}")
print(f"worst max|fmrimod - R reference| = {worst:.3e} (R tolerance 1e-10)")
