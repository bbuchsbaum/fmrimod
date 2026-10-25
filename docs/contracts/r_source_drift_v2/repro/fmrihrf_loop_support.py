"""R test 'loop block support agrees with an independent finite-kernel integral' (daca7ba)
against fmrimod's conv path and its per-event (list-HRF) path, which is fmrimod's analogue of R eval_loop."""
import numpy as np

from fmrimod.hrf import as_hrf
from fmrimod.regressor import regressor

h = as_hrf(lambda t: (np.asarray(t) >= 0).astype(float), name="flat_tail", span=2.0)
grid = np.array([0, 1, 2, 3, 4, 5.0])
expected = np.array([0, 1, 2, 1, 0, 0.0])
conv = regressor([0.0], hrf=h, duration=2.0).evaluate(grid, precision=0.001)
per_event = regressor([0.0], hrf=[h], duration=2.0).evaluate(grid, precision=0.001)
print("expected (finite kernel):", expected.tolist())
print("fmrimod conv             :", np.round(conv, 3).tolist(), "max|diff|", round(float(np.max(np.abs(conv - expected))), 4))
print("fmrimod list-HRF path    :", np.round(per_event, 3).tolist(), "max|diff|", round(float(np.max(np.abs(per_event - expected))), 4))
