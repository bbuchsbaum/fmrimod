"""AR-05: residual-bias correction stable under censoring (0.4.1 5f5fc32, bd4e03f).

fmrimod acvf.py:261 solves A gamma = gamma_raw exactly (np.linalg.solve).
R 0.4.2 .solve_correction pins near-null directions (sv < 0.1 * median sv)
by a zero-tail condition. Fixture = test-audit-regressions.R:566-582.
The R side is emulated by monkeypatching fmrimod.ar.acvf.apply_acvf_correction
with the transcribed R function; everything else is fmrimod's own code.
"""

import sys
import warnings

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from fmriar_common import r_apply_acvf_correction, sim_arma  # noqa: E402

import fmrimod.ar.acvf as acvf_mod  # noqa: E402
from fmrimod.ar import fit_noise  # noqa: E402

warnings.simplefilter("ignore")


def drift_design(n, runs):
    L = n // 2
    t = np.arange(1, L + 1)
    dct = np.column_stack([np.cos(np.pi * j * (t - 0.5) / L) for j in range(1, 7)])
    ind = np.column_stack([(runs == r).astype(float) for r in (1, 2)])
    blk = np.block([[dct, np.zeros((L, 6))], [np.zeros((L, 6)), dct]])
    task = np.tile(np.repeat([0.0, 1.0], 10), n // 20)
    return np.column_stack([ind, blk, task])


rng = np.random.default_rng(9204)
n = 300
runs = np.repeat([1, 2], 150)
X = drift_design(n, runs)
orig = acvf_mod.apply_acvf_correction
res = {"raw": [], "fm": [], "r": [], "fm0": [], "r0": []}
for _ in range(30):
    E = sim_arma(n, 30, [0.4], rng=rng)
    R = E - X @ np.linalg.lstsq(X, E, rcond=None)[0]
    cen = np.sort(rng.choice(n, 30, replace=False))
    res["raw"].append(fit_noise(R, runs=runs, p=1, censor=cen).phi[0][0])
    acvf_mod.apply_acvf_correction = orig
    res["fm"].append(fit_noise(R, runs=runs, p=1, censor=cen, design=X).phi[0][0])
    res["fm0"].append(fit_noise(R, runs=runs, p=1, design=X).phi[0][0])
    acvf_mod.apply_acvf_correction = r_apply_acvf_correction
    res["r"].append(fit_noise(R, runs=runs, p=1, censor=cen, design=X).phi[0][0])
    res["r0"].append(fit_noise(R, runs=runs, p=1, design=X).phi[0][0])
acvf_mod.apply_acvf_correction = orig
for k, lab in (("raw", "uncorrected, 10% censor"), ("fm", "fmrimod corrected, 10% censor"),
               ("r", "R0.4.2 corrected, 10% censor"), ("fm0", "fmrimod corrected, no censor"),
               ("r0", "R0.4.2 corrected, no censor")):
    a = np.array(res[k])
    print(f"{lab:32s} mean={a.mean():.3f} sd={a.std(ddof=1):.3f} rmse={np.sqrt(np.mean((a - 0.4) ** 2)):.3f}")
