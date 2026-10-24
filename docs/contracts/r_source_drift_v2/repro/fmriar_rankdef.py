"""Wave1 carry-over spot-check: acvf_bias_matrix on a rank-deficient design.

R .acvf_bias_by_run uses qr() (dqrdc2, column pivoting of near-collinear
columns) and Q[, seq_len(rank)], so the bias map depends only on the column
space. fmrimod acvf.py:384-386 takes unpivoted numpy QR and Q[:, :rank].
R 0.3.3 ca27f27 tests rank-deficient designs (duplicate + zero columns)."""

import numpy as np

from fmrimod.ar.acvf import acvf_bias_matrix

rng = np.random.default_rng(708)
n = 90
t = np.linspace(-1, 1, n)
X = np.column_stack([np.ones(n), t, t ** 2, t ** 3, rng.standard_normal(n)])
A_full = acvf_bias_matrix(X, max_lag=6)[0]
for name, Xrd in (
    ("dup col 2 appended at end", np.column_stack([X, X[:, 1]])),
    ("dup col 2 inserted at position 2", np.column_stack([X[:, :2], X[:, 1], X[:, 2:]])),
    ("zero col inserted at position 1", np.column_stack([X[:, :1], np.zeros(n), X[:, 1:]])),
):
    A_rd = acvf_bias_matrix(Xrd, max_lag=6)[0]
    print(f"{name:34s} max|A(X_rd) - A(X)| = {np.max(np.abs(A_rd - A_full)):.3g}")
