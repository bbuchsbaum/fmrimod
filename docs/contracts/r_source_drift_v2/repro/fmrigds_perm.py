"""GDS: permutation p-value convention (7c7ecae, test-review-numeric.R:42-94).

Fixed R (perm_onesample_t_cpp / perm_twosample_t_cpp):
  skip_first = generated ? include_observed : identical(mat[1, ], identity_row)
  null rows  = rows (skip_first ? 2 : 1) .. P
  p          = (1 + #{null more extreme than obs}) / (1 + n_null)
  two-sample observed t always from the true labelling `group`.
"""
import numpy as np

from fmrimod.group import (
    SampleLabelSpace,
    group_dataset,
    perm_onesample,
    perm_twosample,
)


def ds(beta_mat):
    n_sub, n_feat = beta_mat.shape
    return group_dataset(
        {"beta": beta_mat.T[:, :, None]},
        space=SampleLabelSpace([f"v{i}" for i in range(n_feat)]),
        subjects=[f"s{i}" for i in range(n_sub)],
        contrasts=["c1"],
    )


def t1(y):
    return y.mean() / (y.std(ddof=1) / np.sqrt(len(y)))


def welch(y, g):
    a, b = y[g == 1], y[g == 0]
    return (a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))


rng = np.random.default_rng(22)
n = 8
beta = rng.normal(0.3, 1.0, size=(n, 2))

print("== one-sample, supplied signs with identity row 1 (R test lines 67-81) ==")
signs = rng.choice([-1, 1], size=(20, n)).astype(np.int8)
signs[0, :] = 1
out = perm_onesample(ds(beta), signs=signs)
for b in range(2):
    tobs = t1(beta[:, b])
    tnull = np.array([t1(s * beta[:, b]) for s in signs[1:]])
    r_p = (1 + np.sum(np.abs(tnull) >= abs(tobs))) / (1 + len(tnull))
    print(f"feature {b}: fmrimod p_perm={out.assay('p_perm').ravel()[b]:.6f}  R p_perm={r_p:.6f}")

print("\n== one-sample, generated, include_observed=True ==")
out_inc = perm_onesample(ds(beta), n_perm=50, seed=4, include_observed=True)
rng2 = np.random.default_rng(4)
sm = rng2.choice(np.array([-1, 1], dtype=np.int8), size=(50, n), replace=True)
sm[0, :] = 1
for b in range(2):
    tobs = t1(beta[:, b])
    tnull = np.array([t1(s * beta[:, b]) for s in sm[1:]])
    r_p = (1 + np.sum(np.abs(tnull) >= abs(tobs))) / (1 + len(tnull))
    print(f"feature {b}: fmrimod p_perm={out_inc.assay('p_perm').ravel()[b]:.6f}  "
          f"R-rule on the same sign matrix={r_p:.6f}")

print("\n== two-sample, include_observed=False (all P rows are null draws in R) ==")
group = np.repeat([0, 1], 4)
rng3 = np.random.default_rng(9)
gm = np.vstack([rng3.permutation(group) for _ in range(15)]).astype(np.int8)
gm[0] = group[::-1]  # row 1 is NOT the observed labelling
out2 = perm_twosample(ds(beta), group=group, group_mat=gm)
for b in range(2):
    tobs = welch(beta[:, b], group)
    tnull_all = np.array([welch(beta[:, b], g) for g in gm])
    r_p = (1 + np.sum(np.abs(tnull_all) >= abs(tobs) - 1e-12)) / (1 + len(tnull_all))
    print(f"feature {b}: fmrimod p_perm={out2.assay('p_perm').ravel()[b]:.6f}  R p_perm={r_p:.6f}"
          f"  (row-1 |t| = {abs(tnull_all[0]):.3f} vs |t_obs| = {abs(tobs):.3f})")
