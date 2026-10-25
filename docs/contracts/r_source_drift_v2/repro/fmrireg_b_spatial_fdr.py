"""REG-B: spatial_fdr vs fixed R (fmrireg c8c6021 + af12983 == fmrigds spatial_fdr_sabha).

Fixed R, p-value input, no neighbors:
  m_g counts finite p only; pi0_g = clamp(#(p>tau) / ((1-tau) m_g), 0, 1),
  empty group -> 1; pi0 = clamp(pi0, min_pi0, 1); w_i = 1/pi0_{g(i)};
  weights normalised so sum over finite-p features = n_valid;
  weighted BH on n_valid tests; q = BH(p / w) over finite p, NA elsewhere.
"""
import numpy as np

from fmrimod.stats.spatial_fdr import spatial_fdr


def r_sabha_p(p, group, alpha=0.05, tau=0.5, min_pi0=0.05):
    p = np.asarray(p, float)
    labels, gid = np.unique(group, return_inverse=True)
    G = len(labels)
    fin = np.isfinite(p)
    m_g = np.bincount(gid[fin], minlength=G)
    tail = np.bincount(gid[fin & (np.nan_to_num(p, nan=0) > tau)], minlength=G)
    with np.errstate(invalid="ignore", divide="ignore"):
        pi0 = np.where(m_g > 0, np.clip(tail / ((1 - tau) * m_g), 0, 1), 1.0)
    pi0 = np.clip(pi0, min_pi0, 1.0)
    w = (1 / pi0)[gid]
    w = np.where(fin, w, 0.0)
    n = fin.sum()
    w = w * n / w[fin].sum()
    q = np.full(p.shape, np.nan)
    pw = p[fin] / w[fin]
    o = np.argsort(pw)
    adj = pw[o] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1].clip(0, 1)
    qq = np.empty(n)
    qq[o] = adj
    q[fin] = qq
    k = np.nonzero(np.sort(pw) <= alpha * np.arange(1, n + 1) / n)[0]
    kmax = k.max() + 1 if k.size else 0
    rej = np.zeros(p.shape, bool)
    if kmax:
        rej[fin] = p[fin] <= alpha * kmax / n * w[fin]
    return dict(pi0=pi0, w=w, q=q, reject=rej, k=kmax)


rng = np.random.default_rng(3)

# 1) Missing p-values in a group (R c8c6021): NaN counted as p<=tau in fmrimod.
grp = np.repeat([0, 1, 2, 3], 250)
p = rng.uniform(size=1000)
p[grp == 0] = rng.uniform(0, 0.02, size=250)  # signal block
p[(grp == 1)][:0]
p[250:250 + 150] = np.nan                     # 60% missing in group 1 (pure null)
py = spatial_fdr(p, group_ids=grp, smooth_lambda=0.0)
ref = r_sabha_p(p, grp)
print("== NaN p-values ==")
print("pi0_raw fmrimod:", py.pi0_raw.round(3), " R:", ref["pi0"].round(3))
print("q all-NaN in fmrimod?", bool(np.all(np.isnan(py.qvalues))),
      "| R finite q count:", int(np.isfinite(ref["q"]).sum()))
print("rejections fmrimod:", int(py.reject.sum()), " R:", int(ref["reject"].sum()))

# 2) Unequal group sizes, no NaN: weight normalisation (group mean vs feature sum).
grp2 = np.concatenate([np.zeros(900, int), np.ones(100, int)])
p2 = rng.uniform(size=1000)
p2[900:] = rng.uniform(0, 0.01, size=100)       # small signal group
p2[:900][:60] = rng.uniform(0, 0.002, size=60)   # a few signals in the big null group
py2 = spatial_fdr(p2, group_ids=grp2, smooth_lambda=0.0)
ref2 = r_sabha_p(p2, grp2)
print("\n== unequal group sizes ==")
print("pi0 fmrimod:", py2.pi0_raw.round(3), " R:", ref2["pi0"].round(3))
print("mean feature weight fmrimod:", round(float(py2.weights.mean()), 3),
      " R:", round(float(ref2["w"].mean()), 3))
print("weights (big, small group) fmrimod:", py2.weights[[0, -1]].round(3),
      " R:", ref2["w"][[0, -1]].round(3))
print("rejections fmrimod:", int(py2.reject.sum()), " R:", int(ref2["reject"].sum()))
print("max |q diff|:", float(np.max(np.abs(py2.qvalues - ref2["q"]))))

# 3) Null calibration with unequal groups: FDR inflation from mean-over-groups normalisation.
fdp_py, fdp_r = [], []
for rep in range(300):
    g = np.concatenate([np.zeros(950, int), np.ones(50, int)])
    truth = np.zeros(1000, bool)
    truth[950:990] = True                       # 40 signals concentrated in the small group
    pp = rng.uniform(size=1000)
    pp[truth] = rng.uniform(0, 1e-3, size=truth.sum())
    a = spatial_fdr(pp, group_ids=g, smooth_lambda=0.0)
    b = r_sabha_p(pp, g)
    for rej, store in ((a.reject, fdp_py), (b["reject"], fdp_r)):
        store.append((rej & ~truth).sum() / max(rej.sum(), 1))
print("\n== simulated FDR at alpha=0.05 (300 reps, 950/50 split, signals in small group) ==")
print("fmrimod FDR:", round(float(np.mean(fdp_py)), 4), " R:", round(float(np.mean(fdp_r)), 4))

# 4) Opposite layout: large signal-rich group, several small null groups.
fdp_py, fdp_r = [], []
for rep in range(300):
    g = np.concatenate([np.zeros(600, int), np.repeat(np.arange(1, 9), 50)])
    truth = np.zeros(1000, bool)
    truth[:480] = True                          # 80% of the big group is signal
    pp = rng.uniform(size=1000)
    pp[truth] = rng.uniform(0, 0.05, size=truth.sum())
    a = spatial_fdr(pp, group_ids=g, smooth_lambda=0.0)
    b = r_sabha_p(pp, g)
    for rej, store in ((a.reject, fdp_py), (b["reject"], fdp_r)):
        store.append((rej & ~truth).sum() / max(rej.sum(), 1))
print("\n== simulated FDR at alpha=0.05 (300 reps, 600 + 8x50 groups, signals in big group) ==")
print("fmrimod FDR:", round(float(np.mean(fdp_py)), 4), " R:", round(float(np.mean(fdp_r)), 4),
      " fmrimod mean feature weight:", round(float(a.weights.mean()), 3))
