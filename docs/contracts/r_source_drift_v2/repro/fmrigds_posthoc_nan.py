"""GDS: BH/BY post-hoc with missing p (R posthoc-registry.R:238 uses stats::p.adjust,
which drops NA from both the ranking and n).  Pre-existing in fmrimod; surfaced because
R's in-range fixes emit more NA p (ols n < p + 1, knownvar var <= 0, fe_reg min_subjects)."""
import numpy as np

from fmrimod.group import SampleLabelSpace, group_dataset, ols_voxelwise, posthoc
from fmrimod.group.ops import posthoc_bh
from fmrimod.stats.inference import fdr_correction

p = np.array([0.001, 0.01, 0.02, np.nan, 0.5])
fin = np.isfinite(p)
o = np.argsort(p[fin])
n = fin.sum()
adj = np.minimum.accumulate((p[fin][o] * n / np.arange(1, n + 1))[::-1])[::-1].clip(0, 1)
r_q = np.full(p.shape, np.nan)
tmp = np.empty(n)
tmp[o] = adj
r_q[fin] = tmp
print("fmrimod posthoc_bh :", posthoc_bh(p))
print("fmrimod fdr_correction:", fdr_correction(p)[1])
print("R p.adjust(BH)     :", r_q)

# End to end: one voxel with too few finite subjects poisons q for the whole map.
rng = np.random.default_rng(1)
beta = rng.normal(0.8, 1.0, size=(50, 6, 1))
beta[0, 1:, 0] = np.nan  # voxel 0 has 1 finite subject -> NA p
ds = group_dataset({"beta": beta}, space=SampleLabelSpace([f"v{i}" for i in range(50)]),
                   subjects=[f"s{i}" for i in range(6)], contrasts=["c"])
fit = ols_voxelwise(ds)
fit_p = fit.with_assays({**fit.assays, "p": fit.assay("p_coef:Intercept")})
q = posthoc(fit_p, "fdr:bh").assay("q")
print("finite q after ols + posthoc:", int(np.isfinite(q).sum()), "of", q.size,
      "(R: 49 of 50)")
