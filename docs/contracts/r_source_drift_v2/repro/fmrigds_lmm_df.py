"""GDS: LMM between/within containment df (7c7ecae; test-review-numeric.R:287-325).

Fixed R (.lmm_between_within_df / .lmm_apply_coef_df):
  a design column constant within every subject block is between-subject;
  df_between = N - rank(X_between at one row per subject);
  p_coef:<between term> = 2 * pt(-|t|, df_between); within terms keep N*K - p.
Fixture mirrors the R test: 20 subjects x 4 conditions, ~ group * condition,
expected df: (Intercept)=18, groupB=18, condition=76, groupB:condition=76.
"""
import numpy as np
import pandas as pd
from scipy import stats

from fmrimod.group import SampleLabelSpace, group_dataset, lmm_ri

rng = np.random.default_rng(91)
n_sub, n_rep = 20, 4
subjects = [f"s{i:02d}" for i in range(n_sub)]
contrasts = [f"c{i}" for i in range(n_rep)]
grp = np.repeat(["A", "B"], n_sub // 2)
cond = np.array([0.0, 1.0, 2.0, 3.0])
u = rng.normal(scale=0.7, size=n_sub)
beta = np.empty((2, n_sub, n_rep))
for s in range(n_sub):
    mu = 0.5 * (grp[s] == "B") + 0.3 * cond + u[s]
    beta[0, s] = mu + rng.normal(scale=0.3, size=n_rep)
    beta[1, s] = mu + rng.normal(scale=0.3, size=n_rep)

ds = group_dataset(
    {"beta": beta},
    space=SampleLabelSpace(["a", "b"]),
    subjects=subjects,
    contrasts=contrasts,
    col_data=pd.DataFrame({"group": grp}, index=subjects),
    contrast_data=pd.DataFrame({"condition": cond}, index=contrasts),
)
out = lmm_ri(ds, formula="~ group * condition", theta_mode="voxelwise")
names = out.metadata["predictor_names"]
print("predictors:", names)
print("df_coef:* assays present:", any(k.startswith("df_coef:") for k in out.assays))
print("fmrimod df_res:", out.assay("df_res").ravel())
r_df = {"Intercept": 18, "group[T.B]": 18, "condition": 76, "group[T.B]:condition": 76}
for name in names:
    t = out.assay(f"t_coef:{name}").ravel()
    p_py = out.assay(f"p_coef:{name}").ravel()
    p_r = 2 * stats.t.sf(np.abs(t), r_df[name])
    print(f"{name:22s} t={np.round(t, 3)}  fmrimod p={p_py}  R p (df={r_df[name]})={p_r}  ratio={np.round(p_r / p_py, 3)}")
