"""REG-B: t-only combine (fmri_meta(combine=...)) vs fixed R combine_t_statistics.

Fixed R (R/metafuns.R combine_t_statistics after aecf0ac; used by
R/fmri_meta_gds.R .fmri_meta_gds_combine at HEAD): per-subject two-sided
p from pt(|t|, df); Fisher X2 = -2 sum log p over finite subjects; z = |qnorm(p/2)|
signed by mean t. Stouffer: z_i = -qnorm(pt(-|t|, df)) * sign(t).

fmrimod HEAD: group_fit(model="meta", combine=...) -> native combine; derive(...)
substitutes t for z (fmrimod/group/ops.py:143-144), so df is ignored.
"""
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from fmrimod.dataset.group_data import group_data_from_csv
from fmrimod.stats.inference import group_fit
from fmrimod.stats.interfaces import GroupFitRequest

warnings.simplefilter("ignore")
rng = np.random.default_rng(5)
S, df = 12, 8
t = rng.standard_t(df, size=S) + 0.8
rows = [{"subject": f"s{i}", "t": t[i], "df": df} for i in range(S)]
frame = pd.DataFrame(rows)


def r_fisher(t, df):
    lp2 = np.log(2) + stats.t.logcdf(-np.abs(t), df)
    X2 = -2 * lp2.sum()
    lpc = stats.chi2.logsf(X2, 2 * len(t))
    z = -stats.norm.ppf(np.exp(lpc - np.log(2)))
    return z * np.sign(t.mean()), np.exp(lpc)


def r_stouffer(t, df):
    zi = -stats.norm.ppf(np.exp(stats.t.logcdf(-np.abs(t), df))) * np.sign(t)
    z = zi.sum() / np.sqrt(len(t))
    return z, 2 * stats.norm.sf(abs(z))


for combine, ref in (("fisher", r_fisher), ("stouffer", r_stouffer)):
    try:
        gd = group_data_from_csv(frame, effect_cols={"t": "t"})
        out = group_fit(GroupFitRequest(data=gd, model="meta", combine=combine))
        got = (float(out.statistic.ravel()[0]), float(out.p.ravel()[0]))
    except Exception as exc:
        got = f"{type(exc).__name__}: {exc}"
    print(f"{combine}: fmrimod -> {got}; fixed R (pt with df={df}) -> "
          f"z={ref(t, df)[0]:.4f}, p={ref(t, df)[1]:.5f}")
