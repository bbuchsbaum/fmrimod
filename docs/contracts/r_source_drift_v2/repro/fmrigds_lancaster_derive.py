"""GDS: Lancaster far tail (7c7ecae) and t/df -> z/p derivation (7c7ecae, e71c9b3).

Fixed R:
  Lancaster  chi2 = sum qchisq(p_i, 2 w_i, lower.tail = FALSE)  (finite for tiny p)
  derive_z   z = -qnorm(pt(-|t|, df, log.p=TRUE), log.p=TRUE) * sign(t)
  derive_p   p = 2 * pt(-|t|, df)        (t distribution, not normal)
  combine:stouffer on a {t, df} source uses z from derive_z.
"""
import numpy as np
from scipy import stats

from fmrimod.group import SampleLabelSpace, combine_lancaster, derive, group_dataset


def ds(assays, n_sub):
    return group_dataset(
        assays,
        space=SampleLabelSpace(["v1"]),
        subjects=[f"s{i}" for i in range(n_sub)],
        contrasts=["c1"],
    )


print("== Lancaster, p = (1e-17, 0.5), dfw = (1, 1)  [R test-review-numeric.R:149] ==")
p = np.array([1e-17, 0.5]).reshape(1, 2, 1)
out = combine_lancaster(ds({"p": p}, 2), dfw=[1, 1])
r_chi2 = stats.chi2.isf(1e-17, 2) + stats.chi2.isf(0.5, 2)
print("fmrimod chi2:", float(out.assay("chi2").ravel()[0]), " p_g:", float(out.assay("p_g").ravel()[0]))
print("R      chi2:", r_chi2, " p_g:", float(stats.chi2.sf(r_chi2, 4)))

print("\n== derive on a {t, df} dataset (NIfTI t/df source shape) ==")
t = np.array([12.0, 1.0, 1.0]).reshape(1, 3, 1)
df = np.full((1, 3, 1), 100.0)
tds = ds({"t": t, "df": df}, 3)
try:
    derive(tds, "z")
    print("derive('z') succeeded")
except Exception as exc:  # noqa: BLE001
    print("derive('z') ->", type(exc).__name__, exc)
dp = derive(tds, "p")
r_p = 2 * stats.t.sf(np.abs(t.ravel()), 100)
print("fmrimod p:", dp.assay("p").ravel(), " (z assay set to t:", dp.assay("z").ravel(), ")")
print("R      p:", r_p)

t_small = np.array([2.5, 2.5, 2.5]).reshape(1, 3, 1)
df_small = np.full((1, 3, 1), 5.0)
dps = derive(ds({"t": t_small, "df": df_small}, 3), "p")
print("t=2.5, df=5: fmrimod p =", float(dps.assay("p").ravel()[0]),
      " R p =", float(2 * stats.t.sf(2.5, 5)))

r_z_log = np.array([
    -stats.norm.ppf(np.exp(stats.t.logsf(abs(x), 100))) * np.sign(x) for x in t.ravel()
])
print("R derive_z (normal-equivalent of t):", r_z_log,
      " -> R stouffer z_g:", float(r_z_log.sum() / np.sqrt(3)))
print("fmrimod has no t->z path; z := t would give stouffer z_g:", float(t.sum() / np.sqrt(3)))
