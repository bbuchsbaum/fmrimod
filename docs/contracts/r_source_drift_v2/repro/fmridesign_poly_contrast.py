"""DES: poly_contrast weights vs R contrast_weights.poly_contrast_spec
(stats::poly over value_map / numeric levels; b7a4fc5 value_map coverage)."""
import warnings

import numpy as np
import pandas as pd

from fmrimod import event_model
from fmrimod.contrast import contrast_weights, poly_contrast
from fmrimod.contrast.contrast_spec import Formula

warnings.simplefilter("ignore")

def r_poly(x, degree):
    """Transcription of stats::poly(x, degree) (orthonormal, sign = diag(R))."""
    x = np.asarray(x, float)
    xc = x - x.mean()
    X = np.vander(xc, degree + 1, increasing=True)
    Q, R = np.linalg.qr(X)
    Z = Q * np.sign(np.diag(R))
    return Z[:, 1:]

df = pd.DataFrame({"onset": np.arange(9) * 6.0 + 2,
                   "intensity": ["low", "medium", "high"] * 3})
em = event_model("onset ~ hrf(intensity)", data=df, tr=2.0, n_scans=40)
et = em._create_event_terms()[0]
conds = et.conditions(drop_empty=False, expand_basis=False)
print("conditions:", conds)
order = [c.split(".", 1)[1] for c in conds]

vm = {"low": 1.0, "medium": 2.0, "high": 4.0}
c = contrast_weights(poly_contrast(Formula("intensity"), name="lin", degree=1, value_map=vm), et)
r = r_poly([vm[o] for o in order], 1)
print("fmrimod lin (value_map low=1,med=2,high=4):", np.round(c.weights.ravel(), 4))
print("R       lin                               :", np.round(r.ravel(), 4))

vm_bad = {"low": 1.0, "high": 3.0}
try:
    c2 = contrast_weights(poly_contrast(Formula("intensity"), name="lin", degree=1, value_map=vm_bad), et)
    print("value_map missing 'medium': no error, weights", np.round(c2.weights.ravel(), 4),
          "(R: error 'value_map does not cover all factor levels')")
except Exception as e:
    print("value_map missing level: ERR", e)
