"""DES: baseline_model nuisance runs with zero columns (R 246da81)."""
import warnings

import numpy as np
import pandas as pd

from fmrimod.baseline import baseline_model, check_nuisance
from fmrimod.sampling import SamplingFrame

sf = SamplingFrame(blocklens=[10, 12], tr=1.0)
rng = np.random.default_rng(11)
cases = {
    "all-dropped run": [pd.DataFrame({"flat": np.ones(10)}),
                        pd.DataFrame({"p": rng.normal(size=12), "q": rng.normal(size=12)})],
    "zero-col run": [np.zeros((10, 0)), pd.DataFrame({"p": np.arange(12.0) ** 2})],
    "all zero-col": [np.zeros((10, 0)), np.zeros((12, 0))],
}
for label, nl in cases.items():
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        try:
            bm = baseline_model(basis="poly", degree=2 if label == "all-dropped run" else 1, sframe=sf,
                                nuisance_list=nl, nuisance_check="drop")
            print(f"{label}: ncol={bm.design_matrix.shape[1]} names={[c for c in bm.column_names if c.startswith('nuis')]}")
        except Exception as e:
            print(f"{label}: ERR {type(e).__name__}: {e}")
    if label == "zero-col run":
        try:
            rep = check_nuisance(nl, sf, basis="poly", degree=1)
            print("   check_nuisance ok:", getattr(rep, "ok", rep.get("ok") if isinstance(rep, dict) else rep))
        except Exception as e:
            print(f"   check_nuisance ERR {type(e).__name__}: {e}")
