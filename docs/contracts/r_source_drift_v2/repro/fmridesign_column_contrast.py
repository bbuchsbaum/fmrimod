"""DES: column_contrast patterns vs R 53772f6 (#24): documented design-matrix
patterns (term_tag_condition_tag[_b##]) must select columns of a term-attached
contrast; term-level patterns are the fallback."""
import warnings

import pandas as pd

from fmrimod import event_model
from fmrimod.contrast import column_contrast, contrast_set, contrast_weights
from fmrimod.formula.base import Term

warnings.simplefilter("ignore")
df = pd.DataFrame({"onset": [2., 20., 40., 60.], "cond": ["A", "B", "A", "B"]})
t = Term("cond", hrf="spmg2")
t.contrast_specs = contrast_set(
    column_contrast(r"^cond_cond\.A_b01$", "design_names", pattern_B=r"^cond_cond\.B_b01$"),
    column_contrast(r"^cond\.A_b01$", "term_level_b01", pattern_B=r"^cond\.B_b01$"),
)
em = event_model([t], data=df, tr=2.0, n_scans=50)
print("columns:", em.column_names)
print("term conditions(expand_basis=True):", em._create_event_terms()[0].conditions(drop_empty=False, expand_basis=True))
for k, c in contrast_weights(em)["cond"].items():
    print(f"  {k:15s}", {n: round(float(v), 3) for n, v in zip(em.column_names, c.offset_weights[:, 0]) if v})
print("R (fixed): both -> +1 on cond_cond.A_b01, -1 on cond_cond.B_b01")
