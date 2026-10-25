"""GDS: axis-frame round trip through fmrimod HDF5 (8b145f9: factor covariates / leading-zero ids)."""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from fmrimod.group import SampleLabelSpace, group_dataset, read_hdf5, write_hdf5

subs = ["001", "002", "010"]
col = pd.DataFrame(
    {"grp": pd.Categorical(["b", "a", "b"], categories=["b", "a"]), "site": ["007", "008", "009"]},
    index=pd.Index(subs, name="subject"),
)
ds = group_dataset(
    {"beta": np.zeros((1, 3, 1))}, space=SampleLabelSpace(["v"]), subjects=subs, contrasts=["c"], col_data=col
)
with tempfile.TemporaryDirectory() as d:
    p = Path(d) / "x.h5"
    write_hdf5(ds, p)
    back = read_hdf5(p)
print("subjects:", back.subjects)
print("col_data index:", list(back.col_data.index), back.col_data.index.dtype)
print("col_data dtypes:", dict(back.col_data.dtypes.astype(str)))
print("site values:", list(back.col_data["site"]))
