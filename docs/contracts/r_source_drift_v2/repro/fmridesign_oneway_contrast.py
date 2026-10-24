"""DES: oneway_contrast(~cond) on a crossed cond x task term. R
(generate_main_effect_contrast) gives the cond main effect: (k_cond - 1) Helmert
columns constant over task. fmrimod ignores the named factor."""
import itertools
import warnings

import numpy as np
import pandas as pd

from fmrimod import event_model
from fmrimod.contrast import contrast_weights, oneway_contrast
from fmrimod.contrast.contrast_spec import Formula

warnings.simplefilter("ignore")
grid = list(itertools.product(["A", "B", "C"], ["x", "y"]))
df = pd.DataFrame(grid * 2, columns=["cond", "task"])
df["onset"] = np.arange(len(df)) * 6.0 + 2
em = event_model("onset ~ hrf(cond:task)", data=df, tr=2.0, n_scans=60)
et = em._create_event_terms()[0]
conds = et.conditions(drop_empty=False, expand_basis=False)
W = contrast_weights(oneway_contrast(Formula("cond"), name="cond_main"), et).weights
print("cells:", conds)
print("fmrimod oneway(~cond) shape", W.shape)
# R: Helmert(3) of cond expanded per cell
H = np.array([[-1., -1.], [1., -1.], [0., 2.]])
lvl = {"A": 0, "B": 1, "C": 2}
R = np.vstack([H[lvl[c.split("_")[0].split(".")[1]]] for c in conds])
print("R shape", R.shape)
# Does fmrimod's space equal the cond main-effect space?
rank_union = np.linalg.matrix_rank(np.hstack([W, R]))
print("rank(W)", np.linalg.matrix_rank(W), "rank(R)", np.linalg.matrix_rank(R), "rank([W R])", rank_union)
task_effect = np.array([1.0 if c.endswith("task.x") else -1.0 for c in conds])
print("max |W' task_effect| (leak onto task main effect):", float(np.max(np.abs(W.T @ task_effect))))
