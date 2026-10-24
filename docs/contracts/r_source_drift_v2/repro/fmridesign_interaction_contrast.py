"""DES: interaction_contrast weights vs R 648bac1 (product of Helmert codings
of every named factor; orthogonal to all lower-order effects)."""
import itertools
import warnings

import numpy as np
import pandas as pd

from fmrimod import event_model
from fmrimod.contrast import contrast_weights, interaction_contrast
from fmrimod.contrast.contrast_spec import Formula

warnings.simplefilter("ignore")

def helmert(k):
    # R contr.helmert(k)
    H = np.zeros((k, k - 1))
    for j in range(k - 1):
        H[: j + 1, j] = -1.0
        H[j + 1, j] = j + 1
    return H

def r_interaction(levels_per_factor):
    """R (fixed) highest-order interaction over cells in last-fastest order."""
    out = helmert(levels_per_factor[0])
    for k in levels_per_factor[1:]:
        out = np.kron(out, helmert(k))
    return out

def lower_order(levels_per_factor):
    """Columns spanning all lower-order effects (incl. intercept)."""
    cols = []
    nf = len(levels_per_factor)
    for r in range(nf):
        for combo in itertools.combinations(range(nf), r):
            mats = [helmert(k) if i in combo else np.ones((k, 1)) for i, k in enumerate(levels_per_factor)]
            M = mats[0]
            for m in mats[1:]:
                M = np.kron(M, m)
            cols.append(M)
    return np.hstack(cols)

for levels, formula in [((2, 2), "a*b"), ((3, 2), "a*b"), ((2, 3), "a*b"), ((2, 2, 2), "a*b*c"), ((3, 2, 2), "a*b*c")]:
    names = "abc"[: len(levels)]
    grid = list(itertools.product(*[[f"{n}{i+1}" for i in range(k)] for n, k in zip(names, levels)]))
    df = pd.DataFrame(grid, columns=list(names))
    df["onset"] = np.arange(len(df)) * 6.0 + 2
    em = event_model(f"onset ~ hrf({':'.join(names)})", data=df, tr=2.0, n_scans=int(len(df) * 3 + 20))
    et = em._create_event_terms()[0]
    con = contrast_weights(interaction_contrast(Formula(formula), name="ix"), et)
    W = con.weights
    L = lower_order(list(levels))
    R = r_interaction(list(levels))
    leak = float(np.max(np.abs(L.T @ W))) if W.size else float("nan")
    same_space = (np.linalg.matrix_rank(np.hstack([W, R])) == R.shape[1] == np.linalg.matrix_rank(W)) if W.size else False
    print(f"{levels}: fmrimod cols={W.shape[1]} R cols={R.shape[1]} "
          f"max|lower-order leak|={leak:.3f} spans R space={same_space}")
