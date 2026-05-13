"""First-trust demo: typed design -> fit -> typed F contrast -> inspection."""

import numpy as np
import pandas as pd

import fmrimod as fm
from fmrimod.contrast import OmnibusContrast
from fmrimod.spec import hrf

rng = np.random.default_rng(0)
bold = rng.standard_normal((120, 16))
events = pd.DataFrame(
    {
        "onset": [10.0, 30.0, 50.0, 70.0, 90.0, 110.0],
        "duration": [2.0] * 6,
        "trial_type": ["face", "scene", "face", "scene", "face", "scene"],
        "run": [1] * 6,
    }
)

dataset = fm.fmri_dataset(bold, tr=2.0, events=events)
fit = fm.fmri_lm(hrf("trial_type", norm="spm"), dataset)

contrast = OmnibusContrast(term="trial_type", levels=("face", "scene"))
result = fit.contrast(contrast)

print(result.explain().to_markdown())
print(fit.provenance)
