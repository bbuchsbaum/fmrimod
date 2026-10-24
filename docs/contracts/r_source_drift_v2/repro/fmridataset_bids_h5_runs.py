"""DS check: BIDS-HDF5 subject datasets label events with the BIDS `run` entity (a within-session label),
and fmri_lm auto-selects `run` as the block column (R 52439b2: 'BIDS run is a within-session label, so a
subject with two sessions each holding run-1 has two distinct acquisitions sharing one label ... silently
merges unrelated acquisitions into a single run').
"""
import pathlib
import tempfile

import numpy as np
import pandas as pd

import fmrimod.dataset as fds
from fmrimod.design.event_model import _parse_block_ids

rng = np.random.default_rng(5)

def ds(onset):
    ev = pd.DataFrame({"onset": [onset], "duration": [1.0], "trial_type": ["go"], "run_id": [1]})
    return fds.matrix_dataset(rng.normal(size=(20, 3)), tr=2.0, event_table=ev)

a, b = ds(4.0), ds(30.0)
# Same subject, two sessions, both BIDS run-1 (what the BIDS-directory path and R-written archives produce).
for d, ses in ((a, "01"), (b, "02")):
    d._bids_task, d._bids_session, d._bids_run = "go", ses, "1"
with tempfile.TemporaryDirectory() as tmp:
    arc = fds.compress_bids_study({"sub-01": a, "sub-01b": b}, pathlib.Path(tmp) / "a.h5")
    # The writer keys scans by the mapping key; give both rows the same subject to mimic one participant.
    # Instead read the manifest of a genuine two-session archive written below.
    print(arc.scan_manifest[["scan_name", "subject", "session", "run"]])

# Build a genuine one-subject, two-session archive: a Sequence input keyed by subject id.
class _Entries(list):
    pass
with tempfile.TemporaryDirectory() as tmp:
    from fmrimod.dataset import bids_h5_write as w
    orig = w._normalise_input
    w._normalise_input = lambda x, **k: [("sub-01", a), ("sub-01", b)]
    try:
        arc = fds.compress_bids_study([a, b], pathlib.Path(tmp) / "b.h5")
    finally:
        w._normalise_input = orig
    print(arc.scan_manifest[["scan_name", "subject", "session", "run"]])
    sub = arc.datasets[arc.subject_ids.index("sub-01")]
    ev = sub.event_table
    print(ev[["onset", "run", "run_id", "session"]])
    print("sampling-frame blocks:", sub.sampling_frame.blocklens.tolist())
    print("block ids fmri_lm would use (block='run'):   ", _parse_block_ids("run", ev).tolist())
    print("block ids from positional run_id:            ", _parse_block_ids("run_id", ev).tolist())
