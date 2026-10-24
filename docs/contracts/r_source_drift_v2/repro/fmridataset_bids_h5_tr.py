"""DS check: BIDS-HDF5 reader collapses per-scan TR to the first scan's TR.

R HEAD (temporal_schema, 52439b2/cce6a44): TR is per run, positive/finite, constant within a run,
'Runs may differ from one another, matching fmrihrf::sampling_frame()'. read_bids_bold() validates
a finite positive TR for every selected file. The fmrimod archive stores /scans/<s>/metadata/tr per
scan, but bids_h5_dataset() reads only the first scan's value.
"""
import pathlib
import tempfile

import numpy as np

import fmrimod.dataset as fds

rng = np.random.default_rng(2)
a = fds.matrix_dataset(rng.normal(size=(20, 3)), tr=2.0)
b = fds.matrix_dataset(rng.normal(size=(20, 3)), tr=1.5)
with tempfile.TemporaryDirectory() as tmp:
    arc = fds.compress_bids_study({"sub-01": a, "sub-02": b}, pathlib.Path(tmp) / "a.h5")
    import h5py
    with h5py.File(pathlib.Path(tmp) / "a.h5") as h:
        stored = {n: float(h[f"scans/{n}/metadata/tr"][()]) for n in h["scans"]}
    print("stored per-scan TR:", stored)
    for sid, d in zip(arc.subject_ids, arc.datasets):
        print(sid, "reader sampling-frame tr:", d.sampling_frame.tr.tolist())

# start_time (slice-timing reference) round trip
c = fds.matrix_dataset(rng.normal(size=(20, 3)), tr=2.0, start_time=0.0)
with tempfile.TemporaryDirectory() as tmp:
    arc = fds.compress_bids_study({"sub-01": c}, pathlib.Path(tmp) / "c.h5")
    print("start_time written 0.0 -> read back", arc.datasets[0].sampling_frame.start_time.tolist())
