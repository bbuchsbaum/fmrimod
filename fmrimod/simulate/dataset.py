"""Convenience helpers for end-to-end simulation datasets."""

from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np
from numpy.typing import NDArray

from ..hrf.core import HRF
from ..sampling import SamplingFrame
from ..regressor import regressor


HrfSpec = Union[HRF, Callable[..., NDArray[np.float64]], str]
"""Accepted HRF specifier: typed HRF, registry key, or callable."""


def simulate_simple_dataset(
    ncond: int,
    nreps: int = 12,
    tr: float = 1.5,
    snr: float = 0.5,
    hrf: Optional[HrfSpec] = None,
    seed: int | None = None,
) -> dict[str, NDArray[np.float64] | list[str]]:
    """Simulate a small multi-condition dataset with additive Gaussian noise.

    Returns keys mirroring ``fmrireg::simulate_simple_dataset``:
    ``clean``, ``noisy``, ``noise``, ``onsets``, ``conditions``.
    """
    if ncond < 1:
        raise ValueError("ncond must be >= 1")
    if nreps < 1:
        raise ValueError("nreps must be >= 1")
    if tr <= 0:
        raise ValueError("tr must be > 0")
    if snr <= 0:
        raise ValueError("snr must be > 0")

    rng = np.random.default_rng(seed)
    total_trials = ncond * nreps
    isi = 4.0 * tr
    onsets = np.arange(total_trials, dtype=np.float64) * isi
    cond_idx = np.tile(np.arange(ncond, dtype=int), nreps)
    rng.shuffle(cond_idx)

    condition_names = [f"cond_{i + 1}" for i in range(ncond)]
    per_cond_onsets = [onsets[cond_idx == i] for i in range(ncond)]

    duration = float(onsets.max() + 32.0)
    n_timepoints = int(np.floor(duration / tr)) + 1
    sf = SamplingFrame(blocklens=n_timepoints, tr=tr)
    time = sf.samples

    clean_cols: list[NDArray[np.float64]] = []
    hrf_spec = "spmg1" if hrf is None else hrf
    for cond_onsets in per_cond_onsets:
        rg = regressor(onsets=cond_onsets, hrf=hrf_spec)
        clean_cols.append(np.asarray(rg.evaluate(time), dtype=np.float64).reshape(-1))
    clean_signal = np.column_stack(clean_cols)

    signal_sd = float(np.std(clean_signal))
    noise_sd = signal_sd / float(snr) if signal_sd > 0 else 1.0 / float(snr)
    noise = rng.normal(loc=0.0, scale=noise_sd, size=clean_signal.shape)
    noisy_signal = clean_signal + noise

    clean = np.column_stack([time, clean_signal])
    noisy = np.column_stack([time, noisy_signal])

    return {
        "clean": clean,
        "noisy": noisy,
        "noise": np.asarray(noise, dtype=np.float64),
        "onsets": np.asarray(onsets, dtype=np.float64),
        "conditions": [condition_names[i] for i in cond_idx],
    }
