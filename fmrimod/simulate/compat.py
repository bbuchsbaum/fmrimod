"""Compatibility simulation helpers mirroring fmrireg names."""

from __future__ import annotations

from typing import Callable, Sequence, Union

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from ..hrf.core import HRF
from ..regressor import regressor
from ..dataset.adapters import NumpyAdapter
from ..dataset import FmriDataset
from ..sampling import SamplingFrame
from .noise import ar_noise


HrfSpec = Union[HRF, Callable[..., NDArray[np.float64]], str]
"""Accepted HRF specifier for the simulate-compat constructors:
a typed :class:`~fmrimod.hrf.HRF` instance, a registry key string,
or a callable returning HRF samples."""


def simulate_bold_signal(
    ncond: int,
    hrf: HrfSpec = "spmg1",
    nreps: int = 12,
    amps: Sequence[float] | None = None,
    isi: Sequence[float] = (3.0, 6.0),
    ampsd: float = 0.0,
    tr: float = 1.5,
    seed: int | None = None,
) -> dict[str, NDArray[np.float64] | list[str]]:
    """Simulate condition-wise BOLD signals for a simple event design.

    Returns keys compatible with fmrireg:
    ``onset``, ``condition``, and ``mat`` (time in first column).
    """
    if ncond < 1:
        raise ValueError("ncond must be positive")
    if nreps < 1:
        raise ValueError("nreps must be positive")
    if tr <= 0:
        raise ValueError("tr must be positive")
    if len(isi) != 2 or float(isi[1]) <= float(isi[0]):
        raise ValueError("isi must be length-2 with isi[1] > isi[0]")

    amps_array = np.ones(ncond, dtype=np.float64) if amps is None else np.asarray(amps, dtype=np.float64)
    if amps_array.shape[0] != ncond:
        raise ValueError("Length of amps must equal ncond")

    rng = np.random.default_rng(seed)
    conditions = [f"Cond{i + 1}" for i in range(ncond)]
    trials = np.array(conditions * nreps, dtype=object)
    rng.shuffle(trials)

    onsets = np.cumsum(
        rng.uniform(float(isi[0]), float(isi[1]), size=trials.shape[0])
    ).astype(np.float64)

    span = float(getattr(hrf, "span", 12.0))
    time = np.arange(0.0, float(onsets.max()) + span + 1e-12, float(tr), dtype=np.float64)

    cols: list[NDArray[np.float64]] = []
    for i, cond_name in enumerate(conditions):
        idx = np.where(trials == cond_name)[0]
        amp_i = float(rng.normal(loc=amps_array[i], scale=float(ampsd)))
        rg = regressor(
            onsets=onsets[idx],
            hrf=hrf,
            amplitude=amp_i,
            span=span,
        )
        cols.append(np.asarray(rg.evaluate(time), dtype=np.float64).reshape(-1))

    mat = np.column_stack([time, np.column_stack(cols)])
    return {
        "onset": onsets,
        "condition": trials.tolist(),
        "mat": mat,
    }


def simulate_noise_vector(
    n: int,
    tr: float = 1.5,
    ar: Sequence[float] = (0.3,),
    ma: Sequence[float] = (0.5,),
    sd: float = 1.0,
    drift_freq: float = 1.0 / 128.0,
    drift_amplitude: float = 2.0,
    physio: bool = True,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Simulate a 1D fMRI-like noise series with ARMA + drift + physiology."""
    if n < 1:
        raise ValueError("n must be positive")
    if tr <= 0:
        raise ValueError("tr must be positive")
    if sd <= 0:
        raise ValueError("sd must be positive")

    rng = np.random.default_rng(seed)
    ar = np.asarray(ar, dtype=np.float64).ravel()
    ma = np.asarray(ma, dtype=np.float64).ravel()
    p = int(ar.shape[0])
    q = int(ma.shape[0])

    innovations = rng.normal(loc=0.0, scale=float(sd), size=n)
    noise = np.zeros(n, dtype=np.float64)

    for t in range(n):
        val = innovations[t]
        for i in range(min(p, t)):
            val += ar[i] * noise[t - i - 1]
        for j in range(min(q, t)):
            val += ma[j] * innovations[t - j - 1]
        noise[t] = val

    time = np.arange(n, dtype=np.float64) * float(tr)
    noise = noise + float(drift_amplitude) * np.sin(2.0 * np.pi * float(drift_freq) * time)

    if physio:
        cardiac = 0.5 * np.sin(2.0 * np.pi * 1.2 * time)
        respiratory = 0.8 * np.sin(2.0 * np.pi * 0.3 * time)
        noise = noise + cardiac + respiratory

    return np.asarray(noise, dtype=np.float64)


def _as_event_vector(x: float | Sequence[float], n_events: int, name: str) -> NDArray[np.float64]:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.shape[0] == 1:
        return np.full(n_events, float(arr[0]), dtype=np.float64)
    if arr.shape[0] != n_events:
        raise ValueError(f"{name} must be length 1 or match n_events")
    return arr


def _resample_param(
    base: NDArray[np.float64],
    sd: float,
    dist: str,
    *,
    allow_negative: bool = False,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    out = np.asarray(base, dtype=np.float64).copy()
    if sd <= 0:
        return out
    for i in range(out.shape[0]):
        mu = float(out[i])
        if dist in ("lognormal", "gamma") and mu <= 0:
            raise ValueError("base values must be > 0 for lognormal or gamma sampling")
        if dist == "lognormal":
            out[i] = rng.lognormal(mean=np.log(mu), sigma=sd)
        elif dist == "gamma":
            shape = (mu ** 2) / (sd ** 2)
            rate = mu / (sd ** 2)
            out[i] = rng.gamma(shape=shape, scale=1.0 / rate)
        elif dist == "gaussian":
            out[i] = rng.normal(loc=mu, scale=sd)
        else:
            raise ValueError(f"Unsupported dist: {dist}")
        if (not allow_negative) and out[i] < 0:
            out[i] = 0.0
    return out


def simulate_fmri_matrix(
    n: int = 1,
    total_time: float = 240.0,
    TR: float = 2.0,
    hrf: HrfSpec = "spmg1",
    n_events: int = 10,
    onsets: Sequence[float] | None = None,
    isi_dist: str = "even",
    isi_min: float = 2.0,
    isi_max: float = 6.0,
    isi_rate: float = 0.25,
    durations: float | Sequence[float] = 0.0,
    duration_sd: float = 0.0,
    duration_dist: str = "lognormal",
    amplitudes: float | Sequence[float] = 1.0,
    amplitude_sd: float = 0.0,
    amplitude_dist: str = "lognormal",
    single_trial: bool = False,
    noise_type: str = "none",
    noise_ar: Sequence[float] | None = None,
    noise_sd: float = 1.0,
    random_seed: int | None = None,
    verbose: bool = False,
    buffer: float = 16.0,
) -> dict[str, object]:
    """Simulate an fMRI matrix dataset plus generation metadata.

    Returns keys mirroring fmrireg:
    ``time_series``, ``ampmat``, ``durmat``, ``hrf_info``, ``noise_params``.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if total_time <= 0:
        raise ValueError("total_time must be positive")
    if TR <= 0:
        raise ValueError("TR must be positive")
    if n_events <= 0:
        raise ValueError("n_events must be positive")
    if isi_max <= isi_min:
        raise ValueError("isi_max must be greater than isi_min")
    if noise_type not in ("none", "white", "ar1", "ar2"):
        raise ValueError("noise_type must be one of: none, white, ar1, ar2")
    if isi_dist not in ("even", "uniform", "exponential"):
        raise ValueError("isi_dist must be one of: even, uniform, exponential")
    if duration_dist not in ("lognormal", "gamma"):
        raise ValueError("duration_dist must be one of: lognormal, gamma")
    if amplitude_dist not in ("lognormal", "gamma", "gaussian"):
        raise ValueError("amplitude_dist must be one of: lognormal, gamma, gaussian")

    rng = np.random.default_rng(random_seed)

    effective_time = float(total_time) - float(buffer)
    if effective_time <= 0:
        raise ValueError("total_time must be greater than buffer")

    if onsets is None:
        if isi_dist == "uniform":
            isi_samples = rng.uniform(low=isi_min, high=isi_max, size=n_events)
        elif isi_dist == "exponential":
            isi_samples = isi_min + rng.exponential(scale=1.0 / isi_rate, size=n_events)
        else:
            isi_samples = np.full(n_events, effective_time / n_events, dtype=np.float64)
        onset_vec = np.cumsum(isi_samples)
        if onset_vec.size > 0 and float(np.max(onset_vec)) > effective_time:
            onset_vec = onset_vec[onset_vec < effective_time]
            if verbose:
                print(f"Reduced to {onset_vec.shape[0]} events to fit within effective time")
    else:
        onset_vec = np.asarray(onsets, dtype=np.float64).reshape(-1)
    if onset_vec.size == 0:
        raise ValueError("No valid onsets generated")

    n_events_eff = int(onset_vec.shape[0])
    n_time_points = int(np.ceil(total_time / TR))
    time_grid = np.arange(n_time_points, dtype=np.float64) * float(TR)

    base_durations = _as_event_vector(durations, n_events_eff, "durations")
    base_amplitudes = _as_event_vector(amplitudes, n_events_eff, "amplitudes")

    def sample_durations() -> NDArray[np.float64]:
        return _resample_param(
            base=base_durations,
            sd=float(duration_sd),
            dist=duration_dist,
            allow_negative=False,
            rng=rng,
        )

    def sample_amplitudes() -> NDArray[np.float64]:
        return _resample_param(
            base=base_amplitudes,
            sd=float(amplitude_sd),
            dist=amplitude_dist,
            allow_negative=(amplitude_dist == "gaussian"),
            rng=rng,
        )

    def gen_noise() -> NDArray[np.float64]:
        if noise_type == "none":
            return np.zeros(n_time_points, dtype=np.float64)
        if noise_type == "white":
            return rng.normal(loc=0.0, scale=noise_sd, size=n_time_points).astype(np.float64)
        if noise_type == "ar1":
            phi = np.asarray([0.3] if noise_ar is None else noise_ar, dtype=np.float64).reshape(-1)
            if phi.shape[0] < 1:
                phi = np.asarray([0.3], dtype=np.float64)
            return ar_noise(n=n_time_points, V=1, phi=phi[:1], sd=noise_sd, rng=rng)[:, 0]
        phi = np.asarray([0.3, 0.2] if noise_ar is None else noise_ar, dtype=np.float64).reshape(-1)
        if phi.shape[0] < 2:
            phi = np.asarray([0.3, 0.2], dtype=np.float64)
        return ar_noise(n=n_time_points, V=1, phi=phi[:2], sd=noise_sd, rng=rng)[:, 0]

    signal_list: list[NDArray[np.float64]] = []
    ampmat = np.zeros((n_events_eff, n), dtype=np.float64)
    durmat = np.zeros((n_events_eff, n), dtype=np.float64)
    for i in range(n):
        this_dur = sample_durations()
        this_amp = sample_amplitudes()
        ampmat[:, i] = this_amp
        durmat[:, i] = this_dur

        if not single_trial:
            rg = regressor(
                onsets=onset_vec,
                hrf=hrf,
                duration=this_dur,
                amplitude=this_amp,
            )
            bold_signal = np.asarray(rg.evaluate(time_grid), dtype=np.float64).reshape(-1)
        else:
            bold_signal = np.zeros(n_time_points, dtype=np.float64)
            for j in range(n_events_eff):
                sreg = regressor(
                    onsets=[float(onset_vec[j])],
                    hrf=hrf,
                    duration=float(this_dur[j]),
                    amplitude=float(this_amp[j]),
                )
                bold_signal = bold_signal + np.asarray(sreg.evaluate(time_grid), dtype=np.float64).reshape(-1)

        signal_list.append(bold_signal + gen_noise())

    sim_matrix = np.column_stack(signal_list)
    event_tab = pd.DataFrame(
        {
            "run": np.ones(n_events_eff, dtype=int),
            "onset": onset_vec,
            "duration": durmat[:, 0],
            "amplitude": ampmat[:, 0],
        }
    )
    sf = SamplingFrame(blocklens=n_time_points, tr=TR)
    ds = FmriDataset(NumpyAdapter(sim_matrix, sf), event_table=event_tab)

    return {
        "time_series": ds,
        "ampmat": ampmat,
        "durmat": durmat,
        "hrf_info": {
            "hrf_type": getattr(hrf, "name", str(hrf)),
            "single_trial": bool(single_trial),
        },
        "noise_params": {
            "type": noise_type,
            "ar": noise_ar,
            "sd": float(noise_sd),
        },
    }
