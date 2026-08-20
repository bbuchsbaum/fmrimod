"""LWU HRF parameter grids for SBHM library construction.

Ports R ``fmrilss::create_lwu_grid()`` so a grid composes directly with
:func:`fmrimod.single.sbhm.library.build_sbhm_library`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from fmrimod.hrf.library import LWUHRF
from fmrimod.hrf.normalization import normalize


@dataclass(frozen=True)
class LwuGrid:
    """LWU parameter grid plus evaluated / callable HRFs.

    ``parameters`` has one row per library member (``tau``, ``sigma``,
    ``rho``). :meth:`to_library_H` evaluates the grid on a time vector so
    the result can be passed to :func:`build_sbhm_library`.
    """

    parameters: pd.DataFrame
    span: float = 30.0

    def __post_init__(self) -> None:
        required = {"tau", "sigma", "rho"}
        missing = required.difference(self.parameters.columns)
        if missing:
            raise ValueError(f"LwuGrid.parameters missing columns: {sorted(missing)}")

    def to_library_H(  # noqa: N802 - H is the SBHM library matrix
        self, t: Sequence[float] | NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Evaluate every grid HRF on ``t``, returning ``(T, n_library)``."""

        t_arr = np.asarray(t, dtype=np.float64)
        cols = [
            _eval_lwu(
                t_arr,
                float(row.tau),
                float(row.sigma),
                float(row.rho),
                span=self.span,
            )
            for row in self.parameters.itertuples(index=False)
        ]
        return np.column_stack(cols)


def create_lwu_grid(
    tau_range: tuple[float, float] = (4.0, 8.0),
    sigma_range: tuple[float, float] = (1.5, 3.5),
    rho_range: tuple[float, float] = (0.1, 0.6),
    n_tau: int = 5,
    n_sigma: int = 3,
    n_rho: int = 3,
    span: float = 30.0,
) -> LwuGrid:
    """Build a Lag-Width-Undershoot HRF parameter grid.

    Matches R ``create_lwu_grid()`` defaults. HRFs are height-normalized
    (``unit_peak``) so the grid composes with SBHM library SVD.
    """

    for name, value in (("n_tau", n_tau), ("n_sigma", n_sigma), ("n_rho", n_rho)):
        if int(value) != value or int(value) < 1:
            raise ValueError(f"{name} must be a positive integer")
    parameters = pd.DataFrame(
        [
            {"tau": tau, "sigma": sigma, "rho": rho}
            for tau in np.linspace(tau_range[0], tau_range[1], int(n_tau))
            for sigma in np.linspace(sigma_range[0], sigma_range[1], int(n_sigma))
            for rho in np.linspace(rho_range[0], rho_range[1], int(n_rho))
        ]
    )
    return LwuGrid(parameters=parameters, span=float(span))


def _eval_lwu(
    t: NDArray[np.float64],
    tau: float,
    sigma: float,
    rho: float,
    *,
    span: float,
) -> NDArray[np.float64]:
    hrf = normalize(LWUHRF(tau=tau, sigma=sigma, rho=rho, span=span), "unit_peak")
    values = np.asarray(hrf(t), dtype=np.float64)
    if values.ndim == 2:
        values = values[:, 0]
    return values
