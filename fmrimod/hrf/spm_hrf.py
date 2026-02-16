"""SPM HRF implementations with analytic derivatives."""

from __future__ import annotations

from typing import Optional, Dict, Any
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .core import HRF
from .functions import spm_canonical
from .derivatives import spmg1_derivative, spmg1_second_derivative


class SPMG1_HRF(HRF):
    """SPM canonical HRF with analytic derivative support."""

    def __init__(self, p1: float = 5.0, p2: float = 15.0, a1: float = 0.0833):
        """Initialize SPM canonical HRF.

        Args:
            p1: First exponent parameter
            p2: Second exponent parameter
            a1: Amplitude scaling factor
        """
        super().__init__(
            name="SPMG1",
            nbasis=1,
            span=24.0,
            params={"p1": p1, "p2": p2, "a1": a1},
            param_names=["p1", "p2", "a1"]
        )

    def __call__(self, t: ArrayLike) -> NDArray[np.float64]:
        """Evaluate the HRF at time points t."""
        return spm_canonical(t, p1=self.params["p1"], p2=self.params["p2"], a1=self.params["a1"])

    def _derivative(self, t: ArrayLike) -> NDArray[np.float64]:
        """Analytic derivative of the HRF."""
        return spmg1_derivative(t, p1=self.params["p1"], p2=self.params["p2"], a1=self.params["a1"])


class SPMG2_HRF(HRF):
    """SPM HRF with temporal derivative basis."""

    def __init__(self, p1: float = 5.0, p2: float = 15.0, a1: float = 0.0833):
        """Initialize SPM HRF with temporal derivative.

        Args:
            p1: First exponent parameter
            p2: Second exponent parameter
            a1: Amplitude scaling factor
        """
        super().__init__(
            name="SPMG2",
            nbasis=2,
            span=24.0,
            params={"p1": p1, "p2": p2, "a1": a1},
            param_names=["p1", "p2", "a1"]
        )

    def __call__(self, t: ArrayLike) -> NDArray[np.float64]:
        """Evaluate the HRF and its temporal derivative."""
        t = np.asarray(t)

        # Canonical HRF
        canonical = spm_canonical(t, p1=self.params["p1"], p2=self.params["p2"], a1=self.params["a1"])

        # Temporal derivative (using analytic formula)
        derivative = spmg1_derivative(t, p1=self.params["p1"], p2=self.params["p2"], a1=self.params["a1"])

        # Stack as columns
        result = np.column_stack([canonical, derivative])

        # Handle single time point case
        if t.ndim == 0 or (t.ndim == 1 and len(t) == 1):
            result = result.reshape(1, -1)

        return result

    def _derivative(self, t: ArrayLike) -> NDArray[np.float64]:
        """Derivatives of both basis functions."""
        t = np.asarray(t)

        # First basis derivative (derivative of canonical)
        first_deriv = spmg1_derivative(t, p1=self.params["p1"], p2=self.params["p2"], a1=self.params["a1"])

        # Second basis derivative (second derivative of canonical)
        second_deriv = spmg1_second_derivative(t, p1=self.params["p1"], p2=self.params["p2"], a1=self.params["a1"])

        # Stack as columns
        result = np.column_stack([first_deriv, second_deriv])

        # Handle single time point case
        if t.ndim == 0 or (t.ndim == 1 and len(t) == 1):
            result = result.reshape(1, -1)

        return result


class SPMG3_HRF(HRF):
    """SPM HRF with temporal and dispersion derivatives."""

    def __init__(self, p1: float = 5.0, p2: float = 15.0, a1: float = 0.0833):
        """Initialize SPM HRF with temporal and dispersion derivatives.

        Args:
            p1: First exponent parameter
            p2: Second exponent parameter
            a1: Amplitude scaling factor
        """
        super().__init__(
            name="SPMG3",
            nbasis=3,
            span=24.0,
            params={"p1": p1, "p2": p2, "a1": a1},
            param_names=["p1", "p2", "a1"]
        )

    def __call__(self, t: ArrayLike) -> NDArray[np.float64]:
        """Evaluate the HRF with temporal and dispersion derivatives."""
        t = np.asarray(t)

        p1, p2, a1 = self.params["p1"], self.params["p2"], self.params["a1"]

        # Canonical HRF
        canonical = spm_canonical(t, p1=p1, p2=p2, a1=a1)

        # Temporal derivative (using analytic formula)
        derivative = spmg1_derivative(t, p1=p1, p2=p2, a1=a1)

        # Dispersion derivative = analytic second temporal derivative
        # (matches R's HRF_SPMG3 which uses hrf_spmg1_second_deriv)
        dispersion = spmg1_second_derivative(t, p1=p1, p2=p2, a1=a1)

        # Stack as columns
        result = np.column_stack([canonical, derivative, dispersion])

        # Handle single time point case
        if t.ndim == 0 or (t.ndim == 1 and len(t) == 1):
            result = result.reshape(1, -1)

        return result

    def _derivative(self, t: ArrayLike) -> NDArray[np.float64]:
        """Derivatives of all three basis functions."""
        t = np.asarray(t)

        p1, p2, a1 = self.params["p1"], self.params["p2"], self.params["a1"]

        # Derivatives of the three basis functions
        # 1. Derivative of canonical
        deriv1 = spmg1_derivative(t, p1=p1, p2=p2, a1=a1)

        # 2. Second derivative (derivative of temporal derivative)
        deriv2 = spmg1_second_derivative(t, p1=p1, p2=p2, a1=a1)

        # 3. Derivative of dispersion basis = derivative of the second derivative
        # Use numerical central difference on the analytic second derivative
        dt = 0.001
        deriv3 = (spmg1_second_derivative(t + dt, p1=p1, p2=p2, a1=a1) -
                  spmg1_second_derivative(t - dt, p1=p1, p2=p2, a1=a1)) / (2 * dt)

        # Stack as columns
        result = np.column_stack([deriv1, deriv2, deriv3])

        # Handle single time point case
        if t.ndim == 0 or (t.ndim == 1 and len(t) == 1):
            result = result.reshape(1, -1)

        return result
