"""FmriModel: combines EventModel + BaselineModel + FmriDataset.

This is the top-level model object that produces the full design matrix
(event + baseline columns) and is passed to :func:`fmri_lm` for fitting.
Ports R's ``fmri_model()`` / ``create_fmri_model()``.
"""

from __future__ import annotations

from typing import Any, List, Optional, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..dataset.protocols import DatasetProtocol
from ..sampling import SamplingFrame


class FmriModel:
    """Full fMRI regression model.

    Combines an :class:`~fmrimod.design.event_model.EventModel` (experimental
    regressors) with a :class:`~fmrimod.baseline.BaselineModel` (drift, block
    intercepts, nuisance) and a dataset satisfying :class:`DatasetProtocol`.

    Parameters
    ----------
    event_model : EventModel
        Event-related design (HRF-convolved regressors).
    baseline_model : BaselineModel
        Baseline design (drift, intercepts, nuisance).
    dataset : DatasetProtocol
        Data source providing per-run time-series.

    Examples
    --------
    >>> from fmrimod import event_model, baseline_model, SamplingFrame
    >>> from fmrimod.dataset import FmriDataset
    >>> from fmrimod.dataset.adapters import NumpyAdapter
    >>> from fmrimod.model import FmriModel
    >>>
    >>> # ... build event_model, baseline_model, dataset ...
    >>> fmod = FmriModel(ev_model, bl_model, dataset)
    >>> X = fmod.design_matrix()          # full design for all runs
    >>> X_run0 = fmod.design_matrix(run=0)  # single run
    """

    def __init__(
        self,
        event_model: object,  # EventModel (avoid circular import)
        baseline_model: object,  # BaselineModel
        dataset: DatasetProtocol,
    ):
        self._event_model = event_model
        self._baseline_model = baseline_model
        self._dataset = dataset

    # -- Accessors --

    @property
    def event_model(self) -> object:
        """The event model component."""
        return self._event_model

    @property
    def baseline_model(self) -> object:
        """The baseline model component."""
        return self._baseline_model

    @property
    def dataset(self) -> DatasetProtocol:
        """The underlying dataset."""
        return self._dataset

    @property
    def sampling_frame(self) -> SamplingFrame:
        """The sampling frame from the dataset."""
        return self._dataset.get_sampling_frame()

    @property
    def n_runs(self) -> int:
        """Number of runs."""
        return int(self._dataset.n_runs)

    @property
    def n_timepoints(self) -> List[int]:
        """Timepoints per run."""
        if hasattr(self._dataset, "run_lengths"):
            return [int(v) for v in self._dataset.run_lengths]
        value = self._dataset.n_timepoints
        if isinstance(value, int):
            return [int(value)]
        return [int(v) for v in value]

    # -- Design matrix construction --

    def design_matrix(
        self,
        run: Optional[int] = None,
    ) -> pd.DataFrame:
        """Build the combined event + baseline design matrix.

        Parameters
        ----------
        run : int, optional
            If given, return the design matrix for a single run only.
            If *None*, return the concatenated design for all runs.

        Returns
        -------
        pd.DataFrame
            Design matrix with named columns.
        """
        ev_dm = self._get_event_dm(run)
        bl_dm = self._get_baseline_dm(run)

        # Combine horizontally
        return pd.concat([ev_dm, bl_dm], axis=1)

    def design_matrix_array(
        self,
        run: Optional[int] = None,
    ) -> NDArray[np.float64]:
        """Return the design matrix as a plain numpy array.

        Parameters
        ----------
        run : int, optional
            If given, return for a single run.

        Returns
        -------
        NDArray
            Shape ``(n_timepoints, n_columns)``.
        """
        return self.design_matrix(run=run).values.astype(np.float64)

    def event_design_matrix(
        self,
        run: Optional[int] = None,
    ) -> pd.DataFrame:
        """Return only the event-related columns."""
        return self._get_event_dm(run)

    def baseline_design_matrix(
        self,
        run: Optional[int] = None,
    ) -> pd.DataFrame:
        """Return only the baseline columns."""
        return self._get_baseline_dm(run)

    def design_columns(self) -> Any:
        """Return typed provenance for realized design columns."""
        from fmrimod.design import DesignColumns

        return DesignColumns.from_model(self)

    @property
    def n_event_columns(self) -> int:
        """Number of event-related design columns."""
        dm = self._get_event_dm(run=0)
        return dm.shape[1]

    @property
    def n_baseline_columns(self) -> int:
        """Number of baseline design columns."""
        dm = self._get_baseline_dm(run=0)
        return dm.shape[1]

    @property
    def n_columns(self) -> int:
        """Total number of design columns."""
        return self.n_event_columns + self.n_baseline_columns

    @property
    def event_column_indices(self) -> NDArray[np.intp]:
        """Column indices of event-related terms in the full design."""
        return np.arange(self.n_event_columns, dtype=np.intp)

    @property
    def baseline_column_indices(self) -> NDArray[np.intp]:
        """Column indices of baseline terms in the full design."""
        start = self.n_event_columns
        return np.arange(start, start + self.n_baseline_columns, dtype=np.intp)

    # -- Contrast weights --

    def contrast_weights(self, **kwargs: object) -> dict[str, Any]:
        """Extract contrast weight specifications from the event model.

        Returns
        -------
        dict
            Contrast specifications.
        """
        if hasattr(self._event_model, "contrast_weights"):
            return cast(
                "dict[str, Any]",
                self._event_model.contrast_weights(**cast("dict[str, Any]", kwargs)),
            )
        return {}

    # -- Internal helpers --

    @staticmethod
    def _as_named_frame(
        dm: object,
        names: object = None,
    ) -> pd.DataFrame:
        """Return *dm* as a DataFrame, preserving supplied column names."""
        frame = dm if isinstance(dm, pd.DataFrame) else pd.DataFrame(cast(Any, dm))
        if names is not None:
            colnames = list(cast(Any, names))
            if len(colnames) == frame.shape[1]:
                frame = frame.copy()
                frame.columns = colnames
        return frame

    def _get_event_dm(self, run: Optional[int]) -> pd.DataFrame:
        """Get event design matrix, optionally for a single run."""
        em = self._event_model
        colnames = getattr(em, "column_names", None)
        if run is not None:
            if hasattr(em, "design_matrix"):
                dm = em.design_matrix
                # Slice to run rows
                starts, ends = self._run_slices()
                if isinstance(dm, np.ndarray):
                    dm_slice = dm[starts[run] : ends[run]]
                else:
                    dm_slice = dm.iloc[starts[run] : ends[run]]
                if isinstance(dm_slice, pd.DataFrame):
                    return self._as_named_frame(
                        dm_slice.reset_index(drop=True),
                        colnames,
                    )
                return self._as_named_frame(dm_slice, colnames)
            raise TypeError("event_model does not have a design_matrix attribute")

        if hasattr(em, "design_matrix"):
            dm = em.design_matrix
            return self._as_named_frame(dm, colnames)
        raise TypeError("event_model does not have a design_matrix attribute")

    def _get_baseline_dm(self, run: Optional[int]) -> pd.DataFrame:
        """Get baseline design matrix, optionally for a single run."""
        bm = self._baseline_model
        if not hasattr(bm, "design_matrix"):
            raise TypeError("baseline_model does not have a design_matrix method")

        dm = None
        colnames = getattr(bm, "column_names", None)

        # Prefer baseline generic dispatch if available; fall back to the
        # object's design_matrix attribute otherwise.
        bl_design_matrix = None
        try:
            from ..baseline import baseline_model as baseline_module

            bl_design_matrix = getattr(baseline_module, "design_matrix", None)
        except Exception:
            bl_design_matrix = None

        if callable(bl_design_matrix):
            try:
                dm = bl_design_matrix(bm, blockid=([run] if run is not None else None))
            except TypeError:
                dm = bl_design_matrix(bm)
        else:
            dm_attr = getattr(bm, "design_matrix")
            dm = dm_attr() if callable(dm_attr) else dm_attr

        if run is not None:
            starts, ends = self._run_slices()
            start = starts[run]
            end = ends[run]
            if isinstance(dm, np.ndarray):
                dm = dm[start:end]
            elif isinstance(dm, pd.DataFrame):
                dm = dm.iloc[start:end].reset_index(drop=True)
            else:
                dm = np.asarray(dm)[start:end]

        return self._as_named_frame(dm, colnames)

    def _run_slices(self) -> tuple[Any, ...]:
        """Compute start/end row indices for each run."""
        lengths = self.n_timepoints
        ends = np.cumsum(lengths)
        starts = np.concatenate([[0], ends[:-1]])
        return starts.tolist(), ends.tolist()

    # -- Display --

    def __repr__(self) -> str:
        return (
            f"FmriModel(\n"
            f"  n_runs={self.n_runs},\n"
            f"  n_timepoints={self.n_timepoints},\n"
            f"  n_event_columns={self.n_event_columns},\n"
            f"  n_baseline_columns={self.n_baseline_columns}\n"
            f")"
        )


def create_fmri_model(
    event_model: object,
    baseline_model: object,
    dataset: DatasetProtocol,
) -> FmriModel:
    """Create an :class:`FmriModel` from its components.

    This is a convenience factory function.

    Parameters
    ----------
    event_model : EventModel
        The event model.
    baseline_model : BaselineModel
        The baseline model.
    dataset : DatasetProtocol
        The data source.

    Returns
    -------
    FmriModel
    """
    return FmriModel(event_model, baseline_model, dataset)
