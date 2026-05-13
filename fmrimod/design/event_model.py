"""Event model implementation for fMRI design matrices."""

from __future__ import annotations

from itertools import product as iter_product
from typing import Dict, List, Optional, Union, Any
import warnings

import numpy as np
import pandas as pd

from ..types import (
    Array,
    EventProtocol,
    HRFProtocol,
    ModelProtocol,
    SamplingInfo,
)
from ..events import (
    EventFactor,
    EventVariable,
    EventMatrix,
    EventBasis,
    create_event,
    events_from_dataframe,
)
from ..events.term import EventTerm, create_interaction
from ..formula.base import Term, EventModelBuilder
from ..covariate import CovariateTerm, CovariateEvent, create_covariate_events
from ..dispatch import get_hrf
from .._warnings import call_safely, suppress_fmrimod_warnings
from ..naming import (
    make_term_tag,
    make_column_names,
    make_unique_colnames,
    make_cond_tag,
    level_token,
    continuous_token,
)
from ..events.cells import (
    cells_event_model,
    conditions_event_model,
)


def _import_fmrimod():
    """Import fmrimod HRF subpackage with warning suppression.

    fmrimod raises noisy warnings during import that we don't want to surface
    from otherwise stable user-facing paths.
    """

    with suppress_fmrimod_warnings():
        from .. import hrf as _hrf_pkg
        from ..hrf import library as _hrf_library
        from ..hrf import registry as _hrf_registry
        from ..regressor import regressor as _regressor_func

    class _FmrimodShim:
        """Shim to provide fmrimod-like interface."""
        SPM_CANONICAL = _hrf_library.SPM_CANONICAL
        SPM_WITH_DERIVATIVE = _hrf_library.SPM_WITH_DERIVATIVE
        SPM_WITH_DISPERSION = _hrf_library.SPM_WITH_DISPERSION
        get_hrf = staticmethod(_hrf_registry.get_hrf)
        regressor = staticmethod(_regressor_func)

    return _FmrimodShim


class EventModel(ModelProtocol):
    """Event model for fMRI design matrix construction.

    ``EventModel`` is the central object in fmrimod. It combines event
    definitions, HRF specifications, and temporal sampling information to
    produce a design matrix suitable for general linear model (GLM) analysis
    of fMRI data. Design matrices are lazily computed on first access and
    then cached.

    Users should typically create instances via the :func:`event_model`
    factory function rather than calling this constructor directly.

    Parameters
    ----------
    terms : list of Term
        Model terms specifying events and transformations.
    events : dict
        Dictionary mapping event names to Event objects
        (``EventFactor``, ``EventVariable``, ``EventMatrix``, or
        ``EventBasis``).
    sampling_info : SamplingInfo or fmrimod.SamplingFrame
        Information about temporal sampling (TR, number of scans,
        block structure).
    name : str, optional
        Human-readable name for the model. Defaults to ``"EventModel"``.
    precision : float, optional
        Temporal precision in seconds for HRF convolution via
        ``fmrimod.regressor().evaluate()``. Smaller values give more
        accurate convolution at the cost of speed. Default is 0.3.
    blockids : array-like, optional
        1-indexed block/run identifier for each event. When provided
        together with a multi-block ``sampling_info``, convolution is
        performed separately within each run to prevent HRF responses
        from bleeding across run boundaries.

    Attributes
    ----------
    terms : list of Term
        Model terms.
    events : dict
        Event objects keyed by name.
    sampling_info : SamplingInfo or SamplingFrame
        Sampling information.
    design_matrix : Array
        Computed design matrix, shape ``(n_timepoints, n_columns)``.
        Lazily computed and cached on first access.
    column_names : list of str
        Names of design matrix columns.
    column_indices : dict
        Mapping from term name to list of column indices.

    See Also
    --------
    event_model : Factory function for creating ``EventModel`` instances.
    EventModelBuilder : Fluent builder interface for model construction.

    Examples
    --------
    >>> import pandas as pd
    >>> from fmrimod import event_model
    >>> df = pd.DataFrame({
    ...     'onset': [1.0, 3.0, 5.0, 7.0],
    ...     'condition': ['A', 'B', 'A', 'B'],
    ...     'duration': [1.0, 1.0, 1.0, 1.0],
    ... })
    >>> model = event_model("condition", data=df, tr=2.0, n_scans=100)
    >>> model.design_matrix.shape
    (100, 2)
    >>> model.column_names
    ['condition[A]', 'condition[B]']
    """

    def __init__(
        self,
        terms: List[Term],
        events: Dict[str, EventProtocol],
        sampling_info: SamplingInfo,
        name: Optional[str] = None,
        precision: Optional[float] = None,
        blockids: Optional[Array] = None,
    ):
        """Initialize event model."""
        self.terms = terms
        self.events = events
        self.sampling_info = sampling_info
        self.name = name or "EventModel"
        self.precision = precision or 0.3
        self._blockids = np.asarray(blockids) if blockids is not None else None

        # Cached values
        self._design_matrix = None
        self._column_names = None
        self._event_terms = None
        self._column_indices = None

    @property
    def n_events(self) -> int:
        """Number of events in the model."""
        return len(self.events)

    @property
    def n_terms(self) -> int:
        """Number of terms in the model."""
        return len(self.terms)

    @property
    def event_names(self) -> List[str]:
        """Names of events in the model."""
        return list(self.events.keys())

    @property
    def sampling_points(self) -> Array:
        """Sampling points for design matrix."""
        if hasattr(self.sampling_info, 'samples'):
            return self.sampling_info.samples
        return self.sampling_info.sampling_points

    @property
    def tr(self) -> float:
        """Repetition time."""
        if hasattr(self.sampling_info, 'TR'):
            tr_val = self.sampling_info.TR
            if hasattr(tr_val, '__len__'):
                return float(tr_val[0])
            return float(tr_val)
        tr_val = self.sampling_info.tr
        if hasattr(tr_val, '__len__'):
            return float(tr_val[0])
        return float(tr_val)

    @property
    def blockids(self) -> Optional[Array]:
        """Block IDs for events (1-indexed)."""
        return self._blockids

    def _create_event_terms(self) -> List[EventTerm]:
        """Create EventTerm objects from Term specifications."""
        if self._event_terms is not None:
            return self._event_terms

        event_terms = []
        for term in self.terms:
            event_term = self._create_single_event_term(term)
            event_terms.append(event_term)

        self._event_terms = event_terms
        return event_terms

    def _create_single_event_term(self, term: Term) -> EventTerm:
        """Create an EventTerm from a single Term specification.

        Parameters
        ----------
        term : Term
            Term specification

        Returns
        -------
        EventTerm
            Created event term
        """
        if hasattr(term, '_is_trialwise') and term._is_trialwise:
            return self._create_trialwise_event_term(term)
        elif isinstance(term, CovariateTerm):
            return self._create_covariate_event_term(term)
        else:
            return self._create_regular_event_term(term)

    def _create_trialwise_event_term(self, term: Term) -> EventTerm:
        """Create a trialwise event term.

        Parameters
        ----------
        term : Term
            Trialwise term specification

        Returns
        -------
        EventTerm
            Trialwise event term
        """
        from ..trialwise import _create_trial_factor

        n_trials, trial_onsets = self._find_trial_info()
        if n_trials == 0:
            raise ValueError("Cannot create trialwise term: no trials found")

        trial_factor = _create_trial_factor(n_trials, trial_onsets)
        self.events['_trial_factor'] = trial_factor

        event_term = EventTerm([trial_factor], name=term.name or 'trial')
        event_term._is_trialwise = True
        event_term._add_sum = term._add_sum
        event_term._trialwise_label = term._trialwise_label
        return event_term

    def _find_trial_info(self):
        """Find trial count and onsets from events.

        Returns
        -------
        tuple
            (n_trials, trial_onsets)
        """
        for event in self.events.values():
            if hasattr(event, 'onsets') and event.onsets is not None and len(event.onsets) > 0:
                return len(event.onsets), event.onsets
        return 0, None

    def _create_covariate_event_term(self, term: CovariateTerm) -> EventTerm:
        """Create a covariate event term.

        Parameters
        ----------
        term : CovariateTerm
            Covariate term specification

        Returns
        -------
        EventTerm
            Covariate event term
        """
        term_events = []
        for cov_name in term.covariates:
            event_name = f"{term.prefix}_{cov_name}" if term.prefix else cov_name
            if event_name in self.events:
                term_events.append(self.events[event_name])
            elif cov_name in self.events:
                term_events.append(self.events[cov_name])
            else:
                raise ValueError(f"Covariate event '{cov_name}' not found in model")

        event_term = EventTerm(term_events, name=term.name)
        event_term._is_covariate = True
        return event_term

    def _create_regular_event_term(self, term: Term) -> EventTerm:
        """Create a regular (non-covariate, non-trialwise) event term.

        Parameters
        ----------
        term : Term
            Regular term specification

        Returns
        -------
        EventTerm
            Regular event term
        """
        term_events = []
        for event_name in term.events:
            if event_name not in self.events:
                raise ValueError(f"Event '{event_name}' not found in model")
            term_events.append(self.events[event_name])

        if len(term_events) == 1:
            if term.basis is not None:
                event = term_events[0]
                if event.event_type == "continuous":
                    term_events = [
                        EventBasis(
                            name=event.name,
                            onsets=event.onsets,
                            values=event.values,
                            durations=event.durations,
                            basis=term.basis,
                        )
                    ]
                elif event.event_type == "basis":
                    # Event was already created with basis expansion.
                    pass
                else:
                    warnings.warn(
                        f"Basis transform for term '{term.name}' cannot be applied "
                        f"to {event.event_type} event '{event.name}'; "
                        "using raw event values"
                    )
            return EventTerm(term_events, name=term.name)
        else:
            return create_interaction(*term_events, name=term.name)

    def _resolve_hrf(self, hrf) -> Any:
        """Resolve HRF specification to a fmrimod HRF object.

        Parameters
        ----------
        hrf : str or HRF object
            HRF specification

        Returns
        -------
        HRF object
            Resolved HRF suitable for fmrimod.regressor()
        """
        try:
            fmrimod = _import_fmrimod()
        except ImportError as err:
            raise ImportError(
                "fmrimod is required for HRF convolution. "
                "Install it with: pip install fmrimod"
            ) from err

        if isinstance(hrf, str):
            return self._resolve_hrf_from_string(hrf, fmrimod)
        else:
            return self._resolve_hrf_from_object(hrf, fmrimod)

    def _resolve_hrf_from_string(self, hrf: str, fmrimod) -> Any:
        """Resolve HRF from string specification.

        Parameters
        ----------
        hrf : str
            HRF name
        fmrimod : module
            fmrimod module

        Returns
        -------
        HRF object
        """
        hrf_lower = hrf.lower()

        # Check standard SPM HRF names
        spm_hrf = self._get_spm_hrf(hrf_lower, fmrimod)
        if spm_hrf is not None:
            return spm_hrf

        # Try fmrimod registry
        try:
            return call_safely(fmrimod.get_hrf, hrf_lower)
        except (KeyError, ValueError):
            pass

        # Try local registry
        return self._resolve_local_hrf(hrf, fmrimod)

    def _get_spm_hrf(self, hrf_lower: str, fmrimod) -> Any:
        """Get SPM HRF by name.

        Parameters
        ----------
        hrf_lower : str
            Lowercase HRF name
        fmrimod : module
            fmrimod module

        Returns
        -------
        HRF object or None
        """
        spm_map = {
            'spm': fmrimod.SPM_CANONICAL,
            'spm_canonical': fmrimod.SPM_CANONICAL,
            'canonical': fmrimod.SPM_CANONICAL,
            'spmg1': fmrimod.SPM_CANONICAL,
            'spmg2': fmrimod.SPM_WITH_DERIVATIVE,
            'spmg3': fmrimod.SPM_WITH_DISPERSION,
            'simple': fmrimod.SPM_CANONICAL,
        }
        return spm_map.get(hrf_lower)

    def _resolve_local_hrf(self, hrf: str, fmrimod) -> Any:
        """Resolve HRF from local registry.

        Parameters
        ----------
        hrf : str
            HRF name
        fmrimod : module
            fmrimod module

        Returns
        -------
        HRF object
        """
        hrf_obj = get_hrf(hrf)
        if hasattr(hrf_obj, '_hrf'):
            return hrf_obj._hrf
        if not hasattr(hrf_obj, 'nbasis'):
            return fmrimod.SPM_CANONICAL
        return hrf_obj

    def _resolve_hrf_from_object(self, hrf, fmrimod) -> Any:
        """Resolve HRF from object.

        Parameters
        ----------
        hrf : object
            HRF object
        fmrimod : module
            fmrimod module

        Returns
        -------
        HRF object
        """
        if hasattr(hrf, 'nbasis') and hasattr(hrf, 'evaluate'):
            if self._is_valid_fmrimod_hrf(hrf, fmrimod):
                return hrf
        return fmrimod.SPM_CANONICAL

    def _is_valid_fmrimod_hrf(self, hrf, fmrimod) -> bool:
        """Check if an HRF object is valid for fmrimod.

        Parameters
        ----------
        hrf : object
            HRF object to check
        fmrimod : module
            fmrimod module

        Returns
        -------
        bool
            True if valid
        """
        try:
            fmrimod.regressor(onsets=np.array([0.0]), hrf=hrf)
            return True
        except Exception:
            return False

    @property
    def design_matrix(self) -> Array:
        """Get the design matrix.

        Returns
        -------
        Array
            Design matrix with shape (n_timepoints, n_columns)
        """
        if self._design_matrix is not None:
            return self._design_matrix

        event_terms = self._create_event_terms()

        columns = []
        column_names = []
        existing_tags = []
        column_indices = {}
        current_col = 0

        for i, (term, event_term) in enumerate(zip(self.terms, event_terms)):
            # Compute design matrix columns for this term
            if isinstance(term, CovariateTerm):
                X_term = event_term.design_matrix(self.sampling_points)
            elif term.hrf is not None:
                # Use proper fmrimod-based convolution
                X_term = self._convolve_term(event_term, term)
            else:
                X_term = event_term.design_matrix(self.sampling_points)

            if X_term.ndim == 1:
                X_term = X_term.reshape(-1, 1)

            # Handle trialwise add_sum
            if hasattr(event_term, '_is_trialwise') and event_term._is_trialwise and event_term._add_sum:
                mean_col = X_term.mean(axis=1, keepdims=True)
                X_term = np.hstack([X_term, mean_col])

            columns.append(X_term)

            # Track column indices for this term
            n_cols = X_term.shape[1]
            term_indices = list(range(current_col, current_col + n_cols))
            column_indices[term.name] = term_indices
            current_col += n_cols

            # Generate term tag
            term_tag = make_term_tag(
                term_name=term.name,
                event_names=term.events,
                hrf_name=term.hrf if isinstance(term.hrf, str) else None,
                basis_type=term.basis.name if hasattr(term, 'basis') and term.basis else None,
                existing_tags=existing_tags
            )
            if term_tag:
                existing_tags.append(term_tag)

            # Generate condition tags
            cond_tags = self._get_condition_tags(event_term)

            # Handle trialwise column names
            if hasattr(event_term, '_is_trialwise') and event_term._is_trialwise:
                n_trials = X_term.shape[1]
                if event_term._add_sum:
                    n_trials -= 1

                label = event_term._trialwise_label or 'trial'
                pad = len(str(n_trials))
                term_col_names = [f"{label}_{i+1:0{pad}d}" for i in range(n_trials)]

                if event_term._add_sum:
                    term_col_names.append(f"{label}_mean")
            else:
                # Get HRF nbasis
                nb = 1
                if term.hrf is not None:
                    try:
                        hrf_obj = self._resolve_hrf(term.hrf)
                        nb = hrf_obj.nbasis
                    except Exception:
                        pass
                # Basis expansion is represented directly via condition tags for
                # non-HRF terms, so keep nb=1 to avoid double-expanding names.

                term_col_names = make_column_names(term_tag, cond_tags, nb)

            column_names.extend(term_col_names)

        self._design_matrix = np.hstack(columns)
        self._column_names = make_unique_colnames(column_names)
        self._column_indices = column_indices

        return self._design_matrix

    def _convolve_term(self, event_term: EventTerm, term: Term) -> Array:
        """Convolve an event term with HRF using fmrimod.

        Uses fmrimod.regressor().evaluate() for proper precision-based
        convolution instead of manual HRF point evaluation.

        Parameters
        ----------
        event_term : EventTerm
            Event term containing event objects
        term : Term
            Term specification with HRF info

        Returns
        -------
        Array
            Convolved design matrix columns
        """
        fmrimod = _import_fmrimod()

        hrf_obj = self._resolve_hrf(term.hrf)
        nb = hrf_obj.nbasis
        sf = self.sampling_info

        # Read normalize/summate from Term
        normalize = getattr(term, 'normalize', False)
        summate = getattr(term, 'summate', True)

        # Check if we have multi-block
        has_multiblock = (
            self._blockids is not None
            and hasattr(sf, 'n_blocks')
            and sf.n_blocks > 1
        )

        if has_multiblock:
            result = self._convolve_term_multiblock(
                event_term, hrf_obj, summate=summate
            )
        else:
            result = self._convolve_term_single(
                event_term, hrf_obj, summate=summate
            )

        # Apply peak normalization after convolution
        if normalize:
            if result.ndim > 1:
                for i in range(result.shape[1]):
                    mx = np.max(np.abs(result[:, i]))
                    if mx > 0:
                        result[:, i] = result[:, i] / mx
            else:
                mx = np.max(np.abs(result))
                if mx > 0:
                    result = result / mx

        return result

    def _convolve_term_single(
        self, event_term: EventTerm, hrf_obj, summate: bool = True
    ) -> Array:
        """Convolve event term for single block design.

        Parameters
        ----------
        event_term : EventTerm
            Event term with event objects
        hrf_obj : HRF
            fmrimod HRF object
        summate : bool
            Whether overlapping HRF responses are summed (default True).

        Returns
        -------
        Array
            Convolved columns, shape (n_samples, n_conditions * nbasis)
        """
        fmrimod = _import_fmrimod()

        grid = self.sampling_points
        nb = hrf_obj.nbasis
        events = event_term.events

        if len(events) == 1 and not event_term.interaction:
            return self._convolve_single_event(
                events[0], hrf_obj, grid, summate=summate
            )
        else:
            return self._convolve_interaction_events(
                event_term, hrf_obj, grid, summate=summate
            )

    def _convolve_single_event(
        self, event, hrf_obj, grid: Array, summate: bool = True
    ) -> Array:
        """Convolve a single event with HRF.

        Parameters
        ----------
        event : EventFactor, EventVariable, EventMatrix, or EventBasis
            Single event object
        hrf_obj : HRF
            fmrimod HRF object
        grid : Array
            Sampling time points
        summate : bool
            Whether overlapping HRF responses are summed (default True).

        Returns
        -------
        Array
            Convolved columns
        """
        fmrimod = _import_fmrimod()

        nb = hrf_obj.nbasis
        n_samples = len(grid)

        event_type_handlers = {
            "categorical": self._convolve_categorical_event,
            "continuous": self._convolve_continuous_event,
            "matrix": self._convolve_matrix_event,
            "basis": self._convolve_basis_event,
        }

        handler = event_type_handlers.get(event.event_type)
        if handler is None:
            raise ValueError(f"Unknown event type: {event.event_type}")

        return handler(event, hrf_obj, grid, nb, n_samples, summate)

    def _convolve_categorical_event(
        self, event, hrf_obj, grid: Array, nb: int, n_samples: int, summate: bool
    ) -> Array:
        """Convolve categorical event with HRF.

        Parameters
        ----------
        event : EventFactor
            Categorical event
        hrf_obj : HRF
            fmrimod HRF object
        grid : Array
            Sampling time points
        nb : int
            Number of basis functions
        n_samples : int
            Number of samples
        summate : bool
            Whether overlapping HRF responses are summed

        Returns
        -------
        Array
            Convolved columns, one per level
        """
        fmrimod = _import_fmrimod()

        per_level = []
        for level in event.levels:
            mask = event.values == level
            if not np.any(mask):
                per_level.append(np.zeros((n_samples, nb)))
                continue

            onsets = event.onsets[mask]
            durs = event.durations[mask]

            reg = fmrimod.regressor(
                onsets=onsets,
                hrf=hrf_obj,
                duration=durs,
                amplitude=1.0,
                summate=summate
            )
            result = reg.evaluate(grid, precision=self.precision)
            if result.ndim == 1:
                result = result.reshape(-1, 1)
            per_level.append(result)

        return np.hstack(per_level)

    def _convolve_continuous_event(
        self, event, hrf_obj, grid: Array, nb: int, n_samples: int, summate: bool
    ) -> Array:
        """Convolve continuous event with HRF.

        Parameters
        ----------
        event : EventVariable
            Continuous event
        hrf_obj : HRF
            fmrimod HRF object
        grid : Array
            Sampling time points
        nb : int
            Number of basis functions
        n_samples : int
            Number of samples
        summate : bool
            Whether overlapping HRF responses are summed

        Returns
        -------
        Array
            Convolved column
        """
        fmrimod = _import_fmrimod()

        reg = fmrimod.regressor(
            onsets=event.onsets,
            hrf=hrf_obj,
            duration=event.durations,
            amplitude=event.values,
            summate=summate
        )
        result = reg.evaluate(grid, precision=self.precision)
        if result.ndim == 1:
            result = result.reshape(-1, 1)
        return result

    def _convolve_matrix_event(
        self, event, hrf_obj, grid: Array, nb: int, n_samples: int, summate: bool
    ) -> Array:
        """Convolve matrix event with HRF.

        Parameters
        ----------
        event : EventMatrix
            Matrix event
        hrf_obj : HRF
            fmrimod HRF object
        grid : Array
            Sampling time points
        nb : int
            Number of basis functions
        n_samples : int
            Number of samples
        summate : bool
            Whether overlapping HRF responses are summed

        Returns
        -------
        Array
            Convolved columns, one per matrix column
        """
        fmrimod = _import_fmrimod()

        cols = []
        for i in range(event.n_columns):
            reg = fmrimod.regressor(
                onsets=event.onsets,
                hrf=hrf_obj,
                duration=event.durations,
                amplitude=event.values[:, i],
                summate=summate
            )
            result = reg.evaluate(grid, precision=self.precision)
            if result.ndim == 1:
                result = result.reshape(-1, 1)
            cols.append(result)
        return np.hstack(cols)

    def _convolve_basis_event(
        self, event, hrf_obj, grid: Array, nb: int, n_samples: int, summate: bool
    ) -> Array:
        """Convolve basis event with HRF.

        Parameters
        ----------
        event : EventBasis
            Basis event
        hrf_obj : HRF
            fmrimod HRF object
        grid : Array
            Sampling time points
        nb : int
            Number of basis functions
        n_samples : int
            Number of samples
        summate : bool
            Whether overlapping HRF responses are summed

        Returns
        -------
        Array
            Convolved columns
        """
        fmrimod = _import_fmrimod()

        if hasattr(event, 'basis_matrix') and event.basis_matrix is not None:
            cols = []
            for i in range(event.basis_matrix.shape[1]):
                reg = fmrimod.regressor(
                    onsets=event.onsets,
                    hrf=hrf_obj,
                    duration=event.durations,
                    amplitude=event.basis_matrix[:, i],
                    summate=summate
                )
                result = reg.evaluate(grid, precision=self.precision)
                if result.ndim == 1:
                    result = result.reshape(-1, 1)
                cols.append(result)
            return np.hstack(cols)
        else:
            # Fallback: treat like continuous with unit amplitude
            reg = fmrimod.regressor(
                onsets=event.onsets,
                hrf=hrf_obj,
                duration=event.durations,
                amplitude=1.0,
                summate=summate
            )
            result = reg.evaluate(grid, precision=self.precision)
            if result.ndim == 1:
                result = result.reshape(-1, 1)
            return result

    def _convolve_interaction_events(
        self, event_term: EventTerm, hrf_obj, grid: Array,
        summate: bool = True
    ) -> Array:
        """Convolve interaction term with HRF.

        Handles crossing of categorical/continuous events and creates
        separate regressors for each combination.

        Parameters
        ----------
        event_term : EventTerm
            Interaction event term
        hrf_obj : HRF
            fmrimod HRF object
        grid : Array
            Sampling time points
        summate : bool
            Whether overlapping HRF responses are summed (default True).

        Returns
        -------
        Array
            Convolved columns for all level combinations
        """
        fmrimod = _import_fmrimod()

        nb = hrf_obj.nbasis
        n_samples = len(grid)
        events = event_term.events

        # Separate categorical and continuous events
        cat_events = [e for e in events if e.event_type == "categorical"]
        cont_events = [e for e in events if e.event_type != "categorical"]

        combos = self._get_level_combinations(cat_events)

        cols = []
        for combo in combos:
            combo_result = self._convolve_interaction_combo(
                combo, cat_events, cont_events, events, hrf_obj, grid, nb, n_samples, summate
            )
            cols.extend(combo_result)

        return np.hstack(cols)

    def _get_level_combinations(self, cat_events):
        """Get all level combinations from categorical events.

        Parameters
        ----------
        cat_events : list
            List of categorical events

        Returns
        -------
        list
            List of level combinations (tuples)
        """
        if cat_events:
            level_lists = [e.levels for e in cat_events]
            return list(iter_product(*level_lists))
        else:
            return [()]

    def _convolve_interaction_combo(
        self, combo, cat_events, cont_events, events, hrf_obj, grid, nb, n_samples, summate
    ):
        """Convolve one combination from interaction term.

        Parameters
        ----------
        combo : tuple
            Level combination for categorical events
        cat_events : list
            Categorical events
        cont_events : list
            Continuous events
        events : list
            All events
        hrf_obj : HRF
            HRF object
        grid : Array
            Sampling time points
        nb : int
            Number of basis functions
        n_samples : int
            Number of samples
        summate : bool
            Whether overlapping HRF responses are summed

        Returns
        -------
        list
            List of convolved result arrays
        """
        fmrimod = _import_fmrimod()

        # Build mask for this combination
        mask = self._build_combination_mask(combo, cat_events, events)

        if not np.any(mask):
            n_cont_cols = self._count_continuous_columns(cont_events)
            return [np.zeros((n_samples, nb * n_cont_cols))]

        onsets = events[0].onsets[mask]
        durations = events[0].durations[mask]

        if cont_events:
            return self._convolve_continuous_in_combo(
                cont_events, mask, onsets, durations, hrf_obj, grid, summate
            )
        else:
            return self._convolve_pure_categorical_combo(
                onsets, durations, hrf_obj, grid, summate
            )

    def _build_combination_mask(self, combo, cat_events, events):
        """Build boolean mask for a level combination.

        Parameters
        ----------
        combo : tuple
            Level combination
        cat_events : list
            Categorical events
        events : list
            All events

        Returns
        -------
        Array
            Boolean mask
        """
        mask = np.ones(len(events[0].onsets), dtype=bool)
        for event, level in zip(cat_events, combo):
            mask &= (event.values == level)
        return mask

    def _count_continuous_columns(self, cont_events):
        """Count total columns from continuous events.

        Parameters
        ----------
        cont_events : list
            Continuous events

        Returns
        -------
        int
            Number of columns
        """
        if cont_events:
            n_cols = 1
            for e in cont_events:
                if getattr(e, 'event_type', None) == 'basis':
                    n_cols *= e.n_basis
                else:
                    n_cols *= getattr(e, 'n_columns', 1)
            return n_cols
        else:
            return 1

    def _convolve_continuous_in_combo(
        self, cont_events, mask, onsets, durations, hrf_obj, grid, summate
    ):
        """Convolve continuous events within an interaction combination.

        Parameters
        ----------
        cont_events : list
            Continuous events
        mask : Array
            Boolean mask for events
        onsets : Array
            Event onsets
        durations : Array
            Event durations
        hrf_obj : HRF
            HRF object
        grid : Array
            Sampling time points
        summate : bool
            Whether overlapping HRF responses are summed

        Returns
        -------
        list
            List of convolved result arrays
        """
        fmrimod = _import_fmrimod()

        cols = []
        for cont_event in cont_events:
            if cont_event.event_type == "matrix":
                for ci in range(cont_event.n_columns):
                    amplitude = cont_event.values[mask, ci]
                    result = self._evaluate_regressor(
                        onsets, durations, amplitude, hrf_obj, grid, summate
                    )
                    cols.append(result)
            elif cont_event.event_type == "basis":
                for ci in range(cont_event.n_basis):
                    amplitude = cont_event.basis_matrix[mask, ci]
                    result = self._evaluate_regressor(
                        onsets, durations, amplitude, hrf_obj, grid, summate
                    )
                    cols.append(result)
            else:
                amplitude = cont_event.values[mask]
                result = self._evaluate_regressor(
                    onsets, durations, amplitude, hrf_obj, grid, summate
                )
                cols.append(result)
        return cols

    def _convolve_pure_categorical_combo(
        self, onsets, durations, hrf_obj, grid, summate
    ):
        """Convolve pure categorical interaction combination.

        Parameters
        ----------
        onsets : Array
            Event onsets
        durations : Array
            Event durations
        hrf_obj : HRF
            HRF object
        grid : Array
            Sampling time points
        summate : bool
            Whether overlapping HRF responses are summed

        Returns
        -------
        list
            List with single convolved result array
        """
        result = self._evaluate_regressor(
            onsets, durations, 1.0, hrf_obj, grid, summate
        )
        return [result]

    def _evaluate_regressor(self, onsets, durations, amplitude, hrf_obj, grid, summate):
        """Evaluate a regressor with given parameters.

        Parameters
        ----------
        onsets : Array
            Event onsets
        durations : Array
            Event durations
        amplitude : float or Array
            Event amplitude(s)
        hrf_obj : HRF
            HRF object
        grid : Array
            Sampling time points
        summate : bool
            Whether overlapping HRF responses are summed

        Returns
        -------
        Array
            Evaluated regressor
        """
        fmrimod = _import_fmrimod()

        reg = fmrimod.regressor(
            onsets=onsets,
            hrf=hrf_obj,
            duration=durations,
            amplitude=amplitude,
            summate=summate
        )
        result = reg.evaluate(grid, precision=self.precision)
        if result.ndim == 1:
            result = result.reshape(-1, 1)
        return result

    def _convolve_term_multiblock(
        self, event_term: EventTerm, hrf_obj, summate: bool = True
    ) -> Array:
        """Per-block convolution for multi-run designs.

        Convolves events separately within each block/run to prevent
        HRF responses from leaking across run boundaries.

        Parameters
        ----------
        event_term : EventTerm
            Event term with event objects
        hrf_obj : HRF
            fmrimod HRF object
        summate : bool
            Whether overlapping HRF responses are summed (default True).

        Returns
        -------
        Array
            Vertically stacked per-block convolved columns
        """
        sf = self.sampling_info
        n_blocks = sf.n_blocks
        global_grid = self.sampling_points
        blocklens = np.asarray(sf.blocklens, dtype=int)

        tr_source = sf.TR if hasattr(sf, "TR") else sf.tr
        tr_vals = np.asarray(tr_source, dtype=float)
        if tr_vals.ndim == 0 or tr_vals.size == 1:
            tr_vals = np.full(n_blocks, float(tr_vals.reshape(-1)[0]))
        elif tr_vals.size != n_blocks:
            tr_vals = np.full(n_blocks, float(tr_vals.reshape(-1)[0]))

        # Match fmridesign/fmrihrf global_onsets semantics:
        # block-local onsets are shifted by cumulative prior block durations.
        onset_offsets = np.zeros(n_blocks, dtype=float)
        if n_blocks > 1:
            onset_offsets[1:] = np.cumsum(blocklens[:-1] * tr_vals[:-1])

        block_results = []
        row_start = 0
        for b in range(n_blocks):
            row_end = row_start + int(blocklens[b])
            block_grid = global_grid[row_start:row_end]
            block_event_mask = (self._blockids == (b + 1))

            # Create block-local event copies
            block_event_term = self._subset_event_term(
                event_term,
                block_event_mask,
                onset_shift=float(onset_offsets[b]),
            )

            if block_event_term is None:
                # No events in this block
                n_cols = self._count_term_columns(event_term, hrf_obj)
                block_results.append(np.zeros((int(blocklens[b]), n_cols)))
            else:
                # Evaluate on this block's global-time sample grid. This matches
                # fmridesign::convolve.event_term, where each block is evaluated
                # separately and the fine-grid origin follows the block samples.
                events = block_event_term.events
                if len(events) == 1 and not block_event_term.interaction:
                    full_result = self._convolve_single_event(
                        events[0], hrf_obj, block_grid,
                        summate=summate,
                    )
                else:
                    full_result = self._convolve_interaction_events(
                        block_event_term, hrf_obj, block_grid,
                        summate=summate,
                    )
                block_results.append(full_result)

            row_start = row_end

        return np.vstack(block_results)

    def _subset_event_term(
        self, event_term: EventTerm, mask: Array, onset_shift: float = 0.0
    ) -> Optional[EventTerm]:
        """Create a subset of an event term for a specific block.

        Parameters
        ----------
        event_term : EventTerm
            Original event term
        mask : Array
            Boolean mask for events to include

        Returns
        -------
        EventTerm or None
            Subset event term, or None if no events match
        """
        if not np.any(mask):
            return None

        subset_events = []
        for event in event_term.events:
            if isinstance(event, EventFactor):
                subset_event = EventFactor(
                    name=event.name,
                    onsets=event.onsets[mask] + onset_shift,
                    values=np.array(event.values)[mask],
                    durations=event.durations[mask],
                    levels=event.levels,
                )
            elif isinstance(event, EventVariable):
                subset_event = EventVariable(
                    name=event.name,
                    onsets=event.onsets[mask] + onset_shift,
                    values=event.raw_values[mask],
                    durations=event.durations[mask],
                    center=False,  # Already transformed
                )
                # Override with pre-transformed values
                subset_event.values = event.values[mask]
            elif isinstance(event, EventMatrix):
                subset_event = EventMatrix(
                    name=event.name,
                    onsets=event.onsets[mask] + onset_shift,
                    values=event.values[mask],
                    durations=event.durations[mask],
                    column_names=event.column_names,
                )
            elif event.event_type == "basis":
                subset_event = EventBasis(
                    name=event.name,
                    onsets=event.onsets[mask] + onset_shift,
                    values=event.values[mask],
                    basis=event.basis,
                    durations=event.durations[mask],
                )
            else:
                # For other types, try generic subsetting
                subset_event = event  # Fallback

            subset_events.append(subset_event)

        return EventTerm(
            subset_events,
            name=event_term.name,
            interaction=event_term.interaction,
        )

    def _count_term_columns(self, event_term: EventTerm, hrf_obj) -> int:
        """Count the number of columns a term will produce.

        Parameters
        ----------
        event_term : EventTerm
            Event term
        hrf_obj : HRF
            HRF object

        Returns
        -------
        int
            Number of design matrix columns
        """
        nb = hrf_obj.nbasis
        n_cond = event_term._get_n_columns()
        return n_cond * nb

    def _get_condition_tags(self, event_term: EventTerm) -> List[str]:
        """Get condition tags for an event term.

        Parameters
        ----------
        event_term : EventTerm
            Event term to get condition tags for

        Returns
        -------
        list of str
            Condition tags
        """
        cond_tags = []

        if event_term.interaction:
            if event_term.is_categorical:
                levels = event_term.get_levels()
                for level_combo in levels:
                    tokens = []
                    for j, (event, level) in enumerate(zip(event_term.events, level_combo)):
                        token = level_token(event.name, level)
                        tokens.append(token)
                    cond_tags.append(make_cond_tag(tokens))
            else:
                cond_tags = event_term.get_column_names()
        else:
            event = event_term.events[0]
            if event.event_type == "categorical":
                for level in event.levels:
                    cond_tags.append(level_token(event.name, level))
            elif event.event_type == "matrix":
                for col_name in event.column_names:
                    cond_tags.append(continuous_token(col_name))
            elif event.event_type == "basis":
                cond_tags.extend(event.basis_names)
            else:
                cond_tags.append(continuous_token(event.name))

        return cond_tags

    @property
    def column_names(self) -> List[str]:
        """Names of design matrix columns."""
        if self._column_names is None:
            _ = self.design_matrix
        return self._column_names

    @property
    def column_indices(self) -> Dict[str, List[int]]:
        """Column indices for each term."""
        if self._column_indices is None:
            _ = self.design_matrix
        return self._column_indices

    # Keep backward compat alias
    def _convolve_hrf(self, X: Array, hrf) -> Array:
        """Legacy convolution method - delegates to _convolve_term.

        This method is kept for backward compatibility but should not
        be called directly. Use _convolve_term instead.
        """
        # If called with old API, fall back to regressor-based approach
        fmrimod = _import_fmrimod()

        hrf_obj = self._resolve_hrf(hrf)
        nb = hrf_obj.nbasis
        grid = self.sampling_points
        n_samples = len(grid)

        if nb == 1:
            X_conv = np.zeros_like(X)
            for i in range(X.shape[1]):
                nz = np.nonzero(X[:, i])[0]
                if len(nz) == 0:
                    continue
                onsets = grid[nz]
                amplitudes = X[nz, i]
                reg = fmrimod.regressor(
                    onsets=onsets, hrf=hrf_obj,
                    duration=0, amplitude=amplitudes
                )
                X_conv[:, i] = reg.evaluate(grid, precision=self.precision).ravel()
            return X_conv
        else:
            X_conv = np.zeros((n_samples, X.shape[1] * nb))
            for i in range(X.shape[1]):
                nz = np.nonzero(X[:, i])[0]
                if len(nz) == 0:
                    continue
                onsets = grid[nz]
                amplitudes = X[nz, i]
                reg = fmrimod.regressor(
                    onsets=onsets, hrf=hrf_obj,
                    duration=0, amplitude=amplitudes
                )
                result = reg.evaluate(grid, precision=self.precision)
                if result.ndim == 1:
                    result = result.reshape(-1, 1)
                for j in range(nb):
                    X_conv[:, i * nb + j] = result[:, j]
            return X_conv

    def get_event_onsets(self, event_name: str) -> Array:
        """Get onset times for a specific event.

        Parameters
        ----------
        event_name : str
            Name of the event whose onsets to retrieve.

        Returns
        -------
        Array
            1-D array of onset times in seconds.

        Raises
        ------
        ValueError
            If ``event_name`` is not found in the model.
        """
        if event_name not in self.events:
            raise ValueError(f"Event '{event_name}' not found")
        return self.events[event_name].onsets

    def get_regressor(self, name: str) -> Array:
        """Get a specific regressor (column) from the design matrix.

        Parameters
        ----------
        name : str or int
            Column name or integer index.

        Returns
        -------
        Array
            1-D array of length ``n_timepoints`` for the requested
            regressor.

        Raises
        ------
        ValueError
            If the named regressor is not found.
        """
        if isinstance(name, int):
            return self.design_matrix[:, name]
        try:
            idx = self.column_names.index(name)
            return self.design_matrix[:, idx]
        except ValueError as err:
            raise ValueError(f"Regressor '{name}' not found") from err

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the design matrix to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns named after ``column_names`` and
            the index set to the sampling time points.
        """
        return pd.DataFrame(
            self.design_matrix,
            columns=self.column_names,
            index=self.sampling_points
        )

    def summary(self) -> str:
        """Get a concise text summary of the model.

        Returns
        -------
        str
            Multi-line summary including event count, term count,
            sampling parameters, and design matrix dimensions.
        """
        lines = [
            f"EventModel: {self.name}",
            f"  Events: {self.n_events} ({', '.join(self.event_names)})",
            f"  Terms: {self.n_terms}",
            f"  Sampling: TR={self.tr}s, {len(self.sampling_points)} timepoints",
            f"  Design matrix: {self.design_matrix.shape[0]} \u00d7 {self.design_matrix.shape[1]}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Rich string representation matching R's print.event_model()."""
        lines = [f"EventModel: {self.name}"]
        lines.append(f"  Design matrix: {self.design_matrix.shape[0]} x {self.design_matrix.shape[1]}")
        lines.append(f"  Sampling: TR={self.tr:.2g}s, {len(self.sampling_points)} timepoints")

        # Terms summary
        lines.append(f"  Terms ({self.n_terms}):")
        for i, term in enumerate(self.terms):
            hrf_str = f" | hrf={term.hrf}" if term.hrf else ""
            basis_str = f" | basis={term.basis.name}" if hasattr(term, 'basis') and term.basis else ""
            # Get column count for this term
            n_cols = len(self.column_indices.get(term.name, [])) if self.column_indices else "?"
            lines.append(f"    {i+1}. {term.name} ({n_cols} cols{hrf_str}{basis_str})")

        # Block info
        sf = self.sampling_info
        if hasattr(sf, 'n_blocks') and sf.n_blocks > 1:
            lines.append(f"  Blocks: {sf.n_blocks}")
            if hasattr(sf, 'blocklens'):
                lens_str = ", ".join(str(b) for b in sf.blocklens)
                lines.append(f"    Lengths: [{lens_str}]")

        # Design matrix preview (first 3 columns)
        col_names = self.column_names
        if col_names:
            preview_cols = min(5, len(col_names))
            preview = ", ".join(col_names[:preview_cols])
            if len(col_names) > preview_cols:
                preview += f", ... ({len(col_names) - preview_cols} more)"
            lines.append(f"  Columns: [{preview}]")

        return "\n".join(lines)

    def cells(self, drop_empty: bool = True) -> List[pd.DataFrame]:
        """Extract cells (factor-level combinations) from all terms.

        For each term containing categorical events, returns a DataFrame
        listing every observed combination of factor levels along with
        event counts.

        Parameters
        ----------
        drop_empty : bool, default True
            If True, omit combinations with zero observations.

        Returns
        -------
        list of pd.DataFrame
            One DataFrame per term. Each has categorical-level columns
            and an ``attrs['count']`` array with per-cell counts.

        See Also
        --------
        conditions : Extract condition name strings instead.
        """
        return cells_event_model(self, drop_empty)

    def conditions(
        self,
        drop_empty: bool = True,
        expand_basis: bool = False
    ) -> List[List[str]]:
        """Extract condition names from all terms.

        Parameters
        ----------
        drop_empty : bool, default True
            If True, omit conditions with zero observations.
        expand_basis : bool, default False
            If True, expand multi-basis HRFs into separate condition
            names (e.g., ``cond_b1``, ``cond_b2``).

        Returns
        -------
        list of list of str
            Nested list: outer list has one entry per term, inner list
            contains condition name strings.

        See Also
        --------
        cells : Extract full factor-level DataFrames.
        """
        return conditions_event_model(self, drop_empty, expand_basis)

    def shortnames(self, acronym: Optional[str] = None) -> List[str]:
        """Get abbreviated column names.

        Parameters
        ----------
        acronym : str, optional
            Acronym to use as a prefix for short names. If None,
            abbreviations are generated automatically.

        Returns
        -------
        list of str
            Shortened column names, one per design-matrix column.
        """
        from ..naming import shortnames as make_shortnames
        return make_shortnames(self.column_names, acronym)

    def longnames(self) -> List[str]:
        """Get full-length column names.

        This is an alias for ``column_names``, provided for symmetry
        with :meth:`shortnames`.

        Returns
        -------
        list of str
            Full column names.
        """
        return self.column_names


def event_model(
    formula: Union[str, List[Term], EventModelBuilder],
    data: Optional[pd.DataFrame] = None,
    block: Optional[Union[str, Array]] = None,
    sampling_frame=None,
    durations: Optional[Union[str, float, Array]] = None,
    drop_empty: bool = True,
    precision: Optional[float] = None,
    # Backward compat
    events: Optional[Dict[str, EventProtocol]] = None,
    sampling_info=None,
    tr: Optional[float] = None,
    n_scans: Optional[int] = None,
    sampling_rate: Optional[float] = None,
    **kwargs
) -> EventModel:
    """Create an event model from a formula, term list, or builder.

    This is the main entry point for constructing fMRI design matrices.
    It accepts multiple specification styles and handles event creation,
    HRF resolution, and sampling-frame construction automatically.

    Parameters
    ----------
    formula : str, list of Term, or EventModelBuilder
        Model specification. Accepted forms:

        * **String formula** -- R-style formula parsed by
          :func:`~fmrimod.formula.parser.parse_formula`, e.g.
          ``"condition + rating:hrf('spm')"``.
        * **List of Term** -- Explicitly constructed
          :class:`~fmrimod.formula.base.Term` objects.
        * **EventModelBuilder** -- A builder produced by the fluent
          :class:`~fmrimod.formula.base.EventModelBuilder` API.
    data : pd.DataFrame, optional
        Event table with at least an onset column and one or more
        event columns. Required when ``events`` is not provided.
    block : str or array-like, optional
        Run/block specification. If a string, the column name in
        ``data``; if array-like, one block ID per row. Block IDs are
        canonicalised to 1-indexed integers.
    sampling_frame : fmrimod.SamplingFrame, optional
        Temporal sampling specification. Takes precedence over
        ``sampling_info``, ``tr``, and ``n_scans``.
    durations : str, float, or array-like, optional
        Event durations. A string names a column in ``data``; a float
        is broadcast to all events; an array gives per-event durations.
    drop_empty : bool, default True
        Whether to drop factor levels with zero observations.
    precision : float, optional
        Temporal precision for HRF convolution in seconds (default 0.3).
    events : dict, optional
        Pre-constructed event objects keyed by name. When provided,
        ``data`` is not required.
    sampling_info : SamplingInfo, optional
        Legacy alias for ``sampling_frame``.
    tr : float, optional
        Repetition time in seconds. Together with ``n_scans``, used to
        auto-construct a ``SamplingFrame`` when ``sampling_frame`` is
        not given.
    n_scans : int, optional
        Number of scan volumes.
    sampling_rate : float, optional
        Alias for ``tr`` (kept for backward compatibility).
    **kwargs
        Forwarded to event-creation helpers (e.g. ``onset_column``).

    Returns
    -------
    EventModel
        Fully constructed event model with lazily-computed design
        matrix.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from fmrimod import event_model

    Minimal model from a DataFrame:

    >>> df = pd.DataFrame({
    ...     'onset': [0, 5, 10, 15],
    ...     'condition': ['face', 'house', 'face', 'house'],
    ...     'duration': [1, 1, 1, 1],
    ... })
    >>> model = event_model("condition", data=df, tr=2.0, n_scans=100)
    >>> model.design_matrix.shape
    (100, 2)

    Formula with explicit HRF and interaction:

    >>> model = event_model(
    ...     "hrf(condition) + hrf(condition:block)",
    ...     data=df, tr=2.0, n_scans=100,
    ... )

    Using the functional (pipe) interface:

    >>> from fmrimod.formula.functional import term, hrf
    >>> terms = [term('condition') | hrf('spmg1')]
    >>> model = event_model(terms, data=df, tr=2.0, n_scans=100)

    See Also
    --------
    EventModel : The class returned by this function.
    EventModelBuilder : Fluent builder alternative.
    """
    terms = _normalize_term_options(_parse_formula_to_terms(formula))
    sf = _resolve_sampling_frame(sampling_frame, sampling_info, tr, n_scans, sampling_rate)
    precision = precision if precision is not None else 0.3
    blockids = _parse_block_ids(block, data)
    kwargs = _parse_durations(durations, data, kwargs)

    if events is None:
        events = _create_events_from_data(data, terms, sf, kwargs)

    return EventModel(
        terms=terms,
        events=events,
        sampling_info=sf,
        precision=precision,
        blockids=blockids,
    )


def _parse_formula_to_terms(formula):
    """Parse formula specification to list of Terms.

    Parameters
    ----------
    formula : str, list of Term, or EventModelBuilder
        Formula specification

    Returns
    -------
    list of Term
        Parsed terms
    """
    if isinstance(formula, str):
        from ..formula.parser import parse_formula
        # Parse in event-model mode so R-style formulas with an LHS
        # (e.g., "onset ~ hrf(condition)") are converted to Term objects
        # without requiring formula-context variable evaluation.
        return parse_formula(formula, for_event_model=True)
    elif isinstance(formula, list):
        return _convert_formula_list_to_terms(formula)
    elif hasattr(formula, 'terms'):
        return formula.terms
    else:
        raise TypeError(
            f"formula must be str, list of Terms, or builder, "
            f"got {type(formula)}"
        )


def _normalize_term_options(terms):
    """Hoist legacy private term kwargs onto the fields EventModel reads."""
    for term in terms:
        extra = getattr(term, '_kwargs', None)
        if not extra:
            continue
        if 'normalize' in extra:
            term.normalize = bool(extra['normalize'])
        if 'summate' in extra:
            term.summate = bool(extra['summate'])
    return terms


def _convert_formula_list_to_terms(formula_list):
    """Convert list of formula items to Terms.

    Parameters
    ----------
    formula_list : list
        List of strings or Terms

    Returns
    -------
    list of Term
        Converted terms
    """
    from ..formula.base import Term as BaseTerm
    terms = []
    for item in formula_list:
        if isinstance(item, str):
            terms.append(BaseTerm(item))
        else:
            terms.append(item)
    return terms


def _resolve_sampling_frame(sampling_frame, sampling_info, tr, n_scans, sampling_rate):
    """Resolve sampling frame from various inputs.

    Parameters
    ----------
    sampling_frame : SamplingFrame or None
        Explicit sampling frame
    sampling_info : SamplingInfo or None
        Legacy sampling info
    tr : float or None
        Repetition time
    n_scans : int or None
        Number of scans
    sampling_rate : float or None
        Alias for tr

    Returns
    -------
    SamplingFrame
        Resolved sampling frame
    """
    if tr is None and sampling_rate is not None:
        tr = sampling_rate

    sf = sampling_frame or sampling_info
    if sf is None:
        if tr is None or n_scans is None:
            raise ValueError(
                "sampling_info or both tr and n_scans must be provided"
            )
        from ..sampling import SamplingFrame
        sf = SamplingFrame(tr=tr, n_scans=n_scans)

    return sf


def _parse_block_ids(block, data):
    """Parse block IDs from block specification.

    Parameters
    ----------
    block : str, array-like, or None
        Block specification
    data : DataFrame or None
        Data containing block column

    Returns
    -------
    Array or None
        1-indexed block IDs
    """
    if block is None:
        return None

    if isinstance(block, str):
        if data is None or block not in data.columns:
            raise ValueError(f"Block column '{block}' not found in data")
        blockids_raw = data[block].values
    else:
        blockids_raw = np.asarray(block)

    # Canonicalize to 1-indexed sequential integers
    unique_blocks = np.unique(blockids_raw)
    block_map = {v: i + 1 for i, v in enumerate(unique_blocks)}
    return np.array([block_map[v] for v in blockids_raw])


def _parse_durations(durations, data, kwargs):
    """Parse durations specification and update kwargs.

    Parameters
    ----------
    durations : str, float, array, or None
        Durations specification
    data : DataFrame or None
        Data containing duration column
    kwargs : dict
        Keyword arguments to update

    Returns
    -------
    dict
        Updated kwargs
    """
    if durations is None or data is None:
        return kwargs

    if isinstance(durations, str):
        if durations not in data.columns:
            raise ValueError(f"Duration column '{durations}' not found in data")
        kwargs['duration_column'] = durations
    elif isinstance(durations, (int, float)):
        # Scalar duration - broadcast later in _create_events_from_data.
        kwargs['_duration_values'] = float(durations)
    else:
        # Per-event duration vector.
        duration_values = np.asarray(durations)
        if len(duration_values) != len(data):
            raise ValueError(
                f"Duration vector length ({len(duration_values)}) must match "
                f"number of rows in data ({len(data)})"
            )
        kwargs['_duration_values'] = duration_values

    return kwargs


def _create_events_from_data(data, terms, sf, kwargs):
    """Create events from data and terms.

    Parameters
    ----------
    data : DataFrame or None
        Event data
    terms : list of Term
        Model terms
    sf : SamplingFrame
        Sampling frame
    kwargs : dict
        Additional arguments

    Returns
    -------
    dict
        Event objects by name
    """
    if data is None:
        raise ValueError("Either events or data must be provided")

    # Separate term types
    covariate_terms, trialwise_terms, regular_terms = _separate_term_types(terms)

    # Extract event specifications from regular terms
    event_specs = _extract_event_specs(regular_terms, data)

    onset_col = kwargs.pop('onset_column', kwargs.pop('onset_col', 'onset'))
    duration_col = kwargs.pop('duration_column', kwargs.pop('duration_col', 'duration'))
    duration_values = kwargs.pop('_duration_values', None)

    # Support scalar/vector duration arguments even when no duration column
    # exists in the input DataFrame.
    if duration_values is not None:
        data = data.copy()
        duration_col = "__duration__"
        if np.isscalar(duration_values):
            data[duration_col] = float(duration_values)
        else:
            data[duration_col] = np.asarray(duration_values, dtype=float)

    # Create regular events
    if event_specs:
        events = events_from_dataframe(
            data, event_specs,
            onset_col=onset_col,
            duration_col=duration_col,
            **kwargs
        )
    else:
        events = {}

    # Add trialwise onset event
    if trialwise_terms and onset_col in data.columns and len(data) > 0:
        events['_onset'] = EventVariable(
            name='_onset',
            onsets=data[onset_col].values,
            values=data[onset_col].values,
            durations=data[duration_col].values if duration_col in data.columns else 0
        )

    # Create covariate events
    for cov_term in covariate_terms:
        cov_events = create_covariate_events(
            data=data,
            covariate_names=cov_term.covariates,
            sampling_info=sf,
            prefix=cov_term.prefix
        )
        events.update(cov_events)

    return events


def _separate_term_types(terms):
    """Separate terms into covariate, trialwise, and regular.

    Parameters
    ----------
    terms : list of Term
        Model terms

    Returns
    -------
    tuple
        (covariate_terms, trialwise_terms, regular_terms)
    """
    covariate_terms = [t for t in terms if isinstance(t, CovariateTerm)]
    trialwise_terms = [t for t in terms if hasattr(t, '_is_trialwise') and t._is_trialwise]
    regular_terms = [
        t for t in terms
        if not isinstance(t, CovariateTerm)
        and not (hasattr(t, '_is_trialwise') and t._is_trialwise)
    ]
    return covariate_terms, trialwise_terms, regular_terms


def _extract_event_specs(regular_terms, data):
    """Extract event specifications from regular terms.

    Parameters
    ----------
    regular_terms : list of Term
        Regular terms
    data : DataFrame
        Event data

    Returns
    -------
    dict
        Event specifications
    """
    event_specs = {}
    for term in regular_terms:
        for event_name in term.events:
            if event_name not in event_specs:
                if event_name in data.columns:
                    col_data = data[event_name]
                    if term.basis is not None and pd.api.types.is_numeric_dtype(col_data):
                        event_specs[event_name] = {
                            'type': 'basis',
                            'basis': term.basis,
                        }
                    elif pd.api.types.is_numeric_dtype(col_data):
                        # Match fmridesign defaults: numeric modulators are
                        # not mean-centered unless explicitly requested.
                        event_specs[event_name] = {'type': 'variable', 'center': False}
                    else:
                        event_specs[event_name] = {'type': 'factor'}
                else:
                    warnings.warn(
                        f"Event '{event_name}' not found in data columns"
                    )
    return event_specs
