"""Main entry point for fMRI GLM fitting.

Provides ``fmri_lm()`` — the primary user-facing function — and the
:class:`FmriLm` result class that holds fitted coefficients, residual
information, and methods for computing contrasts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Union,
    cast,
    runtime_checkable,
)

import numpy as np
from numpy.typing import NDArray

from ..model.config import FmriLmConfig
from .contrasts import (
    ContrastResult,
    contrast_f_vectorized,
    contrast_t,
    contrast_t_batch,
)
from .engine import DEFAULT_ENGINE_OPTIONS, EngineResult, EngineSelector
from .solver import Projection

if TYPE_CHECKING:
    from fmrimod.dataset import FmriDataset
    from fmrimod.dataset.protocols import DatasetProtocol
    from fmrimod.glm.spatial import SpatialContext
    from fmrimod.model.fmri_model import FmriModel
    from fmrimod.spec import Spec, Term


@runtime_checkable
class FmriModelLike(Protocol):
    """Runtime-checkable interface required by GLM fitting engines."""

    @property
    def dataset(self) -> object:
        """Dataset associated with the model."""
        ...

    @property
    def n_runs(self) -> int:
        """Number of runs in the model."""
        ...

    def design_matrix_array(self, run: int = 0) -> NDArray[np.float64]:
        """Return the design matrix for a run."""
        ...


@dataclass
class FmriLm:
    """Result of fitting a GLM to fMRI data.

    Holds beta coefficients, residual information, and pre-computed
    quantities needed for efficient contrast computation.

    Attributes
    ----------
    betas : NDArray
        Coefficient matrix, shape ``(p, V)`` where ``p`` is the number
        of design columns and ``V`` the number of voxels.
    sigma : NDArray
        Residual standard deviation per voxel, shape ``(V,)``.
    residual_df : float
        Residual degrees of freedom.
    XtXinv : NDArray
        ``(X'X)^{-1}`` matrix, shape ``(p, p)``.
    model : object
        The :class:`~fmrimod.model.FmriModel` that was fitted.
    config : FmriLmConfig
        Configuration used for fitting.
    contrasts : dict
        Dictionary of computed :class:`ContrastResult` objects.
    ar_params : NDArray or None
        Estimated AR parameters, shape ``(ar_order, V)`` or ``(ar_order,)``
        for global estimation.
    robust_weights : NDArray or None
        IRLS weights from robust fitting, shape ``(n, V)``.
    run_results : list
        Per-run fitting results (for diagnostics).
    projections : list
        Per-run :class:`Projection` objects.
    """

    betas: NDArray[np.float64]
    sigma: NDArray[np.float64]
    residual_df: float
    XtXinv: NDArray[np.float64]
    model: FmriModelLike
    config: FmriLmConfig
    contrasts: Dict[str, ContrastResult] = field(default_factory=dict)
    ar_params: Optional[NDArray[np.float64]] = None
    robust_weights: Optional[NDArray[np.float64]] = None
    run_results: Optional[List[Any]] = None
    projections: Optional[List[Projection]] = None
    _named_weights_cache: Optional[Dict[str, NDArray[np.float64]]] = field(
        default=None,
        init=False,
        repr=False,
    )

    # -- Accessors --

    def coef(self) -> NDArray[np.float64]:
        """Return the coefficient matrix ``(p, V)``."""
        return self.betas

    def se(self) -> NDArray[np.float64]:
        """Return standard errors for each coefficient, shape ``(p, V)``.

        ``SE_{j,v} = sigma_v * sqrt(XtXinv_{j,j})``
        """
        diag_XtXinv = np.diag(self.XtXinv)
        return np.asarray(
            self.sigma[np.newaxis, :] * np.sqrt(
                np.maximum(diag_XtXinv, 0.0)
            )[:, np.newaxis],
            dtype=np.float64,
        )

    def tstat(self) -> NDArray[np.float64]:
        """Return t-statistics for each coefficient, shape ``(p, V)``."""
        se_vals = self.se()
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(se_vals > 1e-15, self.betas / se_vals, 0.0)

    @property
    def n_voxels(self) -> int:
        """Number of voxels."""
        return self.betas.shape[1]

    @property
    def n_coefficients(self) -> int:
        """Number of regression coefficients."""
        return self.betas.shape[0]

    def design_columns(self):
        """Return typed provenance for the fitted design columns."""
        design_columns = getattr(self.model, "design_columns", None)
        if not callable(design_columns):
            raise TypeError("fitted model does not expose design_columns()")
        return design_columns()

    # -- Contrast computation --

    def contrast(
        self,
        spec: Union[NDArray[np.float64], str, dict[str, Any], "OmnibusContrast"],
        name: Optional[str] = None,
    ) -> ContrastResult:
        """Compute a contrast on the fitted model.

        Parameters
        ----------
        spec : NDArray, str, dict, or OmnibusContrast
            Contrast specification. Can be:
            - A 1-D vector for a t-contrast
            - A 2-D matrix for an F-contrast
            - A string name referring to a pre-defined contrast in
              the event model
            - A dict ``{"weights": array, "name": str}``
            - An :class:`~fmrimod.contrast.OmnibusContrast` typed intent
              value, resolved against the fit's :class:`DesignColumns`
        name : str, optional
            Override contrast name.

        Returns
        -------
        ContrastResult
        """
        from fmrimod.contrast.omnibus import OmnibusContrast

        if isinstance(spec, OmnibusContrast):
            weights = spec.resolve(self.design_columns())
            return self._compute_contrast(
                weights, name=name or spec.display_name
            )

        if isinstance(spec, str):
            if name is None and spec in self.contrasts:
                return self.contrasts[spec]
            # Look up from model's contrast weights
            if self._named_weights_cache is None:
                contrast_weights = getattr(self.model, "contrast_weights", None)
                self._named_weights_cache = (
                    cast(Dict[str, NDArray[np.float64]], contrast_weights())
                    if callable(contrast_weights)
                    else {}
                )
            cw = self._named_weights_cache
            if spec not in cw:
                raise KeyError(f"Unknown contrast name: {spec!r}")
            return self._compute_contrast(cw[spec], name=name or spec)

        if isinstance(spec, dict):
            if "weights" not in spec:
                raise ValueError("Contrast dict spec must contain 'weights'")
            weights = np.asarray(spec["weights"], dtype=np.float64)
            cname = name or spec.get("name", "contrast")
            return self._compute_contrast(weights, name=cname)

        weights = np.asarray(spec, dtype=np.float64)
        return self._compute_contrast(weights, name=name or "contrast")

    def _compute_contrast(
        self,
        weights: NDArray[np.float64],
        name: str,
    ) -> ContrastResult:
        """Dispatch to t or F contrast based on weight dimensions."""
        weights = np.atleast_1d(weights)
        if weights.ndim == 1:
            result = contrast_t(
                weights, self.betas, self.XtXinv, self.sigma,
                self.residual_df, name=name,
            )
        else:
            result = contrast_f_vectorized(
                weights, self.betas, self.XtXinv, self.sigma,
                self.residual_df, name=name,
            )
        result.spatial = self._spatial_context()
        self.contrasts[name] = result
        return result

    def _spatial_context(self) -> SpatialContext | None:
        """Return the spatial context for this fit, if its model exposes one.

        Cached so repeated contrast calls don't re-walk the adapter.
        """
        from fmrimod.glm.spatial import SpatialContext

        if not hasattr(self, "_spatial_context_cache"):
            self._spatial_context_cache = SpatialContext.from_model(self.model)
        return self._spatial_context_cache

    def compute_contrasts(
        self,
        specs: Dict[str, NDArray[np.float64]],
    ) -> Dict[str, ContrastResult]:
        """Compute multiple contrasts, batching t-contrasts for speed.

        Parameters
        ----------
        specs : dict[str, NDArray]
            Mapping of contrast name to weight vector/matrix.

        Returns
        -------
        dict[str, ContrastResult]
            Computed contrast results keyed by name.
        """
        if not specs:
            return {}

        out: Dict[str, ContrastResult] = {}
        t_names: list[str] = []
        t_weights: list[NDArray[np.float64]] = []
        f_items: list[tuple[str, NDArray[np.float64]]] = []

        for cname, w in specs.items():
            if cname in self.contrasts:
                out[cname] = self.contrasts[cname]
                continue
            w_arr = np.asarray(w, dtype=np.float64)
            if w_arr.ndim == 1:
                t_names.append(cname)
                t_weights.append(w_arr)
            else:
                f_items.append((cname, np.atleast_2d(w_arr)))

        ctx = self._spatial_context()
        if t_weights:
            t_mat = np.vstack(t_weights)
            t_results = contrast_t_batch(
                t_mat,
                self.betas,
                self.XtXinv,
                self.sigma,
                self.residual_df,
                names=t_names,
            )
            for res in t_results:
                res.spatial = ctx
                self.contrasts[res.name] = res
                out[res.name] = res

        for cname, w_arr in f_items:
            res = contrast_f_vectorized(
                w_arr,
                self.betas,
                self.XtXinv,
                self.sigma,
                self.residual_df,
                name=cname,
            )
            res.spatial = ctx
            self.contrasts[cname] = res
            out[cname] = res

        return out

    # -- Display --

    def __repr__(self) -> str:
        parts = [
            "FmriLm(",
            f"  n_coefficients={self.n_coefficients},",
            f"  n_voxels={self.n_voxels},",
            f"  residual_df={self.residual_df:.1f},",
            f"  config={self.config!r},",
            f"  contrasts={list(self.contrasts.keys())},",
            ")",
        ]
        return "\n".join(parts)


def fmri_lm(
    spec_or_model: "Spec | Term | str | Sequence[object] | FmriModelLike",
    dataset_or_config: "FmriDataset | FmriLmConfig | None" = None,
    *,
    baseline: object = None,
    block: Optional[Union[str, NDArray[np.float64]]] = None,
    durations: Optional[Union[str, float, NDArray[np.float64]]] = None,
    precision: Optional[float] = None,
    config: Optional[FmriLmConfig] = None,
    engine: EngineSelector = DEFAULT_ENGINE_OPTIONS,
    **engine_kwargs: object,
) -> FmriLm:
    """Fit a GLM to fMRI data.

    Canonical form (matches R ``fmrireg::fmri_lm``)::

        fit = fmri_lm(spec, dataset)
        fit = fmri_lm(spec, dataset, config=FmriLmConfig(ar="ar1"))

    Legacy form (pre-built model)::

        fit = fmri_lm(model)
        fit = fmri_lm(model, FmriLmConfig())

    Parameters
    ----------
    spec_or_model
        Either a *spec* (R-style formula string, list of
        :class:`~fmrimod.formula.base.Term`, or
        :class:`~fmrimod.formula.base.EventModelBuilder`) or a pre-built
        :class:`~fmrimod.model.FmriModel`-like object. The second positional
        argument is interpreted accordingly.
    dataset_or_config
        When ``spec_or_model`` is a spec: the :class:`FmriDataset` to fit
        (must carry an event table). When ``spec_or_model`` is a model: an
        :class:`FmriLmConfig` (back-compat positional). May be omitted in
        either case.
    baseline
        Optional pre-built :class:`~fmrimod.baseline.BaselineModel`. If not
        provided in the spec path, defaults to ``constant`` basis with
        runwise intercepts.
    block
        Block/run column name or array passed to
        :func:`~fmrimod.event_model`. Auto-detected from ``run`` or ``block``
        columns when omitted.
    durations
        Event durations passed to :func:`~fmrimod.event_model`. Auto-detected
        from a ``duration`` column when present.
    precision
        Temporal precision for HRF convolution.
    config
        Fitting configuration. Defaults to plain OLS.
    engine
        Typed engine options object. Built-ins are
        :class:`~fmrimod.glm.engine.RunwiseEngineOptions`,
        :class:`~fmrimod.glm.engine.ChunkwiseEngineOptions`, and
        :class:`~fmrimod.glm.engine.SketchEngineOptions`. Legacy string engine
        names remain accepted for compatibility.
    **engine_kwargs
        Legacy keyword bag forwarded to the engine's ``fit()`` method when
        ``engine`` is a string. Do not mix with typed engine options.

    Returns
    -------
    FmriLm
        Fitted model with coefficients, residuals, and contrast methods.

    Examples
    --------
    Canonical three-line shape::

        >>> ds = fm.fmri_dataset(img, mask=mask, tr=2.0, events=events)
        >>> fit = fm.fmri_lm("hrf(trial_type)", ds)
        >>> fit.contrast("trial_type[listening]")

    Sketch engine::

        >>> from fmrimod.glm import SketchEngineOptions
        >>> fit = fm.fmri_lm(model, engine=SketchEngineOptions(sketch_ratio=0.5))
    """
    from .engine import resolve_engine

    # -- Resolve second positional: dataset vs. config -----------------------
    if isinstance(dataset_or_config, FmriLmConfig):
        if config is not None and config is not dataset_or_config:
            raise ValueError(
                "fmri_lm: `config` was passed both positionally and as a kwarg"
            )
        config = dataset_or_config
        dataset = None
    else:
        dataset = dataset_or_config

    # -- Resolve first positional: spec vs. model ---------------------------
    if _is_fmri_model_like(spec_or_model):
        if dataset is not None:
            raise ValueError(
                "fmri_lm: got a pre-built model and a dataset. "
                "Pass either (spec, dataset, ...) or (model, ...), not both."
            )
        model = cast(FmriModelLike, spec_or_model)
    else:
        if dataset is None:
            raise ValueError(
                "fmri_lm(spec, dataset) requires an FmriDataset as the "
                "second argument when the first argument is a spec."
            )
        model = _build_model_from_spec(
            spec=cast("Spec | Term | str | Sequence[object]", spec_or_model),
            dataset=dataset,
            baseline=baseline,
            block=block,
            durations=durations,
            precision=precision,
        )

    if config is None:
        config = FmriLmConfig()

    # Resolve and run the engine
    eng, fit_kwargs = resolve_engine(engine, engine_kwargs)
    eng.preflight(model, config)
    fit_result: EngineResult = eng.fit(model, config, **fit_kwargs)

    # Handle AR modeling if configured
    ar_params = fit_result.ar_params
    if ar_params is None and config.ar.enabled:
        from ..ar.integration import iterative_gls

        # Build a dict compatible with the legacy interface
        legacy = _engine_result_to_dict(fit_result)
        legacy, ar_params = iterative_gls(model, config, legacy)
        fit_result = _dict_to_engine_result(legacy, fit_result)

    # Handle robust fitting if configured
    robust_weights = fit_result.robust_weights
    if robust_weights is None and config.robust.enabled:
        from ..robust.irls import robust_refit

        legacy = _engine_result_to_dict(fit_result)
        legacy, robust_weights = robust_refit(model, config, legacy)
        fit_result = _dict_to_engine_result(legacy, fit_result)

    return FmriLm(
        betas=fit_result.betas,
        sigma=fit_result.sigma,
        residual_df=fit_result.dfres,
        XtXinv=fit_result.XtXinv,
        model=model,
        config=config,
        ar_params=ar_params,
        robust_weights=robust_weights,
        run_results=fit_result.run_results,
        projections=fit_result.projections,
    )


def _is_fmri_model_like(obj: object) -> bool:
    """Check for the public :class:`FmriModelLike` fitting protocol."""
    return isinstance(obj, FmriModelLike)


def _build_model_from_spec(
    *,
    spec: "Spec | Term | str | Sequence[object]",
    dataset: "FmriDataset",
    baseline: object,
    block: object,
    durations: object,
    precision: Optional[float],
) -> "FmriModel":
    """Build an :class:`FmriModel` from a spec + :class:`FmriDataset`.

    Accepts either a typed :class:`fmrimod.spec.Spec` / :class:`Term` tree or
    a legacy string formula / list-of-Terms. The typed path lowers via
    :func:`fmrimod.spec.compile`. Supported legacy HRF formulas are first
    adapted into the typed spec tree; unsupported legacy forms fall back to
    :func:`fmrimod.event_model` directly.
    """
    from ..baseline.baseline_model import baseline_model as _build_baseline
    from ..design.event_model import event_model as _build_event
    from ..model.fmri_model import FmriModel
    from ..spec import Spec, Term, legacy_formula_to_spec
    from ..spec import compile as _compile_spec

    events_df = getattr(dataset, "event_table", None)
    if events_df is None:
        raise ValueError(
            "fmri_lm(spec, dataset) requires the dataset to carry an event "
            "table; pass `events=` to fmri_dataset(...) or supply a pre-built "
            "FmriModel instead."
        )

    sf = dataset.get_sampling_frame()

    # Auto-detect block column.
    resolved_block = block
    if resolved_block is None:
        if "run" in events_df.columns:
            resolved_block = "run"
        elif "block" in events_df.columns:
            resolved_block = "block"
        else:
            resolved_block = np.ones(len(events_df), dtype=int)

    # Auto-detect durations column.
    resolved_durations = durations
    if resolved_durations is None:
        if "duration" in events_df.columns:
            resolved_durations = "duration"
        else:
            resolved_durations = 0

    # -- Typed Spec / Term path -----------------------------------------
    if isinstance(spec, (Spec, Term)):
        em, default_bm = _compile_spec(
            spec,
            data=events_df,
            sampling_frame=sf,
            block=resolved_block,
            durations=resolved_durations,
            precision=precision,
        )
        bm = baseline if baseline is not None else default_bm
        return FmriModel(em, bm, cast("DatasetProtocol", dataset))

    # -- Convertible legacy HRF formula/list path -----------------------
    try:
        typed_spec = legacy_formula_to_spec(cast(Any, spec))
    except (TypeError, ValueError, NotImplementedError):
        typed_spec = None

    if typed_spec is not None:
        em, default_bm = _compile_spec(
            typed_spec,
            data=events_df,
            sampling_frame=sf,
            block=resolved_block,
            durations=resolved_durations,
            precision=precision,
        )
        bm = baseline if baseline is not None else default_bm
        return FmriModel(em, bm, cast("DatasetProtocol", dataset))

    # -- Legacy string / list path --------------------------------------
    em_kwargs: Dict[str, Any] = dict(
        data=events_df,
        block=resolved_block,
        sampling_frame=sf,
        durations=resolved_durations,
    )
    if precision is not None:
        em_kwargs["precision"] = precision

    em = _build_event(cast(Any, spec), **em_kwargs)

    bm = baseline
    if bm is None:
        bm = _build_baseline(basis="constant", sframe=sf, intercept="runwise")

    return FmriModel(em, bm, cast("DatasetProtocol", dataset))


def _engine_result_to_dict(er: "EngineResult") -> Dict[str, Any]:
    """Convert an EngineResult to the legacy dict format."""
    return {
        "betas": er.betas,
        "sigma": er.sigma,
        "dfres": er.dfres,
        "XtXinv": er.XtXinv,
        "projections": er.projections,
        "run_results": er.run_results,
        "residuals": er.residuals,
        "run_X": er.run_X,
    }


def _dict_to_engine_result(d: Dict[str, Any], original: "EngineResult") -> "EngineResult":
    """Update an EngineResult from a legacy dict (after AR/robust)."""
    from .engine import EngineResult

    return EngineResult(
        betas=d["betas"],
        sigma=d["sigma"],
        dfres=d["dfres"],
        XtXinv=d["XtXinv"],
        projections=d.get("projections"),
        run_results=d.get("run_results"),
        residuals=d.get("residuals"),
        run_X=d.get("run_X"),
        ar_params=getattr(original, "ar_params", None),
        robust_weights=getattr(original, "robust_weights", None),
        extra=getattr(original, "extra", {}),
    )
