"""Pluggable GLM fitting engine registry.

Engines are the core fitting strategies for fMRI GLMs. Built-in engines
(``"runwise"``, ``"sketch"``) are registered lazily on first lookup.
Third-party
packages can register additional engines via Python entry points::

    # In third-party pyproject.toml:
    [project.entry-points."fmrimod.engines"]
    my_engine = "mypkg.engine:MyEngine"

All engines implement the :class:`FittingEngine` protocol.

Examples
--------
>>> from fmrimod.glm.engine import get_engine, list_engines
>>> engine = get_engine("runwise")
>>> result = engine.fit(model, config)

>>> list_engines()
['chunkwise', 'runwise', 'sketch']
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Dict,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Union,
    cast,
    runtime_checkable,
)

import numpy as np
from numpy.typing import DTypeLike, NDArray

from ..model.config import FmriLmConfig

if TYPE_CHECKING:
    from .fmri_lm import FmriModelLike

logger = logging.getLogger(__name__)

EngineName = Literal["runwise", "chunkwise", "sketch"]
SketchKindName = Literal["gaussian", "srht", "countsketch"]
LandmarkMethodName = Literal["kmeans", "random"]

_BUILTIN_ENGINE_NAMES: frozenset[str] = frozenset(("runwise", "chunkwise", "sketch"))


def _normalize_engine_name(name: str) -> EngineName:
    """Validate a built-in engine name and return the typed literal."""
    if name in _BUILTIN_ENGINE_NAMES:
        return cast(EngineName, name)
    raise KeyError(
        f"Unknown built-in engine {name!r}. Built-ins: "
        f"{', '.join(sorted(_BUILTIN_ENGINE_NAMES))}"
    )


def _engine_registry_key(name: EngineName | str) -> str:
    if name in _BUILTIN_ENGINE_NAMES:
        return _normalize_engine_name(name)
    return name


def _validate_positive_int(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _validate_optional_positive_int(value: Optional[int], *, name: str) -> None:
    if value is not None:
        _validate_positive_int(value, name=name)


@dataclass(frozen=True)
class RunwiseEngineOptions:
    """Typed options for the default runwise GLM engine."""

    n_jobs: int = 1
    blas_threads: Optional[int] = None
    compute_dtype: DTypeLike = np.float64
    cache_projections: bool = False
    chunk_size: int = 5000
    name: Literal["runwise"] = field(default="runwise", init=False)

    def __post_init__(self) -> None:
        _validate_positive_int(self.n_jobs, name="n_jobs")
        _validate_optional_positive_int(self.blas_threads, name="blas_threads")
        _validate_positive_int(self.chunk_size, name="chunk_size")

    def fit_kwargs(self) -> Dict[str, object]:
        return {
            "n_jobs": self.n_jobs,
            "blas_threads": self.blas_threads,
            "compute_dtype": self.compute_dtype,
            "cache_projections": self.cache_projections,
            "chunk_size": self.chunk_size,
        }


@dataclass(frozen=True)
class ChunkwiseEngineOptions:
    """Typed options for the chunkwise GLM engine."""

    chunk_size: int = 5000
    n_jobs: int = 1
    blas_threads: Optional[int] = None
    compute_dtype: DTypeLike = np.float64
    cache_projections: bool = False
    name: Literal["chunkwise"] = field(default="chunkwise", init=False)

    def __post_init__(self) -> None:
        _validate_positive_int(self.chunk_size, name="chunk_size")
        _validate_positive_int(self.n_jobs, name="n_jobs")
        _validate_optional_positive_int(self.blas_threads, name="blas_threads")

    def fit_kwargs(self) -> Dict[str, object]:
        return {
            "chunk_size": self.chunk_size,
            "n_jobs": self.n_jobs,
            "blas_threads": self.blas_threads,
            "compute_dtype": self.compute_dtype,
            "cache_projections": self.cache_projections,
        }


@dataclass(frozen=True)
class SketchEngineOptions:
    """Typed options for the sketch/low-rank GLM engine."""

    sketch_kind: SketchKindName = "gaussian"
    sketch_ratio: float = 0.5
    use_landmarks: bool = False
    n_landmarks: int = 500
    landmark_k: int = 6
    landmark_method: LandmarkMethodName = "kmeans"
    ridge: float = 0.0
    seed: Optional[int] = None
    coords: Optional[NDArray[np.float64]] = None
    name: Literal["sketch"] = field(default="sketch", init=False)

    def __post_init__(self) -> None:
        if self.sketch_kind not in ("gaussian", "srht", "countsketch"):
            raise ValueError("sketch_kind must be 'gaussian', 'srht', or 'countsketch'")
        if not (0.0 < float(self.sketch_ratio) <= 1.0):
            raise ValueError("sketch_ratio must be in (0, 1]")
        _validate_positive_int(self.n_landmarks, name="n_landmarks")
        _validate_positive_int(self.landmark_k, name="landmark_k")
        if self.landmark_method not in ("kmeans", "random"):
            raise ValueError("landmark_method must be 'kmeans' or 'random'")
        if self.ridge < 0:
            raise ValueError("ridge must be >= 0")
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise ValueError("seed must be an integer or None")
        if self.coords is not None:
            coords = np.asarray(self.coords, dtype=np.float64)
            if coords.ndim != 2:
                raise ValueError("coords must be a 2-D array")
            if not np.all(np.isfinite(coords)):
                raise ValueError("coords must contain finite values")

    def fit_kwargs(self) -> Dict[str, object]:
        kwargs: Dict[str, object] = {
            "sketch_kind": self.sketch_kind,
            "sketch_ratio": self.sketch_ratio,
            "use_landmarks": self.use_landmarks,
            "n_landmarks": self.n_landmarks,
            "landmark_k": self.landmark_k,
            "landmark_method": self.landmark_method,
            "ridge": self.ridge,
            "seed": self.seed,
        }
        if self.coords is not None:
            kwargs["coords"] = self.coords
        return kwargs


EngineOptions = Union[
    RunwiseEngineOptions,
    ChunkwiseEngineOptions,
    SketchEngineOptions,
]
EngineSelector = Union[EngineName, str, EngineOptions]
DEFAULT_ENGINE_OPTIONS = RunwiseEngineOptions()

# ---------------------------------------------------------------------------
# Engine protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class FittingEngine(Protocol):
    """Protocol that every GLM fitting engine must satisfy.

    Engines encapsulate a fitting strategy (e.g. run-wise OLS, sketched
    low-rank, chunkwise iteration) and are discovered by name via
    :func:`get_engine`.

    Implementations may be stateless classes or carry configuration.
    """

    name: ClassVar[str]
    """Short identifier used in :func:`get_engine` and ``fmri_lm(engine=...)``. """

    def fit(
        self,
        model: "FmriModelLike",
        config: FmriLmConfig,
        **kwargs: object,
    ) -> "EngineResult":
        """Run the fitting strategy and return standardised results.

        Parameters
        ----------
        model : FmriModel
            The model specification (design + data).
        config : FmriLmConfig
            Fitting options (AR, robust, volume-weights, …).
        **kwargs
            Engine-specific options.

        Returns
        -------
        EngineResult
        """
        ...

    def preflight(
        self,
        model: "FmriModelLike",
        config: FmriLmConfig,
    ) -> None:
        """Optional validation before fitting.

        The default implementation is a no-op.  Override to raise early
        errors (e.g. missing data, incompatible config).
        """
        ...


# ---------------------------------------------------------------------------
# Standardised result
# ---------------------------------------------------------------------------


@dataclass
class EngineResult:
    """Standardised output from any :class:`FittingEngine`.

    This is the lingua-franca between engines and :func:`fmri_lm`.
    The top-level ``fmri_lm`` function wraps this into the richer
    :class:`~fmrimod.glm.fmri_lm.FmriLm` object.

    Attributes
    ----------
    betas : NDArray, shape ``(p, V)``
        Coefficient matrix.
    sigma : NDArray, shape ``(V,)``
        Residual standard deviation per voxel.
    dfres : float
        Residual degrees of freedom.
    XtXinv : NDArray, shape ``(p, p)``
        Inverse cross-product for contrast computation.
    projections : list, optional
        Per-run :class:`Projection` objects.
    run_results : list, optional
        Per-run regression results.
    residuals : list, optional
        Per-run residual matrices.
    run_X : list, optional
        Per-run design matrices (post-preprocessing).
    ar_params : NDArray, optional
        Estimated AR parameters.
    robust_weights : NDArray, optional
        IRLS weights from robust fitting.
    extra : dict
        Engine-specific extras (e.g. sketch info, landmarks).
    """

    betas: NDArray[np.float64]
    sigma: NDArray[np.float64]
    dfres: float
    XtXinv: NDArray[np.float64]
    projections: Optional[list[object]] = None
    run_results: Optional[list[object]] = None
    residuals: Optional[list[NDArray[np.float64]]] = None
    run_X: Optional[list[NDArray[np.float64]]] = None
    ar_params: Optional[NDArray[np.float64]] = None
    robust_weights: Optional[NDArray[np.float64]] = None
    extra: Dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_ENGINES: Dict[str, type[FittingEngine]] = {}
_BUILTINS_LOADED = False
_ENTRY_POINTS_LOADED = False


def register_engine(
    name_or_cls: Optional[type] = None,
    *,
    name: Optional[str] = None,
) -> Union[type, Callable[[type], type]]:
    """Register a fitting engine class.

    Can be used as a decorator (with or without arguments) or called
    directly.

    Examples
    --------
    As a decorator with an explicit name::

        @register_engine(name="my_engine")
        class MyEngine:
            name = "my_engine"
            def fit(self, model, config, **kw): ...
            def preflight(self, model, config): ...

    As a decorator inferring the name from ``cls.name``::

        @register_engine
        class MyEngine:
            name = "my_engine"
            ...

    Direct registration::

        register_engine(MyEngine, name="my_engine")
    """
    # Called as @register_engine (no parens)
    if isinstance(name_or_cls, type):
        cls = name_or_cls
        engine_name = name or getattr(cls, "name", None)
        if engine_name is None:
            raise ValueError(
                f"Engine class {cls.__name__} has no 'name' attribute "
                "and no name= was given"
            )
        _ENGINES[engine_name] = cast(type[FittingEngine], cls)
        return cls

    # Called as @register_engine(name="...") or register_engine(cls, name="...")
    if name_or_cls is None:
        # Decorator with keyword: @register_engine(name="foo")
        def decorator(cls: type) -> type:
            engine_name = name or getattr(cls, "name", None)
            if engine_name is None:
                raise ValueError(
                    f"Engine class {cls.__name__} has no 'name' attribute "
                    "and no name= was given"
                )
            _ENGINES[engine_name] = cast(type[FittingEngine], cls)
            return cls
        return decorator

    # Direct call: register_engine("name", SomeCls)  — not supported
    raise TypeError(
        "register_engine() expects a class or name= keyword. "
        f"Got {type(name_or_cls)!r}"
    )


def _load_entry_points() -> None:
    """Discover engines from ``fmrimod.engines`` entry points (lazy)."""
    global _ENTRY_POINTS_LOADED
    if _ENTRY_POINTS_LOADED:
        return
    _ENTRY_POINTS_LOADED = True

    try:
        # Python 3.9 compat
        from importlib.metadata import entry_points as _ep

        eps = _ep()
        # Python 3.12+ returns a SelectableGroups; 3.9 returns a dict
        if isinstance(eps, dict):
            engine_eps = eps.get("fmrimod.engines", [])
        else:
            engine_eps = eps.select(group="fmrimod.engines")

        for ep in engine_eps:
            if ep.name not in _ENGINES:
                try:
                    cls = ep.load()
                    engine_name = getattr(cls, "name", ep.name)
                    _ENGINES[engine_name] = cast(type[FittingEngine], cls)
                    logger.debug("Loaded engine %r from entry point %s", engine_name, ep)
                except Exception:
                    logger.warning(
                        "Failed to load engine entry point %r", ep.name,
                        exc_info=True,
                    )
    except Exception:
        logger.debug("Entry point discovery unavailable", exc_info=True)


def _load_builtin_engines() -> None:
    """Register built-in engines once, lazily.

    Lazy loading avoids circular import paths for modules that only need
    low-level GLM components (e.g. ``fmrimod.glm.solver``).
    """
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return

    from . import engines as _builtin_engines  # noqa: F401

    _BUILTINS_LOADED = True


def get_engine(name: EngineName | str) -> FittingEngine:
    """Look up an engine by name and return an instance.

    Built-in engines are always available.  Third-party engines are
    discovered via ``fmrimod.engines`` entry points on first call.

    Parameters
    ----------
    name : str
        Engine identifier (e.g. ``"runwise"``, ``"sketch"``).

    Returns
    -------
    FittingEngine
        An instantiated engine.

    Raises
    ------
    KeyError
        If no engine with *name* is registered.
    """
    _load_builtin_engines()
    _load_entry_points()
    registry_key = _engine_registry_key(name)

    cls = _ENGINES.get(registry_key)
    if cls is None:
        available = ", ".join(sorted(_ENGINES)) or "(none)"
        raise KeyError(f"Unknown engine {registry_key!r}. Available: {available}")
    return cls()


def resolve_engine(
    engine: EngineSelector = DEFAULT_ENGINE_OPTIONS,
    legacy_kwargs: Mapping[str, object] | None = None,
) -> tuple[FittingEngine, Dict[str, object]]:
    """Resolve a public engine selector into an engine and typed fit kwargs.

    ``str`` selectors preserve the legacy extension/entry-point surface.
    ``*EngineOptions`` selectors are the typed public path and reject extra
    keyword bags so validation happens at option construction time.
    """
    legacy = dict(legacy_kwargs or {})
    if isinstance(engine, str):
        return get_engine(_engine_registry_key(engine)), legacy

    if legacy:
        raise ValueError(
            "fmri_lm: pass either a typed engine options object or legacy "
            "engine keyword arguments, not both"
        )
    return get_engine(_normalize_engine_name(engine.name)), engine.fit_kwargs()


def list_engines() -> list[str]:
    """Return names of all registered engines.

    Triggers entry-point discovery on first call.
    """
    _load_builtin_engines()
    _load_entry_points()
    return sorted(_ENGINES)
