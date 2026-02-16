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
['runwise', 'sketch']
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import (
    Any,
    ClassVar,
    Dict,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

import numpy as np
from numpy.typing import NDArray

from ..model.config import FmriLmConfig

logger = logging.getLogger(__name__)

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
        model: Any,
        config: FmriLmConfig,
        **kwargs: Any,
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
        model: Any,
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
    projections: Optional[list] = None
    run_results: Optional[list] = None
    residuals: Optional[list] = None
    run_X: Optional[list] = None
    ar_params: Optional[NDArray[np.float64]] = None
    robust_weights: Optional[NDArray[np.float64]] = None
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_ENGINES: Dict[str, type] = {}
_BUILTINS_LOADED = False
_ENTRY_POINTS_LOADED = False


def register_engine(name_or_cls: Any = None, *, name: Optional[str] = None):
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
        _ENGINES[engine_name] = cls
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
            _ENGINES[engine_name] = cls
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
                    _ENGINES[engine_name] = cls
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


def get_engine(name: str) -> FittingEngine:
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

    cls = _ENGINES.get(name)
    if cls is None:
        available = ", ".join(sorted(_ENGINES)) or "(none)"
        raise KeyError(
            f"Unknown engine {name!r}. Available: {available}"
        )
    return cls()


def list_engines() -> list[str]:
    """Return names of all registered engines.

    Triggers entry-point discovery on first call.
    """
    _load_builtin_engines()
    _load_entry_points()
    return sorted(_ENGINES)
