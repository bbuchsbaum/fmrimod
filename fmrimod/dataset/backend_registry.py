"""Backend registry for canonical dataset storage backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from .backend_protocol import StorageBackend
from .errors import ConfigError


@dataclass
class _Registration:
    name: str
    factory: Callable[..., StorageBackend]
    description: str
    validate_function: Callable[[StorageBackend], bool] | None
    registered_at: datetime = field(default_factory=datetime.now)


class BackendRegistry:
    """Singleton registry for storage backend factories."""

    _instance: BackendRegistry | None = None

    def __init__(self) -> None:
        self._entries: dict[str, _Registration] = {}

    @classmethod
    def instance(cls) -> BackendRegistry:
        """Return the global registry."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(
        self,
        name: str,
        factory: Callable[..., StorageBackend],
        *,
        description: str | None = None,
        validate_function: Callable[[StorageBackend], bool] | None = None,
        overwrite: bool = False,
    ) -> None:
        """Register a backend factory."""
        if not isinstance(name, str) or not name:
            raise ConfigError("name must be a non-empty string", parameter="name")
        if not callable(factory):
            raise ConfigError("factory must be callable", parameter="factory")
        if validate_function is not None and not callable(validate_function):
            raise ConfigError(
                "validate_function must be callable",
                parameter="validate_function",
            )
        if name in self._entries and not overwrite:
            raise ConfigError(
                f"Backend '{name}' is already registered. "
                "Use overwrite=True to replace.",
                parameter="name",
            )

        self._entries[name] = _Registration(
            name=name,
            factory=factory,
            description=description or f"Backend: {name}",
            validate_function=validate_function,
        )

    def create(
        self,
        name: str,
        *,
        validate: bool = True,
        **kwargs: Any,
    ) -> StorageBackend:
        """Create a backend instance by registered name."""
        if not isinstance(name, str) or not name:
            raise ConfigError("name must be a non-empty string", parameter="name")
        if name not in self._entries:
            raise ConfigError(
                f"Backend '{name}' is not registered. "
                f"Available: {', '.join(self.list_names())}",
                parameter="name",
            )

        reg = self._entries[name]
        try:
            backend = reg.factory(**kwargs)
        except Exception as exc:
            raise ConfigError(f"Failed to create backend '{name}': {exc}") from exc

        if validate:
            backend.validate()
            if reg.validate_function is not None and not reg.validate_function(backend):
                raise ConfigError(f"Backend '{name}' failed custom validation")
        return backend

    def list_names(self) -> list[str]:
        """Return sorted registered backend names."""
        return sorted(self._entries)

    def is_registered(self, name: str) -> bool:
        """Return whether a backend name is registered."""
        if not isinstance(name, str):
            raise ConfigError("name must be a string", parameter="name")
        return name in self._entries

    def unregister(self, name: str) -> bool:
        """Remove a backend registration."""
        if not isinstance(name, str):
            raise ConfigError("name must be a string", parameter="name")
        return self._entries.pop(name, None) is not None

    def get_info(self, name: str) -> dict[str, Any]:
        """Return registration metadata for a backend."""
        if not isinstance(name, str) or not name:
            raise ConfigError("name must be a non-empty string", parameter="name")
        if name not in self._entries:
            raise ConfigError(f"Backend '{name}' is not registered")
        reg = self._entries[name]
        return {
            "name": reg.name,
            "description": reg.description,
            "has_validate": reg.validate_function is not None,
            "registered_at": reg.registered_at,
        }


def _register_builtins() -> None:
    """Register built-in backends that are implemented in this package."""
    from .backends.latent_backend import LatentBackend
    from .backends.matrix_backend import MatrixBackend

    def _latent_factory(
        source: str,
        preload: bool = False,
    ) -> LatentBackend:
        return LatentBackend(source=source, preload=preload)

    registry = BackendRegistry.instance()
    registry.register(
        "latent",
        _latent_factory,
        description="HDF5 latent decomposition backend",
        overwrite=True,
    )
    registry.register(
        "matrix",
        lambda **kwargs: MatrixBackend(**kwargs),
        description="In-memory matrix backend",
        overwrite=True,
    )


def get_backend_registry(name: str | None = None) -> BackendRegistry | dict[str, Any]:
    """Return the global registry or metadata for one backend."""
    registry = BackendRegistry.instance()
    if name is None:
        return registry
    return registry.get_info(name)


def register_backend(
    name: str,
    factory: Callable[..., StorageBackend],
    *,
    description: str | None = None,
    validate_function: Callable[[StorageBackend], bool] | None = None,
    overwrite: bool = False,
) -> None:
    """Register a backend factory in the global registry."""
    BackendRegistry.instance().register(
        name,
        factory,
        description=description,
        validate_function=validate_function,
        overwrite=overwrite,
    )


def create_backend(
    name: str,
    *,
    validate: bool = True,
    **kwargs: Any,
) -> StorageBackend:
    """Create a backend from the global registry."""
    return BackendRegistry.instance().create(name, validate=validate, **kwargs)


def is_backend_registered(name: str) -> bool:
    """Return whether a backend name is registered."""
    return BackendRegistry.instance().is_registered(name)


def list_backend_names() -> list[str]:
    """Return registered backend names."""
    return BackendRegistry.instance().list_names()


def unregister_backend(name: str) -> bool:
    """Unregister a backend by name."""
    return BackendRegistry.instance().unregister(name)


_register_builtins()
