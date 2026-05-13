"""Registries for native group-analysis components."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Generic, TypeVar

from .errors import GroupRegistryError

T = TypeVar("T")


@dataclass(frozen=True)
class RegistryEntry(Generic[T]):
    """Metadata for a registered group-analysis component."""

    name: str
    factory: Callable[..., T]
    description: str
    validate_function: Callable[[T], bool] | None
    registered_at: datetime = field(default_factory=datetime.now)


class GroupRegistry(Generic[T]):
    """Registry for adapters, reducers, posthoc methods, and map families."""

    def __init__(self, kind: str) -> None:
        if not isinstance(kind, str) or not kind:
            raise GroupRegistryError("kind must be a non-empty string", parameter="kind")
        self.kind = kind
        self._entries: dict[str, RegistryEntry[T]] = {}

    def register(
        self,
        name: str,
        factory: Callable[..., T],
        *,
        description: str | None = None,
        validate_function: Callable[[T], bool] | None = None,
        overwrite: bool = False,
    ) -> None:
        """Register a named component factory."""
        if not isinstance(name, str) or not name:
            raise GroupRegistryError("name must be a non-empty string", parameter="name")
        if not callable(factory):
            raise GroupRegistryError("factory must be callable", parameter="factory")
        if validate_function is not None and not callable(validate_function):
            raise GroupRegistryError(
                "validate_function must be callable",
                parameter="validate_function",
            )
        if name in self._entries and not overwrite:
            raise GroupRegistryError(
                f"{self.kind} '{name}' is already registered. "
                "Use overwrite=True to replace.",
                parameter="name",
                value=name,
            )

        self._entries[name] = RegistryEntry(
            name=name,
            factory=factory,
            description=description or f"{self.kind}: {name}",
            validate_function=validate_function,
        )

    def create(self, name: str, *, validate: bool = True, **kwargs: Any) -> T:
        """Create a registered component by name."""
        entry = self._get_entry(name)
        try:
            component = entry.factory(**kwargs)
        except Exception as exc:
            raise GroupRegistryError(
                f"Failed to create {self.kind} '{name}': {exc}",
                parameter="name",
                value=name,
            ) from exc

        validator = entry.validate_function
        if validate and validator is not None and not validator(component):
            raise GroupRegistryError(
                f"{self.kind} '{name}' failed custom validation",
                parameter="name",
                value=name,
            )
        return component

    def get(self, name: str) -> Callable[..., T]:
        """Return a registered component factory."""
        return self._get_entry(name).factory

    def list_names(self) -> list[str]:
        """Return sorted registered component names."""
        return sorted(self._entries)

    def is_registered(self, name: str) -> bool:
        """Return whether a component is registered."""
        if not isinstance(name, str):
            raise GroupRegistryError("name must be a string", parameter="name")
        return name in self._entries

    def unregister(self, name: str) -> bool:
        """Remove a registration."""
        if not isinstance(name, str):
            raise GroupRegistryError("name must be a string", parameter="name")
        return self._entries.pop(name, None) is not None

    def get_info(self, name: str) -> dict[str, Any]:
        """Return registration metadata."""
        entry = self._get_entry(name)
        return {
            "name": entry.name,
            "kind": self.kind,
            "description": entry.description,
            "has_validate": entry.validate_function is not None,
            "registered_at": entry.registered_at,
        }

    def _get_entry(self, name: str) -> RegistryEntry[T]:
        if not isinstance(name, str) or not name:
            raise GroupRegistryError("name must be a non-empty string", parameter="name")
        if name not in self._entries:
            available = ", ".join(self.list_names()) or "<none>"
            raise GroupRegistryError(
                f"{self.kind} '{name}' is not registered. Available: {available}",
                parameter="name",
                value=name,
            )
        return self._entries[name]


adapter_registry: GroupRegistry[Any] = GroupRegistry("adapter")
reducer_registry: GroupRegistry[Any] = GroupRegistry("reducer")
posthoc_registry: GroupRegistry[Any] = GroupRegistry("posthoc")
map_family_registry: GroupRegistry[Any] = GroupRegistry("map_family")

