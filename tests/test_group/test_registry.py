"""Tests for native group-analysis registries."""

from __future__ import annotations

import pytest

from fmrimod.group import GroupRegistry, GroupRegistryError


class Component:
    def __init__(self, value: int = 1) -> None:
        self.value = value


def test_group_registry_create_and_metadata() -> None:
    registry: GroupRegistry[Component] = GroupRegistry("reducer")
    registry.register(
        "demo",
        lambda value=1: Component(value),
        description="demo reducer",
        validate_function=lambda component: component.value > 0,
    )

    assert registry.is_registered("demo")
    assert registry.list_names() == ["demo"]
    component = registry.create("demo", value=3)
    assert component.value == 3
    info = registry.get_info("demo")
    assert info["kind"] == "reducer"
    assert info["description"] == "demo reducer"
    assert info["has_validate"] is True


def test_group_registry_rejects_duplicates_and_bad_validation() -> None:
    registry: GroupRegistry[Component] = GroupRegistry("adapter")
    registry.register("demo", Component)

    with pytest.raises(GroupRegistryError, match="already registered"):
        registry.register("demo", Component)

    registry.register(
        "bad",
        lambda: Component(0),
        validate_function=lambda component: component.value > 0,
    )
    with pytest.raises(GroupRegistryError, match="failed custom validation"):
        registry.create("bad")


def test_group_registry_unregister() -> None:
    registry: GroupRegistry[Component] = GroupRegistry("posthoc")
    registry.register("demo", Component)

    assert registry.unregister("demo") is True
    assert registry.unregister("demo") is False
    with pytest.raises(GroupRegistryError, match="not registered"):
        registry.create("demo")

