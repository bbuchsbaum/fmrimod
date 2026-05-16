"""Pinned namespace-shadowing states for the public surface.

Enforces ``docs/contracts/public_api_policy_v1.md`` § "Namespace
shadowing": every top-level name that collides with a same-named
submodule is a *ratified, pinned* policy state, not an inline source
comment. A new top-level/submodule collision outside the enumerated
set fails here until it is classified in the policy and added to the
pinned set in the same commit.

Cheap-pass disqualified: asserting ``hasattr(fmrimod, "hrf") is
False``. Importing a submodule attaches it as an attribute of the
``fmrimod`` package even when the name is absent from ``__all__``, so
the naive ``hasattr`` assertion is wrong. These tests pin *behavioural*
states (``__all__`` membership, callability, and ``sys.modules``
submodule resolution) instead.

Submodule detection is case-*exact* (``os.listdir``), not
``Path.exists()``: the dogfood dev platform is case-insensitive APFS,
where ``Path("fmrimod/HRF").exists()`` spuriously matches the ``hrf``
submodule and would misclassify the ``HRF`` class as a collision.

Refs: bd-01KRHTHY5309X3VZT515S9E3H3 (this policy state),
bd-01KRFMD3F66TENJMP6BQYE32HC (the ``contrast`` rebind ergonomic),
tests/test_contrast/test_polymorphic_predicates.py (the deeper
``contrast`` dual-resolution anchor this file points at).
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from types import ModuleType

import fmrimod

# ── Pinned allowed set ────────────────────────────────────────────────
# A name that collides with a submodule must appear in exactly one of
# these. Adding a member is a visible diff that must land with the
# matching policy classification in docs/contracts/public_api_policy_v1.md.

# rebind: a top-level callable intentionally shadows the same-named
# submodule; ``import fmrimod.<name>`` still resolves the *distinct*
# submodule object via sys.modules.
REBIND_NAMES = frozenset({"contrast", "regressor"})

# withheld: a name deliberately kept out of ``__all__`` to protect the
# submodule slot; the Spec-tree builder is reached via
# ``from fmrimod.spec import <name>``.
WITHHELD_NAMES = frozenset({"hrf", "trialwise"})

# callable_submodule: the submodule object *is itself* callable; the
# name resolves to one object that is both the importable submodule and
# a callable (no separate shadowing binding).
CALLABLE_SUBMODULE_NAMES = frozenset({"stats"})

_PKG_DIR = Path(fmrimod.__file__).resolve().parent
_ENTRIES = frozenset(os.listdir(_PKG_DIR))


def _is_submodule(name: str) -> bool:
    """True if ``fmrimod/<name>/`` or ``fmrimod/<name>.py`` exists (case-exact)."""
    if name in _ENTRIES and (_PKG_DIR / name / "__init__.py").exists():
        return True
    return f"{name}.py" in _ENTRIES


def test_rebind_names_shadow_callable_yet_submodule_resolves() -> None:
    """Each rebind name: in ``__all__``, callable, distinct submodule resolves."""
    for name in sorted(REBIND_NAMES):
        assert _is_submodule(name), (
            f"{name!r} is pinned as a rebind but no fmrimod/{name} submodule "
            f"exists; remove it from REBIND_NAMES or restore the submodule."
        )
        assert name in fmrimod.__all__, (
            f"{name!r} is pinned as a rebind but is absent from "
            f"fmrimod.__all__; a rebind must expose the top-level callable."
        )
        attr = getattr(fmrimod, name)
        assert callable(attr), (
            f"fmrimod.{name} is pinned as a rebind callable but is "
            f"{type(attr)!r}."
        )
        submodule = importlib.import_module(f"fmrimod.{name}")
        assert isinstance(submodule, ModuleType), (
            f"import fmrimod.{name} did not resolve the submodule "
            f"(got {type(submodule)!r}); the rebind broke sys.modules "
            f"submodule resolution."
        )
        assert attr is not submodule, (
            f"fmrimod.{name} attribute and the fmrimod.{name} submodule "
            f"are the same object; this is a callable_submodule, not a "
            f"rebind — reclassify it."
        )


def test_withheld_names_absent_from_all_yet_submodule_resolves() -> None:
    """Each withheld name: absent from ``__all__``, submodule still resolves."""
    for name in sorted(WITHHELD_NAMES):
        assert _is_submodule(name), (
            f"{name!r} is pinned as withheld but no fmrimod/{name} "
            f"submodule exists to protect."
        )
        assert name not in fmrimod.__all__, (
            f"{name!r} is pinned as withheld but appears in "
            f"fmrimod.__all__; either remove the top-level binding or "
            f"reclassify it as a rebind in the policy + REBIND_NAMES."
        )
        submodule = importlib.import_module(f"fmrimod.{name}")
        assert isinstance(submodule, ModuleType), (
            f"import fmrimod.{name} did not resolve the submodule "
            f"(got {type(submodule)!r})."
        )


def test_callable_submodule_names_are_one_callable_module() -> None:
    """Each callable_submodule name: in ``__all__``; attribute *is* the
    submodule and is callable."""
    for name in sorted(CALLABLE_SUBMODULE_NAMES):
        assert _is_submodule(name), (
            f"{name!r} is pinned as a callable_submodule but no "
            f"fmrimod/{name} submodule exists."
        )
        assert name in fmrimod.__all__, (
            f"{name!r} is pinned as a callable_submodule but is absent "
            f"from fmrimod.__all__."
        )
        attr = getattr(fmrimod, name)
        submodule = importlib.import_module(f"fmrimod.{name}")
        assert isinstance(attr, ModuleType), (
            f"fmrimod.{name} is pinned as a callable_submodule but the "
            f"attribute is {type(attr)!r}, not a module."
        )
        assert attr is submodule, (
            f"fmrimod.{name} attribute and the fmrimod.{name} submodule "
            f"are distinct objects; this is a rebind, not a "
            f"callable_submodule — reclassify it."
        )
        assert callable(attr), (
            f"fmrimod.{name} is pinned as a callable_submodule but the "
            f"module object is not callable."
        )


def test_no_unclassified_top_level_submodule_collision() -> None:
    """No name in ``__all__`` shadows a submodule unless it is pinned.

    This is the guard that makes the policy enforceable: a *new*
    collision cannot land silently — it fails here until it is
    classified in docs/contracts/public_api_policy_v1.md § 'Namespace
    shadowing' and added to the matching pinned set in the same commit.
    """
    ratified = REBIND_NAMES | CALLABLE_SUBMODULE_NAMES
    colliding = {n for n in fmrimod.__all__ if _is_submodule(n)}
    unclassified = colliding - ratified
    assert not unclassified, (
        "top-level names shadow a submodule without being a ratified "
        f"rebind/callable_submodule: {sorted(unclassified)}. Classify "
        "each in docs/contracts/public_api_policy_v1.md § 'Namespace "
        "shadowing' and add it to the matching pinned set in the same "
        "commit, or stop exporting it at top level."
    )
