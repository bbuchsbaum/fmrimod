"""Typed contrast errors."""

from __future__ import annotations


class DesignProvenanceError(ValueError):
    """Raised when a typed contrast cannot resolve against the realized design.

    Carries the names of fields whose provenance is inferred or missing and a
    pointer to the repair path (design compiler / construction-time facts).
    """

    def __init__(
        self,
        message: str,
        *,
        weak_fields: tuple[str, ...] = (),
        repair_path: str | None = None,
    ) -> None:
        super().__init__(message)
        self.weak_fields = tuple(weak_fields)
        self.repair_path = repair_path

    def __str__(self) -> str:
        message = str(self.args[0]) if self.args else ""
        if self.repair_path:
            return f"{message}\n  Repair: {self.repair_path}"
        return message

    def __repr__(self) -> str:
        return (
            f"DesignProvenanceError(message={self.args[0]!r}, "
            f"weak_fields={self.weak_fields!r}, "
            f"repair_path={self.repair_path!r})"
        )


__all__ = ["DesignProvenanceError"]
