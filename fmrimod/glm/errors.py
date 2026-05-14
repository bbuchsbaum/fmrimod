"""Typed GLM fitting errors."""

from __future__ import annotations


class FmriCapabilityError(NotImplementedError):
    """Raised when a requested fMRImod capability is intentionally unsupported."""

    def __init__(
        self,
        *,
        feature: str,
        current_capability: tuple[str, ...] = (),
        repair: str | None = None,
        message: str | None = None,
    ) -> None:
        self.feature = feature
        self.current_capability = current_capability
        self.repair = repair

        if message is None:
            detail = f"{feature} is not supported by the current implementation"
            if current_capability:
                detail += f"; current capability: {', '.join(current_capability)}"
            if repair:
                detail += f". Repair: {repair}"
            message = detail

        super().__init__(message)


class UnsupportedEngineConfiguration(FmriCapabilityError):
    """Raised when a fitting engine cannot honor a requested configuration."""

    def __init__(
        self,
        *,
        engine: str,
        feature: str,
        repair: str,
        current_capability: tuple[str, ...] = (),
    ) -> None:
        self.engine = engine
        capability = (
            f" Current capability: {', '.join(current_capability)}."
            if current_capability
            else ""
        )
        super().__init__(
            feature=f"{engine}:{feature}",
            current_capability=current_capability,
            repair=repair,
            message=(
                f"{engine!r} engine does not support {feature}."
                f"{capability} "
                f"Repair: {repair}."
            ),
        )
