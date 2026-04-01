from __future__ import annotations

from typing import Final


KUKA_ENV_ID: Final[str] = "ColoredChestKuka-v0"


def ensure_kuka_registered() -> None:
    import colored_chest_kuka_env  # noqa: F401
