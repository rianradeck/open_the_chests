from __future__ import annotations

import importlib.util
from typing import Any

import gymnasium as gym

from open_the_chests.utils.seeding import seed_everything

from .kuka import KUKA_ENV_ID, ensure_kuka_registered


_ALIAS_TO_ID: dict[str, str] = {
    "kuka": KUKA_ENV_ID,
    "coloredchestkuka": KUKA_ENV_ID,
    "colored_chest_kuka": KUKA_ENV_ID,
    "otc-v0": "OpenTheChests-v0",
    "otc-v1": "OpenTheChests-v1",
    "otc-v2": "OpenTheChests-v2",
}


def _normalize_env_id(env_id: str) -> str:
    key = env_id.strip()
    key_lower = key.lower()
    return _ALIAS_TO_ID.get(key_lower, key)


def _ensure_registered(env_id: str) -> None:
    if env_id == KUKA_ENV_ID:
        ensure_kuka_registered()
        return

    if env_id.startswith("OpenTheChests-"):
        if importlib.util.find_spec("openthechests") is None:
            raise ModuleNotFoundError(
                "openthechests não está instalado, necessário para OpenTheChests-*"
            )
        from .otc_registry import register_custom_envs

        register_custom_envs()
        return


def get_env(
    env_id: str,
    *,
    seed: int | None = None,
    render_mode: str | None = None,
    **kwargs: Any,
) -> gym.Env:
    normalized_id = _normalize_env_id(env_id)
    _ensure_registered(normalized_id)

    final_kwargs: dict[str, Any] = dict(kwargs)
    if render_mode is not None:
        final_kwargs["render_mode"] = render_mode

    seed_everything(seed)

    env = gym.make(normalized_id, **final_kwargs)

    if seed is not None:
        env.reset(seed=seed)

    return env
