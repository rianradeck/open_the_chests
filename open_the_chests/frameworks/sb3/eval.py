from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from collections.abc import Mapping

import statistics

from open_the_chests.envs.factory import get_env
from open_the_chests.utils.seeding import seed_everything


AlgoName = Literal["ppo", "sac"]


@dataclass(frozen=True)
class EvalMetrics:
    mean_reward: float
    success_rate: float | None
    mean_final_distance: float | None
    mean_ep_len: float


def _extract_success(info: Mapping[str, Any]) -> bool | None:
    if "is_success" in info:
        return bool(info.get("is_success"))
    return None


def _extract_final_distance(info: Mapping[str, Any]) -> float | None:
    value = info.get("distance_to_target")
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def eval_model(
    *,
    env_id: str,
    model: Any,
    n_episodes: int,
    seed: int | None = None,
    deterministic: bool = True,
    env_kwargs: Mapping[str, Any] | None = None,
) -> EvalMetrics:
    seed_everything(seed)

    env = get_env(env_id, seed=seed, **(dict(env_kwargs) if env_kwargs else {}))

    episode_returns: list[float] = []
    episode_lens: list[int] = []
    successes: list[bool] = []
    final_distances: list[float] = []

    try:
        for ep in range(n_episodes):
            ep_seed = (seed + ep) if seed is not None else None
            obs, _info = env.reset(seed=ep_seed)

            done = False
            total_reward = 0.0
            steps = 0
            last_info: Mapping[str, Any] = {}

            while not done:
                action, _state = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                last_info = info

                total_reward += float(reward)
                steps += 1
                done = bool(terminated) or bool(truncated)

            episode_returns.append(total_reward)
            episode_lens.append(steps)

            s = _extract_success(last_info)
            if s is not None:
                successes.append(s)

            d = _extract_final_distance(last_info)
            if d is not None:
                final_distances.append(d)

        mean_reward = float(statistics.fmean(episode_returns)) if episode_returns else 0.0
        mean_ep_len = float(statistics.fmean(episode_lens)) if episode_lens else 0.0

        success_rate: float | None
        if successes:
            success_rate = float(statistics.fmean([1.0 if x else 0.0 for x in successes]))
        else:
            success_rate = None

        mean_final_distance: float | None
        if final_distances:
            mean_final_distance = float(statistics.fmean(final_distances))
        else:
            mean_final_distance = None

        return EvalMetrics(
            mean_reward=mean_reward,
            success_rate=success_rate,
            mean_final_distance=mean_final_distance,
            mean_ep_len=mean_ep_len,
        )
    finally:
        env.close()


def load_model(
    *,
    model_path: str | Path,
    env_id: str,
    algo: AlgoName | None = None,
    device: str = "cpu",
    env_kwargs: Mapping[str, Any] | None = None,
) -> Any:
    model_path = Path(model_path)
    env = get_env(env_id, seed=None, **(dict(env_kwargs) if env_kwargs else {}))

    try:
        if algo == "ppo":
            from stable_baselines3 import PPO

            return PPO.load(str(model_path), env=env, device=device)
        if algo == "sac":
            from stable_baselines3 import SAC

            return SAC.load(str(model_path), env=env, device=device)

        last_error: Exception | None = None
        for candidate in ("ppo", "sac"):
            try:
                return load_model(
                    model_path=model_path,
                    env_id=env_id,
                    algo=candidate,  # type: ignore[arg-type]
                    device=device,
                    env_kwargs=env_kwargs,
                )
            except Exception as e:
                last_error = e

        raise RuntimeError("Falha ao carregar o modelo") from last_error
    finally:
        env.close()
