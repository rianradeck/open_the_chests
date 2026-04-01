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
    tb_log_dir: str | Path | None = None,
    tb_run_name: str = "eval",
) -> EvalMetrics:
    seed_everything(seed)

    writer = None
    if tb_log_dir is not None:
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore[import-not-found]

            writer = SummaryWriter(log_dir=str(tb_log_dir))
        except Exception as e:
            raise RuntimeError(
                "Falha ao inicializar TensorBoard SummaryWriter. "
                "Verifique se `tensorboard` está instalado no seu ambiente (pip install tensorboard)."
            ) from e

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

            if writer is not None:
                writer.add_scalar(f"{tb_run_name}/episode_reward", float(total_reward), ep)
                writer.add_scalar(f"{tb_run_name}/episode_len", float(steps), ep)
                if s is not None:
                    writer.add_scalar(f"{tb_run_name}/episode_success", 1.0 if s else 0.0, ep)
                if d is not None:
                    writer.add_scalar(f"{tb_run_name}/episode_final_distance", float(d), ep)

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

        metrics = EvalMetrics(
            mean_reward=mean_reward,
            success_rate=success_rate,
            mean_final_distance=mean_final_distance,
            mean_ep_len=mean_ep_len,
        )
        if writer is not None:
            writer.add_scalar(f"{tb_run_name}/mean_reward", float(metrics.mean_reward), 0)
            writer.add_scalar(f"{tb_run_name}/mean_ep_len", float(metrics.mean_ep_len), 0)
            if metrics.success_rate is not None:
                writer.add_scalar(f"{tb_run_name}/success_rate", float(metrics.success_rate), 0)
            if metrics.mean_final_distance is not None:
                writer.add_scalar(
                    f"{tb_run_name}/mean_final_distance", float(metrics.mean_final_distance), 0
                )
            writer.flush()

        return metrics
    finally:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
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
