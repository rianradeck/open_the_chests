from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from collections.abc import Mapping

from open_the_chests.frameworks.sb3.eval import EvalMetrics, eval_model
from open_the_chests.envs.factory import get_env
from open_the_chests.utils.runs import RunPaths, create_run_dir, write_config
from open_the_chests.utils.results import write_results_json
from open_the_chests.utils.seeding import seed_everything


AlgoName = Literal["ppo", "sac"]


@dataclass(frozen=True)
class TrainOutput:
    run_paths: RunPaths
    model_path: Path
    eval_metrics: EvalMetrics


def train_sb3(
    *,
    env_id: str,
    algo: AlgoName,
    total_timesteps: int,
    seed: int | None,
    device: str,
    run_name: str | None,
    env_kwargs: Mapping[str, Any] | None = None,
    eval_episodes: int = 50,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
) -> TrainOutput:
    seed_everything(seed)

    run_paths = create_run_dir(run_name=run_name or f"{env_id}_{algo}", seed=seed)

    config: dict[str, Any] = {
        "env_id": env_id,
        "algo": algo,
        "total_timesteps": int(total_timesteps),
        "seed": seed,
        "device": device,
        "eval_episodes": int(eval_episodes),
        "learning_rate": float(learning_rate),
        "gamma": float(gamma),
        "env_kwargs": dict(env_kwargs) if env_kwargs else {},
    }
    write_config(run_paths.config_path, config)

    env = get_env(env_id, seed=seed, **(dict(env_kwargs) if env_kwargs else {}))

    tb_train_dir = run_paths.run_dir / "tb" / "train"

    try:
        if algo == "ppo":
            from stable_baselines3 import PPO

            model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                learning_rate=learning_rate,
                gamma=gamma,
                seed=seed,
                device=device,
                tensorboard_log=str(tb_train_dir),
            )
        elif algo == "sac":
            from stable_baselines3 import SAC

            model = SAC(
                "MlpPolicy",
                env,
                verbose=1,
                learning_rate=learning_rate,
                gamma=gamma,
                seed=seed,
                device=device,
                tensorboard_log=str(tb_train_dir),
            )
        else:
            raise ValueError(f"algo inválido: {algo}")

        model.learn(total_timesteps=total_timesteps, tb_log_name=str(algo))

        model_path = run_paths.models_dir / "final_model.zip"
        model.save(str(model_path))
    finally:
        env.close()

    eval_metrics = eval_model(
        env_id=env_id,
        model=model,
        n_episodes=eval_episodes,
        seed=seed,
        deterministic=True,
        env_kwargs=env_kwargs,
    )

    results: dict[str, Any] = {
        "env_id": env_id,
        "algo": algo,
        "seed": seed,
        "device": device,
        "total_timesteps": int(total_timesteps),
        "mean_reward": float(eval_metrics.mean_reward),
        "success_rate": eval_metrics.success_rate,
        "mean_final_distance": eval_metrics.mean_final_distance,
        "mean_ep_len": float(eval_metrics.mean_ep_len),
        "model_path": str(model_path),
        "run_dir": str(run_paths.run_dir),
    }
    write_results_json(run_paths.results_path, results)

    return TrainOutput(run_paths=run_paths, model_path=model_path, eval_metrics=eval_metrics)
