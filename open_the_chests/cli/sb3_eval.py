from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from open_the_chests.frameworks.sb3.eval import eval_model, load_model
from open_the_chests.utils.results import write_results_json
from open_the_chests.utils.runs import create_run_dir, write_config

# python -m open_the_chests.cli.sb3_eval --env-id ColoredChestKuka-v0 --model-path runs/test_run/20260401_092604_seed42/models/final_model.zip --algo sac --episodes 50 --seed 42 --device cpu --total-timesteps 100000 --max-steps 200
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--algo", choices=["ppo", "sac"], default=None)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--run-dir", type=str, default=None)
    p.add_argument("--total-timesteps", type=int, default=None)

    p.add_argument(
        "--tb-eval-dir",
        type=str,
        default=None,
        help="Diretório para gravar logs TensorBoard da avaliação (default: <run_dir>/tb/eval)",
    )

    p.add_argument("--reward-type", type=str, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--observation-space", type=str, choices=["default", "extended"], default="extended")

    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    env_kwargs: dict[str, Any] = {}
    if args.reward_type is not None:
        env_kwargs["reward_type"] = args.reward_type
    if args.max_steps is not None:
        env_kwargs["max_steps"] = int(args.max_steps)
    if args.observation_space is not None:
        env_kwargs["observation_space"] = args.observation_space

    if args.run_dir is None:
        run_paths = create_run_dir(run_name=f"eval_{args.env_id}", seed=args.seed)
        run_dir = run_paths.run_dir
        results_path = run_paths.results_path
        config_path = run_paths.config_path
    else:
        run_dir = Path(args.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        results_path = run_dir / "results.json"
        config_path = run_dir / "config.json"

    config = {
        "env_id": str(args.env_id),
        "algo": args.algo,
        "seed": args.seed,
        "device": str(args.device),
        "episodes": int(args.episodes),
        "model_path": str(args.model_path),
        "env_kwargs": dict(env_kwargs),
        "total_timesteps": args.total_timesteps,
        "observation_space": args.observation_space,
    }
    write_config(config_path, config)

    model = load_model(
        model_path=str(args.model_path),
        env_id=str(args.env_id),
        algo=args.algo,  # type: ignore[arg-type]
        device=str(args.device),
        env_kwargs=env_kwargs,
    )

    if args.tb_eval_dir is None:
        tb_eval_dir = run_dir / "tb" / "eval"
    else:
        tb_eval_dir = Path(str(args.tb_eval_dir))
    tb_eval_dir.mkdir(parents=True, exist_ok=True)

    metrics = eval_model(
        env_id=str(args.env_id),
        model=model,
        n_episodes=int(args.episodes),
        seed=args.seed,
        deterministic=True,
        env_kwargs=env_kwargs,
        tb_log_dir=tb_eval_dir,
        tb_run_name="eval",
    )

    results = {
        "env_id": str(args.env_id),
        "algo": args.algo,
        "seed": args.seed,
        "device": str(args.device),
        "total_timesteps": args.total_timesteps,
        "mean_reward": float(metrics.mean_reward),
        "success_rate": metrics.success_rate,
        "mean_final_distance": metrics.mean_final_distance,
        "mean_ep_len": float(metrics.mean_ep_len),
        "model_path": str(args.model_path),
        "run_dir": str(run_dir),
    }
    write_results_json(results_path, results)

    print(str(run_dir))
    print(str(results_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
