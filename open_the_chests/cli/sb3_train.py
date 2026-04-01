from __future__ import annotations

import argparse
from typing import Any

from open_the_chests.frameworks.sb3.train import train_sb3

# python -m open_the_chests.cli.sb3_train --env-id ColoredChestKuka-v0 --algo sac --timesteps 100000 --seed 42 --device cpu --run-name test_run --max-steps 200 --observation-space extended
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", required=True)
    p.add_argument("--algo", required=True, choices=["ppo", "sac"])
    p.add_argument("--timesteps", required=True, type=int)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--run-name", type=str, default=None)

    p.add_argument("--eval-episodes", type=int, default=50)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)

    p.add_argument("--reward-type", type=str, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--observation-space", type=str, choices=["default", "extended"], default="default")

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

    out = train_sb3(
        env_id=str(args.env_id),
        algo=str(args.algo),  # type: ignore[arg-type]
        total_timesteps=int(args.timesteps),
        seed=args.seed,
        device=str(args.device),
        run_name=args.run_name,
        env_kwargs=env_kwargs,
        eval_episodes=int(args.eval_episodes),
        learning_rate=float(args.learning_rate),
        gamma=float(args.gamma),
    )

    print(str(out.run_paths.run_dir))
    print(str(out.run_paths.results_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
