from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from open_the_chests.utils.results import read_results_json


def _safe_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in s)


def plot_results(*, results_paths: list[Path], output_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for p in results_paths:
        try:
            r = read_results_json(p)
            r["_results_path"] = str(p)
            rows.append(r)
        except Exception:
            continue

    by_env: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        env_id = str(r.get("env_id", "unknown"))
        by_env.setdefault(env_id, []).append(r)

    metrics = [
        ("mean_reward", "Mean Reward"),
        ("success_rate", "Success Rate"),
        ("mean_final_distance", "Mean Final Distance"),
    ]

    # print(by_env)
    for env_id, items in by_env.items():
        run_dir = by_env[env_id][0].get("run_dir", None)
        if output_dir is None:
            output_dir = Path(run_dir) / "plots" if run_dir else Path("./plots")
        print("Plotting results for env:", env_id, " - found in:", output_dir)
        
        series_by_algo: dict[str, list[dict[str, Any]]] = {}
        for it in items:
            algo = str(it.get("algo") or "unknown")
            series_by_algo.setdefault(algo, []).append(it)

        for key, title in metrics:
            plt.figure(figsize=(10, 6))

            for algo, algo_items in series_by_algo.items():
                points: list[tuple[float, float]] = []
                for it in algo_items:
                    x = it.get("total_timesteps")
                    y = it.get(key)
                    if x is None or y is None:
                        continue
                    try:
                        xf = float(x)
                        yf = float(y)
                    except Exception:
                        continue
                    points.append((xf, yf))

                points.sort(key=lambda t: t[0])
                if not points:
                    continue

                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                plt.plot(xs, ys, marker="o", label=algo)

            plt.title(f"{env_id} - {title}")
            plt.xlabel("total_timesteps")
            plt.ylabel(key)
            plt.grid(True)
            plt.legend()

            fname = f"{_safe_name(env_id)}_{key}_vs_timesteps.png"
            out = output_dir / fname
            plt.tight_layout()
            plt.savefig(out)
            plt.close()
