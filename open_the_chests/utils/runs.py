from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from collections.abc import Mapping


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    models_dir: Path
    plots_dir: Path
    config_path: Path
    results_path: Path


def create_run_dir(
    *,
    base_dir: str | Path = "runs",
    run_name: str | None = None,
    seed: int | None = None,
) -> RunPaths:
    base_path = Path(base_dir)
    safe_name = (run_name or "run").strip().replace(" ", "_")
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    seed_part = f"seed{seed}" if seed is not None else "seedNone"

    run_dir = base_path / safe_name / f"{ts}_{seed_part}"
    models_dir = run_dir / "models"
    plots_dir = run_dir / "plots"

    models_dir.mkdir(parents=True, exist_ok=False)
    plots_dir.mkdir(parents=True, exist_ok=True)

    return RunPaths(
        run_dir=run_dir,
        models_dir=models_dir,
        plots_dir=plots_dir,
        config_path=run_dir / "config.json",
        results_path=run_dir / "results.json",
    )


def write_config(path: Path, config: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(dict(config), f, indent=2, sort_keys=True)
        f.write("\n")
