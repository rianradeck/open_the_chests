from __future__ import annotations

import argparse
from pathlib import Path

from open_the_chests.viz.training_curves import plot_results

# python -m open_the_chests.cli.plot --run-dir /home/rianradeck/open_the_chests/runs/test_run/20260401_092604_seed42
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", action="append", default=[])
    p.add_argument("--runs-root", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    run_dirs = [Path(p) for p in args.run_dir]

    results_paths: list[Path] = []
    for d in run_dirs:
        p = d / "results.json"
        if p.exists():
            results_paths.append(p)

    if args.runs_root is not None:
        root = Path(args.runs_root)
        results_paths.extend(root.glob("**/results.json"))

    if not results_paths:
        raise SystemExit("Nenhum results.json encontrado")

    try:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        output_dir = None

    plot_results(results_paths=results_paths, output_dir=output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
