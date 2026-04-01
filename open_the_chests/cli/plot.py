from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

# python -m open_the_chests.cli.plot --run-dir runs/test_run/20260401_092604_seed42
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", action="append", default=[])
    p.add_argument("--runs-root", type=str, default=None)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=6006)
    return p


def _find_tb_dirs(*, run_dirs: list[Path]) -> list[Path]:
    tb_dirs: list[Path] = []
    for run_dir in run_dirs:
        candidates = [run_dir / "tb", run_dir / "tensorboard"]
        for c in candidates:
            if c.exists() and c.is_dir():
                tb_dirs.append(c)
                break
    return tb_dirs


def _build_logdir_spec(*, tb_dirs: list[Path]) -> str:
    parts: list[str] = []
    for tb_dir in tb_dirs:
        run_dir = tb_dir.parent
        name = run_dir.name
        parts.append(f"{name}:{tb_dir}")
    return ",".join(parts)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    run_dirs = [Path(p) for p in args.run_dir]

    if args.runs_root is not None:
        root = Path(args.runs_root)
        for tb_dir in root.glob("**/tb"):
            if tb_dir.is_dir():
                run_dirs.append(tb_dir.parent)
        for tb_dir in root.glob("**/tensorboard"):
            if tb_dir.is_dir():
                run_dirs.append(tb_dir.parent)

    # de-dup
    seen: set[str] = set()
    uniq_run_dirs: list[Path] = []
    for d in run_dirs:
        key = str(d.resolve())
        if key in seen:
            continue
        seen.add(key)
        uniq_run_dirs.append(d)

    tb_dirs = _find_tb_dirs(run_dirs=uniq_run_dirs)
    if not tb_dirs:
        raise SystemExit("Nenhum diretório de TensorBoard encontrado (procure por <run_dir>/tb)")

    logdir_spec = _build_logdir_spec(tb_dirs=tb_dirs)

    cmd = [
        "tensorboard",
        "--logdir_spec",
        logdir_spec,
        "--host",
        str(args.host),
        "--port",
        str(args.port),
    ]

    print("Executando:", " ".join(cmd))
    print(f"Acesse: http://{args.host}:{args.port}")

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as e:
        raise SystemExit(
            "tensorboard não encontrado. Instale com `pip install tensorboard` (no seu venv)."
        ) from e
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
