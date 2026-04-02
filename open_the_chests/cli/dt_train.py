from __future__ import annotations

import argparse

from open_the_chests.frameworks.pytorch_transformer.train import train


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the Decision Transformer on OpenTheChests sequences.")
    p.add_argument("--num-sequences", type=int, default=1000)
    p.add_argument("--n-events",      type=int, default=200)
    p.add_argument("--epochs",        type=int, default=100)
    p.add_argument("--batch-size",    type=int, default=32)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--device",        type=str, default=None)
    p.add_argument("--save-path",     type=str, default=None,
                   help="Path to save model (default: {env}_decision_transformer.pt)")
    p.add_argument("--env",           type=str, default="medium", choices=["easy", "medium", "hard"])
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    train(
        num_sequences=args.num_sequences,
        n_events=args.n_events,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        save_path=args.save_path,
        env=args.env,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
