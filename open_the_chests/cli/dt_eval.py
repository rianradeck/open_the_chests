from __future__ import annotations

import argparse

from open_the_chests.models.eval import evaluate, test_sequence


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate or test the Decision Transformer.")
    sub = p.add_subparsers(dest="command", required=True)

    # --- evaluate ---
    ev = sub.add_parser("evaluate", help="Compute accuracy, precision, recall, F1.")
    ev.add_argument("--model-path",    type=str,   required=True)
    ev.add_argument("--num-sequences", type=int,   default=200)
    ev.add_argument("--n-events",      type=int,   default=200)
    ev.add_argument("--batch-size",    type=int,   default=32)
    ev.add_argument("--threshold",     type=float, default=0.5)
    ev.add_argument("--device",        type=str,   default=None)
    ev.add_argument("--env",           type=str,   default="medium", choices=["easy", "medium", "hard"])

    # --- test ---
    ts = sub.add_parser("test", help="Generate one sequence and plot predictions.")
    ts.add_argument("--model-path",      type=str,   required=True)
    ts.add_argument("--n-events",        type=int,   default=50)
    ts.add_argument("--train-n-events",  type=int,   default=None)
    ts.add_argument("--threshold",       type=float, default=0.5)
    ts.add_argument("--save-path",       type=str,   default="test_sequence.png")
    ts.add_argument("--device",          type=str,   default=None)
    ts.add_argument("--env",             type=str,   default="medium", choices=["easy", "medium", "hard"])

    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "evaluate":
        metrics = evaluate(
            load_path=args.model_path,
            num_sequences=args.num_sequences,
            n_events=args.n_events,
            batch_size=args.batch_size,
            threshold=args.threshold,
            device=args.device,
            env=args.env,
        )
        print(metrics)

    elif args.command == "test":
        test_sequence(
            load_path=args.model_path,
            n_events=args.n_events,
            train_n_events=args.train_n_events,
            threshold=args.threshold,
            save_path=args.save_path,
            device=args.device,
            env=args.env,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
