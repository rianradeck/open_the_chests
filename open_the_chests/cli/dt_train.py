from __future__ import annotations

import argparse


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the Decision Transformer on OpenTheChests sequences.")
    p.add_argument("--run-name",      type=str, default="dt_train")
    p.add_argument("--seed",          type=int, default=None)
    p.add_argument("--num-sequences", type=int, default=1000)
    p.add_argument("--n-events",      type=int, default=200)
    p.add_argument("--context-len",   type=int, default=None, help="Context window length K (default: full sequence)")
    p.add_argument("--epochs",        type=int, default=100)
    p.add_argument("--batch-size",    type=int, default=32)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--device",        type=str, default=None)
    p.add_argument("--env",           type=str, default="medium", choices=["easy", "medium", "hard"])

    p.add_argument("--model-type", choices=["scratch", "pretrained"], default="scratch")
    p.add_argument(
        "--pretrained-name",
        type=str,
        default="distilgpt2",
        help="HuggingFace causal backbone name (e.g. distilgpt2, gpt2). Used when --model-type=pretrained",
    )
    p.add_argument(
        "--freeze-backbone",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze HF backbone weights (recommended to start). Used when --model-type=pretrained",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # Lazy import so `--help` works even if optional deps are missing.
    from open_the_chests.frameworks.pytorch_transformer.train import train_dt

    out = train_dt(
        run_name=str(args.run_name),
        seed=args.seed,
        num_sequences=int(args.num_sequences),
        n_events=int(args.n_events),
        context_len=args.context_len,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        device=args.device,
        env=str(args.env),
        model_type=str(args.model_type),  # type: ignore[arg-type]
        pretrained_name=str(args.pretrained_name),
        freeze_backbone=bool(args.freeze_backbone),
    )

    print(str(out.run_paths.run_dir))
    print(str(out.run_paths.results_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
