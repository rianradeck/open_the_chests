from __future__ import annotations

import argparse
from pathlib import Path
import json

from open_the_chests.utils.runs import create_run_dir, write_config
from open_the_chests.utils.results import write_results_json


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate or test the Decision Transformer.")
    sub = p.add_subparsers(dest="command", required=True)

    # --- evaluate ---
    ev = sub.add_parser("evaluate", help="Compute accuracy, precision, recall, F1.")
    ev.add_argument("--model-path",    type=str,   default=None,
                   help="Path to model (default: {env}_decision_transformer.pt)")
    ev.add_argument("--num-sequences", type=int,   default=200)
    ev.add_argument("--n-events",      type=int,   default=200)
    ev.add_argument("--context-len",   type=int,   default=None, help="Context window length K (default: infer from config.json when available)")
    ev.add_argument("--batch-size",    type=int,   default=32)
    ev.add_argument("--threshold",     type=float, default=0.5)
    ev.add_argument("--device",        type=str,   default=None)
    ev.add_argument("--env",           type=str,   default="medium", choices=["easy", "medium", "hard"])
    ev.add_argument("--run-dir",       type=str,   default=None)
    ev.add_argument("--run-name",      type=str,   default="dt_eval")
    ev.add_argument("--seed",          type=int,   default=None)
    ev.add_argument("--model-type",    choices=["scratch", "pretrained"], default="scratch")
    ev.add_argument(
        "--pretrained-name",
        type=str,
        default=None,
        help="HuggingFace backbone name used when --model-type=pretrained (can be inferred from <run_dir>/config.json)",
    )

    # --- test ---
    ts = sub.add_parser("test", help="Generate one sequence and plot predictions.")
    ts.add_argument("--model-path",      type=str,   default=None)
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
        # Lazy import so `--help` works even if optional deps are missing.
        from open_the_chests.frameworks.pytorch_transformer.eval import evaluate_dt

        if args.run_dir is None:
            run_paths = create_run_dir(run_name=f"{args.run_name}_{args.env}", seed=args.seed)
            run_dir = run_paths.run_dir
            results_path = run_paths.results_path
            config_path = run_paths.config_path
        else:
            run_dir = Path(args.run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            results_path = run_dir / "results.json"
            config_path = run_dir / "config.json"

        inferred_pretrained = args.pretrained_name
        inferred_context_len = args.context_len
        inferred_model_hparams: dict[str, object] = {}
        if inferred_pretrained is None and args.run_dir is not None:
            cfg_path = Path(args.run_dir) / "config.json"
            if cfg_path.exists():
                try:
                    with cfg_path.open("r", encoding="utf-8") as f:
                        cfg = json.load(f)
                    model_cfg = (cfg.get("model", {}) or {})
                    inferred_pretrained = model_cfg.get("pretrained_name")
                    if inferred_context_len is None:
                        inferred_context_len = cfg.get("context_len")

                    # scratch hyperparams (safe defaults if absent)
                    for k in ("d_model", "nhead", "num_layers", "dropout"):
                        if k in model_cfg and model_cfg.get(k) is not None:
                            inferred_model_hparams[k] = model_cfg.get(k)
                except Exception:
                    inferred_pretrained = None

        config = {
            "framework": "pytorch_transformer",
            "algo": "decision_transformer",
            "command": "evaluate",
            "env": str(args.env),
            "seed": args.seed,
            "model_path": args.model_path,
            "model_type": str(args.model_type),
            "pretrained_name": inferred_pretrained,
            "num_sequences": int(args.num_sequences),
            "n_events": int(args.n_events),
            "batch_size": int(args.batch_size),
            "threshold": float(args.threshold),
            "device": args.device,
        }
        write_config(config_path, config)

        metrics = evaluate_dt(
            load_path=args.model_path,
            num_sequences=int(args.num_sequences),
            n_events=int(args.n_events),
            K=(int(inferred_context_len) if inferred_context_len is not None else None),
            batch_size=int(args.batch_size),
            threshold=float(args.threshold),
            device=args.device,
            env=str(args.env),
            tb_log_dir=run_dir / "tb" / "eval",
            model_type=str(args.model_type),  # type: ignore[arg-type]
            pretrained_name=inferred_pretrained,
            d_model=int(inferred_model_hparams.get("d_model", 128)),
            nhead=int(inferred_model_hparams.get("nhead", 4)),
            num_layers=int(inferred_model_hparams.get("num_layers", 3)),
            dropout=float(inferred_model_hparams.get("dropout", 0.1)),
        )

        results = dict(config)
        results.update(metrics)
        results["run_dir"] = str(run_dir)
        write_results_json(results_path, results)

        print(str(run_dir))
        print(str(results_path))
        print(metrics)

    elif args.command == "test":
        from open_the_chests.frameworks.pytorch_transformer.eval import test_sequence

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
