"""Command-line interface: ``synbiotorch <command> <config.yaml>``."""

from __future__ import annotations

import argparse
import sys

from synbiotorch.config import RunConfig, TaskConfig
from synbiotorch.data.corpus import build_corpus
from synbiotorch.data.materialize import materialize
from synbiotorch.generate import generate_sequence
from synbiotorch.models import build_model
from synbiotorch.pipeline import run_training
from synbiotorch.tokenize.base import build_tokenizer


def _cmd_ingest(args: argparse.Namespace) -> int:
    config = RunConfig.from_yaml(args.config)
    corpus = build_corpus(config.corpus)
    result = materialize(corpus, config.corpus.cache_dir, force=args.force)
    print(f"materialized {result.count} objects -> {result.path}")
    print(f"fingerprint: {result.fingerprint}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    config = RunConfig.from_yaml(args.config)
    metrics = run_training(config, resume_from=args.resume)
    print("final metrics:", metrics)
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    config = RunConfig.from_yaml(args.config)
    tokenizer = build_tokenizer(config.tokenizer)
    model = build_model(
        config.model, TaskConfig(kind="causal"), vocab_size=tokenizer.vocab_size, pad_token_id=tokenizer.pad_token_id
    )
    model.eval()
    for i in range(args.num_samples):
        seed = None if args.seed is None else args.seed + i
        print(
            generate_sequence(
                model,
                tokenizer,
                args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                max_context=config.model.arch.max_position_embeddings,
                seed=seed,
            )
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="synbiotorch", description="Train transformer models on SBOL data")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Materialize a corpus to the local Parquet cache")
    p_ingest.add_argument("config", help="Path to a run config YAML")
    p_ingest.add_argument("--force", action="store_true", help="Re-materialize even if cached")
    p_ingest.set_defaults(func=_cmd_ingest)

    p_train = sub.add_parser("train", help="Run training from a config")
    p_train.add_argument("config", help="Path to a run config YAML")
    p_train.add_argument(
        "--resume", metavar="CKPT", default=None, help="Resume from a checkpoint (e.g. an output dir's last.pt)"
    )
    p_train.set_defaults(func=_cmd_train)

    p_generate = sub.add_parser("generate", help="Generate sequences from a trained causal-LM backbone")
    p_generate.add_argument("config", help="Path to a run config YAML (model.backbone points at the saved backbone)")
    p_generate.add_argument("--prompt", default="", help="Seed sequence to continue (empty starts from scratch)")
    p_generate.add_argument("--max-new-tokens", type=int, default=128)
    p_generate.add_argument("--temperature", type=float, default=1.0, help="<=0 is greedy")
    p_generate.add_argument("--top-k", type=int, default=0)
    p_generate.add_argument("--top-p", type=float, default=1.0)
    p_generate.add_argument("--num-samples", type=int, default=1)
    p_generate.add_argument("--seed", type=int, default=None)
    p_generate.set_defaults(func=_cmd_generate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
