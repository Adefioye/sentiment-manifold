"""Command-line interface."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import ReproductionConfig
from .data import load_toy_movie_review
from .devices import resolve_device
from .experiment import run_reproduction
from .models import CausalLMAdapter
from .plotting import plot_run


def _load_with_overrides(args) -> ReproductionConfig:
    config = ReproductionConfig.load(args.config)
    if getattr(args, "model", None):
        config.model.name = args.model
    if getattr(args, "device", None):
        config.model.device = args.device
    if getattr(args, "output_dir", None):
        config.experiment.output_dir = str(Path(args.output_dir).resolve())
    if getattr(args, "with_openwebtext", False):
        config.experiment.evaluate_openwebtext = True
    return config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="sentiment-manifold")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect-data", help="summarize ToyMovieReview")
    inspect_parser.add_argument("--config", default="configs/reproduction.yaml")
    inspect_parser.add_argument("--model", default=None)
    inspect_parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default=None)

    reproduce = subparsers.add_parser("reproduce", help="fit and evaluate all configured layers")
    reproduce.add_argument("--config", default="configs/reproduction.yaml")
    reproduce.add_argument("--model", choices=["gpt2-small", "qwen-0.6b"], default=None)
    reproduce.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default=None)
    reproduce.add_argument("--output-dir", default=None)
    reproduce.add_argument("--with-openwebtext", action="store_true")

    plot_parser = subparsers.add_parser("plot", help="render plots from a completed run")
    plot_parser.add_argument("--run-dir", required=True)

    args = parser.parse_args(argv)
    if args.command == "inspect-data":
        config = _load_with_overrides(args)
        toy = load_toy_movie_review(config.data.toy_config)
        for split, examples in (("train", toy.train), ("test", toy.test)):
            positive = sum(example.label == 1 for example in examples)
            negative = len(examples) - positive
            print(f"{split}: {len(examples)} examples ({positive} positive, {negative} negative)")
        if args.model:
            adapter = CausalLMAdapter.from_pretrained(
                config.model.hub_name,
                resolve_device(config.model.device, config.model.dtype),
                revision=config.model.revision,
            )
            for split, examples in (("train", toy.train), ("test", toy.test)):
                retained = sum(adapter.focus_is_single_token(example) for example in examples)
                print(
                    f"{split}: tokenizer retained {retained}/{len(examples)} one-token adjectives"
                )
            answer_ids = {
                label: adapter.tokenizer(answer, add_special_tokens=False)["input_ids"]
                for label, answer in toy.answers.items()
            }
            print(f"answer token ids: {answer_ids}")
    elif args.command == "reproduce":
        run_dir = run_reproduction(_load_with_overrides(args))
        print(f"Completed run: {run_dir}")
    elif args.command == "plot":
        for path in plot_run(args.run_dir):
            print(path)


if __name__ == "__main__":
    main()
