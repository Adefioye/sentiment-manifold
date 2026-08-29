"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .config import ReproductionConfig
from .data import load_toy_movie_review, preprocess_sst
from .devices import resolve_device
from .experiment import run_reproduction
from .models import CausalLMAdapter
from .plotting import plot_run


def _load_with_overrides(args) -> ReproductionConfig:
    config = ReproductionConfig.load(args.config)
    if getattr(args, "model", None):
        if config.model.name != args.model:
            config.model.revision = None
        config.model.name = args.model
    if getattr(args, "device", None):
        config.model.device = args.device
    if getattr(args, "output_dir", None):
        config.experiment.output_dir = str(Path(args.output_dir).resolve())
    if getattr(args, "with_openwebtext_resample_ablation", False):
        config.experiment.openwebtext_resample_ablation = True
    if getattr(args, "with_openwebtext", False):
        config.experiment.evaluate_openwebtext = True
    return config


def _token_from_environment(variable_name: str) -> str | None:
    token = os.environ.get(variable_name)
    if token:
        return token
    token_path = os.environ.get("HF_TOKEN_PATH")
    if token_path:
        path = Path(token_path).expanduser()
        if not path.is_file():
            raise ValueError(f"HF_TOKEN_PATH does not point to a readable file: {path}")
        token = path.read_text(encoding="utf-8").strip()
        if token:
            return token
    return None


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
    reproduce.add_argument(
        "--with-openwebtext-resample-ablation",
        action="store_true",
        help="run the optional exploratory OpenWebText loss diagnostic",
    )
    reproduce.add_argument("--with-openwebtext", action="store_true", help=argparse.SUPPRESS)

    plot_parser = subparsers.add_parser("plot", help="render plots from a completed run")
    plot_parser.add_argument("--run-dir", required=True)

    preprocess = subparsers.add_parser(
        "preprocess-sst",
        help="build Pythia-1.4B-filtered SST datasets and optionally publish them",
    )
    preprocess.add_argument(
        "--sst-root",
        default="../eliciting-latent-sentiment/stanfordSentimentTreebank",
    )
    preprocess.add_argument("--output-dir", default="data/processed/sst-pythia-1.4b")
    preprocess.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    preprocess.add_argument(
        "--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto"
    )
    preprocess.add_argument("--batch-size", type=int, default=16)
    preprocess.add_argument("--revision", default=None)
    preprocess.add_argument(
        "--binarization",
        choices=["both", "tigges", "neutral_removed"],
        default="both",
        help="save both label policies by default, or only the selected policy",
    )
    preprocess.add_argument("--push-to-hub", action="store_true")
    preprocess.add_argument(
        "--hub-repo-id",
        default=None,
        help="defaults to <authenticated-user>/sentiment-manifold-sst-pythia-1.4b",
    )
    preprocess.add_argument(
        "--hf-token-env",
        default="HF_TOKEN",
        help="environment variable containing the Hub token; the token is never logged",
    )
    visibility = preprocess.add_mutually_exclusive_group()
    visibility.add_argument("--public", action="store_true", help="publish a public dataset")
    visibility.add_argument("--private", action="store_true", help="publish privately (default)")

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
                prepend_bos=config.model.prepend_bos,
            )
            filtered_toy = toy.tokenizer_filtered(adapter.tokenizer)
            for split, examples, filtered_examples in (
                ("train", toy.train, filtered_toy.train),
                ("test", toy.test, filtered_toy.test),
            ):
                print(
                    f"{split}: tokenizer retained {len(filtered_examples)}/{len(examples)} "
                    "one-token adjectives with tokenizer-filtered verbs"
                )
            answer_ids = {
                label: [adapter.single_token_id(answer) for answer in answers]
                for label, answers in filtered_toy.answers.items()
            }
            print(f"answer token ids: {answer_ids}")
            print(f"runtime provenance: {json.dumps(adapter.provenance(), sort_keys=True)}")
    elif args.command == "reproduce":
        run_dir = run_reproduction(_load_with_overrides(args))
        print(f"Completed run: {run_dir}")
    elif args.command == "plot":
        for path in plot_run(args.run_dir):
            print(path)
    elif args.command == "preprocess-sst":
        hf_token = _token_from_environment(args.hf_token_env) if args.push_to_hub else None
        if args.push_to_hub and not hf_token:
            parser.error(
                f"--push-to-hub requires a token in {args.hf_token_env!r} or HF_TOKEN_PATH"
            )
        result = preprocess_sst(
            sst_root=args.sst_root,
            output_dir=args.output_dir,
            device=args.device,
            dtype=args.dtype,
            batch_size=args.batch_size,
            revision=args.revision,
            binarization=args.binarization,
            push_to_hub=args.push_to_hub,
            hub_repo_id=args.hub_repo_id,
            private=not args.public,
            hf_token=hf_token,
        )
        print(f"Saved SST datasets: {result.output_dir}")
        print(json.dumps(result.metadata["counts"], indent=2, sort_keys=True))
        if result.hub_repo_id:
            print(f"Published dataset: https://huggingface.co/datasets/{result.hub_repo_id}")


if __name__ == "__main__":
    main()
