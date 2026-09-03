"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .config import MODEL_ALIASES, ReproductionConfig
from .data import (
    load_toy_movie_review,
    preprocess_ait,
    preprocess_dynasent,
    preprocess_imdb,
    preprocess_sst,
)
from .data.preprocessing.common import DEFAULT_PROMPT_TEMPLATE, PAIRING_MODEL_SPECS
from .devices import resolve_device
from .directions import list_fitters
from .experiment import run_reproduction
from .models import (
    CausalLMAdapter,
    DEFAULT_PYTHIA_FILTER_MODEL,
    PYTHIA_FILTER_MODELS,
)
from .plotting import plot_run
from .tuning import TUNABLE_METHODS, run_confirmation, run_tuning


MODEL_CHOICES = tuple(MODEL_ALIASES)


def _add_pairing_arguments(
    parser: argparse.ArgumentParser, *, include_prompt_template: bool = True
) -> None:
    parser.add_argument(
        "--pairing-model",
        action="append",
        choices=["all", *PAIRING_MODEL_SPECS],
        help=(
            "tokenizer used to build equal-full-prompt-length pairs; repeat for several "
            "models (default: all four)"
        ),
    )
    parser.add_argument(
        "--pairing-revision",
        action="append",
        default=None,
        metavar="MODEL=REVISION",
        help="optional immutable tokenizer revision; repeat per model",
    )
    if include_prompt_template:
        parser.add_argument(
            "--prompt-template",
            default=DEFAULT_PROMPT_TEMPLATE,
            help="prompt scaffold containing the literal placeholder {text}",
        )


def _add_publish_arguments(parser: argparse.ArgumentParser, default_repo: str) -> None:
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument(
        "--hub-repo-id",
        default=None,
        help=f"defaults to <authenticated-user>/{default_repo}",
    )
    parser.add_argument(
        "--hf-token-env",
        default="HF_TOKEN",
        help="environment variable containing the Hub token; the token is never logged",
    )
    visibility = parser.add_mutually_exclusive_group()
    visibility.add_argument(
        "--public",
        action="store_true",
        help="explicitly opt into a public dataset repository",
    )
    visibility.add_argument("--private", action="store_true", help="publish privately (default)")


def _add_correctness_filter_arguments(
    parser: argparse.ArgumentParser, *, revision_alias: bool = False
) -> None:
    parser.add_argument(
        "--filter-model",
        choices=list(PYTHIA_FILTER_MODELS),
        default=DEFAULT_PYTHIA_FILTER_MODEL,
        help="Pythia model used for zero-shot correctness selection (default: pythia-2.8b)",
    )
    revision_flags = ("--filter-revision", "--revision") if revision_alias else ("--filter-revision",)
    parser.add_argument(
        *revision_flags,
        dest="filter_revision",
        default=None,
        help="optional immutable revision of the Pythia correctness-filter model",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument(
        "--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto"
    )
    parser.add_argument("--batch-size", type=int, default=16)


def _publish_token_or_error(parser: argparse.ArgumentParser, args) -> str | None:
    token = _token_from_environment(args.hf_token_env) if args.push_to_hub else None
    if args.push_to_hub and not token:
        parser.error(f"--push-to-hub requires a token in {args.hf_token_env!r} or HF_TOKEN_PATH")
    return token


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
    if getattr(args, "checkpoint_dir", None):
        config.experiment.checkpoint_dir = str(Path(args.checkpoint_dir).resolve())
    if getattr(args, "seed", None) is not None:
        config.seed = int(args.seed)
    methods = getattr(args, "method", None)
    if methods:
        config.experiment.methods = [methods] if isinstance(methods, str) else list(methods)
    if getattr(args, "layer", None):
        config.experiment.layers = list(args.layer)
    if getattr(args, "kmeans_n_init", None) is not None:
        config.fitting.kmeans_n_init = int(args.kmeans_n_init)
    if getattr(args, "logistic_c", None) is not None:
        config.fitting.logistic_c = float(args.logistic_c)
    if getattr(args, "logistic_max_iter", None) is not None:
        config.fitting.logistic_max_iter = int(args.logistic_max_iter)
    if getattr(args, "logistic_tol", None) is not None:
        config.fitting.logistic_tol = float(args.logistic_tol)
    if getattr(args, "das_learning_rate", None) is not None:
        config.das.learning_rate = float(args.das_learning_rate)
    if getattr(args, "das_weight_decay", None) is not None:
        config.das.weight_decay = float(args.das_weight_decay)
    if getattr(args, "das_epochs", None) is not None:
        config.das.epochs = int(args.das_epochs)
    if getattr(args, "das_batch_size", None) is not None:
        config.das.batch_size = int(args.das_batch_size)
    if getattr(args, "das_max_grad_norm", None) is not None:
        config.das.max_grad_norm = float(args.das_max_grad_norm)
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
    reproduce.add_argument("--model", choices=MODEL_CHOICES, default=None)
    reproduce.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default=None)
    reproduce.add_argument("--output-dir", default=None)
    reproduce.add_argument("--checkpoint-dir", default=None)
    reproduce.add_argument("--seed", type=int, default=None)
    reproduce.add_argument(
        "--method",
        action="append",
        choices=list_fitters(),
        help="run only this method; repeat to run more than one",
    )
    reproduce.add_argument(
        "--layer",
        action="append",
        type=int,
        help="run only this residual boundary; repeat to run more than one",
    )
    reproduce.add_argument("--kmeans-n-init", type=int, default=None)
    reproduce.add_argument("--logistic-c", type=float, default=None)
    reproduce.add_argument("--logistic-max-iter", type=int, default=None)
    reproduce.add_argument("--logistic-tol", type=float, default=None)
    reproduce.add_argument("--das-learning-rate", type=float, default=None)
    reproduce.add_argument("--das-weight-decay", type=float, default=None)
    reproduce.add_argument("--das-epochs", type=int, default=None)
    reproduce.add_argument("--das-batch-size", type=int, default=None)
    reproduce.add_argument("--das-max-grad-norm", type=float, default=None)
    reproduce.add_argument(
        "--with-openwebtext-resample-ablation",
        action="store_true",
        help="run the optional exploratory OpenWebText loss diagnostic",
    )
    reproduce.add_argument("--with-openwebtext", action="store_true", help=argparse.SUPPRESS)

    plot_parser = subparsers.add_parser("plot", help="render plots from a completed run")
    plot_parser.add_argument("--run-dir", required=True)

    tune_parser = subparsers.add_parser(
        "tune",
        help="select one method's layer and hyperparameters on Toy training-validation only",
    )
    tune_parser.add_argument("--config", default="configs/tuning.yaml")
    tune_parser.add_argument("--model", choices=MODEL_CHOICES, default=None)
    tune_parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default=None)
    tune_parser.add_argument("--output-dir", default=None)
    tune_parser.add_argument("--checkpoint-dir", default=None)
    tune_parser.add_argument("--seed", type=int, default=None)
    tune_parser.add_argument("--method", choices=TUNABLE_METHODS, required=True)
    tune_parser.add_argument(
        "--layer",
        action="append",
        type=int,
        help="tune only this residual boundary; repeat to tune more than one",
    )

    confirm_parser = subparsers.add_parser(
        "confirm",
        help="refit one validation-selected configuration and evaluate locked test/OOD data",
    )
    confirm_parser.add_argument("--config", default="configs/reproduction.yaml")
    confirm_parser.add_argument("--selection", required=True)
    confirm_parser.add_argument("--method", choices=TUNABLE_METHODS, default=None)
    confirm_parser.add_argument("--model", choices=MODEL_CHOICES, default=None)
    confirm_parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default=None)
    confirm_parser.add_argument("--output-dir", required=True)
    confirm_parser.add_argument("--checkpoint-dir", default=None)
    confirm_parser.add_argument("--seed", type=int, default=None)

    preprocess = subparsers.add_parser(
        "preprocess-sst",
        help="build Pythia-correctness-filtered SST datasets and optionally publish them",
    )
    preprocess.add_argument(
        "--sst-root",
        default="../eliciting-latent-sentiment/stanfordSentimentTreebank",
    )
    preprocess.add_argument("--output-dir", default="data/processed/sst-pythia-2.8b")
    _add_correctness_filter_arguments(preprocess, revision_alias=True)
    preprocess.add_argument(
        "--binarization",
        choices=["both", "tigges", "neutral_removed"],
        default="both",
        help="save both label policies by default, or only the selected policy",
    )
    _add_pairing_arguments(preprocess, include_prompt_template=False)
    _add_publish_arguments(preprocess, "sentiment-manifold-sst-pythia-2.8b")

    ait = subparsers.add_parser(
        "preprocess-ait",
        help="build binary AIT Valence-oc datasets and equal-length prompt pairs",
    )
    ait.add_argument("--ait-root", default=None)
    ait.add_argument("--train-file", default=None)
    ait.add_argument("--validation-file", default=None)
    ait.add_argument("--test-file", default=None)
    ait.add_argument("--output-dir", default="data/processed/ait-valence-binary")
    ait.add_argument(
        "--pairing-split",
        action="append",
        choices=["train", "validation", "test"],
        help="split to pair; repeat as needed (default: all labeled splits)",
    )
    _add_pairing_arguments(ait)
    _add_publish_arguments(ait, "sentiment-manifold-ait-valence-binary")

    imdb = subparsers.add_parser(
        "preprocess-imdb",
        help="build IMDb binary datasets and equal-length prompt pairs",
    )
    imdb.add_argument("--dataset-name", default="stanfordnlp/imdb")
    imdb.add_argument("--dataset-revision", default=None)
    imdb.add_argument("--output-dir", default="data/processed/imdb-pythia-2.8b")
    _add_correctness_filter_arguments(imdb)
    imdb.add_argument(
        "--pairing-split",
        action="append",
        choices=["train", "validation", "test"],
        help="split to pair; repeat as needed (default: test)",
    )
    _add_pairing_arguments(imdb)
    _add_publish_arguments(imdb, "sentiment-manifold-imdb-pythia-2.8b")

    dynasent = subparsers.add_parser(
        "preprocess-dynasent",
        help="build DynaSent R1/R2 binary datasets and equal-length prompt pairs",
    )
    dynasent.add_argument("--dynasent-root", required=True)
    dynasent.add_argument(
        "--output-dir", default="data/processed/dynasent-r1-r2-pythia-2.8b"
    )
    _add_correctness_filter_arguments(dynasent)
    dynasent.add_argument(
        "--round",
        action="append",
        type=int,
        choices=[1, 2],
        help="round to include; repeat for both (default: both)",
    )
    dynasent.add_argument(
        "--pairing-split",
        action="append",
        choices=["train", "validation", "test"],
        help="split to pair; repeat as needed (default: test)",
    )
    _add_pairing_arguments(dynasent)
    _add_publish_arguments(dynasent, "sentiment-manifold-dynasent-r1-r2-pythia-2.8b")

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
    elif args.command == "tune":
        run_dir = run_tuning(_load_with_overrides(args), args.method)
        print(f"Completed validation tuning: {run_dir}")
    elif args.command == "confirm":
        run_dir = run_confirmation(
            _load_with_overrides(args),
            args.selection,
            method=args.method,
            confirmation_seed=args.seed,
        )
        print(f"Completed frozen confirmation: {run_dir}")
    elif args.command == "preprocess-sst":
        hf_token = _publish_token_or_error(parser, args)
        result = preprocess_sst(
            sst_root=args.sst_root,
            output_dir=args.output_dir,
            device=args.device,
            dtype=args.dtype,
            batch_size=args.batch_size,
            filter_revision=args.filter_revision,
            filter_model=args.filter_model,
            binarization=args.binarization,
            pairing_models=args.pairing_model,
            pairing_revisions=args.pairing_revision,
            push_to_hub=args.push_to_hub,
            hub_repo_id=args.hub_repo_id,
            private=not args.public,
            hf_token=hf_token,
        )
        print(f"Saved SST datasets: {result.output_dir}")
        print(json.dumps(result.metadata["counts"], indent=2, sort_keys=True))
        if result.hub_repo_id:
            print(f"Published dataset: https://huggingface.co/datasets/{result.hub_repo_id}")
    elif args.command == "preprocess-ait":
        hf_token = _publish_token_or_error(parser, args)
        explicit_files = {
            split: path
            for split, path in (
                ("train", args.train_file),
                ("validation", args.validation_file),
                ("test", args.test_file),
            )
            if path
        }
        if explicit_files and len(explicit_files) != 3:
            parser.error("provide all three AIT split files, or use --ait-root for discovery")
        result = preprocess_ait(
            ait_root=args.ait_root,
            files=explicit_files or None,
            output_dir=args.output_dir,
            pairing_models=args.pairing_model,
            pairing_revisions=args.pairing_revision,
            pairing_splits=args.pairing_split or ("train", "validation", "test"),
            prompt_template=args.prompt_template,
            push_to_hub=args.push_to_hub,
            hub_repo_id=args.hub_repo_id,
            private=not args.public,
            hf_token=hf_token,
        )
        print(f"Saved AIT datasets: {result.output_dir}")
        print(json.dumps(result.metadata["counts"], indent=2, sort_keys=True))
        if result.hub_repo_id:
            print(f"Published dataset: https://huggingface.co/datasets/{result.hub_repo_id}")
    elif args.command == "preprocess-imdb":
        hf_token = _publish_token_or_error(parser, args)
        result = preprocess_imdb(
            output_dir=args.output_dir,
            dataset_name=args.dataset_name,
            dataset_revision=args.dataset_revision,
            pairing_models=args.pairing_model,
            pairing_revisions=args.pairing_revision,
            pairing_splits=args.pairing_split or ("test",),
            prompt_template=args.prompt_template,
            filter_model=args.filter_model,
            filter_revision=args.filter_revision,
            device=args.device,
            dtype=args.dtype,
            batch_size=args.batch_size,
            push_to_hub=args.push_to_hub,
            hub_repo_id=args.hub_repo_id,
            private=not args.public,
            hf_token=hf_token,
        )
        print(f"Saved IMDb datasets: {result.output_dir}")
        print(json.dumps(result.metadata["counts"], indent=2, sort_keys=True))
        if result.hub_repo_id:
            print(f"Published dataset: https://huggingface.co/datasets/{result.hub_repo_id}")
    elif args.command == "preprocess-dynasent":
        hf_token = _publish_token_or_error(parser, args)
        result = preprocess_dynasent(
            dynasent_root=args.dynasent_root,
            output_dir=args.output_dir,
            rounds=args.round or (1, 2),
            pairing_models=args.pairing_model,
            pairing_revisions=args.pairing_revision,
            pairing_splits=args.pairing_split or ("test",),
            prompt_template=args.prompt_template,
            filter_model=args.filter_model,
            filter_revision=args.filter_revision,
            device=args.device,
            dtype=args.dtype,
            batch_size=args.batch_size,
            push_to_hub=args.push_to_hub,
            hub_repo_id=args.hub_repo_id,
            private=not args.public,
            hf_token=hf_token,
        )
        print(f"Saved DynaSent datasets: {result.output_dir}")
        print(json.dumps(result.metadata["counts"], indent=2, sort_keys=True))
        if result.hub_repo_id:
            print(f"Published dataset: https://huggingface.co/datasets/{result.hub_repo_id}")


if __name__ == "__main__":
    main()
