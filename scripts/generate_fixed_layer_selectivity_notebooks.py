"""Generate the four thin Colab notebooks for fixed-layer selectivity runs."""

from __future__ import annotations

import json
from hashlib import sha1
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"

METHODS = {
    "mean_diff": {
        "number": 12,
        "title": "Mean-difference random-label selectivity",
        "filename": "colab_fixed_layer_mean_difference_selectivity",
        "description": (
            "Mean difference has no hyperparameter search. Its learned centroid direction "
            "is evaluated with the training-class midpoint decision rule."
        ),
        "settings": "MANUAL_TRIALS = []  # Mean difference has no tunable hyperparameters.",
        "override": "pass",
    },
    "logistic_regression": {
        "number": 13,
        "title": "Logistic-regression random-label selectivity",
        "filename": "colab_fixed_layer_logistic_regression_selectivity",
        "description": (
            "Three explicit logistic-regression trials are compared on real validation "
            "balanced accuracy. The selected C, penalty, and class weight are then frozen."
        ),
        "settings": dedent(
            """
            MANUAL_TRIALS = [
                {"c": 0.1, "penalty": "l2", "class_weight": None},
                {"c": 1.0, "penalty": "l2", "class_weight": None},
                {"c": 10.0, "penalty": "l2", "class_weight": None},
            ]
            """
        ).strip(),
        "override": "config.logistic_regression.trials = list(MANUAL_TRIALS)",
    },
    "das": {
        "number": 14,
        "title": "One-dimensional DAS random-label selectivity",
        "filename": "colab_fixed_layer_das_selectivity",
        "description": (
            "Three explicit DAS trials vary learning rate, weight decay, and epoch budget. "
            "Validation IIA is primary; recovery is only a tie-breaker."
        ),
        "settings": dedent(
            """
            MANUAL_TRIALS = [
                {"learning_rate": 0.0003, "weight_decay": 0.0, "epochs": 64},
                {"learning_rate": 0.001, "weight_decay": 0.0, "epochs": 64},
                {"learning_rate": 0.001, "weight_decay": 0.001, "epochs": 96},
            ]
            """
        ).strip(),
        "override": "config.das.trials = list(MANUAL_TRIALS)",
    },
    "mlp1": {
        "number": 15,
        "title": "MLP-1 random-label selectivity",
        "filename": "colab_fixed_layer_mlp1_selectivity",
        "description": (
            "Three explicit one-hidden-layer MLP trials vary hidden width, learning rate, "
            "and weight decay. Selection uses real validation balanced accuracy."
        ),
        "settings": dedent(
            """
            MANUAL_TRIALS = [
                {"hidden_size": 4, "learning_rate": 0.001, "weight_decay": 0.01},
                {"hidden_size": 10, "learning_rate": 0.001, "weight_decay": 0.01},
                {"hidden_size": 32, "learning_rate": 0.001, "weight_decay": 0.01},
            ]
            """
        ).strip(),
        "override": "config.mlp1.trials = list(MANUAL_TRIALS)",
    },
}


def markdown(source: str) -> dict:
    normalized = dedent(source).strip()
    return {
        "cell_type": "markdown",
        "id": sha1(f"markdown:{normalized}".encode()).hexdigest()[:8],
        "metadata": {},
        "source": normalized.splitlines(keepends=True),
    }


def code(source: str) -> dict:
    normalized = dedent(source).strip()
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": sha1(f"code:{normalized}".encode()).hexdigest()[:8],
        "metadata": {},
        "outputs": [],
        "source": normalized.splitlines(keepends=True),
    }


def notebook(method: str, details: dict) -> dict:
    title = details["title"]
    description = details["description"]
    settings = details["settings"]
    override = details["override"]
    settings_source = (
        dedent(
            f"""
            PROJECT_URL = "https://github.com/Adefioye/sentiment-manifold.git"
            PROJECT_REVISION = None  # Optional immutable commit or tag.
            DRIVE_ROOT = "/content/drive/MyDrive/sentiment-geometry/selectivity"
            RUN_ID = "fixed-layer-selectivity-v1"  # Use this exact value in all four notebooks.

            METHOD = "{method}"
            DEVICE = "cuda"
            DTYPE = "auto"
            RUN_EXPERIMENT = True
            SHOW_PROGRESS = True
            """
        ).strip()
        + "\n\n"
        + settings
    )
    cells = [
        markdown(
            f"""
            # {title}

            This Colab notebook runs **{method}** on ToyMovieReview and full AIT using
            last-non-padding-token residual activations. It has separate GPT-2 Small and
            Qwen3-0.6B Base sections, but both write beneath one shared Google Drive run ID.

            {description}

            Every random-label seed is paired with a real-task run using the same optimization
            seed. Labels are permuted independently within train, validation, and test while
            preserving each split's class counts. Hyperparameters are selected on the real
            validation split only; test data never affects selection. Result tables are written
            only after both the real and paired random-label fits finish. Combined plots are
            created only when all four methods and both models are present.
            """
        ),
        markdown(
            """
            ## 1. Settings

            Keep the same `RUN_ID` in all four notebooks. Change it before starting a new study.
            The configured fixed residual boundaries are:

            | Dataset | GPT-2 Small | Qwen3-0.6B Base |
            |---|---:|---:|
            | ToyMovieReview | 10 | 26 |
            | Full AIT | 11 | 26 |
            """
        ),
        code(settings_source),
        markdown(
            """
            ## 2. Install the package, mount Drive, and authenticate

            Add `HF_TOKEN` to Colab Secrets with read access to the AIT dataset. The notebook
            calls the package API; it does not reimplement extraction, fitting, randomization,
            evaluation, or persistence.
            """
        ),
        code(
            """
            import importlib
            import os
            import subprocess
            import sys
            from pathlib import Path

            from google.colab import drive, userdata

            PROJECT_ROOT = Path("/content/sentiment-manifold")
            if not (PROJECT_ROOT / ".git").is_dir():
                subprocess.run(["git", "clone", PROJECT_URL, str(PROJECT_ROOT)], check=True)
            if PROJECT_REVISION is not None:
                subprocess.run(
                    ["git", "checkout", "--detach", PROJECT_REVISION],
                    cwd=PROJECT_ROOT,
                    check=True,
                )
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-e", f"{PROJECT_ROOT}[notebooks]"],
                check=True,
            )
            os.chdir(PROJECT_ROOT)
            root_string = str(PROJECT_ROOT.resolve())
            if root_string in sys.path:
                sys.path.remove(root_string)
            sys.path.insert(0, root_string)
            for module_name in list(sys.modules):
                if module_name == "sentiment_geometry" or module_name.startswith("sentiment_geometry."):
                    del sys.modules[module_name]
            importlib.invalidate_caches()

            drive.mount("/content/drive")
            hf_token = userdata.get("HF_TOKEN")
            if not hf_token:
                raise RuntimeError("Add HF_TOKEN to Colab Secrets before running the notebook.")
            os.environ["HF_TOKEN"] = hf_token
            RUN_ROOT = Path(DRIVE_ROOT) / RUN_ID
            RUN_ROOT.mkdir(parents=True, exist_ok=True)
            print("Shared run root:", RUN_ROOT)
            """
        ),
        markdown(
            """
            ## 3. Verify the protocol and define the reusable runner

            A trial list contains complete hand-chosen configurations, not a grid. There are no
            more than three user-adjusted hyperparameters for this method. Each model run includes
            both datasets and all paired real/random-label seeds before any result table is shown.
            """
        ),
        code(
            f"""
            import pandas as pd
            from IPython.display import Image, display

            from sentiment_geometry.experiments.selectivity import (
                FixedLayerSelectivityConfig,
                run_fixed_layer_selectivity,
            )
            from sentiment_geometry.reporting import (
                combine_fixed_layer_selectivity_runs,
                load_fixed_layer_selectivity_report,
                plot_fixed_layer_selectivity,
                plot_fixed_layer_run_diagnostics,
            )

            CONFIG_PATH = PROJECT_ROOT / "configs/selectivity/fixed_layer.yaml"
            EXPECTED_LAYERS = {{
                "gpt2-small": {{"toy_movie_review": 10, "full_ait": 11}},
                "qwen-0.6b": {{"toy_movie_review": 26, "full_ait": 26}},
            }}

            def display_table(title, frame, columns=None):
                print(title)
                selected = frame
                if columns is not None:
                    selected = frame[[column for column in columns if column in frame.columns]]
                display(selected.round(4))

            def configured_run(model_name):
                config = FixedLayerSelectivityConfig.load(CONFIG_PATH)
                config.models = [model for model in config.models if model.name == model_name]
                if len(config.models) != 1:
                    raise ValueError(f"Expected exactly one config for {{model_name}}")
                model = config.models[0]
                model.device = DEVICE
                model.dtype = DTYPE
                actual_layers = {{
                    dataset: model.layer_for(dataset) for dataset in config.data.datasets
                }}
                if actual_layers != EXPECTED_LAYERS[model_name]:
                    raise ValueError(
                        f"Layer contract changed for {{model_name}}: {{actual_layers}}"
                    )
                config.methods = [METHOD]
                config.output.output_dir = str(RUN_ROOT / "methods" / METHOD)
                config.output.run_id = model_name
                config.progress.enabled = SHOW_PROGRESS
                {override}
                config.validate(require_layers=True)
                return config

            def run_and_display(model_name):
                config = configured_run(model_name)
                run_dir = Path(config.output.output_dir) / config.output.run_id
                if RUN_EXPERIMENT:
                    run_dir = run_fixed_layer_selectivity(config)
                if not (run_dir / "metrics.csv").is_file():
                    raise FileNotFoundError(f"No completed results at {{run_dir}}")

                report = load_fixed_layer_selectivity_report(run_dir)
                if not report.tuning_trials.empty:
                    display_table(
                        "Real-validation tuning trials (selected rows are marked):",
                        report.tuning_trials,
                        [
                            "dataset", "layer", "trial_index", "hyperparameters",
                            "validation_native_accuracy",
                            "validation_native_balanced_accuracy",
                            "validation_midpoint_balanced_accuracy",
                            "validation_loss", "validation_recovery",
                            "validation_logit_flip", "validation_sign_flip", "selected",
                        ],
                    )

                metric_columns = [
                    "model", "dataset", "layer", "task", "split", "n_seeds",
                    "native_balanced_accuracy_mean", "native_balanced_accuracy_std",
                    "midpoint_balanced_accuracy_mean", "midpoint_balanced_accuracy_std",
                    "prediction_agreement_mean",
                ]
                display_table(
                    "Training memorization diagnostic (real and random labels):",
                    report.training_memorization,
                    metric_columns,
                )
                display_table(
                    "All-split performance, mean and standard deviation across paired seeds:",
                    report.performance,
                    metric_columns,
                )
                display_table(
                    "Native versus midpoint gaps and prediction agreement:",
                    report.native_midpoint_comparison,
                    [
                        "model", "dataset", "layer", "task", "split", "n_seeds",
                        "accuracy_gap_midpoint_minus_native_mean",
                        "balanced_gap_midpoint_minus_native_mean",
                        "prediction_agreement_mean",
                    ],
                )
                balanced_selectivity = report.selectivity[
                    report.selectivity["metric"].isin(
                        ["native_balanced_accuracy", "midpoint_balanced_accuracy"]
                    )
                ]
                display_table(
                    "Paired selectivity, including train, validation, and test:",
                    balanced_selectivity,
                    [
                        "model", "dataset", "layer", "split", "metric",
                        "selectivity_mean", "selectivity_std", "selectivity_ci95_low",
                        "selectivity_ci95_high", "n_seeds",
                    ],
                )
                display_table(
                    "Fit diagnostics and selected training epochs:",
                    report.fit_diagnostics,
                    [
                        "model", "dataset", "layer", "task", "seed", "hyperparameters",
                        "n_parameters", "best_epoch", "epochs_completed_from_history",
                        "best_validation_loss", "best_training_loss",
                    ],
                )
                if not report.das_causal_metrics.empty:
                    display_table(
                        "DAS IIA, patched/clean/corrupted accuracy, recovery, "
                        "logit flip, and sign flip:",
                        report.das_causal_metrics,
                        [
                            "model", "dataset", "layer", "task", "split", "n_seeds",
                            "iia_mean", "iia_std", "patched_accuracy_mean",
                            "clean_accuracy_mean", "corrupted_accuracy_mean",
                            "recovery_mean", "recovery_std", "logit_flip_mean",
                            "logit_flip_std", "sign_flip_rate_mean",
                            "sign_flip_rate_std",
                        ],
                    )

                print("Per-model diagnostic plots:")
                for figure in plot_fixed_layer_run_diagnostics(run_dir):
                    print(" -", figure)
                    display(Image(filename=str(figure)))
                return run_dir
            """
        ),
        markdown("## 4. GPT-2 Small"),
        code(
            """
            GPT2_RUN_DIR = run_and_display("gpt2-small")
            print("GPT-2 artifacts:", GPT2_RUN_DIR)
            """
        ),
        markdown("## 5. Qwen3-0.6B Base"),
        code(
            """
            QWEN_RUN_DIR = run_and_display("qwen-0.6b")
            print("Qwen artifacts:", QWEN_RUN_DIR)
            """
        ),
        markdown(
            """
            ## 6. Aggregate and plot only after every notebook is complete

            This cell refuses to aggregate partial studies. Once all eight method/model folders
            exist, it writes combined CSVs and the cross-method plots under `RUN_ROOT/combined`.
            It is safe to rerun this cell after completing another method notebook.
            """
        ),
        code(
            """
            try:
                COMBINED_DIR = combine_fixed_layer_selectivity_runs(RUN_ROOT)
            except FileNotFoundError as error:
                print(error)
                print("No combined tables or plots were created from this partial study.")
            else:
                FIGURES = plot_fixed_layer_selectivity(COMBINED_DIR)
                print("Combined tables:", COMBINED_DIR)
                print("Figures:")
                for figure in FIGURES:
                    print(" -", figure)
                    display(Image(filename=str(figure)))
                display(pd.read_csv(COMBINED_DIR / "selectivity_summary.csv"))
            finally:
                os.environ.pop("HF_TOKEN", None)
                hf_token = None
            """
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": f"{details['number']:02d}_{details['filename']}.ipynb"},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    for method, details in METHODS.items():
        name = f"{details['number']:02d}_{details['filename']}.ipynb"
        destination = NOTEBOOK_DIR / name
        destination.write_text(
            json.dumps(notebook(method, details), indent=1) + "\n",
            encoding="utf-8",
        )
        print(destination.relative_to(ROOT))


if __name__ == "__main__":
    main()
