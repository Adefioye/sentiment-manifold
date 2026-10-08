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
            "Ten explicit DAS trials vary only learning rate across two orders of magnitude. "
            "A second five-trial sweep then varies only the epoch budget at each dataset's "
            "selected learning rate. Validation IIA remains the primary selection metric."
        ),
        "tuning_note": (
            "**Two-stage tuning note.** The first sweep holds every trial at 64 epochs so learning "
            "rate is the only changing factor. The next section freezes the separately selected "
            "ToyMovieReview and full-AIT learning rates, then compares epoch budgets of 32, 64, "
            "96, 128, and 160 using validation IIA only. The original learning-rate artifacts are "
            "preserved."
        ),
        "settings": dedent(
            """
            MANUAL_TRIALS = [
                {"learning_rate": 0.0001, "weight_decay": 0.0, "epochs": 64},
                {"learning_rate": 0.0002, "weight_decay": 0.0, "epochs": 64},
                {"learning_rate": 0.0003, "weight_decay": 0.0, "epochs": 64},
                {"learning_rate": 0.0005, "weight_decay": 0.0, "epochs": 64},
                {"learning_rate": 0.0007, "weight_decay": 0.0, "epochs": 64},
                {"learning_rate": 0.001, "weight_decay": 0.0, "epochs": 64},
                {"learning_rate": 0.002, "weight_decay": 0.0, "epochs": 64},
                {"learning_rate": 0.003, "weight_decay": 0.0, "epochs": 64},
                {"learning_rate": 0.005, "weight_decay": 0.0, "epochs": 64},
                {"learning_rate": 0.01, "weight_decay": 0.0, "epochs": 64},
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
    tuning_note = details.get("tuning_note", "")
    settings = details["settings"]
    override = details["override"]
    selection_example = {
        "mean_diff": "# Mean difference has no hyperparameters to override.",
        "logistic_regression": (
            '# GPT2_SELECTIONS["gpt2-small"]["toy_movie_review"][METHOD]["c"] = 1.0'
        ),
        "das": (
            '# GPT2_SELECTIONS["gpt2-small"]["toy_movie_review"][METHOD]["learning_rate"] = 0.001'
        ),
        "mlp1": ('# GPT2_SELECTIONS["gpt2-small"]["toy_movie_review"][METHOD]["hidden_size"] = 10'),
    }[method]
    qwen_selection_example = selection_example.replace("GPT2", "QWEN").replace(
        '"gpt2-small"', '"qwen-0.6b"'
    )
    das_settings = ""
    if method == "das":
        das_settings = dedent(
            """
            # GPT-2 learning-rate tuning is already complete on Drive. Keep this False.
            RUN_LEARNING_RATE_TUNING = {
                "gpt2-small": False,
                "qwen-0.6b": True,
            }
            RUN_DAS_EPOCH_TUNING = True
            DAS_EPOCH_BUDGETS = [32, 64, 96, 128, 160]
            """
        ).strip()
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
            RUN_TUNING = True
            RUN_FINAL_TRAINING = True
            SHOW_PROGRESS = True
            HF_TOKEN_SOURCE = "prompt"  # "prompt" (hidden entry) or "colab_secret".
            """
        ).strip()
        + "\n\n"
        + settings
        + (("\n\n" + das_settings) if das_settings else "")
    )
    gpt2_tuning_call = 'tuning_and_display("gpt2-small")'
    qwen_tuning_call = 'tuning_and_display("qwen-0.6b")'
    gpt2_epoch_cells: list[dict] = []
    qwen_epoch_cells: list[dict] = []
    gpt2_inspect_number = "5b"
    gpt2_training_number = "5c"
    qwen_inspect_number = "6b"
    qwen_training_number = "6c"
    if method == "das":
        gpt2_tuning_call = (
            'tuning_and_display("gpt2-small", run_tuning=RUN_LEARNING_RATE_TUNING["gpt2-small"])'
        )
        qwen_tuning_call = (
            'tuning_and_display("qwen-0.6b", run_tuning=RUN_LEARNING_RATE_TUNING["qwen-0.6b"])'
        )
        gpt2_inspect_number = "5c"
        gpt2_training_number = "5d"
        qwen_inspect_number = "6c"
        qwen_training_number = "6d"
        gpt2_epoch_cells = [
            markdown(
                """
                ### 5b. Tune the epoch budget at GPT-2's selected learning rates

                This stage does **not** rerun the completed learning-rate sweep. It loads the
                saved, dataset-specific learning rates and varies only the epoch budget over
                `32, 64, 96, 128, 160`. Selection remains validation-only. Results are saved after
                each dataset in separate `das_epoch_*` artifacts, so the original learning-rate
                trials remain intact.
                """
            ),
            code(
                """
                GPT2_EPOCH_TUNING_DIR, GPT2_SELECTIONS = das_epoch_tuning_and_display(
                    "gpt2-small", GPT2_SELECTIONS
                )
                print("GPT-2 epoch-tuning artifacts:", GPT2_EPOCH_TUNING_DIR)
                """
            ),
        ]
        qwen_epoch_cells = [
            markdown(
                """
                ### 6b. Tune the epoch budget at Qwen's selected learning rates

                Run this after Qwen's learning-rate sweep finishes. It freezes the separately
                selected ToyMovieReview and full-AIT learning rates and varies only the epoch
                budget over `32, 64, 96, 128, 160`, using validation IIA for selection.
                """
            ),
            code(
                """
                QWEN_EPOCH_TUNING_DIR, QWEN_SELECTIONS = das_epoch_tuning_and_display(
                    "qwen-0.6b", QWEN_SELECTIONS
                )
                print("Qwen epoch-tuning artifacts:", QWEN_EPOCH_TUNING_DIR)
                """
            ),
        ]
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
            validation split only; test data never affects selection. Each model section first
            finishes tuning, displays the trial table and tuning plot, and freezes the selection.
            Paired real/random training starts in a separate cell. Final result tables are written
            only after both paired tasks finish. Combined plots are created only when all four
            methods and both models are present.
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
            ## 2. Install the package, verify imports, and mount Drive

            The notebook imports the editable checkout, verifies its filesystem location, imports
            every package module, and checks the public APIs used below. This catches stale Colab
            modules before a long run starts. The notebook calls package APIs rather than
            reimplementing extraction, fitting, randomization, evaluation, or persistence.
            """
        ),
        code(
            """
            import importlib
            import os
            import pkgutil
            import subprocess
            import sys
            from pathlib import Path

            from google.colab import drive

            PROJECT_ROOT = Path("/content/sentiment-manifold")
            if not (PROJECT_ROOT / ".git").is_dir():
                subprocess.run(["git", "clone", PROJECT_URL, str(PROJECT_ROOT)], check=True)
            if PROJECT_REVISION is not None:
                subprocess.run(
                    ["git", "checkout", "--detach", PROJECT_REVISION],
                    cwd=PROJECT_ROOT,
                    check=True,
                )
            project_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
            ).strip()
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

            sentiment_geometry = importlib.import_module("sentiment_geometry")
            expected_package_root = (PROJECT_ROOT / "sentiment_geometry").resolve()
            imported_package_root = Path(sentiment_geometry.__file__).resolve().parent
            if imported_package_root != expected_package_root:
                raise ImportError(
                    f"Imported sentiment_geometry from {imported_package_root}, "
                    f"expected {expected_package_root}. Restart the runtime and rerun from the top."
                )
            module_names = sorted(
                module.name
                for module in pkgutil.walk_packages(
                    sentiment_geometry.__path__, prefix="sentiment_geometry."
                )
                if module.name != "sentiment_geometry.__main__"
            )
            for module_name in module_names:
                importlib.import_module(module_name)
            required_apis = {
                "sentiment_geometry.experiments.selectivity": {
                    "FixedLayerSelectivityConfig",
                    "run_fixed_layer_selectivity_with_frozen_hyperparameters",
                    "tune_fixed_layer_das_epoch_budgets",
                    "tune_fixed_layer_selectivity",
                },
                "sentiment_geometry.reporting": {
                    "combine_fixed_layer_selectivity_runs",
                    "load_fixed_layer_das_epoch_selections",
                    "load_fixed_layer_das_epoch_tuning_trials",
                    "load_fixed_layer_hyperparameter_selections",
                    "load_fixed_layer_tuning_trials",
                    "load_fixed_layer_selectivity_report",
                    "plot_fixed_layer_selectivity",
                    "plot_fixed_layer_das_epoch_tuning",
                    "plot_fixed_layer_hyperparameter_tuning",
                    "plot_fixed_layer_run_diagnostics",
                },
            }
            for module_name, api_names in required_apis.items():
                module = importlib.import_module(module_name)
                missing_apis = sorted(name for name in api_names if not hasattr(module, name))
                if missing_apis:
                    raise ImportError(f"{module_name} is missing {missing_apis}.")

            drive.mount("/content/drive")
            RUN_ROOT = Path(DRIVE_ROOT) / RUN_ID
            RUN_ROOT.mkdir(parents=True, exist_ok=True)
            print("Project commit:", project_commit)
            print("Imported package from:", imported_package_root)
            print(f"Imported and checked {len(module_names)} package modules.")
            print("Shared run root:", RUN_ROOT)
            """
        ),
        markdown(
            """
            ## 3. Authenticate to Hugging Face

            The default `HF_TOKEN_SOURCE = "prompt"` asks for the token through a hidden manual
            prompt, matching the earlier notebooks. To use Colab Secrets instead, add a secret
            named `HF_TOKEN` and set `HF_TOKEN_SOURCE = "colab_secret"`. The token is kept only in
            runtime memory, is never printed or written to Drive, and is placed in the environment
            only while a model run is actively loading AIT data.
            """
        ),
        code(
            """
            import gc
            from getpass import getpass

            from huggingface_hub import HfApi
            from huggingface_hub.utils import reset_sessions

            _RUNTIME_SECRETS = {}

            def get_runtime_secret(name):
                if name in _RUNTIME_SECRETS:
                    return _RUNTIME_SECRETS[name]
                if HF_TOKEN_SOURCE == "prompt":
                    value = getpass(f"Enter {name} (input hidden): ").strip()
                elif HF_TOKEN_SOURCE == "colab_secret":
                    from google.colab import userdata

                    value = (userdata.get(name) or "").strip()
                else:
                    raise ValueError("HF_TOKEN_SOURCE must be 'prompt' or 'colab_secret'.")
                if not value:
                    raise RuntimeError(f"{name} was not provided.")
                _RUNTIME_SECRETS[name] = value
                return value

            def release_hf_environment(env_name="HF_TOKEN"):
                os.environ.pop(env_name, None)
                reset_sessions()
                gc.collect()

            def clear_hf_credentials(env_name="HF_TOKEN"):
                release_hf_environment(env_name)
                value = _RUNTIME_SECRETS.pop("HF_TOKEN", None)
                if value is not None:
                    del value

            _token = get_runtime_secret("HF_TOKEN")
            try:
                hf_account = HfApi(token=_token).whoami()["name"]
            except BaseException:
                clear_hf_credentials()
                raise
            finally:
                del _token
            print(
                f"Authenticated to Hugging Face as {hf_account}. "
                "Token value was not displayed."
            )
            """
        ),
        markdown(
            """
            ## 4. Verify the protocol and define the staged runners

            A trial list contains complete hand-chosen configurations, not a grid. There are no
            more than three user-adjusted hyperparameters for this method. The tuning runner stops
            after validation-only selection and visualization. The final runner starts separately
            and uses exactly the frozen settings shown between the two stages.
            """
        ),
        code(
            f"""
            import pandas as pd
            from IPython.display import Image, display

            from sentiment_geometry.experiments.selectivity import (
                FixedLayerSelectivityConfig,
                run_fixed_layer_selectivity_with_frozen_hyperparameters,
                tune_fixed_layer_das_epoch_budgets,
                tune_fixed_layer_selectivity,
            )
            from sentiment_geometry.reporting import (
                combine_fixed_layer_selectivity_runs,
                load_fixed_layer_das_epoch_selections,
                load_fixed_layer_das_epoch_tuning_trials,
                load_fixed_layer_hyperparameter_selections,
                load_fixed_layer_tuning_trials,
                load_fixed_layer_selectivity_report,
                plot_fixed_layer_selectivity,
                plot_fixed_layer_das_epoch_tuning,
                plot_fixed_layer_hyperparameter_tuning,
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

            def tuning_and_display(model_name, run_tuning=RUN_TUNING):
                config = configured_run(model_name)
                run_dir = Path(config.output.output_dir) / config.output.run_id
                if run_tuning:
                    token_env = config.data.hf_token_env
                    os.environ[token_env] = get_runtime_secret("HF_TOKEN")
                    try:
                        run_dir = tune_fixed_layer_selectivity(config)
                    finally:
                        release_hf_environment(token_env)
                selections = load_fixed_layer_hyperparameter_selections(run_dir)
                try:
                    trials = load_fixed_layer_tuning_trials(run_dir)
                except FileNotFoundError:
                    print("This method has no hyperparameter trials to compare.")
                else:
                    display_table(
                        "Real-validation tuning trials (selected rows are marked):",
                        trials,
                        [
                            "dataset", "layer", "trial_index", "hyperparameters",
                            "validation_native_accuracy",
                            "validation_native_balanced_accuracy",
                            "validation_midpoint_balanced_accuracy",
                            "validation_loss", "validation_recovery",
                            "validation_logit_flip", "validation_sign_flip", "selected",
                            "best_epoch", "epoch_budget", "best_epoch_near_budget",
                        ],
                    )
                    tuning_figure = plot_fixed_layer_hyperparameter_tuning(run_dir)
                    print("Validation-only tuning plot:", tuning_figure)
                    display(Image(filename=str(tuning_figure)))
                print("Frozen settings (inspect before starting final training):")
                display(pd.json_normalize(selections, sep=" → ").T.rename(columns={{0: "value"}}))
                return run_dir, selections

            def das_epoch_tuning_and_display(model_name, learning_rate_selections):
                if METHOD != "das":
                    raise ValueError("Epoch-budget tuning is only defined for DAS.")
                config = configured_run(model_name)
                run_dir = Path(config.output.output_dir) / config.output.run_id
                if RUN_DAS_EPOCH_TUNING:
                    token_env = config.data.hf_token_env
                    os.environ[token_env] = get_runtime_secret("HF_TOKEN")
                    try:
                        run_dir = tune_fixed_layer_das_epoch_budgets(
                            config,
                            learning_rate_selections,
                            DAS_EPOCH_BUDGETS,
                        )
                    finally:
                        release_hf_environment(token_env)
                selections = load_fixed_layer_das_epoch_selections(run_dir)
                trials = load_fixed_layer_das_epoch_tuning_trials(run_dir)
                display_table(
                    "Validation-only epoch-budget trials (selected rows are marked):",
                    trials,
                    [
                        "dataset", "layer", "trial_index", "hyperparameters",
                        "validation_native_accuracy",
                        "validation_native_balanced_accuracy",
                        "validation_loss", "validation_recovery",
                        "validation_logit_flip", "validation_sign_flip", "selected",
                        "best_epoch", "epoch_budget", "best_epoch_near_budget",
                    ],
                )
                figure = plot_fixed_layer_das_epoch_tuning(run_dir)
                print("Validation-only DAS epoch-budget plot:", figure)
                display(Image(filename=str(figure)))
                print("Frozen learning-rate plus epoch selections:")
                display(pd.json_normalize(selections, sep=" → ").T.rename(columns={{0: "value"}}))
                return run_dir, selections

            def final_training_and_display(model_name, selections):
                config = configured_run(model_name)
                run_dir = Path(config.output.output_dir) / config.output.run_id
                if RUN_FINAL_TRAINING:
                    token_env = config.data.hf_token_env
                    os.environ[token_env] = get_runtime_secret("HF_TOKEN")
                    try:
                        run_dir = run_fixed_layer_selectivity_with_frozen_hyperparameters(
                            config, selections
                        )
                    finally:
                        release_hf_environment(token_env)
                if not (run_dir / "metrics.csv").is_file():
                    raise FileNotFoundError(f"No completed final results at {{run_dir}}")

                report = load_fixed_layer_selectivity_report(run_dir)

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
        markdown(
            f"""
            ## 5. GPT-2 Small

            ### 5a. Tune on real validation data and visualize every trial

            This stage does not train either final real-label or random-label model.

            {tuning_note}
            """
        ),
        code(
            f"""
            GPT2_TUNING_DIR, GPT2_SELECTIONS = {gpt2_tuning_call}
            print("GPT-2 tuning artifacts:", GPT2_TUNING_DIR)
            """
        ),
        *gpt2_epoch_cells,
        markdown(
            f"""
            ### {gpt2_inspect_number}. Inspect or deliberately override the frozen GPT-2 settings

            The automatically selected values are already frozen in `GPT2_SELECTIONS`. If you
            deliberately override one, edit the nested dictionary here before final training and
            document the reason. Do not consult test performance when making that decision.
            """
        ),
        code(
            f"""
            # Example only (leave commented for validation-selected settings):
            {selection_example}
            display(pd.json_normalize(GPT2_SELECTIONS, sep=" → ").T.rename(columns={{0: "value"}}))
            """
        ),
        markdown(
            f"""
            ### {gpt2_training_number}. Train paired real/random-label tasks with the frozen GPT-2 settings

            Final metrics and diagnostic plots appear only after every paired seed finishes.
            """
        ),
        code(
            """
            GPT2_RUN_DIR = final_training_and_display("gpt2-small", GPT2_SELECTIONS)
            print("GPT-2 final artifacts:", GPT2_RUN_DIR)
            """
        ),
        markdown(
            f"""
            ## 6. Qwen3-0.6B Base

            ### 6a. Tune on real validation data and visualize every trial

            This repeats the complete selection process independently for Qwen and both datasets.

            {tuning_note}
            """
        ),
        code(
            f"""
            QWEN_TUNING_DIR, QWEN_SELECTIONS = {qwen_tuning_call}
            print("Qwen tuning artifacts:", QWEN_TUNING_DIR)
            """
        ),
        *qwen_epoch_cells,
        markdown(
            f"""
            ### {qwen_inspect_number}. Inspect or deliberately override the frozen Qwen settings

            Leave the dictionary unchanged for the automatic validation-selected workflow.
            """
        ),
        code(
            f"""
            # Example only (leave commented for validation-selected settings):
            {qwen_selection_example}
            display(pd.json_normalize(QWEN_SELECTIONS, sep=" → ").T.rename(columns={{0: "value"}}))
            """
        ),
        markdown(
            f"""
            ### {qwen_training_number}. Train paired real/random-label tasks with the frozen Qwen settings

            Final metrics and diagnostic plots appear only after every paired seed finishes.
            """
        ),
        code(
            """
            QWEN_RUN_DIR = final_training_and_display("qwen-0.6b", QWEN_SELECTIONS)
            print("Qwen final artifacts:", QWEN_RUN_DIR)
            """
        ),
        markdown(
            """
            ## 7. Aggregate and plot only after every notebook is complete

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
                clear_hf_credentials()
                if os.environ.get("HF_TOKEN") is not None:
                    raise RuntimeError("HF_TOKEN remains in the notebook environment.")
                if "HF_TOKEN" in _RUNTIME_SECRETS:
                    raise RuntimeError("HF_TOKEN remains in the runtime secret cache.")
                print("Verified: HF_TOKEN was cleared from runtime memory and the environment.")
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
