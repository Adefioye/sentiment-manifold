"""Generate the read-only cross-method fixed-layer selectivity analysis notebook."""

from __future__ import annotations

import json
from hashlib import sha1
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).parents[1]
DESTINATION = ROOT / "notebooks/16_colab_analyze_fixed_layer_selectivity.ipynb"


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


def notebook() -> dict:
    cells = [
        markdown(
            """
            # Cross-method random-label selectivity analysis

            This read-only notebook combines the completed mean-difference, logistic-regression,
            DAS, and MLP-1 runs for GPT-2 Small and Qwen3-0.6B Base. It does not tune or train any
            probe and does not require a Hugging Face token.

            The primary analysis follows the selectivity presentation from Hewitt and Liang:
            each method has a real-task accuracy point and a random-label accuracy point, joined
            by a vertical segment. The segment is selectivity, `real accuracy − random-label
            accuracy`. Native and midpoint decision rules are kept separate.
            """
        ),
        markdown(
            """
            ## 1. Settings

            Use the same Drive root and run ID as Notebooks 12–15. Held-out test accuracy is the
            primary result. Set `USE_BALANCED_ACCURACY_FOR_PLOTS = True` for a class-imbalance
            sensitivity plot; both ordinary and balanced accuracy are always retained in the
            saved tables.
            """
        ),
        code(
            """
            PROJECT_URL = "https://github.com/Adefioye/sentiment-manifold.git"
            PROJECT_REVISION = None  # Optional immutable commit or tag.
            DRIVE_ROOT = "/content/drive/MyDrive/sentiment-geometry/selectivity"
            RUN_ID = "fixed-layer-selectivity-v1"
            ANALYSIS_SPLIT = "test"
            USE_BALANCED_ACCURACY_FOR_PLOTS = False
            """
        ),
        markdown(
            """
            ## 2. Install the reporting package and mount Drive

            This installs the editable project checkout and verifies only the reporting APIs used
            below. No model is loaded and no experiment output is overwritten.
            """
        ),
        code(
            """
            import importlib
            import os
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
                if module_name == "sentiment_geometry" or module_name.startswith(
                    "sentiment_geometry."
                ):
                    del sys.modules[module_name]
            importlib.invalidate_caches()

            reporting = importlib.import_module("sentiment_geometry.reporting")
            required = {
                "build_fixed_layer_selectivity_accuracy_tables",
                "combine_fixed_layer_selectivity_runs",
                "plot_fixed_layer_selectivity_accuracy_gaps",
                "save_fixed_layer_selectivity_accuracy_tables",
            }
            missing = sorted(name for name in required if not hasattr(reporting, name))
            if missing:
                raise ImportError(f"sentiment_geometry.reporting is missing {missing}.")

            drive.mount("/content/drive")
            RUN_ROOT = Path(DRIVE_ROOT) / RUN_ID
            if not RUN_ROOT.is_dir():
                raise FileNotFoundError(f"No selectivity run directory at {RUN_ROOT}")
            print("Project commit:", project_commit)
            print("Shared run root:", RUN_ROOT)
            """
        ),
        markdown(
            """
            ## 3. Combine the eight completed method/model runs

            Aggregation is deliberately strict: it stops if any method/model run lacks its final
            metrics or selectivity summary. This prevents a partial study from looking complete.
            The combined CSV files are written under the same Drive run ID in `combined/`.
            """
        ),
        code(
            """
            from sentiment_geometry.reporting import combine_fixed_layer_selectivity_runs

            COMBINED_DIR = combine_fixed_layer_selectivity_runs(RUN_ROOT)
            print("Combined artifacts:", COMBINED_DIR)
            """
        ),
        markdown(
            """
            ## 4. Native-rule accuracy and selectivity table

            Each row is one model/dataset/method combination on the held-out split. Means and
            standard deviations are computed across paired seeds. The 95% confidence interval is
            computed from the paired per-seed selectivity differences, not by subtracting two
            independently aggregated intervals.
            """
        ),
        code(
            """
            import pandas as pd
            from IPython.display import Image, display

            from sentiment_geometry.reporting import (
                build_fixed_layer_selectivity_accuracy_tables,
                plot_fixed_layer_selectivity_accuracy_gaps,
                save_fixed_layer_selectivity_accuracy_tables,
            )

            NATIVE_TABLE, MIDPOINT_TABLE = build_fixed_layer_selectivity_accuracy_tables(
                COMBINED_DIR,
                split=ANALYSIS_SPLIT,
            )
            TABLE_PATHS = save_fixed_layer_selectivity_accuracy_tables(
                COMBINED_DIR,
                split=ANALYSIS_SPLIT,
            )

            TABLE_COLUMNS = [
                "model",
                "dataset",
                "layer",
                "method_label",
                "accuracy_real_mean",
                "accuracy_real_std",
                "accuracy_random_mean",
                "accuracy_random_std",
                "accuracy_selectivity_mean",
                "accuracy_selectivity_ci95_low",
                "accuracy_selectivity_ci95_high",
                "balanced_accuracy_real_mean",
                "balanced_accuracy_random_mean",
                "balanced_accuracy_selectivity_mean",
                "accuracy_n_seeds",
            ]
            print("Native-rule table saved to:", TABLE_PATHS[0])
            display(NATIVE_TABLE[TABLE_COLUMNS].round(4))
            """
        ),
        markdown(
            """
            ## 5. Midpoint-rule accuracy and selectivity table

            This table has the same rows and uncertainty summaries as the native table, but every
            prediction uses the training-class midpoint threshold.
            """
        ),
        code(
            """
            print("Midpoint-rule table saved to:", TABLE_PATHS[1])
            display(MIDPOINT_TABLE[TABLE_COLUMNS].round(4))
            """
        ),
        markdown(
            """
            ## 6. Hewitt-style native and midpoint plots

            Blue circles are real-task means, orange squares are random-label means, and each
            vertical connector is the paired selectivity gap. Error bars show standard deviation
            across seeds. The dotted line is binary chance accuracy. One figure is saved for each
            decision rule.
            """
        ),
        code(
            """
            FIGURE_PATHS = plot_fixed_layer_selectivity_accuracy_gaps(
                COMBINED_DIR,
                split=ANALYSIS_SPLIT,
                use_balanced_accuracy=USE_BALANCED_ACCURACY_FOR_PLOTS,
            )
            for figure_path in FIGURE_PATHS:
                print(figure_path)
                display(Image(filename=str(figure_path)))
            """
        ),
        markdown(
            """
            ## 7. Interpretation checklist

            - Treat held-out test accuracy and selectivity as the primary comparison.
            - Confirm in Notebooks 12–15 that random-label training accuracy is high enough to
              demonstrate probe capacity; low control training accuracy can indicate failed
              optimization rather than strong selectivity.
            - Check uncertainty across paired seeds rather than ranking methods from point
              estimates alone.
            - Use the native-versus-midpoint agreement diagnostics in Notebooks 12–15 to identify
              conclusions that depend on the decision threshold.
            - Keep DAS IIA, recovery, logit flip, and sign flip as separate causal evidence. They
              should not be merged into the probe-accuracy selectivity ranking.
            """
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "colab": {"name": DESTINATION.name},
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
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(json.dumps(notebook(), indent=1) + "\n", encoding="utf-8")
    print(DESTINATION.relative_to(ROOT))


if __name__ == "__main__":
    main()
