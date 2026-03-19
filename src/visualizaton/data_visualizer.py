from pathlib import Path
import shutil
import pandas as pd
import matplotlib.pyplot as plt

from configs.config_loader import (
    summary_csv_path,
    MODEL_NAMES,
    METRICS_LIST,
    graphs_path,
    SINGLE_MODELS,
    ENSEMBLE_MODELS
)


class grapher:
    def __init__(self, summary_csv_path):
        self.df = pd.read_csv(summary_csv_path)
        self.models = MODEL_NAMES
        self.metrics = METRICS_LIST
        self.output_dir = Path(graphs_path)

        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_cv_sd(self):

        for metric in self.metrics:
            plt.figure()
            values = []

            for model in self.models:
                col = f"{model}_{metric}_val_sd"
                values.append(self.df[col].mean())

            plt.bar(self.models, values)
            plt.xlabel("Models")
            plt.ylabel("Validation SD")
            plt.title(f"{metric.upper()} CV Standard Deviation")
            plt.xticks(rotation=30)
            plt.tight_layout()

            plt.savefig(self.output_dir / f"cv_sd_{metric}.png", dpi=300)
            plt.close()

    def plot_variance_grouped(self):

        groups = {
            "single": SINGLE_MODELS,
            "ensemble": ENSEMBLE_MODELS
        }

        for metric in self.metrics:
            plt.figure()
            values = []

            for group_models in groups.values():
                group_vals = []

                for model in group_models:
                    col = f"{model}_{metric}_val_sd"
                    group_vals.append(self.df[col].mean())

                values.append(sum(group_vals) / len(group_vals))

            plt.bar(groups.keys(), values)
            plt.xlabel("Model Type")
            plt.ylabel("Mean SD")
            plt.title(f"{metric.upper()} Variance by Model Type")
            plt.tight_layout()

            plt.savefig(self.output_dir / f"variance_grouped_{metric}.png", dpi=300)
            plt.close()

    def plot_generalisation_gap(self):

        plt.figure()
        values = []

        for model in self.models:
            col = f"{model}_gen_gap"
            values.append(self.df[col].mean())

        plt.bar(self.models, values)
        plt.xlabel("Models")
        plt.ylabel("Generalisation Gap")
        plt.title("Generalisation Gap by Model")
        plt.xticks(rotation=30)
        plt.tight_layout()

        plt.savefig(self.output_dir / "generalisation_gap.png", dpi=300)
        plt.close()

    def plot_train_vs_validation(self):

        for metric in self.metrics:
            plt.figure()

            for model in self.models:
                train_col = f"{model}_{metric}_train_mean"
                val_col = f"{model}_{metric}_val_mean"

                train_mean = self.df[train_col].mean()
                val_mean = self.df[val_col].mean()

                plt.plot(["train", "validation"], [train_mean, val_mean], marker='o', label=model)

            plt.xlabel("Dataset Split")
            plt.ylabel(metric.upper())
            plt.title(f"Train vs Validation ({metric.upper()})")
            plt.legend()
            plt.tight_layout()

            plt.savefig(self.output_dir / f"train_vs_val_{metric}.png", dpi=300)
            plt.close()