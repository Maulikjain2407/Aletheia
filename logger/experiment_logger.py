import os
import csv
import json
from datetime import datetime
from pathlib import Path


class experiment_logger:

    def __init__(self, base_dir="utils"):

        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.summary_file = self.base_dir / "summary.csv"

        self.run_id = self.generate_run_id()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def generate_run_id(self):
        if not self.summary_file.exists():
            return "E0001"

        with self.summary_file.open("r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        next_id = len(rows) + 1
        return f"E{next_id:04d}"

    def _build_csv_schema(self, trained_models):
        fields = [
            "run_id", "timestamp",
            "dataset", "test_size", "seed",
            "cv", "scoring", "n_jobs"
        ]

        for model_key in trained_models.keys():
            for metric in ["accuracy", "f1", "logloss"]:
                fields.extend([
                    f"{model_key}_{metric}_train_mean",
                    f"{model_key}_{metric}_train_sd",
                    f"{model_key}_{metric}_val_mean",
                    f"{model_key}_{metric}_val_sd",
                ])

            fields.append(f"{model_key}_gen_gap")

        return fields

    def _build_csv_row(self, trained_models, dataset_name, test_size, seed, scoring, n_jobs, cv):

        row = {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "dataset": dataset_name,
            "test_size": float(test_size),
            "seed": int(seed),
            "cv": int(cv),
            "scoring": scoring,
            "n_jobs": int(n_jobs)
        }

        for model_key, model_data in trained_models.items():

            evals = model_data["evaluations"]

            for metric in ["accuracy", "f1", "logloss"]:
                t = evals[metric]["training"]
                v = evals[metric]["validation"]

                row[f"{model_key}_{metric}_train_mean"] = float(f"{t['mean']:.3f}")
                row[f"{model_key}_{metric}_train_sd"] = float(f"{t['sd']:.3f}")
                row[f"{model_key}_{metric}_val_mean"] = float(f"{v['mean']:.3f}")
                row[f"{model_key}_{metric}_val_sd"] = float(f"{v['sd']:.3f}")

            gap = model_data["generalisation_gap"]

            row[f"{model_key}_gen_gap"] = float(f"{gap:.3f}")

        return row

    def append_csv(self, trained_models, dataset_name, test_size, seed, scoring, n_jobs, cv):

        fieldnames = self._build_csv_schema(trained_models)
        row = self._build_csv_row(
            trained_models, dataset_name, test_size, seed, scoring, n_jobs, cv
        )

        exists = self.summary_file.exists()

        with self.summary_file.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if not exists:
                writer.writeheader()

            writer.writerow(row)

    def json_writer(self, trained_models, dataset_name, test_size, seed, scoring, n_jobs, cv):

        model_family_map = {
            "logistic": "linear model",
            "decision_trees": "tree model",
            "random_forest": "ensemble model",
            "XG_boost": "boosting model"
        }

        data = {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "dataset": {
                "name": dataset_name
            },
            "model": {},
            "metadata": {
                "random_seed": seed,
                "n_jobs": n_jobs,
                "GridSearchCV_scoring": scoring,
                "CV": cv,
                "test_size": test_size
            }
        }

        for model_key, model_data in trained_models.items():

            evals = model_data["evaluations"]
            Lmetrics = {}

            for metric, splits in evals.items():

                t = splits["training"]
                v = splits["validation"]

                Lmetrics[metric] = {
                        "training": {
                            "mean": float(f"{t['mean']:.3f}"),
                            "sd": float(f"{t['sd']:.3f}")
                        },
                        "validation": {
                            "mean": float(f"{v['mean']:.3f}"),
                            "sd": float(f"{v['sd']:.3f}")
                        }
                }

            gap = model_data["generalisation_gap"]

            data["model"][model_key] = {
                "model_name": model_key.replace("_", " "),
                "model_family": model_family_map.get(model_key, ""),
                "hyperparameters": model_data["best_params"],
                "metrics": Lmetrics,
                "generalisation_gap": float(f"{gap:.3f}"),
                "test_metrics": model_data.get("test_metrics", {})
            }

        json_path = self.base_dir / f"{self.run_id}.json"

        with json_path.open("w") as f:
            json.dump(data, f, indent=4)

        self.append_csv(
            trained_models=trained_models,
            dataset_name=dataset_name,
            test_size=test_size,
            seed=seed,
            scoring=scoring,
            n_jobs=n_jobs,
            cv=cv
        )

        return json_path