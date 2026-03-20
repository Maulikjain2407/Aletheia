# Aletheia: Model Evaluation and Analysis Pipeline

**ALETHEIA** — **A**nalytical **L**earning and **E**valuation **T**ool for **H**ypothesis-driven **E**xperimentation and **I**nsight **A**nalysis

---

## 📌 Overview

This project implements a reproducible machine learning experimentation pipeline designed to:

* Train and tune multiple models
* Evaluate performance using cross-validation
* Measure stability (variance) and generalization behavior
* Log structured results (JSON + CSV)
* Generate automated research-ready visualizations

The system is modular, configurable, and scalable for repeated experiments.

---

## 🎯 Research Objectives

This artifact is built to test the following hypotheses:

1. Ensemble models exhibit lower variance than single models.
2. Decision trees exhibit larger generalization gaps compared to logistic regression under fixed conditions.

---

## 🧱 Project Structure

```bash
project_root/
│
├── src/
│   ├── models.py
│   ├── data_cleaner.py
│   ├── tuning.py
│   ├── metrics.py
│   ├── experiments.py
│   ├── visualization/
│   │   └── data_visualizer.py
│
├── configs/
│   └── config_loader.py
│
├── logger/
│   └── experiment_logger.py
│
├── utils/
│   ├── summary.csv
│   └── graphs/
│
├── main.py
```

---

## ⚙️ Configuration

All paths and parameters are defined in a YAML configuration file and loaded via `config_loader`.

### Example

```yaml
path:
  data_dir: data
  training_path: train.csv

summary_path:
  utils_folder: utils
  summary_csv: summary.csv

models:
  names: ["logistic", "decision_trees", "random_forest", "XG_boost"]

metrics:
  list: ["accuracy", "f1", "logloss"]
```

---

## 🚀 Pipeline Workflow

### 1. Data Splitting

* Stratified train-test split

### 2. Model Training + Tuning

* GridSearchCV with configurable CV strategy

### 3. Evaluation

* Cross-validation metrics:

  * mean
  * standard deviation
* Test set evaluation
* Generalization gap computation

### 4. Logging

* JSON: detailed run logs
* CSV: aggregated metrics across runs

### 5. Visualization

* Automatic graph generation from CSV

---

## 📊 Generated Graphs

All graphs are saved in:

```
utils/graphs/
```

The folder is deleted and recreated each run to ensure clean outputs.

### Included Visualizations

1. CV Standard Deviation (per metric)
2. Variance by Model Type
3. Generalization Gap
4. Train vs Validation Performance

---

## 📁 CSV Schema

Each run appends a new row with columns like:

```
logistic_accuracy_train_mean  
logistic_accuracy_val_mean  
logistic_accuracy_val_sd  
...  
random_forest_f1_val_sd  
...  
XG_boost_logloss_val_sd  
...  
logistic_gen_gap  
```

### Naming Convention

```
{model}_{metric}_{split}_{stat}
```

---

## 🧪 How to Run

```bash
python main.py
```

---

## ⚠️ Important Notes

### 1. Naming Consistency

Model names must match across:

* model definitions
* logger
* grapher
* config

Example:

```
random_forest ✓  
random_forests ✗  
```

---

### 2. Graph Overwriting

* `utils/graphs/` is fully deleted each run
* Prevents stale outputs

---

### 3. Data Growth

* CSV is append-only
* Graphs adapt automatically to more runs

---

## 🧠 Design Principles

* **Reproducibility**: YAML-driven configuration
* **Modularity**: separation of concerns
* **Scalability**: supports increasing runs
* **Automation**: end-to-end pipeline

---

## 🔧 Dependencies

* Python 3.10+
* pandas
* matplotlib
* scikit-learn

---

## ✅ Summary

This artifact provides a complete experimental framework for:

* analyzing model stability
* evaluating generalization behavior
* generating research-ready insights

It supports iterative research workflows with minimal manual intervention.
