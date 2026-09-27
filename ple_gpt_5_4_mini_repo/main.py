## main.py
"""Top-level experiment orchestration for tabular deep learning reproduction.

This entrypoint coordinates the full paper-style workflow:

1. Load configuration from YAML.
2. Set deterministic behavior and global seeds.
3. Load and split the dataset.
4. Preprocess the splits.
5. Optionally tune hyperparameters with Optuna.
6. Train the selected model over 15 seeds.
7. Aggregate test metrics and optional ensembles.
8. Optionally run CatBoost/XGBoost baselines.
9. Persist all artifacts.

The file intentionally contains no model-specific learning logic; it only
coordinates the modules defined elsewhere in the project.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch

from baselines.gbdt import GBDDBaselineRunner
from data.dataset_loader import DatasetLoader
from data.preprocessing import PreprocessedData, Preprocessor
from evaluation.ensemble import EnsembleEvaluator
from evaluation.metrics import MetricsEvaluator
from models.tabular_model import ModelFactory, TabularModel
from training.trainer import Trainer
from training.tuning import Tuner
from utils.io import load_config, save_json, save_pickle
from utils.reproducibility import get_device, set_deterministic, set_seed


@dataclass(frozen=True)
class RunArtifacts:
    """Container for key outputs of one experiment run."""

    dataset_name: str
    model_name: str
    task_type: str
    seed_list: List[int]
    best_config: Dict[str, Any]
    per_seed_metrics: List[Dict[str, float]]
    per_seed_predictions: List[np.ndarray]
    per_seed_targets: List[np.ndarray]
    aggregated_metrics: Dict[str, float]
    ensemble_metrics: Optional[Dict[str, Any]]
    baselines: Dict[str, Any]


def _resolve_dataset_name(config: Mapping[str, Any], explicit: Optional[str]) -> str:
    """Resolves the dataset name from CLI or config."""
    if explicit is not None and str(explicit).strip():
        return str(explicit)
    if "dataset_name" in config:
        return str(config["dataset_name"])
    experiment_cfg = config.get("experiment", {})
    if isinstance(experiment_cfg, Mapping) and "dataset_name" in experiment_cfg:
        return str(experiment_cfg["dataset_name"])
    raise KeyError(
        "Missing dataset name. Provide --dataset-name or set config['dataset_name'] "
        "or config['experiment']['dataset_name']."
    )


def _resolve_model_name(config: Mapping[str, Any], explicit: Optional[str]) -> str:
    """Resolves the model name from CLI or config."""
    if explicit is not None and str(explicit).strip():
        return str(explicit)
    if "model_name" in config:
        return str(config["model_name"])
    experiment_cfg = config.get("experiment", {})
    if isinstance(experiment_cfg, Mapping) and "model_name" in experiment_cfg:
        return str(experiment_cfg["model_name"])
    raise KeyError(
        "Missing model name. Provide --model-name or set config['model_name'] "
        "or config['experiment']['model_name']."
    )


def _resolve_task_type(config: Mapping[str, Any], dataset_name: str, explicit: Optional[str]) -> str:
    """Resolves the task type from CLI, config, or dataset name."""
    if explicit is not None and str(explicit).strip():
        return str(explicit).lower()
    if "task_type" in config:
        return str(config["task_type"]).lower()
    experiment_cfg = config.get("experiment", {})
    if isinstance(experiment_cfg, Mapping) and "task_type" in experiment_cfg:
        return str(experiment_cfg["task_type"]).lower()
    if dataset_name.lower() in {"ca", "ho", "fb", "mi"}:
        return "regression"
    return "classification"


def _ensure_runtime_fields(
    config: Dict[str, Any],
    dataset_name: str,
    model_name: str,
    task_type: str,
    device: str,
) -> Dict[str, Any]:
    """Returns a deep-copied config with runtime fields populated."""
    cfg = copy.deepcopy(config)
    cfg["dataset_name"] = dataset_name
    cfg["model_name"] = model_name
    cfg["task_type"] = task_type
    cfg["device"] = device

    cfg.setdefault("experiment", {})
    if isinstance(cfg["experiment"], MutableMapping):
        cfg["experiment"]["dataset_name"] = dataset_name
        cfg["experiment"]["model_name"] = model_name
        cfg["experiment"]["task_type"] = task_type

    return cfg


def _make_seed_list(base_seed: int, num_seeds: int) -> List[int]:
    """Builds the paper-style list of run seeds."""
    return [int(base_seed + idx) for idx in range(int(num_seeds))]


def _to_float(value: Any) -> float:
    """Converts a scalar-like value to float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _summarize_values(values: Sequence[float], higher_is_better: bool) -> Dict[str, float]:
    """Summarizes a list of scalar metric values with mean and std."""
    arr = np.asarray(list(values), dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan")}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "best": float(np.max(arr) if higher_is_better else np.min(arr)),
    }


def _extract_task_metric_name(task_type: str, metrics: MetricsEvaluator) -> str:
    """Resolves the canonical metric name for the task."""
    return metrics.select_metric(task_type)


def _compute_metrics(
    metrics: MetricsEvaluator,
    task_type: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Computes the canonical metric dictionary."""
    if task_type == "regression":
        return metrics.compute_regression(y_true=y_true, y_pred=y_pred)
    return metrics.compute_classification(y_true=y_true, y_pred=y_pred)


def _build_model(
    config: Dict[str, Any],
    data: PreprocessedData,
) -> TabularModel:
    """Constructs a model using the shared factory."""
    factory = ModelFactory(config)
    return factory.create(config, data)


def _train_one_seed(
    config: Dict[str, Any],
    data: PreprocessedData,
    seed: int,
) -> Dict[str, Any]:
    """Trains a single model instance and returns artifacts."""
    set_seed(seed)
    model = _build_model(config, data)
    trainer = Trainer(model=model, config=config)
    result = trainer.train(data)
    result["model"] = model
    result["trainer"] = trainer
    return result


def _run_tuning(config: Dict[str, Any], data: PreprocessedData) -> Dict[str, Any]:
    """Runs Optuna tuning if configured."""
    tuner = Tuner(config)
    tuning_result = tuner.run(data)
    return tuning_result


def _run_deep_experiment(
    config: Dict[str, Any],
    data: PreprocessedData,
    task_type: str,
    output_dir: Path,
    run_tuning: bool,
    run_ensemble: bool,
) -> RunArtifacts:
    """Runs the full deep-learning experiment for one model-dataset pair."""
    metrics = MetricsEvaluator()
    base_seed = int(config["experiment"]["seed"])
    num_seeds = int(config["experiment"]["num_seeds"])
    seed_list = _make_seed_list(base_seed, num_seeds)

    best_config = copy.deepcopy(config)
    tuning_summary: Dict[str, Any] = {}
    if run_tuning:
        tuning_summary = _run_tuning(best_config, data)
        best_config = copy.deepcopy(tuning_summary["best_config"])
    best_config = _ensure_runtime_fields(
        best_config,
        dataset_name=str(best_config.get("dataset_name", config["dataset_name"])),
        model_name=str(best_config.get("model_name", config["model_name"])),
        task_type=task_type,
        device=str(config.get("device", "cpu")),
    )

    per_seed_metrics: List[Dict[str, float]] = []
    per_seed_predictions: List[np.ndarray] = []
    per_seed_targets: List[np.ndarray] = []

    metric_name = _extract_task_metric_name(task_type, metrics)

    for seed in seed_list:
        set_seed(seed)
        train_result = _train_one_seed(best_config, data, seed)
        y_pred = np.asarray(train_result["test_predictions"])
        y_true = np.asarray(train_result["test_targets"])
        seed_metrics = _compute_metrics(metrics, task_type, y_true, y_pred)
        per_seed_metrics.append(seed_metrics)
        per_seed_predictions.append(y_pred)
        per_seed_targets.append(y_true)

    metric_values = [float(entry[metric_name]) for entry in per_seed_metrics]
    aggregated = _summarize_values(
        metric_values,
        higher_is_better=metrics.is_higher_better(task_type),
    )
    aggregated_metrics = {
        metric_name: aggregated["mean"],
        f"{metric_name}_std": aggregated["std"],
        f"{metric_name}_best": aggregated["best"],
    }

    ensemble_metrics: Optional[Dict[str, Any]] = None
    if run_ensemble:
        ensemble_evaluator = EnsembleEvaluator(num_groups=int(config["experiment"]["num_ensembles"]))
        ensemble_predictions = ensemble_evaluator.build_ensembles(per_seed_predictions)
        # Use the first target array as reference; all seeds share the same test targets.
        ensemble_metrics = ensemble_evaluator.evaluate_ensembles(
            y_true=per_seed_targets[0],
            ensembles=ensemble_predictions,
            task_type=task_type,
        )

    artifacts = RunArtifacts(
        dataset_name=str(best_config["dataset_name"]),
        model_name=str(best_config["model_name"]),
        task_type=task_type,
        seed_list=seed_list,
        best_config=best_config,
        per_seed_metrics=per_seed_metrics,
        per_seed_predictions=per_seed_predictions,
        per_seed_targets=per_seed_targets,
        aggregated_metrics=aggregated_metrics,
        ensemble_metrics=ensemble_metrics,
        baselines={},
    )

    _save_deep_artifacts(output_dir=output_dir, artifacts=artifacts, tuning_summary=tuning_summary)
    return artifacts


def _save_deep_artifacts(
    output_dir: Path,
    artifacts: RunArtifacts,
    tuning_summary: Dict[str, Any],
) -> None:
    """Saves all deep-learning artifacts to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "dataset_name": artifacts.dataset_name,
        "model_name": artifacts.model_name,
        "task_type": artifacts.task_type,
        "seed_list": artifacts.seed_list,
        "best_config": artifacts.best_config,
        "aggregated_metrics": artifacts.aggregated_metrics,
        "per_seed_metrics": artifacts.per_seed_metrics,
        "ensemble_metrics": artifacts.ensemble_metrics,
        "tuning_summary": tuning_summary,
    }
    save_json(summary, output_dir / "deep_summary.json")
    save_pickle(artifacts.per_seed_predictions, output_dir / "test_predictions.pkl")
    save_pickle(artifacts.per_seed_targets, output_dir / "test_targets.pkl")
    save_pickle(artifacts.best_config, output_dir / "best_config.pkl")


def _run_baselines(
    config: Dict[str, Any],
    data: PreprocessedData,
    output_dir: Path,
) -> Dict[str, Any]:
    """Runs CatBoost and XGBoost baselines."""
    runner = GBDDBaselineRunner(config)
    results: Dict[str, Any] = {}

    results["catboost"] = runner.run_catboost(data)
    results["xgboost"] = runner.run_xgboost(data)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(results, output_dir / "gbdt_baselines.json")
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    """Builds the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Reproduce the paper 'On Embeddings for Numerical Features in Tabular Deep Learning'."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Dataset name override (e.g. ca, ad, mi).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model name override (e.g. MLP, MLP-Q-LR, Transformer-PLR).",
    )
    parser.add_argument(
        "--task-type",
        type=str,
        default=None,
        choices=("classification", "regression"),
        help="Task type override.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Optional override for the data root directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory where experiment artifacts will be saved.",
    )
    parser.add_argument(
        "--no-tuning",
        action="store_true",
        help="Disable Optuna hyperparameter tuning.",
    )
    parser.add_argument(
        "--no-ensemble",
        action="store_true",
        help="Disable ensemble evaluation.",
    )
    parser.add_argument(
        "--run-baselines",
        action="store_true",
        help="Run CatBoost and XGBoost baselines.",
    )
    return parser


def main() -> None:
    """Main CLI entrypoint."""
    parser = build_arg_parser()
    args = parser.parse_args()

    config = load_config(args.config)

    if args.dataset_name is not None:
        config["dataset_name"] = args.dataset_name
    if args.model_name is not None:
        config["model_name"] = args.model_name
    if args.task_type is not None:
        config["task_type"] = args.task_type
    if args.data_root is not None:
        config["data_root"] = args.data_root

    dataset_name = _resolve_dataset_name(config, args.dataset_name)
    model_name = _resolve_model_name(config, args.model_name)
    task_type = _resolve_task_type(config, dataset_name, args.task_type)

    device = str(get_device(config.get("device", "cpu")))
    config = _ensure_runtime_fields(
        config=config,
        dataset_name=dataset_name,
        model_name=model_name,
        task_type=task_type,
        device=device,
    )

    set_seed(int(config["experiment"]["seed"]))
    set_deterministic()

    output_dir = Path(args.output_dir).expanduser().resolve()
    run_dir = output_dir / dataset_name.lower() / model_name.replace("/", "_")
    run_dir.mkdir(parents=True, exist_ok=True)

    save_json(config, run_dir / "resolved_config.json")

    loader = DatasetLoader(config)
    splits = loader.load_and_split()

    preprocessor = Preprocessor(config)
    preprocessed = preprocessor.fit_transform(splits)

    deep_artifacts = _run_deep_experiment(
        config=config,
        data=preprocessed,
        task_type=task_type,
        output_dir=run_dir / "deep",
        run_tuning=not args.no_tuning,
        run_ensemble=not args.no_ensemble,
    )

    baseline_results: Dict[str, Any] = {}
    if args.run_baselines:
        baseline_results = _run_baselines(
            config=config,
            data=preprocessed,
            output_dir=run_dir / "baselines",
        )

    final_summary: Dict[str, Any] = {
        "dataset_name": deep_artifacts.dataset_name,
        "model_name": deep_artifacts.model_name,
        "task_type": deep_artifacts.task_type,
        "seed_list": deep_artifacts.seed_list,
        "aggregated_metrics": deep_artifacts.aggregated_metrics,
        "ensemble_metrics": deep_artifacts.ensemble_metrics,
        "baseline_results": baseline_results,
    }
    save_json(final_summary, run_dir / "final_summary.json")

    print(json.dumps(final_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
