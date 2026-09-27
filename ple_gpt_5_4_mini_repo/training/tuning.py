## training/tuning.py
"""Optuna-based hyperparameter tuning for tabular deep learning experiments.

This module implements the paper's tuning protocol with the following goals:

- Use Optuna TPE for Bayesian optimization.
- Tune on validation score only.
- Respect dataset-specific search spaces from ``config.yaml``.
- Support model families:
  * MLP
  * ResNet
  * Transformer
- Support embedding variants:
  * linear
  * PLE quantile / target-aware
  * periodic
- Keep the configuration immutable across trials by deep-copying the base config.
- Return the best tuned configuration and study metadata for downstream use.

The tuner intentionally depends only on:
- ``models.tabular_model.ModelFactory`` for model construction
- ``training.trainer.Trainer`` for training/evaluation
- ``utils.reproducibility`` for deterministic execution
- the shared config schema provided by ``utils.io.load_config``

Public API:
  - Tuner.__init__(config)
  - Tuner.objective(trial)
  - Tuner.run(data)
  - Tuner.sample_hyperparameters(trial)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import optuna

from evaluation.metrics import MetricsEvaluator
from models.tabular_model import ModelFactory
from training.trainer import Trainer
from utils.reproducibility import set_seed


@dataclass
class TuningResult:
    """Container for the outcome of one tuning run."""

    best_config: Dict[str, Any]
    best_trial_number: int
    best_validation_score: float
    study_direction: str
    study_name: str


class Tuner:
    """Optuna-based tuner for paper-faithful tabular DL experiments."""

    _SPECIAL_TRANSFORMER_DATASETS = {"sa", "co", "mi"}

    def __init__(self, config: Mapping[str, Any]) -> None:
        """Initializes the tuner.

        Args:
          config: Shared experiment configuration mapping.

        Raises:
          TypeError: If config is not mapping-like.
          KeyError: If required keys are missing.
        """
        if not isinstance(config, Mapping):
            raise TypeError(
                f"Expected config to be a mapping, got {type(config).__name__}."
            )

        self._base_config: Dict[str, Any] = copy.deepcopy(dict(config))
        self._experiment_cfg: Dict[str, Any] = dict(self._base_config.get("experiment", {}))
        self._tuning_cfg: Dict[str, Any] = dict(
            self._base_config.get("hyperparameter_tuning", {})
        )
        self._model_cfg: Dict[str, Any] = dict(self._base_config.get("models", {}))
        self._training_cfg: Dict[str, Any] = dict(self._base_config.get("training", {}))
        self._evaluation_cfg: Dict[str, Any] = dict(self._base_config.get("evaluation", {}))

        self._dataset_name: str = self._resolve_dataset_name().lower()
        self._model_name: str = self._resolve_model_name()
        self._task_type: str = self._resolve_task_type().lower()

        self._seed: int = int(self._experiment_cfg.get("seed", 42))
        self._n_trials: int = int(self._tuning_cfg.get("trials", 100))
        self._method: str = str(self._tuning_cfg.get("method", "optuna_tpe")).lower()
        self._validation_selection: str = str(
            self._evaluation_cfg.get("validation_selection", "best_validation_score")
        ).lower()

        self._metrics = MetricsEvaluator()
        self._study: Optional[optuna.Study] = None
        self._best_config: Optional[Dict[str, Any]] = None
        self._best_validation_score: Optional[float] = None
        self._best_trial_number: Optional[int] = None

        set_seed(self._seed)

    def objective(self, trial: optuna.Trial, data: Any) -> float:
        """Optuna objective using a single tuning trial.

        Args:
          trial: Optuna trial object.
          data: Preprocessed data container.

        Returns:
          Validation metric value. Accuracy is maximized, RMSE is minimized.
        """
        trial_config = self._build_trial_config(trial)

        model_factory = ModelFactory(trial_config)
        model = model_factory.create(trial_config, data)
        trainer = Trainer(model=model, config=trial_config)

        train_result = trainer.train(data)

        # The trainer already follows early stopping and returns test predictions,
        # but tuning must rely only on validation selection.
        val_score = self._extract_validation_score(
            trainer=trainer,
            train_result=train_result,
            data=data,
        )

        # Store per-trial diagnostics for later retrieval.
        trial.set_user_attr("dataset_name", self._dataset_name)
        trial.set_user_attr("model_name", self._model_name)
        trial.set_user_attr("task_type", self._task_type)
        trial.set_user_attr("validation_score", float(val_score))

        return float(val_score)

    def run(self, data: Any) -> Dict[str, Any]:
        """Runs Optuna TPE tuning and returns the best configuration.

        Args:
          data: Preprocessed data container.

        Returns:
          Dictionary with best configuration and study metadata.
        """
        if self._method != "optuna_tpe":
            raise ValueError(
                f"Unsupported tuning method '{self._method}'. Expected 'optuna_tpe'."
            )

        direction = "maximize" if self._task_type == "classification" else "minimize"
        sampler = optuna.samplers.TPESampler(seed=self._seed, multivariate=True)

        study_name = f"{self._dataset_name}_{self._model_name}".replace(" ", "_")
        self._study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            sampler=sampler,
        )

        objective_fn = lambda trial: self.objective(trial=trial, data=data)
        self._study.optimize(objective_fn, n_trials=self._n_trials, show_progress_bar=False)

        if self._study.best_trial is None:
            raise RuntimeError("Optuna study did not produce a best trial.")

        self._best_trial_number = int(self._study.best_trial.number)
        self._best_validation_score = float(self._study.best_value)
        self._best_config = self._build_trial_config(self._study.best_trial)

        result = TuningResult(
            best_config=copy.deepcopy(self._best_config),
            best_trial_number=self._best_trial_number,
            best_validation_score=self._best_validation_score,
            study_direction=direction,
            study_name=study_name,
        )
        return {
            "best_config": result.best_config,
            "best_trial_number": result.best_trial_number,
            "best_validation_score": result.best_validation_score,
            "study_direction": result.study_direction,
            "study_name": result.study_name,
            "study": self._study,
        }

    def sample_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Samples hyperparameters for the current model/dataset combination.

        Args:
          trial: Optuna trial object.

        Returns:
          Dictionary of sampled hyperparameters.
        """
        hyperparams: Dict[str, Any] = {}

        backbone = self._resolve_backbone_name(self._model_name)
        embedding_code = self._resolve_embedding_code(self._model_name)

        # Common optimizer settings.
        hyperparams["optimizer"] = "adamw"
        hyperparams["lr_schedule"] = "none"
        hyperparams["patience"] = int(
            self._training_cfg.get("early_stopping", {}).get("patience", 16)
        )
        hyperparams["min_delta"] = float(
            self._training_cfg.get("early_stopping", {}).get("min_delta", 0.0)
        )
        hyperparams["max_epochs"] = int(self._training_cfg.get("max_epochs", 100))

        # Backbone-specific search spaces.
        if backbone == "mlp":
            mlp_cfg = dict(self._model_cfg.get("mlp", {}))
            hyperparams["mlp_layers"] = trial.suggest_int(
                "mlp_layers",
                int(mlp_cfg.get("layers", [1, 16])[0]),
                int(mlp_cfg.get("layers", [1, 16])[1]),
            )
            hyperparams["mlp_layer_size"] = trial.suggest_int(
                "mlp_layer_size",
                int(mlp_cfg.get("layer_size", [1, 1024])[0]),
                int(mlp_cfg.get("layer_size", [1, 1024])[1]),
            )
            hyperparams["mlp_dropout"] = trial.suggest_float(
                "mlp_dropout",
                float(mlp_cfg.get("dropout", [0.0, 0.5])[0]),
                float(mlp_cfg.get("dropout", [0.0, 0.5])[1]),
            )
            hyperparams["mlp_weight_decay"] = trial.suggest_float(
                "mlp_weight_decay",
                float(mlp_cfg.get("weight_decay", [0.0, 1.0e-3])[0]),
                float(mlp_cfg.get("weight_decay", [0.0, 1.0e-3])[1]),
                log=True,
            )
            hyperparams["learning_rate"] = trial.suggest_float(
                "learning_rate",
                self._resolve_lr_bounds("mlp")[0],
                self._resolve_lr_bounds("mlp")[1],
                log=True,
            )

        elif backbone == "resnet":
            resnet_cfg = dict(self._model_cfg.get("resnet", {}))
            hyperparams["resnet_layers"] = trial.suggest_int(
                "resnet_layers",
                int(resnet_cfg.get("layers", [1, 8])[0]),
                int(resnet_cfg.get("layers", [1, 8])[1]),
            )
            hyperparams["resnet_layer_size"] = trial.suggest_int(
                "resnet_layer_size",
                int(resnet_cfg.get("layer_size", [32, 512])[0]),
                int(resnet_cfg.get("layer_size", [32, 512])[1]),
            )
            hyperparams["resnet_hidden_factor"] = trial.suggest_float(
                "resnet_hidden_factor",
                float(resnet_cfg.get("hidden_factor", [1.0, 4.0])[0]),
                float(resnet_cfg.get("hidden_factor", [1.0, 4.0])[1]),
            )
            hyperparams["resnet_hidden_dropout"] = trial.suggest_float(
                "resnet_hidden_dropout",
                float(resnet_cfg.get("hidden_dropout", [0.0, 0.5])[0]),
                float(resnet_cfg.get("hidden_dropout", [0.0, 0.5])[1]),
            )
            hyperparams["resnet_residual_dropout"] = trial.suggest_float(
                "resnet_residual_dropout",
                float(resnet_cfg.get("residual_dropout", [0.0, 0.5])[0]),
                float(resnet_cfg.get("residual_dropout", [0.0, 0.5])[1]),
            )
            hyperparams["resnet_weight_decay"] = trial.suggest_float(
                "resnet_weight_decay",
                float(resnet_cfg.get("weight_decay", [0.0, 1.0e-3])[0]),
                float(resnet_cfg.get("weight_decay", [0.0, 1.0e-3])[1]),
                log=True,
            )
            hyperparams["learning_rate"] = trial.suggest_float(
                "learning_rate",
                self._resolve_lr_bounds("resnet")[0],
                self._resolve_lr_bounds("resnet")[1],
                log=True,
            )

        elif backbone == "transformer":
            transformer_cfg = dict(self._model_cfg.get("transformer", {}))
            group_key = self._resolve_transformer_group_key(self._dataset_name)

            hyperparams["transformer_layers"] = trial.suggest_int(
                "transformer_layers",
                int(transformer_cfg.get("layers", {}).get(group_key, [1, 4])[0]),
                int(transformer_cfg.get("layers", {}).get(group_key, [1, 4])[1]),
            )
            hyperparams["transformer_embedding_size"] = trial.suggest_int(
                "transformer_embedding_size",
                int(transformer_cfg.get("embedding_size", {}).get(group_key, [96, 512])[0]),
                int(transformer_cfg.get("embedding_size", {}).get(group_key, [96, 512])[1]),
            )

            residual_cfg = transformer_cfg.get("residual_dropout", {})
            residual_default = residual_cfg.get(group_key, [0.0, 0.2])
            if isinstance(residual_default, (int, float)):
                hyperparams["transformer_residual_dropout"] = float(residual_default)
            else:
                hyperparams["transformer_residual_dropout"] = trial.suggest_float(
                    "transformer_residual_dropout",
                    float(residual_default[0]),
                    float(residual_default[1]),
                )

            hyperparams["transformer_attention_dropout"] = trial.suggest_float(
                "transformer_attention_dropout",
                float(transformer_cfg.get("attention_dropout", [0.0, 0.5])[0]),
                float(transformer_cfg.get("attention_dropout", [0.0, 0.5])[1]),
            )
            hyperparams["transformer_ffn_dropout"] = trial.suggest_float(
                "transformer_ffn_dropout",
                float(transformer_cfg.get("ffn_dropout", [0.0, 0.5])[0]),
                float(transformer_cfg.get("ffn_dropout", [0.0, 0.5])[1]),
            )
            hyperparams["transformer_ffn_factor"] = trial.suggest_float(
                "transformer_ffn_factor",
                float(transformer_cfg.get("ffn_factor", [0.6666667, 2.6666667])[0]),
                float(transformer_cfg.get("ffn_factor", [0.6666667, 2.6666667])[1]),
            )
            weight_decay_cfg = transformer_cfg.get("weight_decay", {})
            weight_decay_default = weight_decay_cfg.get(group_key, [1.0e-6, 1.0e-4])
            if isinstance(weight_decay_default, (int, float)):
                hyperparams["transformer_weight_decay"] = float(weight_decay_default)
            else:
                hyperparams["transformer_weight_decay"] = trial.suggest_float(
                    "transformer_weight_decay",
                    float(weight_decay_default[0]),
                    float(weight_decay_default[1]),
                    log=True,
                )
            hyperparams["learning_rate"] = trial.suggest_float(
                "learning_rate",
                self._resolve_lr_bounds("transformer")[0],
                self._resolve_lr_bounds("transformer")[1],
                log=True,
            )

        else:
            raise ValueError(f"Unsupported backbone '{backbone}' parsed from model name.")

        # Embedding-specific search spaces.
        if embedding_code in {"L", "LR", "LRLR", "PL", "PLR", "PLLR"}:
            # The paper tunes linear output dimensions when linear layers are used.
            linear_bounds = self._get_range(
                self._base_config
                .get("hyperparameter_tuning", {})
                .get("search_spaces", {})
                .get("linear_embedding_output_dim", [1, 128]),
                default_low=1,
                default_high=128,
            )
            hyperparams["linear_embedding_output_dim"] = trial.suggest_int(
                "linear_embedding_output_dim",
                linear_bounds[0],
                linear_bounds[1],
            )

        if "Q" in embedding_code:
            q_bounds = self._get_range(
                self._base_config
                .get("hyperparameter_tuning", {})
                .get("search_spaces", {})
                .get("ple_quantiles", [2, 256]),
                default_low=2,
                default_high=256,
            )
            hyperparams["ple_quantiles"] = trial.suggest_int(
                "ple_quantiles",
                q_bounds[0],
                q_bounds[1],
            )

        if "T" in embedding_code:
            ple_tree_cfg = (
                self._base_config
                .get("hyperparameter_tuning", {})
                .get("search_spaces", {})
                .get("ple_tree", {})
            )
            max_leaves = self._get_range(
                ple_tree_cfg.get("max_leaves", [2, 256]),
                default_low=2,
                default_high=256,
            )
            min_items = self._get_range(
                ple_tree_cfg.get("min_items_per_leaf", [1, 128]),
                default_low=1,
                default_high=128,
            )
            min_gain = self._get_range(
                ple_tree_cfg.get("min_information_gain", [1.0e-9, 0.01]),
                default_low=1.0e-9,
                default_high=0.01,
            )
            hyperparams["ple_max_leaves"] = trial.suggest_int(
                "ple_max_leaves",
                max_leaves[0],
                max_leaves[1],
            )
            hyperparams["ple_min_items_per_leaf"] = trial.suggest_int(
                "ple_min_items_per_leaf",
                min_items[0],
                min_items[1],
            )
            hyperparams["ple_min_information_gain"] = trial.suggest_float(
                "ple_min_information_gain",
                min_gain[0],
                min_gain[1],
                log=True,
            )

        if embedding_code.startswith("P"):
            periodic_bounds = self._get_range(
                self._base_config
                .get("hyperparameter_tuning", {})
                .get("search_spaces", {})
                .get("periodic_k", [1, 128]),
                default_low=1,
                default_high=128,
            )
            hyperparams["periodic_k"] = trial.suggest_int(
                "periodic_k",
                periodic_bounds[0],
                periodic_bounds[1],
            )

        return hyperparams

    def _build_trial_config(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Builds a full experiment config for a trial."""
        trial_params = self.sample_hyperparameters(trial)
        trial_config: Dict[str, Any] = copy.deepcopy(self._base_config)

        trial_config["dataset_name"] = self._dataset_name
        trial_config["model_name"] = self._model_name
        trial_config["task_type"] = self._task_type
        trial_config["seed"] = self._seed

        trial_config.setdefault("training", {})
        trial_config["training"] = dict(trial_config["training"])
        trial_config["training"]["optimizer"] = trial_params["optimizer"]
        trial_config["training"]["lr_schedule"] = trial_params["lr_schedule"]
        trial_config["training"]["max_epochs"] = int(trial_params["max_epochs"])
        trial_config["training"]["early_stopping"] = {
            "patience": int(trial_params["patience"]),
            "min_delta": float(trial_params["min_delta"]),
        }

        # Keep batch size and optimizer settings from the base config.
        if "batch_size" not in trial_config["training"]:
            trial_config["training"]["batch_size"] = dict(
                self._training_cfg.get("batch_size", {})
            )

        # Model params are inserted at the top level so ModelFactory can read them.
        trial_config.setdefault("model", {})
        trial_config["model"] = dict(trial_config["model"])

        backbone = self._resolve_backbone_name(self._model_name)
        embedding_code = self._resolve_embedding_code(self._model_name)

        trial_config["model"]["backbone"] = backbone
        trial_config["model"]["embedding_code"] = embedding_code
        trial_config["model"]["learning_rate"] = float(trial_params["learning_rate"])

        if backbone == "mlp":
            trial_config["model"]["layers"] = int(trial_params["mlp_layers"])
            trial_config["model"]["layer_size"] = int(trial_params["mlp_layer_size"])
            trial_config["model"]["dropout"] = float(trial_params["mlp_dropout"])
            trial_config["model"]["weight_decay"] = float(trial_params["mlp_weight_decay"])
        elif backbone == "resnet":
            trial_config["model"]["layers"] = int(trial_params["resnet_layers"])
            trial_config["model"]["layer_size"] = int(trial_params["resnet_layer_size"])
            trial_config["model"]["hidden_factor"] = float(trial_params["resnet_hidden_factor"])
            trial_config["model"]["hidden_dropout"] = float(trial_params["resnet_hidden_dropout"])
            trial_config["model"]["residual_dropout"] = float(
                trial_params["resnet_residual_dropout"]
            )
            trial_config["model"]["weight_decay"] = float(trial_params["resnet_weight_decay"])
        elif backbone == "transformer":
            trial_config["model"]["layers"] = int(trial_params["transformer_layers"])
            trial_config["model"]["embedding_size"] = int(
                trial_params["transformer_embedding_size"]
            )
            trial_config["model"]["residual_dropout"] = float(
                trial_params["transformer_residual_dropout"]
            )
            trial_config["model"]["attention_dropout"] = float(
                trial_params["transformer_attention_dropout"]
            )
            trial_config["model"]["ffn_dropout"] = float(
                trial_params["transformer_ffn_dropout"]
            )
            trial_config["model"]["ffn_factor"] = float(trial_params["transformer_ffn_factor"])
            trial_config["model"]["weight_decay"] = float(
                trial_params["transformer_weight_decay"]
            )

        # Embedding parameters.
        trial_config["model"]["embedding_output_dim"] = int(
            trial_params.get("linear_embedding_output_dim", 1)
        )
        trial_config["model"]["ple_quantiles"] = int(trial_params.get("ple_quantiles", 16))
        trial_config["model"]["ple_max_leaves"] = int(trial_params.get("ple_max_leaves", 16))
        trial_config["model"]["ple_min_items_per_leaf"] = int(
            trial_params.get("ple_min_items_per_leaf", 1)
        )
        trial_config["model"]["ple_min_information_gain"] = float(
            trial_params.get("ple_min_information_gain", 1.0e-9)
        )
        trial_config["model"]["periodic_k"] = int(trial_params.get("periodic_k", 8))

        # Retain task/data settings.
        trial_config["evaluation"] = copy.deepcopy(self._evaluation_cfg)
        trial_config["experiment"] = copy.deepcopy(self._experiment_cfg)
        trial_config["training"]["batch_size"] = copy.deepcopy(
            self._training_cfg.get("batch_size", {})
        )
        trial_config["data"] = copy.deepcopy(self._base_config.get("data", {}))
        trial_config["models"] = copy.deepcopy(self._model_cfg)
        trial_config["hyperparameter_tuning"] = copy.deepcopy(self._tuning_cfg)
        trial_config["baselines"] = copy.deepcopy(self._base_config.get("baselines", {}))

        return trial_config

    def _extract_validation_score(
        self,
        trainer: Trainer,
        train_result: Mapping[str, Any],
        data: Any,
    ) -> float:
        """Extracts the validation score in the correct optimization direction."""
        # Prefer the trainer's recorded best validation score.
        if "best_val_score" in train_result:
            score = float(train_result["best_val_score"])
        else:
            # Fallback to the trainer's history.
            history = trainer.get_history()
            score = float(history.get("best_val_score", np.inf))

        if self._task_type == "classification":
            # Higher is better.
            return float(score)

        # Regression: lower RMSE is better, keep raw RMSE because the study is minimize.
        return float(score)

    def _resolve_dataset_name(self) -> str:
        """Resolves dataset name from config."""
        dataset_name = self._base_config.get("dataset_name")
        if dataset_name is None:
            dataset_name = self._experiment_cfg.get("dataset_name")
        if dataset_name is None:
            raise KeyError(
                "Missing dataset_name. Expected config['dataset_name'] or "
                "config['experiment']['dataset_name']."
            )
        return str(dataset_name)

    def _resolve_model_name(self) -> str:
        """Resolves model name from config."""
        model_name = self._base_config.get("model_name")
        if model_name is None:
            model_name = self._experiment_cfg.get("model_name")
        if model_name is None:
            raise KeyError(
                "Missing model_name. Expected config['model_name'] or "
                "config['experiment']['model_name']."
            )
        return str(model_name)

    def _resolve_task_type(self) -> str:
        """Resolves task type from config."""
        task_type = self._base_config.get("task_type")
        if task_type is None:
            task_type = self._experiment_cfg.get("task_type")
        if task_type is None:
            dataset_name = self._resolve_dataset_name().lower()
            if dataset_name in {"ca", "ho", "fb", "mi"}:
                task_type = "regression"
            else:
                task_type = "classification"
        return str(task_type)

    def _resolve_backbone_name(self, model_name: str) -> str:
        """Parses backbone family from model name."""
        backbone = str(model_name).split("-", 1)[0].strip().lower()
        if backbone not in {"mlp", "resnet", "transformer"}:
            raise ValueError(
                f"Unsupported backbone '{backbone}' parsed from model name '{model_name}'."
            )
        return backbone

    def _resolve_embedding_code(self, model_name: str) -> str:
        """Parses embedding code from model name."""
        parts = str(model_name).split("-", 1)
        if len(parts) == 1:
            return ""
        return parts[1].strip().upper()

    def _resolve_transformer_group_key(self, dataset_name: str) -> str:
        """Returns the transformer hyperparameter group key."""
        if dataset_name.lower() in self._SPECIAL_TRANSFORMER_DATASETS:
            return "sa_co_mi"
        return "other"

    def _resolve_lr_bounds(self, backbone: str) -> Tuple[float, float]:
        """Returns learning-rate bounds for a backbone."""
        lr_cfg = dict(self._training_cfg.get("learning_rate", {}))
        backbone = str(backbone).lower()

        if backbone in {"mlp", "resnet"}:
            bounds = lr_cfg.get(backbone, [5.0e-05, 0.005])
            return self._get_range(bounds, default_low=5.0e-05, default_high=0.005)

        if backbone == "transformer":
            group_key = self._resolve_transformer_group_key(self._dataset_name)
            bounds = lr_cfg.get(group_key, [1.0e-05, 1.0e-03])
            if group_key == "sa_co_mi":
                bounds = lr_cfg.get(group_key, [1.0e-05, 3.0e-04])
            return self._get_range(bounds, default_low=1.0e-05, default_high=1.0e-03)

        return 1.0e-04, 1.0e-03

    def _get_range(
        self,
        value: Any,
        default_low: float,
        default_high: float,
    ) -> Tuple[float, float]:
        """Normalizes a range-like config value into a 2-tuple."""
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            low = float(value[0])
            high = float(value[1])
            return low, high
        if isinstance(value, (int, float)):
            scalar = float(value)
            return scalar, scalar
        return float(default_low), float(default_high)

