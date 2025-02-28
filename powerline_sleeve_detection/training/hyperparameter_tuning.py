import os
import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

import yaml
import torch
import numpy as np
import optuna
from optuna.trial import Trial
from optuna.samplers import TPESampler

from powerline_sleeve_detection.system.config import Config
from powerline_sleeve_detection.training.trainer import SleeveModelTrainer

logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """Performs hyperparameter tuning for sleeve detection models."""

    def __init__(self, config: Config):
        """Initialize the hyperparameter tuner.

        Args:
            config: Application configuration
        """
        self.config = config
        self.dataset_yaml = config.get('training.dataset_yaml')
        self.base_model = config.get('training.base_model', 'yolov8n.pt')
        self.study = None
        self.best_params = None
        self.best_score = None
        self.output_dir = Path(config.get(
            'training.tuning_output_dir', 'runs/tuning'))
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def run_study(self,
                  n_trials: int = 20,
                  timeout: Optional[int] = None,
                  study_name: str = "sleeve_detection_tuning") -> Dict[str, Any]:
        """Run a hyperparameter optimization study.

        Args:
            n_trials: Number of trials to run
            timeout: Timeout in seconds (None for no timeout)
            study_name: Name of the study

        Returns:
            Dictionary with best parameters and score
        """
        if not self.dataset_yaml:
            raise ValueError("Dataset YAML path not specified in config")

        # Create optuna study with TPE sampler
        sampler = TPESampler(seed=42)
        self.study = optuna.create_study(
            direction="maximize",  # Maximize mAP
            sampler=sampler,
            study_name=study_name
        )

        # Run optimization
        logger.info(
            f"Starting hyperparameter tuning with {n_trials} trials...")
        self.study.optimize(
            self._objective, n_trials=n_trials, timeout=timeout)

        # Get best parameters
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value

        logger.info(f"Best trial: {self.study.best_trial.number}")
        logger.info(f"Best mAP: {self.best_score:.4f}")
        logger.info(f"Best parameters: {self.best_params}")

        # Save best parameters to file
        self._save_best_params()

        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "best_trial": self.study.best_trial.number
        }

    def _objective(self, trial: Trial) -> float:
        """Objective function for optimization.

        Args:
            trial: Optuna trial

        Returns:
            Validation mAP score
        """
        # Sample hyperparameters
        params = self._sample_parameters(trial)

        # Create trainer
        trainer = SleeveModelTrainer(self.config)

        try:
            # Initialize model
            trainer.initialize_model(self.base_model)

            # Train model with current hyperparameters
            training_dir = self.output_dir / f"trial_{trial.number}"
            self.config.set('training.output_dir', str(training_dir))
            self.config.set('training.experiment_name',
                            f"trial_{trial.number}")

            # Train for fewer epochs during tuning to save time
            # Cap at 50 epochs for tuning
            max_epochs = min(params.get('epochs', 50), 50)

            results = trainer.train(
                dataset_yaml=self.dataset_yaml,
                epochs=max_epochs,
                batch_size=params.get('batch_size'),
                image_size=params.get('image_size'),
                patience=params.get('patience'),
                learning_rate=params.get('learning_rate')
            )

            # Evaluate on validation set
            val_results = trainer.validate()
            mAP50 = val_results.box.map50

            # Report intermediate metric value
            trial.report(mAP50, step=max_epochs)

            # Save trial details
            self._save_trial_results(trial, params, mAP50)

            return mAP50

        except Exception as e:
            logger.error(f"Error in trial {trial.number}: {e}")
            return 0.0  # Return lowest possible score

    def _sample_parameters(self, trial: Trial) -> Dict[str, Any]:
        """Sample hyperparameters for a trial.

        Args:
            trial: Optuna trial

        Returns:
            Dictionary of hyperparameters
        """
        # Define parameter ranges
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [4, 8, 16, 32]),
            'image_size': trial.suggest_categorical('image_size', [416, 512, 640, 768]),
            'patience': trial.suggest_int('patience', 10, 30),
            'epochs': trial.suggest_int('epochs', 30, 100)
        }

        # Add augmentation parameters
        params['mosaic'] = trial.suggest_float('mosaic', 0.0, 1.0)
        params['mixup'] = trial.suggest_float('mixup', 0.0, 1.0)

        return params

    def _save_trial_results(self,
                            trial: Trial,
                            params: Dict[str, Any],
                            score: float) -> None:
        """Save trial results to disk.

        Args:
            trial: Optuna trial
            params: Hyperparameters
            score: Validation score
        """
        trial_dir = self.output_dir / f"trial_{trial.number}"
        trial_dir.mkdir(exist_ok=True, parents=True)

        trial_info = {
            "trial_number": trial.number,
            "parameters": params,
            "score": float(score)
        }

        with open(trial_dir / "params.json", 'w') as f:
            json.dump(trial_info, f, indent=2)

    def _save_best_params(self) -> None:
        """Save best parameters to disk."""
        if self.best_params is None:
            logger.warning("No best parameters available to save")
            return

        best_dir = self.output_dir / "best"
        best_dir.mkdir(exist_ok=True, parents=True)

        best_info = {
            "best_score": float(self.best_score),
            "parameters": self.best_params
        }

        with open(best_dir / "best_params.json", 'w') as f:
            json.dump(best_info, f, indent=2)

        # Update config with best parameters
        self.config.set('training.best_params', self.best_params)
        self.config.set('training.best_score', float(self.best_score))
        self.config.save()

        logger.info(
            f"Best parameters saved to {best_dir / 'best_params.json'}")

    def apply_best_params(self) -> Dict[str, Any]:
        """Apply best parameters to config for model training.

        Returns:
            Dictionary of best parameters
        """
        if self.best_params is None:
            # Try to load from saved file
            best_file = self.output_dir / "best" / "best_params.json"
            if best_file.exists():
                with open(best_file, 'r') as f:
                    best_info = json.load(f)
                    self.best_params = best_info.get('parameters', {})
                    self.best_score = best_info.get('best_score', 0.0)
            else:
                raise ValueError(
                    "No best parameters available. Run study first.")

        # Apply parameters to config
        for key, value in self.best_params.items():
            self.config.set(f'training.{key}', value)

        self.config.save()
        logger.info(f"Applied best parameters to config: {self.best_params}")

        return self.best_params
