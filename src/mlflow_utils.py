"""
MLflow experiment tracking utilities for SECOM yield prediction project
Author: Victoria A.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import mlflow
import mlflow.sklearn
from sklearn.base import BaseEstimator

from .config import MLFLOW_EXPERIMENT_NAME

logger = logging.getLogger('secom')

# SQLAlchemy exceptions that indicate database contention (safe to retry)
try:
    from sqlalchemy.exc import OperationalError, DatabaseError
    _RETRYABLE_EXCEPTIONS = (OperationalError, DatabaseError)
except ImportError:
    # SQLAlchemy not available - fall back to generic Exception
    _RETRYABLE_EXCEPTIONS = (Exception,)


def setup_mlflow(max_retries: int = 3) -> None:
    """
    Initialize MLflow experiment tracking with retry for parallel workers.

    MLflow defaults to sqlite:///mlflow.db when no mlruns/ directory exists.
    Parallel processes can race to create the schema, so we retry on failure.

    Args:
        max_retries: Number of retry attempts for initialization

    Raises:
        RuntimeError: If initialization fails after all retries

    Note: Logging levels for mlflow/alembic are set in config.setup_logging().
    """
    last_exception: Exception | None = None

    for attempt in range(max_retries):
        try:
            mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
            mlflow.sklearn.autolog(disable=True)
            logger.debug(f"MLflow experiment: {MLFLOW_EXPERIMENT_NAME}")
            return
        except _RETRYABLE_EXCEPTIONS as e:
            # Database contention - safe to retry
            last_exception = e
            logger.debug(f"MLflow setup attempt {attempt + 1}/{max_retries} failed ({type(e).__name__}): {e}")
            if attempt < max_retries - 1:
                time.sleep(1 * (attempt + 1))  # Exponential backoff
        except Exception as e:
            # Non-retryable error - fail immediately
            logger.exception(f"MLflow setup failed with non-retryable error: {type(e).__name__}")
            raise RuntimeError(f"MLflow setup failed: {e}") from e

    # All retries exhausted
    raise RuntimeError(f"MLflow setup failed after {max_retries} attempts: {last_exception}") from last_exception


def log_experiment(
    run_name: str,
    pipeline: BaseEstimator,
    cv_result: dict[str, Any],
    holdout_result: dict[str, Any],
    feature_set: str,
    resampler: str,
    model_type: str,
    hyperparams: dict[str, Any] | None = None
) -> None:
    """
    Log experiment to MLflow.

    Args:
        run_name: Unique name for this run
        pipeline: Trained sklearn pipeline
        cv_result: Cross-validation results with keys: cv_gmean_mean, cv_gmean_std
        holdout_result: Holdout evaluation with keys: gmean, sensitivity, specificity, etc.
        feature_set: Feature set identifier ('all', 'pca', 'lasso')
        resampler: Resampling strategy ('native', 'smote', 'adasyn')
        model_type: Model class name
        hyperparams: Optional dict of hyperparameters to log
    """
    # Validate required cv_result keys
    required_cv_keys = {'cv_gmean_mean', 'cv_gmean_std'}
    if not required_cv_keys.issubset(cv_result.keys()):
        missing = required_cv_keys - cv_result.keys()
        raise ValueError(f"cv_result missing required keys: {missing}")

    # Validate required holdout_result keys
    required_holdout_keys = {'gmean', 'sensitivity', 'specificity', 'precision', 'f1_score', 'accuracy'}
    if not required_holdout_keys.issubset(holdout_result.keys()):
        missing = required_holdout_keys - holdout_result.keys()
        raise ValueError(f"holdout_result missing required keys: {missing}")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("feature_set", feature_set)
        mlflow.log_param("resampler", resampler)
        mlflow.log_param("model_type", model_type)

        if hyperparams:
            for key, value in hyperparams.items():
                mlflow.log_param(key, value)

        mlflow.log_metric("cv_gmean_mean", cv_result['cv_gmean_mean'])
        mlflow.log_metric("cv_gmean_std", cv_result['cv_gmean_std'])

        mlflow.log_metric("test_gmean", holdout_result['gmean'])
        mlflow.log_metric("test_sensitivity", holdout_result['sensitivity'])
        mlflow.log_metric("test_specificity", holdout_result['specificity'])
        mlflow.log_metric("test_precision", holdout_result['precision'])
        mlflow.log_metric("test_f1_score", holdout_result['f1_score'])
        mlflow.log_metric("test_accuracy", holdout_result['accuracy'])

        if 'auc_roc' in holdout_result:
            mlflow.log_metric("test_auc_roc", holdout_result['auc_roc'])
        if 'auc_pr' in holdout_result:
            mlflow.log_metric("test_auc_pr", holdout_result['auc_pr'])

        mlflow.sklearn.log_model(pipeline, name="model")
        mlflow.set_tag("feature_set", feature_set)
        mlflow.set_tag("resampler", resampler)
        mlflow.set_tag("model_type", model_type)
