"""
Hyperparameter tuning utilities for SECOM yield prediction.
"""

from __future__ import annotations

import logging
import warnings
from collections import Counter
from typing import Any, Callable

from numpy.typing import ArrayLike, NDArray
import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning
from imblearn.pipeline import Pipeline as ImbPipeline

from .config import (
    RANDOM_STATE, compute_class_ratio, MIN_SAMPLES_SMOTE, MIN_SAMPLES_ADASYN,
    SamplingStrategy, OPTUNA_STORAGE_PATH, OPTUNA_N_TRIALS_DEFAULT,
)
from .evaluation import gmean_scorer, get_cv_splitter
from .models.registry import MODEL_FACTORIES, SUPPORTED_MODELS, create_xgboost_model
from .models.factory import get_sampler

logger = logging.getLogger('secom')

# Suppress sklearn ConvergenceWarning during hyperparameter tuning.
# RATIONALE: During Optuna optimization, many trial configurations will not converge
# (e.g., weak regularization with small max_iter). This is expected behavior - Optuna
# will naturally steer toward better configurations. The warnings add noise without
# actionable information. The final selected model should be validated separately.
warnings.filterwarnings('ignore', category=ConvergenceWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _check_resampling_feasibility(
    y_train: ArrayLike,
    sampling_strategy: SamplingStrategy
) -> bool:
    """
    Check if resampling is feasible given minority class size.

    ADASYN and SMOTE require minimum minority samples to work.
    This pre-check avoids relying on fragile exception string matching.

    Args:
        y_train: Training labels
        sampling_strategy: Resampling strategy to check

    Returns:
        True if resampling is feasible, False otherwise
    """
    if sampling_strategy == 'native':
        return True

    minority_count = min(Counter(y_train).values())

    if sampling_strategy == 'smote':
        return minority_count >= MIN_SAMPLES_SMOTE
    elif sampling_strategy == 'adasyn':
        return minority_count >= MIN_SAMPLES_ADASYN

    return True


def create_objective(
    model_name: str,
    X_train: NDArray,
    y_train: ArrayLike,
    sampling_strategy: SamplingStrategy = 'native',
    cv_n_jobs: int = 1
) -> Callable[[optuna.Trial], float]:
    """
    Create Optuna objective function for hyperparameter optimization.

    This factory function returns an objective callable that Optuna uses to
    evaluate different hyperparameter configurations. Each call to the returned
    objective runs stratified cross-validation and returns the mean G-Mean score.

    The objective handles:
    - Model creation with Optuna-suggested hyperparameters
    - Class balancing (native mode only - avoids double-correction with resampling)
    - Dynamic class ratio for XGBoost's scale_pos_weight
    - Resampling pipeline construction (SMOTE/ADASYN)
    - Failed trial handling (marks trials as pruned with failure reason)

    Args:
        model_name: Model type from SUPPORTED_MODELS
            - 'LogReg': L1/L2 regularized logistic regression
            - 'RandomForest': Random forest classifier
            - 'XGBoost': Gradient boosting classifier
            - 'SVM': Support vector machine with RBF/linear kernel
        X_train: Training features (n_samples, n_features)
        y_train: Training labels (n_samples,), binary classification
        sampling_strategy: How to handle class imbalance
            - 'native': Use model's built-in class weighting (class_weight='balanced')
            - 'smote': Synthetic Minority Over-sampling Technique
            - 'adasyn': Adaptive Synthetic Sampling
        cv_n_jobs: Number of parallel jobs for cross-validation (-1 for all cores)

    Returns:
        Objective function with signature (trial: optuna.Trial) -> float
        Returns mean G-Mean from 5-fold stratified CV. Sets trial user attributes:
        - cv_std: Standard deviation of CV scores
        - failed: Boolean indicating if trial failed
        - failure_reason: String describing failure (if failed)

    Raises:
        optuna.TrialPruned: When CV fails or resampling is infeasible
    """
    model_factory = MODEL_FACTORIES[model_name]
    use_class_balancing = (sampling_strategy == 'native')

    # Compute actual class ratio for XGBoost (more accurate than hardcoded default)
    class_ratio = compute_class_ratio(y_train)

    def objective(trial: optuna.Trial) -> float:
        # Use dynamic class ratio for XGBoost
        if model_name == 'XGBoost':
            model = create_xgboost_model(trial, with_class_balancing=use_class_balancing, class_ratio=class_ratio)
        else:
            model = model_factory(trial, with_class_balancing=use_class_balancing)

        # Build pipeline using centralized sampler factory (DRY)
        sampler = get_sampler(sampling_strategy)
        if sampler is None:
            pipeline = Pipeline([('classifier', model)])
        else:
            pipeline = ImbPipeline([('sampler', sampler), ('classifier', model)])

        cv = get_cv_splitter()

        # Pre-check resampling feasibility to avoid fragile exception string matching
        if not _check_resampling_feasibility(y_train, sampling_strategy):
            logger.debug(f"Resampling infeasible for {model_name}/{sampling_strategy}: insufficient minority samples")
            trial.set_user_attr('cv_std', 0.0)
            trial.set_user_attr('failed', True)
            trial.set_user_attr('failure_reason', 'insufficient_minority_samples')
            raise optuna.TrialPruned(f"Resampling infeasible: insufficient minority samples")

        try:
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=gmean_scorer, n_jobs=cv_n_jobs)
        except ValueError as e:
            # Handle CV fold edge cases (e.g., fold has too few minority samples)
            logger.warning(f"CV failed for {model_name}/{sampling_strategy}: {type(e).__name__}: {e}")
            trial.set_user_attr('cv_std', 0.0)
            trial.set_user_attr('failed', True)
            trial.set_user_attr('failure_reason', f'{type(e).__name__}: {str(e)[:80]}')
            raise optuna.TrialPruned(f"CV failed: {type(e).__name__}")

        trial.set_user_attr('cv_std', cv_scores.std())
        trial.set_user_attr('failed', False)
        return float(cv_scores.mean())

    return objective


def get_optuna_storage_url() -> str:
    """
    Get Optuna SQLite storage URL.

    Returns:
        SQLite connection string for Optuna storage
    """
    return f"sqlite:///{OPTUNA_STORAGE_PATH}"


def get_study_progress(study_name: str) -> dict[str, Any] | None:
    """
    Check progress of an existing Optuna study.

    Args:
        study_name: Name of the study to check

    Returns:
        Dict with trial counts and best value, or None if study doesn't exist
    """
    storage_url = get_optuna_storage_url()
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
        completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        failed = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])

        best_value = study.best_value if completed > 0 else None

        return {
            'study_name': study_name,
            'n_trials_completed': completed,
            'n_trials_pruned': pruned,
            'n_trials_failed': failed,
            'n_trials_total': len(study.trials),
            'best_value': best_value,
        }
    except KeyError:
        return None


def list_existing_studies() -> list[str]:
    """
    List all existing Optuna studies in storage.

    Returns:
        List of study names
    """
    storage_url = get_optuna_storage_url()
    if not OPTUNA_STORAGE_PATH.exists():
        return []
    try:
        return optuna.study.get_all_study_names(storage=storage_url)
    except Exception:
        return []


def delete_study(study_name: str) -> bool:
    """
    Delete an existing Optuna study.

    Args:
        study_name: Name of the study to delete

    Returns:
        True if deleted, False if study didn't exist
    """
    storage_url = get_optuna_storage_url()
    try:
        optuna.delete_study(study_name=study_name, storage=storage_url)
        logger.info(f"Deleted study: {study_name}")
        return True
    except KeyError:
        return False


def run_optuna_study(
    model_name: str,
    X_train: NDArray,
    y_train: ArrayLike,
    sampling_strategy: SamplingStrategy = 'native',
    feature_set: str = 'unknown',
    n_trials: int = OPTUNA_N_TRIALS_DEFAULT,
    resume: bool = True,
    fresh: bool = False,
) -> dict[str, Any]:
    """
    Run Optuna hyperparameter optimization with persistent SQLite storage.

    Studies are persisted to disk, enabling:
    - Crash recovery: Resume interrupted tuning runs
    - Progress monitoring: Query trial counts while running
    - Reproducibility: Inspect historical optimization results

    Args:
        model_name: Model type ('LogReg', 'RandomForest', 'XGBoost', 'SVM')
        X_train: Training features
        y_train: Training labels
        sampling_strategy: Resampling method ('native', 'smote', 'adasyn')
        feature_set: Feature set name for unique study identification
        n_trials: Total number of trials to run (including any completed)
        resume: If True, continue from existing study progress
        fresh: If True, delete existing study and start fresh (overrides resume)

    Returns:
        Dict with best parameters, CV G-Mean, and study reference
    """
    # Unique study name includes feature set to avoid collisions
    study_name = f"{model_name}_{feature_set}_{sampling_strategy}"
    storage_url = get_optuna_storage_url()

    # Handle fresh start request
    if fresh:
        delete_study(study_name)
        resume = False

    # Check existing progress
    existing = get_study_progress(study_name)
    if existing and resume:
        completed = existing['n_trials_completed']
        if completed >= n_trials:
            logger.info(f"Study '{study_name}' already complete ({completed}/{n_trials} trials)")
            study = optuna.load_study(study_name=study_name, storage=storage_url)
            return {
                'model': model_name,
                'feature_set': feature_set,
                'sampling_strategy': sampling_strategy,
                'best_params': study.best_params,
                'cv_gmean': study.best_value,
                'cv_std': study.best_trial.user_attrs.get('cv_std', 0.0),
                'study': study,
                'resumed': True,
                'trials_run': 0,
            }
        remaining = n_trials - completed
        logger.info(f"Resuming study '{study_name}': {completed}/{n_trials} complete, running {remaining} more")
    else:
        remaining = n_trials
        if existing and not resume:
            logger.info(f"Starting fresh study '{study_name}' (ignoring {existing['n_trials_total']} existing trials)")

    objective = create_objective(model_name, X_train, y_train, sampling_strategy, cv_n_jobs=1)

    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=RANDOM_STATE),
        study_name=study_name,
        storage=storage_url,
        load_if_exists=resume,
    )

    study.optimize(objective, n_trials=remaining, show_progress_bar=False)

    logger.debug(f"{study_name}: G-Mean={study.best_value:.4f} ({len(study.trials)} trials)")

    return {
        'model': model_name,
        'feature_set': feature_set,
        'sampling_strategy': sampling_strategy,
        'best_params': study.best_params,
        'cv_gmean': study.best_value,
        'cv_std': study.best_trial.user_attrs.get('cv_std', 0.0),
        'study': study,
        'resumed': resume and existing is not None,
        'trials_run': remaining,
    }


def run_all_experiments_for_sampling(
    sampling_strategy: SamplingStrategy,
    feature_sets: dict[str, tuple[NDArray, NDArray]],
    y_train: ArrayLike,
    n_trials: int = OPTUNA_N_TRIALS_DEFAULT,
    resume: bool = True,
    fresh: bool = False,
) -> list[dict[str, Any]]:
    """
    Run all 12 experiments (4 models x 3 feature sets) for one sampling strategy.

    Args:
        sampling_strategy: Resampling method ('native', 'smote', 'adasyn')
        feature_sets: Dict mapping feature set names to (X_train, X_test) tuples
        y_train: Training labels
        n_trials: Number of Optuna trials per experiment
        resume: If True, resume from existing study progress (default: True)
        fresh: If True, delete existing studies and start fresh (default: False)

    Returns:
        List of result dicts with best params and CV G-Mean for each experiment
    """
    results: list[dict[str, Any]] = []

    for model_name in SUPPORTED_MODELS:
        for feature_name, (X_train, _) in feature_sets.items():
            study_result = run_optuna_study(
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
                sampling_strategy=sampling_strategy,
                feature_set=feature_name,
                n_trials=n_trials,
                resume=resume,
                fresh=fresh,
            )

            result: dict[str, Any] = {
                'model': model_name,
                'feature_set': feature_name,
                'n_features': X_train.shape[1],
                'sampling_strategy': sampling_strategy,
                'cv_gmean': study_result['cv_gmean'],
                'cv_gmean_std': study_result['cv_std'],
            }

            for param_name, param_value in study_result['best_params'].items():
                result[f'param_{param_name}'] = param_value

            results.append(result)

    return results
