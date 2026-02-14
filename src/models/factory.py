"""
Model factory for SECOM yield prediction project.
Author: Victoria A.

Provides factory functions to build models and pipelines from config rows,
avoiding code duplication across notebooks and pipelines.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, ADASYN

from ..config import RANDOM_STATE, SAMPLING_STRATEGIES, SamplingStrategy
from .registry import (
    MODEL_CLASSES,
    FIXED_PARAMS,
    CLASS_BALANCE_PARAMS,
    MODEL_FACTORIES,
    SUPPORTED_MODELS,
    INT_PARAMS,
)


def build_model(
    row_or_params: dict[str, Any] | pd.Series,
    model_name: str | None = None,
    with_class_balancing: bool = True
) -> BaseEstimator:
    """
    Build model from config row or params dict, optionally with class balancing.

    Supports three input patterns:
    1. DataFrame row from tuning results (has 'model' and 'param_*' columns)
    2. Dict with 'model' key and 'param_*' keys (same structure as row)
    3. Dict of raw params with model_name provided separately

    Args:
        row_or_params: Model configuration in one of these formats:
            - pd.Series: Row from tuning results DataFrame
            - dict: Either {'model': 'XGBoost', 'param_learning_rate': 0.1, ...}
                    or raw params {'learning_rate': 0.1, ...} with model_name arg
        model_name: Model type, required if row_or_params doesn't have 'model' key.
                    One of: 'LogReg', 'RandomForest', 'XGBoost', 'SVM'
        with_class_balancing: Whether to apply class weighting.
            - True: Use for 'native' sampling (class_weight='balanced')
            - False: Use for 'smote'/'adasyn' (avoids double-correction)

    Returns:
        Configured sklearn/xgboost model instance ready for fitting

    Raises:
        ValueError: If model_name is not in SUPPORTED_MODELS

    Examples:
        # From tuning results DataFrame row:
        best_row = results_df.loc[results_df['cv_gmean'].idxmax()]
        model = build_model(best_row)

        # From dict with param_* keys:
        config = {'model': 'XGBoost', 'param_learning_rate': 0.1, 'param_max_depth': 6}
        model = build_model(config)

        # From raw params dict:
        params = {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 200}
        model = build_model(params, model_name='XGBoost')

        # Without class balancing (for use with SMOTE/ADASYN):
        model = build_model(best_row, with_class_balancing=False)
    """
    # Handle different input formats
    if isinstance(row_or_params, dict) and 'model' not in row_or_params and model_name:
        # Direct params dict with model_name provided separately
        params = dict(row_or_params)
    else:
        # Row-like input (DataFrame row or dict with 'model' key)
        row = row_or_params
        model_name = model_name or row['model']

        # Handle both DataFrame row (has .index) and dict
        # Use INT_PARAMS from registry for float->int conversion
        items = row.index if hasattr(row, 'index') else row.keys()
        params = {
            col.removeprefix('param_'): (
                int(row[col]) if col.removeprefix('param_') in INT_PARAMS else row[col]
            )
            for col in items
            if col.startswith('param_') and pd.notna(row[col])
        }

    # Validate model name
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unknown model '{model_name}'. Valid options: {SUPPORTED_MODELS}"
        )

    # Get model class and fixed params
    model_cls = MODEL_CLASSES[model_name]
    params.update(FIXED_PARAMS[model_name])

    if with_class_balancing:
        params.update(CLASS_BALANCE_PARAMS[model_name])

    return model_cls(**params)


def build_model_from_trial(
    model_name: str,
    trial: Any,  # optuna.Trial
    with_class_balancing: bool = True
) -> BaseEstimator:
    """
    Build model from Optuna trial.

    Args:
        model_name: One of 'LogReg', 'RandomForest', 'XGBoost', 'SVM'
        trial: Optuna trial object
        with_class_balancing: Whether to apply class balancing

    Returns:
        sklearn/xgboost model instance

    Raises:
        ValueError: If model_name is not in SUPPORTED_MODELS
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unknown model '{model_name}'. Valid options: {SUPPORTED_MODELS}"
        )
    model_factory = MODEL_FACTORIES[model_name]
    return model_factory(trial, with_class_balancing=with_class_balancing)


def build_pipeline(
    row: dict[str, Any] | pd.Series,
    sampling_strategy: SamplingStrategy | None = None
) -> Pipeline | ImbPipeline:
    """
    Build complete pipeline (sampler + model) from config row.

    Args:
        row: DataFrame row with 'model', 'sampling_strategy', and param_* columns
        sampling_strategy: Override sampling strategy (default: use row['sampling_strategy'])

    Returns:
        sklearn/imblearn pipeline ready for training

    Raises:
        ValueError: If sampling_strategy is not valid
    """
    sampling = sampling_strategy or row.get('sampling_strategy', 'native')
    use_balancing = (sampling == 'native')
    model = build_model(row, with_class_balancing=use_balancing)

    if sampling == 'native':
        return Pipeline([('classifier', model)])
    elif sampling == 'smote':
        sampler = SMOTE(random_state=RANDOM_STATE)
        return ImbPipeline([('sampler', sampler), ('classifier', model)])
    elif sampling == 'adasyn':
        sampler = ADASYN(random_state=RANDOM_STATE)
        return ImbPipeline([('sampler', sampler), ('classifier', model)])
    else:
        raise ValueError(
            f"Unknown sampling_strategy: '{sampling}'. "
            f"Valid options: {SAMPLING_STRATEGIES}"
        )


def get_sampler(sampling_strategy: SamplingStrategy) -> SMOTE | ADASYN | None:
    """
    Get sampler instance for given strategy.

    Args:
        sampling_strategy: 'native', 'smote', or 'adasyn'

    Returns:
        Sampler instance or None for native strategy

    Raises:
        ValueError: If sampling_strategy is not valid
    """
    if sampling_strategy == 'native':
        return None
    elif sampling_strategy == 'smote':
        return SMOTE(random_state=RANDOM_STATE)
    elif sampling_strategy == 'adasyn':
        return ADASYN(random_state=RANDOM_STATE)
    else:
        raise ValueError(
            f"Unknown sampling_strategy: '{sampling_strategy}'. "
            f"Valid options: {SAMPLING_STRATEGIES}"
        )
