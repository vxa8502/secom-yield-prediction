"""
Model registry and hyperparameter spaces for SECOM yield prediction.
Author: Victoria A.

Defines supported models, their search spaces for Optuna, and default parameters.

METHODOLOGY NOTES:
- Class imbalance ratio is computed dynamically when y_train is available
- DEFAULT_CLASS_IMBALANCE_RATIO (14) is used as fallback
- XGBoost scale_pos_weight uses the actual computed ratio for accuracy
"""

from __future__ import annotations

from typing import Any, Callable

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from ..config import RANDOM_STATE, DEFAULT_CLASS_IMBALANCE_RATIO

# Supported model types
SUPPORTED_MODELS: list[str] = ['LogReg', 'RandomForest', 'XGBoost', 'SVM']

# Parameters that must be integers (used for type conversion from Optuna floats)
INT_PARAMS: frozenset[str] = frozenset({
    'max_iter', 'n_estimators', 'max_depth', 'min_samples_split',
    'min_samples_leaf', 'min_child_weight'
})

# Default parameters for each model.
# PURPOSE: Quick testing, baseline comparisons, and fallback when Optuna tuning fails.
# NOTE: These are NOT used in the tuning pipeline (which uses Optuna search spaces).
#       For tuned models, use build_model() with params from tuning results.
DEFAULT_PARAMS = {
    'LogReg': {
        'l1_ratio': 0.5,
        'C': 1.0,
        'solver': 'saga',
        'max_iter': 1000,
        'random_state': RANDOM_STATE,
    },
    'RandomForest': {
        'n_estimators': 200,
        'max_depth': 20,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'random_state': RANDOM_STATE,
        'n_jobs': 1,
    },
    'XGBoost': {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.01,
        'reg_lambda': 1.0,
        'random_state': RANDOM_STATE,
        'n_jobs': 1,
        'eval_metric': 'logloss',
    },
    'SVM': {
        'kernel': 'rbf',
        'C': 1.0,
        'gamma': 'scale',
        'probability': True,
        'random_state': RANDOM_STATE,
    },
}

# Optuna search spaces for hyperparameter tuning
SEARCH_SPACES = {
    'LogReg': {
        'l1_ratio': ('float', 0.0, 1.0),
        'C': ('float_log', 1e-5, 1000),
        'max_iter': ('int_step', 1000, 5000, 100),
    },
    'RandomForest': {
        'n_estimators': ('int_step', 100, 500, 50),
        'max_depth': ('int_step', 5, 50, 5),
        'min_samples_split': ('int_step', 2, 20, 2),
        'min_samples_leaf': ('int', 1, 10),
        'max_features': ('categorical', ['sqrt', 'log2', None]),
    },
    'XGBoost': {
        'n_estimators': ('int_step', 50, 500, 50),
        'max_depth': ('int', 3, 10),
        'learning_rate': ('float_log', 0.001, 0.5),
        'subsample': ('float', 0.6, 1.0),
        'colsample_bytree': ('float', 0.6, 1.0),
        'min_child_weight': ('int', 1, 10),
        'gamma': ('float_log', 1e-8, 1.0),
        'reg_alpha': ('float_log', 1e-8, 1.0),
        'reg_lambda': ('float_log', 1e-8, 1.0),
    },
    'SVM': {
        'kernel': ('categorical', ['linear', 'rbf']),
        'C': ('float_log', 1e-4, 1000),
        'gamma': ('float_log', 1e-4, 1.0),  # Only used for RBF kernel
    },
}


def suggest_param(trial: Any, param_name: str, space: tuple) -> Any:
    """
    Suggest a hyperparameter value from Optuna trial using SEARCH_SPACES format.

    Args:
        trial: Optuna trial object
        param_name: Name of the parameter
        space: Tuple from SEARCH_SPACES (type, *args)

    Returns:
        Suggested parameter value

    Raises:
        ValueError: If space type is unknown
    """
    ptype, *args = space

    if ptype == 'int':
        return trial.suggest_int(param_name, args[0], args[1])
    elif ptype == 'int_step':
        return trial.suggest_int(param_name, args[0], args[1], step=args[2])
    elif ptype == 'float':
        return trial.suggest_float(param_name, args[0], args[1])
    elif ptype == 'float_log':
        return trial.suggest_float(param_name, args[0], args[1], log=True)
    elif ptype == 'categorical':
        return trial.suggest_categorical(param_name, args[0])
    else:
        raise ValueError(f"Unknown parameter type: {ptype}")

# Class balancing parameters for each model type
# Note: XGBoost scale_pos_weight should be computed dynamically when possible
CLASS_BALANCE_PARAMS = {
    'LogReg': {'class_weight': 'balanced'},
    'RandomForest': {'class_weight': 'balanced'},
    'XGBoost': {'scale_pos_weight': DEFAULT_CLASS_IMBALANCE_RATIO},  # Use dynamic ratio when available
    'SVM': {'class_weight': 'balanced'},
}


def get_class_balance_params(model_name: str, class_ratio: float | None = None) -> dict[str, Any]:
    """
    Get class balancing parameters for a model.

    Args:
        model_name: Model type
        class_ratio: Actual class imbalance ratio (computed from y_train)
                    If None, uses DEFAULT_CLASS_IMBALANCE_RATIO

    Returns:
        Dict of class balancing parameters
    """
    if model_name == 'XGBoost':
        ratio = class_ratio if class_ratio is not None else DEFAULT_CLASS_IMBALANCE_RATIO
        return {'scale_pos_weight': ratio}
    return CLASS_BALANCE_PARAMS.get(model_name, {})

# Fixed parameters that are always applied
FIXED_PARAMS = {
    'LogReg': {'solver': 'saga', 'random_state': RANDOM_STATE},
    'RandomForest': {'random_state': RANDOM_STATE, 'n_jobs': 1},
    'XGBoost': {'random_state': RANDOM_STATE, 'n_jobs': 1, 'eval_metric': 'logloss'},
    'SVM': {'random_state': RANDOM_STATE, 'probability': True},
}

# Model class mapping
MODEL_CLASSES = {
    'LogReg': LogisticRegression,
    'RandomForest': RandomForestClassifier,
    'XGBoost': XGBClassifier,
    'SVM': SVC,
}


def create_logreg_model(trial: Any, with_class_balancing: bool = True) -> LogisticRegression:
    """Create LogisticRegression with Optuna-suggested hyperparameters."""
    space = SEARCH_SPACES['LogReg']
    params = {
        'l1_ratio': suggest_param(trial, 'l1_ratio', space['l1_ratio']),
        'C': suggest_param(trial, 'C', space['C']),
        'max_iter': suggest_param(trial, 'max_iter', space['max_iter']),
        **FIXED_PARAMS['LogReg']
    }
    if with_class_balancing:
        params['class_weight'] = 'balanced'
    return LogisticRegression(**params)


def create_randomforest_model(trial: Any, with_class_balancing: bool = True) -> RandomForestClassifier:
    """Create RandomForestClassifier with Optuna-suggested hyperparameters."""
    space = SEARCH_SPACES['RandomForest']
    params = {
        'n_estimators': suggest_param(trial, 'n_estimators', space['n_estimators']),
        'max_depth': suggest_param(trial, 'max_depth', space['max_depth']),
        'min_samples_split': suggest_param(trial, 'min_samples_split', space['min_samples_split']),
        'min_samples_leaf': suggest_param(trial, 'min_samples_leaf', space['min_samples_leaf']),
        'max_features': suggest_param(trial, 'max_features', space['max_features']),
        **FIXED_PARAMS['RandomForest']
    }
    if with_class_balancing:
        params['class_weight'] = 'balanced'
    return RandomForestClassifier(**params)


def create_xgboost_model(
    trial: Any,
    with_class_balancing: bool = True,
    class_ratio: float | None = None
) -> XGBClassifier:
    """
    Create XGBClassifier with Optuna-suggested hyperparameters.

    Args:
        trial: Optuna trial object
        with_class_balancing: Whether to apply class balancing
        class_ratio: Actual class imbalance ratio (computed from y_train)
                    If None, uses DEFAULT_CLASS_IMBALANCE_RATIO
    """
    space = SEARCH_SPACES['XGBoost']
    params = {
        'n_estimators': suggest_param(trial, 'n_estimators', space['n_estimators']),
        'max_depth': suggest_param(trial, 'max_depth', space['max_depth']),
        'learning_rate': suggest_param(trial, 'learning_rate', space['learning_rate']),
        'subsample': suggest_param(trial, 'subsample', space['subsample']),
        'colsample_bytree': suggest_param(trial, 'colsample_bytree', space['colsample_bytree']),
        'min_child_weight': suggest_param(trial, 'min_child_weight', space['min_child_weight']),
        'gamma': suggest_param(trial, 'gamma', space['gamma']),
        'reg_alpha': suggest_param(trial, 'reg_alpha', space['reg_alpha']),
        'reg_lambda': suggest_param(trial, 'reg_lambda', space['reg_lambda']),
        **FIXED_PARAMS['XGBoost']
    }
    if with_class_balancing:
        ratio = class_ratio if class_ratio is not None else DEFAULT_CLASS_IMBALANCE_RATIO
        params['scale_pos_weight'] = ratio
    return XGBClassifier(**params)


def create_svm_model(trial: Any, with_class_balancing: bool = True) -> SVC:
    """Create SVC with Optuna-suggested hyperparameters."""
    space = SEARCH_SPACES['SVM']
    kernel = suggest_param(trial, 'kernel', space['kernel'])
    params = {
        'kernel': kernel,
        'C': suggest_param(trial, 'C', space['C']),
        **FIXED_PARAMS['SVM']
    }
    if kernel == 'rbf':
        params['gamma'] = suggest_param(trial, 'gamma', space['gamma'])
    if with_class_balancing:
        params['class_weight'] = 'balanced'
    return SVC(**params)


# Model factory functions for Optuna
MODEL_FACTORIES: dict[str, Callable[[Any, bool], BaseEstimator]] = {
    'LogReg': create_logreg_model,
    'RandomForest': create_randomforest_model,
    'XGBoost': create_xgboost_model,
    'SVM': create_svm_model
}
