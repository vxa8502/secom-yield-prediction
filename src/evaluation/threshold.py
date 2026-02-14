"""
Threshold optimization for SECOM yield prediction.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict

from ..config import DEFAULT_CLASSIFICATION_THRESHOLD, DEFAULT_COST_RATIO
from .metrics import calculate_gmean, calculate_weighted_gmean, get_cv_splitter, unpack_confusion_matrix

logger = logging.getLogger('secom')


MAX_THRESHOLDS = 1000  # Safety limit to prevent memory exhaustion


def _compute_gmean(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    cost_ratio: float = DEFAULT_COST_RATIO
) -> float:
    """
    Compute G-Mean with optional cost weighting.

    Uses standard G-Mean when cost_ratio=1.0, weighted otherwise.
    This helper centralizes the branching logic.
    """
    if cost_ratio == 1.0:
        gmean, _, _ = calculate_gmean(y_true, y_pred)
    else:
        gmean, _, _ = calculate_weighted_gmean(y_true, y_pred, cost_ratio)
    return gmean


def _validate_threshold_range(
    threshold_range: tuple[float, float],
    step: float,
    max_thresholds: int = MAX_THRESHOLDS
) -> None:
    """
    Validate threshold range parameters.

    Args:
        threshold_range: (min, max) bounds for threshold search
        step: Step size between thresholds
        max_thresholds: Maximum allowed number of thresholds (default: 1000)

    Raises:
        ValueError: If parameters are invalid or would generate too many thresholds
    """
    min_thresh, max_thresh = threshold_range

    if not (0.0 <= min_thresh < max_thresh <= 1.0):
        raise ValueError(
            f"threshold_range must satisfy 0 <= min < max <= 1, "
            f"got ({min_thresh}, {max_thresh})"
        )

    if step <= 0 or step > (max_thresh - min_thresh):
        raise ValueError(
            f"step must be positive and <= range width, "
            f"got step={step} for range ({min_thresh}, {max_thresh})"
        )

    # Check total threshold count to prevent memory exhaustion
    n_thresholds = int(np.ceil((max_thresh - min_thresh) / step)) + 1
    if n_thresholds > max_thresholds:
        raise ValueError(
            f"Step size {step} generates {n_thresholds} thresholds, "
            f"exceeding limit of {max_thresholds}. Use a larger step size."
        )


def find_optimal_threshold(
    y_true: ArrayLike,
    y_pred_proba: ArrayLike,
    metric: Literal['gmean', 'f1', 'sensitivity'] = 'gmean',
    threshold_range: tuple[float, float] = (0.01, 0.99),
    step: float = 0.01,
    cost_ratio: float = DEFAULT_COST_RATIO
) -> dict[str, Any]:
    """
    Find optimal classification threshold by sweeping thresholds.

    CRITICAL: Use this function BEFORE final model selection to ensure
    fair comparison between models. Different models have different
    probability calibration - comparing at default threshold=0.5 is unfair.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities for positive class
        metric: Metric to optimize ('gmean', 'f1', 'sensitivity')
        threshold_range: Min/max bounds for threshold search (default 0.01-0.99)
        step: Step size for threshold grid (default 0.01)
        cost_ratio: Relative cost of FN vs FP for weighted G-Mean (default 1.0).
                    Only applies when metric='gmean'. Higher values penalize
                    missed defects more heavily.

    Returns:
        dict with optimal_threshold, optimal_value, thresholds, metric_values

    Raises:
        ValueError: If threshold_range or step are invalid
    """
    _validate_threshold_range(threshold_range, step)
    thresholds = np.arange(threshold_range[0], threshold_range[1] + step, step)
    metric_values = []

    for threshold in thresholds:
        y_pred_thresh = (y_pred_proba >= threshold).astype(int)

        if metric == 'gmean':
            metric_values.append(_compute_gmean(y_true, y_pred_thresh, cost_ratio))
        elif metric == 'f1':
            metric_values.append(f1_score(y_true, y_pred_thresh, zero_division=0))
        elif metric == 'sensitivity':
            tn, fp, fn, tp = unpack_confusion_matrix(y_true, y_pred_thresh)
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            metric_values.append(sensitivity)

    # Find optimal threshold
    optimal_idx = np.argmax(metric_values)
    optimal_threshold = thresholds[optimal_idx]
    optimal_value = metric_values[optimal_idx]

    return {
        'optimal_threshold': optimal_threshold,
        'optimal_value': optimal_value,
        'thresholds': thresholds,
        'metric_values': np.array(metric_values),
        'metric_name': metric,
        'cost_ratio': cost_ratio
    }


def get_cv_predictions(
    pipeline: BaseEstimator,
    X: ArrayLike,
    y: ArrayLike,
    method: Literal['predict_proba', 'decision_function'] = 'predict_proba'
) -> NDArray[np.floating]:
    """
    Get cross-validated predictions for threshold tuning.

    CRITICAL: Always tune threshold on CV predictions, NEVER on test set.
    Using test set for threshold tuning = data leakage.

    Args:
        pipeline: Trained sklearn/imblearn pipeline
        X: Features
        y: Labels
        method: 'predict_proba' or 'decision_function'

    Returns:
        array: CV predictions (probabilities if method='predict_proba')
    """
    cv = get_cv_splitter()

    if method == 'predict_proba':
        cv_proba = cross_val_predict(pipeline, X, y, cv=cv, method='predict_proba', n_jobs=-1)
        return cv_proba[:, 1]
    else:
        return cross_val_predict(pipeline, X, y, cv=cv, method=method, n_jobs=-1)


def _sweep_thresholds(
    y_true: ArrayLike,
    y_proba: NDArray,
    thresholds: NDArray,
    cost_ratio: float = DEFAULT_COST_RATIO
) -> tuple[NDArray, int]:
    """
    Compute G-Mean scores across all thresholds.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        thresholds: Array of thresholds to evaluate
        cost_ratio: Relative cost of FN vs FP (default 1.0 = standard G-Mean)

    Returns:
        Tuple of (gmean_scores array, optimal_idx)
    """
    gmean_scores = np.array([
        _compute_gmean(y_true, (y_proba >= thresh).astype(int), cost_ratio)
        for thresh in thresholds
    ])
    optimal_idx = int(np.argmax(gmean_scores))
    return gmean_scores, optimal_idx


def _get_metrics_at_threshold(
    y_true: ArrayLike,
    y_proba: NDArray,
    threshold: float
) -> tuple[float, float, float]:
    """
    Compute G-Mean, sensitivity, and specificity at a specific threshold.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        threshold: Classification threshold

    Returns:
        Tuple of (gmean, sensitivity, specificity)
    """
    y_pred = (y_proba >= threshold).astype(int)
    return calculate_gmean(y_true, y_pred)


def _build_config_string(
    model_name: str | None,
    feature_set: str | None,
    sampling_strategy: str | None
) -> str:
    """Build configuration string for logging."""
    config_str = model_name or "model"
    if feature_set:
        config_str += f"+{feature_set}"
    if sampling_strategy:
        config_str += f"+{sampling_strategy}"
    return config_str


def find_optimal_threshold_cv(
    pipeline: BaseEstimator,
    X_train: ArrayLike,
    y_train: ArrayLike,
    threshold_range: tuple[float, float] = (0.01, 0.99),
    step: float = 0.01,
    model_name: str | None = None,
    feature_set: str | None = None,
    sampling_strategy: str | None = None,
    cost_ratio: float = DEFAULT_COST_RATIO
) -> dict[str, Any]:
    """
    Find optimal classification threshold using cross-validation predictions.

    Uses CV predictions to avoid data leakage - threshold is tuned on
    out-of-fold predictions, not the same data used to fit the model.

    Args:
        pipeline: Trained sklearn/imblearn pipeline with predict_proba support
        X_train: Training features (used for CV, not held-out test set)
        y_train: Training labels
        threshold_range: Min/max bounds for threshold search, default (0.01, 0.99)
        step: Step size for threshold grid, default 0.01 (99 thresholds)
        model_name: Optional identifier for logging (e.g., 'LogReg', 'XGBoost')
        feature_set: Optional identifier for logging (e.g., 'lasso', 'pca')
        sampling_strategy: Optional identifier for logging (e.g., 'SMOTE', 'ADASYN')
        cost_ratio: Relative cost of FN vs FP (default 1.0 = standard G-Mean).
                    Use higher values (e.g., 5.0, 10.0) to penalize missed defects
                    more heavily in manufacturing contexts.

    Returns:
        dict containing:
            optimal_threshold: Best threshold maximizing (weighted) G-Mean
            cv_gmean: G-Mean at optimal threshold
            cv_gmean_at_default: G-Mean at threshold=0.5 (baseline)
            threshold_improvement: Gain from optimization (cv_gmean - default)
            cv_sensitivity: Sensitivity at optimal threshold
            cv_specificity: Specificity at optimal threshold
            cv_sensitivity_at_default: Sensitivity at threshold=0.5
            cv_specificity_at_default: Specificity at threshold=0.5
            threshold_curve: Dict with 'thresholds' and 'gmean_scores' arrays
            cost_ratio: The cost ratio used for optimization

    Raises:
        ValueError: If threshold_range or step are invalid
    """
    _validate_threshold_range(threshold_range, step)
    cv_proba = get_cv_predictions(pipeline, X_train, y_train, method='predict_proba')
    thresholds = np.arange(threshold_range[0], threshold_range[1] + step, step)

    # Sweep all thresholds with cost-sensitive metric
    gmean_scores, optimal_idx = _sweep_thresholds(y_train, cv_proba, thresholds, cost_ratio)
    optimal_threshold = thresholds[optimal_idx]
    optimal_gmean = gmean_scores[optimal_idx]

    # Baseline at default threshold
    default_idx = int(np.argmin(np.abs(thresholds - DEFAULT_CLASSIFICATION_THRESHOLD)))
    gmean_at_default = gmean_scores[default_idx]
    _, sens_default, spec_default = _get_metrics_at_threshold(
        y_train, cv_proba, thresholds[default_idx]
    )

    # Metrics at optimal threshold
    _, sensitivity, specificity = _get_metrics_at_threshold(
        y_train, cv_proba, optimal_threshold
    )

    improvement = optimal_gmean - gmean_at_default
    config_str = _build_config_string(model_name, feature_set, sampling_strategy)

    logger.debug(f"{config_str}: thresh={optimal_threshold:.3f} gmean={optimal_gmean:.4f} (default={gmean_at_default:.4f})")

    return {
        'optimal_threshold': optimal_threshold,
        'cv_gmean': optimal_gmean,
        'cv_gmean_at_default': gmean_at_default,
        'threshold_improvement': improvement,
        'cv_sensitivity': sensitivity,
        'cv_specificity': specificity,
        'cv_sensitivity_at_default': sens_default,
        'cv_specificity_at_default': spec_default,
        'threshold_curve': {
            'thresholds': thresholds,
            'gmean_scores': np.array(gmean_scores)
        },
        'cost_ratio': cost_ratio
    }
