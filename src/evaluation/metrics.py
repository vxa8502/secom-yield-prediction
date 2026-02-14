"""
Core evaluation metrics for SECOM yield prediction.
Author: Victoria A.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score,
    f1_score, roc_auc_score, average_precision_score, make_scorer
)
from sklearn.model_selection import cross_val_score, StratifiedKFold

from ..config import RANDOM_STATE, CV_FOLDS

logger = logging.getLogger('secom')


def get_cv_splitter(n_splits: int | None = None) -> StratifiedKFold:
    """
    Get configured StratifiedKFold splitter.

    Centralizes CV configuration to ensure consistency across all evaluation code.

    Args:
        n_splits: Number of folds (default: CV_FOLDS from config)

    Returns:
        Configured StratifiedKFold instance
    """
    return StratifiedKFold(
        n_splits=n_splits or CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )


def unpack_confusion_matrix(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[int, int, int, int]:
    """
    Unpack confusion matrix into (tn, fp, fn, tp).

    Centralizes CM unpacking to ensure consistent label ordering.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Tuple of (tn, fp, fn, tp)
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return cm.ravel()


def _compute_sensitivity_specificity(
    y_true: ArrayLike,
    y_pred: ArrayLike
) -> tuple[float, float]:
    """
    Compute sensitivity and specificity from predictions.

    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels

    Returns:
        tuple: (sensitivity, specificity)
    """
    tn, fp, fn, tp = unpack_confusion_matrix(y_true, y_pred)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return sensitivity, specificity


def calculate_gmean(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[float, float, float]:
    """
    Calculate G-Mean (geometric mean of sensitivity and specificity).

    Returns:
        tuple: (gmean, sensitivity, specificity)
    """
    sensitivity, specificity = _compute_sensitivity_specificity(y_true, y_pred)
    gmean = np.sqrt(sensitivity * specificity)
    return gmean, sensitivity, specificity


def calculate_weighted_gmean(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    cost_ratio: float = 1.0
) -> tuple[float, float, float]:
    """
    Calculate weighted G-Mean for cost-sensitive classification.

    In manufacturing, missed defects (FN) typically cost 2-10x more than
    false alarms (FP). This weighted G-Mean penalizes FN more heavily.

    Formula: (sensitivity^beta * specificity)^(1/(1+beta))
    where beta = cost_ratio. Higher ratio = penalize FN more.

    Note: When cost_ratio=1.0, this reduces to standard G-Mean.

    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        cost_ratio: Relative cost of FN vs FP (default 1.0 = standard G-Mean)
                    Use 5.0 for typical manufacturing, 10.0 for critical defects

    Returns:
        tuple: (weighted_gmean, sensitivity, specificity)

    Examples:
        cost_ratio=1.0: Standard G-Mean (sensitivity and specificity equal)
        cost_ratio=5.0: Sensitivity weighted 5x (catches more defects)
        cost_ratio=10.0: Sensitivity weighted 10x (critical defect detection)
    """
    if cost_ratio <= 0:
        raise ValueError(f"cost_ratio must be positive, got {cost_ratio}")

    sensitivity, specificity = _compute_sensitivity_specificity(y_true, y_pred)

    if sensitivity == 0 or specificity == 0:
        weighted_gmean = 0.0
    else:
        # Formula: (sens^beta * spec)^(1/(1+beta))
        # When beta=1: (sens * spec)^0.5 = sqrt(sens * spec) = standard G-Mean
        weighted_gmean = (sensitivity ** cost_ratio * specificity) ** (1 / (1 + cost_ratio))

    return weighted_gmean, sensitivity, specificity


def gmean_scorer_func(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """G-Mean scorer function for cross_val_score."""
    gmean, _, _ = calculate_gmean(y_true, y_pred)
    return gmean


gmean_scorer = make_scorer(gmean_scorer_func)


def evaluate_at_threshold(
    y_true: ArrayLike,
    y_proba: ArrayLike,
    threshold: float
) -> dict[str, Any]:
    """
    Evaluate predictions at a specific threshold.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities for positive class
        threshold: Classification threshold

    Returns:
        dict: Evaluation metrics
    """
    y_pred = (y_proba >= threshold).astype(int)
    return evaluate_model(y_true, y_pred, y_proba, f"threshold={threshold:.3f}")


def evaluate_model(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    y_pred_proba: ArrayLike | None = None,
    model_name: str = "Model"
) -> dict[str, Any]:
    """
    Comprehensive model evaluation.

    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        y_pred_proba: Predicted probabilities for positive class (should be in [0, 1])
        model_name: Identifier for logging

    Returns:
        dict with evaluation metrics (gmean, sensitivity, specificity, precision, etc.)
    """
    gmean, sensitivity, specificity = calculate_gmean(y_true, y_pred)

    results = {
        'model': model_name,
        'gmean': gmean,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'accuracy': accuracy_score(y_true, y_pred),
    }

    if y_pred_proba is not None:
        # Validate probabilities are in expected range
        proba_arr = np.asarray(y_pred_proba)
        if np.any(proba_arr < 0) or np.any(proba_arr > 1):
            min_val, max_val = proba_arr.min(), proba_arr.max()
            logger.warning(
                f"y_pred_proba contains values outside [0, 1] range "
                f"(min={min_val:.3f}, max={max_val:.3f}). "
                f"Possible causes: "
                f"(1) Using decision_function instead of predict_proba, "
                f"(2) Model not calibrated, "
                f"(3) Data type mismatch. "
                f"AUC metrics may be unreliable."
            )
        results['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
        results['auc_pr'] = average_precision_score(y_true, y_pred_proba)

    return results


def run_cv_evaluation(
    pipeline: BaseEstimator,
    X_train: ArrayLike,
    y_train: ArrayLike,
    experiment_name: str
) -> dict[str, Any]:
    """Run stratified cross-validation using CV_FOLDS from config."""
    cv = get_cv_splitter()
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=gmean_scorer, n_jobs=-1)

    logger.debug(f"{experiment_name}: CV G-Mean={cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    return {
        'model': experiment_name,
        'cv_gmean_mean': cv_scores.mean(),
        'cv_gmean_std': cv_scores.std(),
        'cv_gmean_min': cv_scores.min(),
        'cv_gmean_max': cv_scores.max()
    }


def benchmark_prediction_latency(
    model: BaseEstimator,
    X_test: ArrayLike,
    n_runs: int = 100
) -> dict[str, float | int]:
    """Benchmark prediction latency."""
    n_samples = X_test.shape[0]

    # Warm-up
    _ = model.predict(X_test)

    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = model.predict(X_test)
        latencies.append(time.perf_counter() - start)

    latencies = np.array(latencies)
    mean_latency = np.mean(latencies)
    per_sample = mean_latency / n_samples

    logger.info(f"Latency: {per_sample*1000:.4f} ms/sample, {n_samples/mean_latency:.0f} samples/sec")

    return {
        'mean_total_latency_ms': mean_latency * 1000,
        'std_total_latency_ms': np.std(latencies) * 1000,
        'mean_per_sample_latency_ms': per_sample * 1000,
        'throughput_per_second': n_samples / mean_latency,
        'n_samples': n_samples,
        'n_runs': n_runs
    }
