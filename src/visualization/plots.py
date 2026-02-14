"""
Visualization utilities for SECOM yield prediction.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import (
    RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay,
    roc_auc_score, average_precision_score, confusion_matrix, precision_score
)
from sklearn.calibration import calibration_curve
from scipy.stats import ks_2samp

from ..config import VIZ_CONFIG
from ..evaluation.metrics import calculate_gmean

logger = logging.getLogger('secom')


def finalize_figure(save_path: str | Path | None = None, close: bool = True) -> None:
    """
    Finalize figure with consistent save settings.

    Centralizes the save/close pattern used across all visualization functions.

    Args:
        save_path: Optional path to save the figure. If None, figure is not saved.
        close: Whether to close the figure after saving (default: True)
    """
    if save_path is not None:
        plt.savefig(save_path, dpi=VIZ_CONFIG['dpi'], bbox_inches='tight')
    if close:
        plt.close()

# Suppress sklearn FutureWarning about kwargs deprecation (fixed in sklearn 1.9+)
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn.utils._plotting')


def _extract_column(X: ArrayLike, col_idx: int) -> NDArray:
    """
    Extract column from ndarray or DataFrame.

    Args:
        X: Input array (numpy ndarray or pandas DataFrame)
        col_idx: Column index to extract

    Returns:
        Column values as numpy array
    """
    if isinstance(X, np.ndarray):
        return X[:, col_idx]
    else:
        return X.iloc[:, col_idx].values


def _compute_ks_statistics(
    X_train: ArrayLike,
    X_test: ArrayLike,
    feature_names: list[str]
) -> list[dict[str, Any]]:
    """
    Compute KS test statistics for all features.

    Args:
        X_train: Training features
        X_test: Test features
        feature_names: List of feature names

    Returns:
        List of dicts with KS test results per feature
    """
    n_features = X_train.shape[1]
    ks_results = []

    for i in range(n_features):
        train_col = _extract_column(X_train, i)
        test_col = _extract_column(X_test, i)

        ks_stat, p_value = ks_2samp(train_col, test_col)
        ks_results.append({
            'feature': feature_names[i],
            'ks_statistic': ks_stat,
            'p_value': p_value,
            'drift': p_value < 0.05
        })

    return ks_results


def _plot_feature_distributions(
    X_train: ArrayLike,
    X_test: ArrayLike,
    ks_results_sorted: list[dict[str, Any]],
    feature_names: list[str],
    max_features: int,
    save_path: str | Path
) -> None:
    """
    Plot histograms of top drifting features.

    Args:
        X_train: Training features
        X_test: Test features
        ks_results_sorted: KS results sorted by statistic (descending)
        feature_names: List of feature names
        max_features: Maximum number of features to plot
        save_path: Path to save the figure
    """
    n_plot = min(max_features, len(ks_results_sorted))
    n_cols = min(3, n_plot)
    n_rows = (n_plot + n_cols - 1) // n_cols

    _, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for idx, result in enumerate(ks_results_sorted[:n_plot]):
        feature_idx = feature_names.index(result['feature'])
        train_col = _extract_column(X_train, feature_idx)
        test_col = _extract_column(X_test, feature_idx)

        axes[idx].hist(train_col, bins=30, alpha=0.5, label='Train', density=True)
        axes[idx].hist(test_col, bins=30, alpha=0.5, label='Test', density=True)
        axes[idx].set_title(f"{result['feature']}\nKS={result['ks_statistic']:.3f}")
        axes[idx].legend()

    # Hide unused axes
    for idx in range(n_plot, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    finalize_figure(save_path)


def compare_feature_distributions(
    X_train: ArrayLike,
    X_test: ArrayLike,
    feature_names: list[str] | None = None,
    max_features: int = 10,
    save_path: str | Path | None = None
) -> list[dict[str, Any]]:
    """
    Compare feature distributions between train and test sets using KS test.

    Args:
        X_train: Training features array
        X_test: Test features array
        feature_names: Optional list of feature names
        max_features: Maximum number of features to plot (plots top by KS statistic)
        save_path: Optional path to save the figure

    Returns:
        List of dicts with KS test results for each feature
    """
    n_features = X_train.shape[1]

    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(n_features)]

    # Compute KS statistics for all features
    ks_results = _compute_ks_statistics(X_train, X_test, feature_names)

    # Sort by KS statistic and log drift count
    ks_results_sorted = sorted(ks_results, key=lambda x: x['ks_statistic'], reverse=True)
    drift_count = sum(1 for r in ks_results if r['drift'])
    logger.debug(f"Drift detected in {drift_count}/{n_features} features")

    # Plot if save_path provided
    if save_path is not None:
        _plot_feature_distributions(
            X_train, X_test, ks_results_sorted, feature_names, max_features, save_path
        )

    return ks_results


def plot_calibration_curve(
    y_true: ArrayLike,
    y_pred_proba: ArrayLike,
    model_name: str = "Model",
    n_bins: int = 10,
    save_path: str | Path | None = None
) -> dict[str, Any]:
    """
    Plot calibration curve to assess probability calibration quality.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities for positive class
        model_name: Model name for plot title
        n_bins: Number of bins for calibration curve
        save_path: Optional path to save the figure

    Returns:
        Dict with prob_true, prob_pred arrays and calibration_error
    """
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins, strategy='uniform')
    calibration_error = np.mean(np.abs(prob_true - prob_pred))

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
    ax.plot(prob_pred, prob_true, 's-', label=f'{model_name} (ECE={calibration_error:.3f})', linewidth=2, markersize=8)

    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives', fontsize=12)
    ax.set_title(f'Calibration Curve: {model_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    finalize_figure(save_path)

    return {
        'prob_true': prob_true,
        'prob_pred': prob_pred,
        'calibration_error': calibration_error
    }


def plot_roc_curve(
    y_true: ArrayLike,
    y_pred_proba: ArrayLike,
    model_name: str = "Model",
    save_path: str | Path | None = None
) -> dict[str, Any]:
    """
    Plot ROC curve and calculate AUC-ROC.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities for positive class
        model_name: Model name for plot title
        save_path: Optional path to save the figure

    Returns:
        Dict with auc_roc score and fpr/tpr arrays
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    display = RocCurveDisplay.from_predictions(
        y_true, y_pred_proba,
        ax=ax,
        name=model_name,
        color=VIZ_CONFIG['roc_color'],
        plot_chance_level=True
    )

    ax.set_title(f'ROC Curve: {model_name}', fontsize=VIZ_CONFIG['title_fontsize'], fontweight='bold')
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=VIZ_CONFIG['label_fontsize'])
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=VIZ_CONFIG['label_fontsize'])
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right', fontsize=10)

    auc_roc = roc_auc_score(y_true, y_pred_proba)
    ax.text(0.6, 0.2, f'AUC-ROC = {auc_roc:.4f}',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=11, verticalalignment='top')

    plt.tight_layout()
    finalize_figure(save_path)

    return {
        'auc_roc': auc_roc,
        'fpr': display.fpr,
        'tpr': display.tpr
    }


def plot_precision_recall_curve(
    y_true: ArrayLike,
    y_pred_proba: ArrayLike,
    model_name: str = "Model",
    save_path: str | Path | None = None
) -> dict[str, Any]:
    """
    Plot Precision-Recall curve and calculate AUC-PR.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities for positive class
        model_name: Model name for plot title
        save_path: Optional path to save the figure

    Returns:
        Dict with auc_pr score and precision/recall arrays
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    display = PrecisionRecallDisplay.from_predictions(
        y_true, y_pred_proba,
        ax=ax,
        name=model_name,
        color=VIZ_CONFIG['pr_color'],
        plot_chance_level=True
    )

    ax.set_title(f'Precision-Recall Curve: {model_name}', fontsize=VIZ_CONFIG['title_fontsize'], fontweight='bold')
    ax.set_xlabel('Recall (Sensitivity)', fontsize=VIZ_CONFIG['label_fontsize'])
    ax.set_ylabel('Precision', fontsize=VIZ_CONFIG['label_fontsize'])
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)

    auc_pr = average_precision_score(y_true, y_pred_proba)
    baseline = np.sum(y_true) / len(y_true)
    ax.text(0.4, 0.95, f'AUC-PR = {auc_pr:.4f}\nBaseline = {baseline:.4f}',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=11, verticalalignment='top')

    plt.tight_layout()
    finalize_figure(save_path)

    return {
        'auc_pr': auc_pr,
        'precision': display.precision,
        'recall': display.recall
    }


def plot_confusion_matrix(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    model_name: str = "Model",
    threshold: float = 0.5,
    save_path: str | Path | None = None
) -> dict[str, int]:
    """
    Plot confusion matrix heatmap with counts and percentages.

    Args:
        y_true: True labels
        y_pred: Predicted labels (binary)
        model_name: Model name for plot title
        threshold: Classification threshold used (for display)
        save_path: Optional path to save the figure

    Returns:
        Dict with tn, fp, fn, tp counts and total
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Pass', 'Fail'])
    disp.plot(ax=ax, cmap=VIZ_CONFIG['heatmap_cmap'], values_format='d', colorbar=True)

    ax.set_title(f'Confusion Matrix: {model_name}\n(Threshold = {threshold:.3f})',
                 fontsize=VIZ_CONFIG['title_fontsize'], fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=VIZ_CONFIG['label_fontsize'])
    ax.set_ylabel('True Label', fontsize=VIZ_CONFIG['label_fontsize'])

    tn, fp, fn, tp = cm.ravel()
    total = cm.sum()

    ax.text(0, 0, f'\n{tn/total*100:.1f}%', ha='center', va='center', fontsize=10, color='gray')
    ax.text(1, 0, f'\n{fp/total*100:.1f}%', ha='center', va='center', fontsize=10, color='gray')
    ax.text(0, 1, f'\n{fn/total*100:.1f}%', ha='center', va='center', fontsize=10, color='gray')
    ax.text(1, 1, f'\n{tp/total*100:.1f}%', ha='center', va='center', fontsize=10, color='gray')

    # Use centralized metric calculation (DRY)
    _, sensitivity, specificity = calculate_gmean(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)

    metrics_text = f'Sens: {sensitivity:.3f}\nSpec: {specificity:.3f}\nPrec: {prec:.3f}'
    ax.text(1.35, 0.5, metrics_text, transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10, verticalalignment='center')

    plt.tight_layout()
    finalize_figure(save_path)

    return {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp), 'total': int(total)}


def plot_threshold_curve(
    threshold_results: dict[str, Any],
    model_name: str = "Model",
    save_path: str | Path | None = None
) -> None:
    """
    Plot G-Mean vs threshold curve from threshold tuning results.

    Args:
        threshold_results: Dict from find_optimal_threshold_cv containing
                          threshold_curve, optimal_threshold, cv_gmean
        model_name: Model name for plot title
        save_path: Optional path to save the figure
    """
    thresholds = threshold_results['threshold_curve']['thresholds']
    gmean_scores = threshold_results['threshold_curve']['gmean_scores']
    optimal_threshold = threshold_results['optimal_threshold']
    optimal_gmean = threshold_results['cv_gmean']

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(thresholds, gmean_scores, color=VIZ_CONFIG['threshold_color'], linewidth=2, label='G-Mean')
    ax.axvline(x=optimal_threshold, color=VIZ_CONFIG['optimal_marker'], linestyle='--', linewidth=2,
               label=f'Optimal = {optimal_threshold:.3f}')
    ax.axvline(x=0.5, color=VIZ_CONFIG['neutral'], linestyle=':', linewidth=1.5, label='Default = 0.5')

    ax.scatter([optimal_threshold], [optimal_gmean], color=VIZ_CONFIG['optimal_marker'], s=100, zorder=5)
    ax.annotate(f'{optimal_gmean:.4f}',
                xy=(optimal_threshold, optimal_gmean),
                xytext=(optimal_threshold + 0.1, optimal_gmean - 0.05),
                fontsize=10,
                arrowprops=dict(arrowstyle='->', color=VIZ_CONFIG['optimal_marker']))

    ax.set_xlabel('Classification Threshold', fontsize=12)
    ax.set_ylabel('G-Mean (CV)', fontsize=12)
    ax.set_title(f'Threshold Optimization: {model_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, max(gmean_scores) * 1.1])

    plt.tight_layout()
    finalize_figure(save_path)
