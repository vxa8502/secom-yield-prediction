#!/usr/bin/env python
"""
SECOM Production Model Selection Pipeline

Usage:
    python -m pipelines.select
    python -m pipelines.select --top-n=4 --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator

# Add project root to path for module imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import VIZ_CONFIG, MODELS_DIR, REPORTS_DIR, FIGURES_DIR, DATA_DIR, setup_logging, ensure_directories, DEFAULT_COST_RATIO
from src.data import load_labels, load_all_feature_sets
from src.evaluation import evaluate_model, benchmark_prediction_latency, find_optimal_threshold_cv
from src.visualization import (
    plot_confusion_matrix,
    plot_calibration_curve,
    plot_roc_curve,
    plot_precision_recall_curve,
)
from src.visualization import create_shap_explainer, compute_shap_values, plot_shap_summary, get_top_shap_features
from src.models import build_pipeline
from src.mlflow_utils import setup_mlflow

logger = logging.getLogger('secom')


def load_all_tuning_results() -> pd.DataFrame:
    """Load and combine results from all tuning runs."""
    dfs = []
    for strategy in ['native', 'smote', 'adasyn']:
        path = REPORTS_DIR / f'tuning_{strategy}_results.csv'
        if path.exists():
            dfs.append(pd.read_csv(path))

    if not dfs:
        raise FileNotFoundError(
            "No tuning results found. Run 'make tune' first."
        )

    all_results = pd.concat(dfs, ignore_index=True)
    all_results.to_csv(REPORTS_DIR / 'all_tuning_results.csv', index=False)
    logger.info(f"Loaded {len(all_results)} experiments")
    return all_results


def run_threshold_tuning(
    top_n_df: pd.DataFrame,
    feature_sets: dict[str, tuple[NDArray, NDArray]],
    y_train: NDArray,
    cost_ratio: float = DEFAULT_COST_RATIO
) -> pd.DataFrame:
    """Run threshold tuning for top N candidates."""
    results = []

    for _, row in top_n_df.iterrows():
        X_train, _ = feature_sets[row['feature_set']]
        pipeline = build_pipeline(row)

        with mlflow.start_run(run_name=f"threshold_{row['model']}_{row['feature_set']}_{row['sampling_strategy']}"):
            mlflow.log_param("model", row['model'])
            mlflow.log_param("feature_set", row['feature_set'])
            mlflow.log_param("sampling_strategy", row['sampling_strategy'])
            mlflow.log_param("cost_ratio", cost_ratio)

            threshold_result = find_optimal_threshold_cv(
                pipeline=pipeline,
                X_train=X_train,
                y_train=y_train,
                threshold_range=(0.01, 0.99),
                step=0.01,
                model_name=row['model'],
                feature_set=row['feature_set'],
                sampling_strategy=row['sampling_strategy'],
                cost_ratio=cost_ratio
            )

            mlflow.log_metric("optimal_threshold", threshold_result['optimal_threshold'])
            mlflow.log_metric("cv_gmean_at_optimal", threshold_result['cv_gmean'])

            result = {
                'model': row['model'],
                'feature_set': row['feature_set'],
                'n_features': X_train.shape[1],
                'sampling_strategy': row['sampling_strategy'],
                'optimal_threshold': threshold_result['optimal_threshold'],
                'cv_gmean_at_default': threshold_result['cv_gmean_at_default'],
                'cv_gmean_at_optimal': threshold_result['cv_gmean'],
                'cv_sensitivity': threshold_result['cv_sensitivity'],
                'cv_specificity': threshold_result['cv_specificity'],
                'threshold_curve': threshold_result['threshold_curve'],
                'cost_ratio': cost_ratio
            }

            for col in row.index:
                if col.startswith('param_'):
                    result[col] = row[col]

            results.append(result)

    return pd.DataFrame(results)


def plot_threshold_curves(threshold_df: pd.DataFrame, save_path: Path) -> None:
    """Plot threshold tuning curves."""
    n = len(threshold_df)
    _, axes = plt.subplots((n + 1) // 2, 2, figsize=(12, 4 * ((n + 1) // 2)))
    axes = axes.flatten() if n > 1 else [axes]

    # Use enumerate for 0-based indexing (iterrows idx is DataFrame index, not sequential)
    for plot_idx, (_, row) in enumerate(threshold_df.iterrows()):
        ax = axes[plot_idx]
        thresholds = row['threshold_curve']['thresholds']
        gmean_scores = row['threshold_curve']['gmean_scores']

        ax.plot(thresholds, gmean_scores, linewidth=2, color=VIZ_CONFIG['primary'])
        ax.axvline(row['optimal_threshold'], color=VIZ_CONFIG['fail_color'], linestyle='--', linewidth=2)
        ax.scatter([row['optimal_threshold']], [row['cv_gmean_at_optimal']],
                   color=VIZ_CONFIG['fail_color'], s=80, zorder=5)

        ax.set_xlabel('Threshold')
        ax.set_ylabel('CV G-Mean')
        ax.set_title(f"{row['model']} ({row['feature_set']}) - opt={row['optimal_threshold']:.2f}")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)

    for idx in range(len(threshold_df), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=VIZ_CONFIG['dpi'], bbox_inches='tight')
    plt.close()


def _extract_sklearn_feature_importance(
    classifier: BaseEstimator,
    feature_names: list[str]
) -> pd.DataFrame | None:
    """
    Extract feature importance from sklearn model as SHAP fallback.

    Supports tree-based models (feature_importances_) and linear models (coef_).
    Returns None if model doesn't support feature importance extraction.
    """
    importance_values = None

    if hasattr(classifier, 'feature_importances_'):
        importance_values = classifier.feature_importances_
        importance_type = 'gini_importance'
    elif hasattr(classifier, 'coef_'):
        importance_values = np.abs(classifier.coef_).flatten()
        importance_type = 'abs_coefficient'

    if importance_values is None:
        return None

    # Validate feature_names matches importance_values length
    if len(feature_names) != len(importance_values):
        logger.warning(
            f"Feature names length ({len(feature_names)}) doesn't match "
            f"importance values length ({len(importance_values)}). Skipping fallback."
        )
        return None

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_values,
        'importance_type': importance_type
    })
    importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
    importance_df['rank'] = range(1, len(importance_df) + 1)

    return importance_df.head(10)


def generate_shap_explanations(
    pipeline: BaseEstimator,
    production_row: pd.Series,
    X_train: NDArray,
    X_test: NDArray
) -> pd.DataFrame | None:
    """
    Generate SHAP explanations with sklearn fallback.

    If SHAP computation fails (e.g., unsupported model type), falls back to
    sklearn feature_importances_ or coef_ extraction.

    Returns None only if both SHAP and sklearn fallback fail.
    """
    classifier = pipeline.named_steps.get('classifier', pipeline.steps[-1][1])

    if production_row['feature_set'] == 'lasso':
        lasso_features_path = DATA_DIR / 'lasso_selected_features.txt'
        with open(lasso_features_path) as f:
            feature_names = [line.strip() for line in f if line.strip()]
    elif production_row['feature_set'] == 'pca':
        feature_names = [f"PC{i+1}" for i in range(X_test.shape[1])]
    else:
        feature_names = [f"feature_{i}" for i in range(X_test.shape[1])]

    # Try SHAP first
    try:
        explainer = create_shap_explainer(classifier, X_train, max_background_samples=100)
        shap_values, sample_indices = compute_shap_values(explainer, X_test, max_samples=200)

        # Use correct subset of X_test that matches shap_values
        if sample_indices is not None:
            X_explain = X_test[sample_indices]
        else:
            X_explain = X_test

        plot_shap_summary(shap_values, X_explain, feature_names,
                          plot_type='bar', max_display=15, save_path=FIGURES_DIR / 'shap_importance.png')

        top_shap = get_top_shap_features(shap_values, feature_names, top_n=10)
        top_shap.to_csv(REPORTS_DIR / 'shap_feature_importance.csv', index=False)
        logger.info("SHAP explanations generated successfully")

        return top_shap

    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}")
        logger.info("Attempting sklearn feature importance fallback")

        # Fallback to sklearn feature importance
        fallback_importance = _extract_sklearn_feature_importance(classifier, feature_names)

        if fallback_importance is not None:
            fallback_importance.to_csv(REPORTS_DIR / 'sklearn_feature_importance.csv', index=False)
            logger.info(f"Sklearn feature importance extracted ({fallback_importance['importance_type'].iloc[0]})")
            return fallback_importance

        logger.warning("No feature importance available for this model type")
        return None


def _safe_json_write(data: dict[str, Any], path: Path) -> None:
    """
    Write JSON with atomic write pattern and error handling.

    Uses a temporary file and atomic replace to prevent partial writes.
    If the write fails, the original file (if any) remains intact.

    Args:
        data: Dictionary to serialize as JSON
        path: Target file path

    Raises:
        RuntimeError: If write fails after cleanup
    """
    # Use .tmp suffix with unique path to avoid collisions
    temp_path = path.with_name(f".{path.name}.tmp")

    try:
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write to temp file with explicit encoding
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        # Atomic replace (works on same filesystem)
        temp_path.replace(path)
        logger.debug(f"Wrote {path}")

    except (IOError, OSError, TypeError, json.JSONDecodeError) as e:
        # Cleanup temp file on failure
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            logger.warning(f"Could not clean up temp file: {temp_path}")

        logger.error(f"Failed to write {path}: {type(e).__name__}: {e}")
        raise RuntimeError(f"Failed to write JSON to {path}: {e}") from e


def save_artifacts(
    pipeline: BaseEstimator,
    production_row: pd.Series,
    test_results: dict[str, Any],
    training_time: float,
    threshold_df: pd.DataFrame
) -> None:
    """
    Save production artifacts with error handling.

    Raises:
        IOError: If file write fails
    """
    # Ensure directories exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save model with joblib
    model_path = MODELS_DIR / 'production_model.pkl'
    try:
        joblib.dump(pipeline, model_path)
        logger.debug(f"Saved model: {model_path}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        raise

    cost_ratio = float(production_row.get('cost_ratio', DEFAULT_COST_RATIO))
    threshold_info = {
        'optimal_threshold': float(production_row['optimal_threshold']),
        'model_name': str(production_row['model']),
        'feature_set': str(production_row['feature_set']),
        'n_features': int(production_row['n_features']),
        'sampling_strategy': str(production_row['sampling_strategy']),
        'cv_gmean': float(production_row['cv_gmean_at_optimal']),
        'cost_ratio': cost_ratio
    }
    _safe_json_write(threshold_info, MODELS_DIR / 'production_threshold.json')

    metadata = {
        **threshold_info,
        # Complete holdout metrics for production record
        'test_gmean': float(test_results['gmean']),
        'test_sensitivity': float(test_results['sensitivity']),
        'test_specificity': float(test_results['specificity']),
        'test_precision': float(test_results['precision']),
        'test_f1_score': float(test_results['f1_score']),
        'test_accuracy': float(test_results['accuracy']),
        'test_auc_roc': float(test_results['auc_roc']),
        'test_auc_pr': float(test_results.get('auc_pr', 0.0)),
        'training_time_seconds': float(training_time),
        'timestamp': pd.Timestamp.now().isoformat()
    }
    _safe_json_write(metadata, MODELS_DIR / 'production_model_metadata.json')

    summary = threshold_df[['model', 'feature_set', 'sampling_strategy', 'cv_gmean_at_optimal', 'optimal_threshold']]
    summary.to_csv(REPORTS_DIR / 'finalists_summary.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description='SECOM production model selection')
    parser.add_argument('--top-n', type=int, default=4, help='Top N candidates for threshold tuning')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--cost-ratio', type=float, default=DEFAULT_COST_RATIO,
                        help=f'FN/FP cost ratio for weighted G-Mean (default: {DEFAULT_COST_RATIO})')
    args = parser.parse_args()

    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    ensure_directories()
    setup_mlflow()

    # Load data (DRY: use centralized loader)
    y_train, y_test = load_labels()
    feature_sets = load_all_feature_sets()

    # Load and rank experiments
    all_results = load_all_tuning_results()
    ranked = all_results.sort_values('cv_gmean', ascending=False).reset_index(drop=True)
    top_n = ranked.head(args.top_n).copy()

    logger.info(f"Top {args.top_n}: " + ", ".join(
        f"{r['model']}+{r['feature_set']}={r['cv_gmean']:.4f}" for _, r in top_n.iterrows()))

    # Threshold tuning
    logger.info(f"Running threshold optimization (cost_ratio={args.cost_ratio})")
    threshold_df = run_threshold_tuning(top_n, feature_sets, y_train, cost_ratio=args.cost_ratio)
    threshold_df.drop(columns=['threshold_curve']).to_csv(REPORTS_DIR / 'threshold_results.csv', index=False)
    plot_threshold_curves(threshold_df, FIGURES_DIR / 'threshold_curves.png')

    # Select production model
    production_row = threshold_df.loc[threshold_df['cv_gmean_at_optimal'].idxmax()]

    logger.info(f"Selected: {production_row['model']}+{production_row['feature_set']}+{production_row['sampling_strategy']} "
                f"thresh={production_row['optimal_threshold']:.3f} gmean={production_row['cv_gmean_at_optimal']:.4f}")

    # Train final model
    X_train, X_test = feature_sets[production_row['feature_set']]
    pipeline = build_pipeline(production_row)

    start = time.time()
    pipeline.fit(X_train, y_train)
    training_time = time.time() - start

    # Evaluate on holdout
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= production_row['optimal_threshold']).astype(int)
    test_results = evaluate_model(y_test, y_pred, y_proba, production_row['model'])

    logger.info(f"Holdout: gmean={test_results['gmean']:.4f} auc={test_results['auc_roc']:.4f}")

    # Visualizations
    plot_confusion_matrix(y_test, y_pred, production_row['model'],
                          production_row['optimal_threshold'], FIGURES_DIR / 'confusion_matrix.png')
    plot_calibration_curve(y_test, y_proba, production_row['model'],
                           save_path=FIGURES_DIR / 'calibration_curve.png')
    plot_roc_curve(y_test, y_proba, production_row['model'],
                   save_path=FIGURES_DIR / 'roc_curve.png')
    plot_precision_recall_curve(y_test, y_proba, production_row['model'],
                                save_path=FIGURES_DIR / 'pr_curve.png')

    # SHAP
    logger.info("Computing SHAP values")
    generate_shap_explanations(pipeline, production_row, X_train, X_test)

    # Save artifacts
    save_artifacts(pipeline, production_row, test_results, training_time, threshold_df)

    # MLflow
    with mlflow.start_run(run_name=f"PRODUCTION_{production_row['model']}"):
        mlflow.log_param("model", production_row['model'])
        mlflow.log_param("feature_set", production_row['feature_set'])
        mlflow.log_param("threshold", production_row['optimal_threshold'])
        mlflow.log_metric("cv_gmean", production_row['cv_gmean_at_optimal'])
        mlflow.log_metric("test_gmean", test_results['gmean'])
        mlflow.log_metric("test_auc_roc", test_results['auc_roc'])
        mlflow.sklearn.log_model(pipeline, "model")
        mlflow.set_tag("production_model", "true")

    # Latency benchmark
    latency = benchmark_prediction_latency(pipeline, X_test, n_runs=100)
    pd.DataFrame([latency]).to_csv(REPORTS_DIR / 'latency_benchmark.csv', index=False)

    logger.info("Artifacts written to models/ and reports/")


if __name__ == '__main__':
    main()
