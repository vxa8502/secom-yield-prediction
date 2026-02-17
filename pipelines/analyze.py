#!/usr/bin/env python
"""
SECOM Interpretability Analysis Pipeline

Generates comprehensive model analysis reports:
- Misclassification analysis (waterfall plots for false negatives)
- Residual distribution analysis
- Performance stratified by missingness
- False negative cluster analysis

Usage:
    python -m pipelines.analyze
    python -m pipelines.analyze --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from scipy import stats

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import (
    MODELS_DIR, REPORTS_DIR, FIGURES_DIR, SECOM_RAW_DIR, RANDOM_STATE, DATA_DIR,
    VIZ_CONFIG, setup_logging, ensure_directories, load_json
)
from src.data import load_labels, load_features
from src.evaluation import calculate_gmean
from src.visualization import (
    finalize_figure,
    create_shap_explainer,
    compute_shap_values,
    plot_shap_waterfall,
    get_shap_base_value,
    get_sample_feature_contributions,
)

logger = logging.getLogger('secom')


def load_production_artifacts() -> tuple:
    """Load production model, threshold, and metadata."""
    model = joblib.load(MODELS_DIR / 'production_model.pkl')
    threshold_info = load_json(MODELS_DIR / 'production_threshold.json')
    metadata = load_json(MODELS_DIR / 'production_model_metadata.json')
    return model, threshold_info, metadata


def load_feature_names(feature_set: str, n_features: int) -> list[str]:
    """Load feature names for the given feature set."""
    if feature_set == 'lasso':
        lasso_features_path = DATA_DIR / 'lasso_selected_features.txt'
        with open(lasso_features_path) as f:
            return [line.strip() for line in f if line.strip()]
    elif feature_set == 'pca':
        return [f"PC{i+1}" for i in range(n_features)]
    else:
        return [f"feature_{i}" for i in range(n_features)]


def analyze_misclassifications(
    model,
    X_test: NDArray,
    y_test: NDArray,
    y_pred: NDArray,
    y_proba: NDArray,
    feature_names: list[str],
    threshold: float,
    n_examples: int = 5
) -> pd.DataFrame:
    """
    Analyze misclassified samples, focusing on false negatives (missed defects).

    Generates SHAP waterfall plots for the worst false negatives.
    """
    logger.info("Analyzing misclassifications...")

    fn_mask = (y_test == 1) & (y_pred == 0)
    fp_mask = (y_test == 0) & (y_pred == 1)

    fn_indices = np.where(fn_mask)[0]
    fp_indices = np.where(fp_mask)[0]

    logger.info(f"False Negatives (missed defects): {len(fn_indices)}")
    logger.info(f"False Positives (false alarms): {len(fp_indices)}")

    # Sort FNs by predicted probability (lowest = most confident wrong predictions)
    if len(fn_indices) > 0:
        fn_proba = y_proba[fn_indices]
        fn_sorted = fn_indices[np.argsort(fn_proba)]
        worst_fn = fn_sorted[:n_examples]

        # Create SHAP explainer
        classifier = model.named_steps.get('classifier', model.steps[-1][1])
        explainer = create_shap_explainer(classifier, X_test, max_background_samples=100)
        shap_values, _ = compute_shap_values(explainer, X_test)

        # Get base value using centralized helper (DRY)
        base_value = get_shap_base_value(explainer)

        # Generate waterfall plots for worst FNs
        (FIGURES_DIR / 'misclassification').mkdir(exist_ok=True)

        fn_analysis = []
        for i, idx in enumerate(worst_fn):
            plot_shap_waterfall(
                shap_values, X_test, sample_idx=idx,
                feature_names=feature_names,
                base_value=base_value,
                save_path=FIGURES_DIR / 'misclassification' / f'fn_waterfall_{i+1}.png'
            )

            fn_analysis.append({
                'rank': i + 1,
                'test_index': int(idx),
                'true_label': 1,
                'predicted_label': 0,
                'predicted_proba': float(y_proba[idx]),
                'error_type': 'false_negative',
                'distance_to_threshold': float(threshold - y_proba[idx])
            })

            # Log top contributing features using centralized helper (DRY)
            top_features = get_sample_feature_contributions(
                shap_values, X_test, feature_names, sample_idx=idx, top_n=3
            )
            logger.info(f"FN #{i+1} (idx={idx}, proba={y_proba[idx]:.3f}): "
                       f"Top factors: {', '.join(top_features['feature'].tolist())}")

        fn_df = pd.DataFrame(fn_analysis)
        fn_df.to_csv(REPORTS_DIR / 'false_negative_analysis.csv', index=False)
        logger.info(f"Saved {len(fn_analysis)} FN waterfall plots to reports/figures/misclassification/")

        return fn_df

    logger.warning("No false negatives found in test set")
    return pd.DataFrame()


def analyze_residuals(
    y_test: NDArray,
    y_proba: NDArray,
    y_pred: NDArray
) -> dict:
    """
    Analyze residual distributions for systematic patterns.

    Residual = y_true - y_pred_proba
    For correct predictions, residuals should cluster near 0 (passes) or near 0 (failures predicted correctly).
    """
    logger.info("Analyzing residual distributions...")

    residuals = y_test - y_proba

    # Split by true class
    pass_residuals = residuals[y_test == 0]
    fail_residuals = residuals[y_test == 1]

    # Summary statistics
    analysis = {
        'overall': {
            'mean': float(np.mean(residuals)),
            'std': float(np.std(residuals)),
            'median': float(np.median(residuals)),
            'skewness': float(stats.skew(residuals)),
            'kurtosis': float(stats.kurtosis(residuals))
        },
        'passes': {
            'mean': float(np.mean(pass_residuals)),
            'std': float(np.std(pass_residuals)),
            'n_samples': int(len(pass_residuals))
        },
        'failures': {
            'mean': float(np.mean(fail_residuals)),
            'std': float(np.std(fail_residuals)),
            'n_samples': int(len(fail_residuals))
        }
    }

    # Check for systematic bias
    # For passes (y=0): ideal residual is 0 - proba = -proba (negative, close to 0)
    # For fails (y=1): ideal residual is 1 - proba = 1-proba (positive, close to 0)
    analysis['interpretation'] = {
        'pass_bias': 'underconfident' if analysis['passes']['mean'] < -0.3 else
                     'overconfident' if analysis['passes']['mean'] > -0.1 else 'calibrated',
        'fail_bias': 'underconfident' if analysis['failures']['mean'] > 0.3 else
                     'overconfident' if analysis['failures']['mean'] < 0.1 else 'calibrated'
    }

    # Plot residual distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Overall distribution
    axes[0].hist(residuals, bins=30, color=VIZ_CONFIG['primary'], alpha=0.7, edgecolor='white')
    axes[0].axvline(0, color=VIZ_CONFIG['fail_color'], linestyle='--', linewidth=2)
    axes[0].set_xlabel('Residual (y_true - y_proba)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Overall Residual Distribution')

    # By class
    axes[1].hist(pass_residuals, bins=20, color=VIZ_CONFIG['pass_color'], alpha=0.7,
                 label='Passes (y=0)', edgecolor='white')
    axes[1].hist(fail_residuals, bins=20, color=VIZ_CONFIG['fail_color'], alpha=0.7,
                 label='Failures (y=1)', edgecolor='white')
    axes[1].axvline(0, color='black', linestyle='--', linewidth=1)
    axes[1].set_xlabel('Residual')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Residuals by True Class')
    axes[1].legend()

    # Q-Q plot for normality check
    stats.probplot(residuals, dist="norm", plot=axes[2])
    axes[2].set_title('Q-Q Plot (Normality Check)')

    plt.tight_layout()
    finalize_figure(FIGURES_DIR / 'residual_analysis.png')

    # Save analysis
    with open(REPORTS_DIR / 'residual_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)

    logger.info(f"Residual mean: {analysis['overall']['mean']:.4f}, std: {analysis['overall']['std']:.4f}")
    logger.info(f"Pass bias: {analysis['interpretation']['pass_bias']}, "
               f"Fail bias: {analysis['interpretation']['fail_bias']}")

    return analysis


def analyze_missingness_performance(
    X_test: NDArray,
    y_test: NDArray,
    y_pred: NDArray,
    y_proba: NDArray
) -> pd.DataFrame:
    """
    Analyze if missingness patterns correlate with prediction quality.

    Uses the raw data to compute missingness per sample, then checks if
    samples with more missing values have systematically different performance.
    """
    logger.info("Analyzing performance by missingness...")

    # Load raw data to get original missingness patterns
    raw_data_path = SECOM_RAW_DIR / 'secom.data'
    raw_labels_path = SECOM_RAW_DIR / 'secom_labels.data'

    if not raw_data_path.exists():
        logger.warning("Raw data not found, skipping missingness analysis")
        return pd.DataFrame()

    # Load raw data
    raw_df = pd.read_csv(raw_data_path, sep=' ', header=None, na_values=['NaN'])
    raw_labels = pd.read_csv(raw_labels_path, sep=' ', header=None, usecols=[0])
    raw_labels.columns = ['label']
    raw_labels['label'] = (raw_labels['label'] == -1).astype(int)

    # Compute missingness per sample
    missing_counts = raw_df.isna().sum(axis=1).values
    missing_pct = missing_counts / raw_df.shape[1] * 100

    # Split into train/test (80/20 stratified - match preprocessing)
    _, test_missing_pct, _, _ = train_test_split(
        missing_pct, raw_labels['label'].values,
        test_size=0.2, stratify=raw_labels['label'].values,
        random_state=RANDOM_STATE
    )

    # Stratify by missingness quartiles
    quartiles = pd.qcut(test_missing_pct, q=4, labels=['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)'])

    results = []
    for q in ['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)']:
        mask = quartiles == q
        if mask.sum() == 0:
            continue

        q_y_true = y_test[mask]
        q_y_pred = y_pred[mask]
        q_y_proba = y_proba[mask]

        gmean, sens, spec = calculate_gmean(q_y_true, q_y_pred)

        results.append({
            'quartile': q,
            'n_samples': int(mask.sum()),
            'n_failures': int(q_y_true.sum()),
            'fail_rate': float(q_y_true.mean()),
            'mean_missing_pct': float(test_missing_pct[mask].mean()),
            'gmean': gmean,
            'sensitivity': sens,
            'specificity': spec,
            'mean_proba': float(q_y_proba.mean())
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(REPORTS_DIR / 'missingness_stratified_performance.csv', index=False)

    # Plot
    if len(results_df) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        x = range(len(results_df))
        axes[0].bar(x, results_df['gmean'], color=VIZ_CONFIG['primary'], alpha=0.8)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(results_df['quartile'])
        axes[0].set_ylabel('G-Mean')
        axes[0].set_xlabel('Missingness Quartile')
        axes[0].set_title('Model Performance by Missingness Level')
        axes[0].set_ylim([0, 1])

        axes[1].bar(x, results_df['mean_missing_pct'], color=VIZ_CONFIG['secondary'], alpha=0.8)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(results_df['quartile'])
        axes[1].set_ylabel('Mean Missing %')
        axes[1].set_xlabel('Missingness Quartile')
        axes[1].set_title('Average Missingness per Quartile')

        plt.tight_layout()
        finalize_figure(FIGURES_DIR / 'missingness_performance.png')

        logger.info("Missingness analysis complete:")
        for _, row in results_df.iterrows():
            logger.info(f"  {row['quartile']}: G-Mean={row['gmean']:.3f}, n={row['n_samples']}")

    return results_df


def analyze_fn_clusters(
    X_test: NDArray,
    y_test: NDArray,
    y_pred: NDArray,
    y_proba: NDArray,
    feature_names: list[str],
    n_clusters: int = 3
) -> pd.DataFrame:
    """
    Cluster false negatives to identify systematic failure patterns.

    Uses K-means to find if missed defects share common feature profiles.
    """
    logger.info("Analyzing false negative clusters...")

    fn_mask = (y_test == 1) & (y_pred == 0)
    fn_indices = np.where(fn_mask)[0]

    if len(fn_indices) < n_clusters:
        logger.warning(f"Only {len(fn_indices)} false negatives, insufficient for clustering")
        return pd.DataFrame()

    X_fn = X_test[fn_indices]

    # Cluster FNs
    kmeans = KMeans(n_clusters=min(n_clusters, len(fn_indices)), random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_fn)

    # Analyze each cluster
    cluster_analysis = []
    for c in range(kmeans.n_clusters):
        cluster_mask = clusters == c
        cluster_features = X_fn[cluster_mask]

        # Mean feature values for this cluster
        cluster_means = cluster_features.mean(axis=0)

        # Compare to overall test set means
        overall_means = X_test.mean(axis=0)
        deviations = cluster_means - overall_means

        # Find most distinctive features (largest absolute deviation)
        top_deviation_idx = np.argsort(np.abs(deviations))[-3:][::-1]
        distinctive_features = [
            f"{feature_names[i]} ({deviations[i]:+.2f})"
            for i in top_deviation_idx
        ]

        cluster_analysis.append({
            'cluster': c,
            'n_samples': int(cluster_mask.sum()),
            'mean_proba': float(y_proba[fn_indices][cluster_mask].mean()),
            'distinctive_features': ', '.join(distinctive_features)
        })

    cluster_df = pd.DataFrame(cluster_analysis)
    cluster_df.to_csv(REPORTS_DIR / 'fn_cluster_analysis.csv', index=False)

    # Plot cluster feature profiles
    fig, ax = plt.subplots(figsize=(12, 6))

    cluster_centers = kmeans.cluster_centers_
    x = range(len(feature_names))

    for c in range(kmeans.n_clusters):
        ax.plot(x, cluster_centers[c], marker='o', label=f'Cluster {c} (n={cluster_analysis[c]["n_samples"]})')

    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.set_ylabel('Feature Value (standardized)')
    ax.set_xlabel('Feature')
    ax.set_title('False Negative Cluster Profiles')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    finalize_figure(FIGURES_DIR / 'fn_cluster_profiles.png')

    logger.info("FN Cluster Analysis:")
    for _, row in cluster_df.iterrows():
        logger.info(f"  Cluster {row['cluster']}: n={row['n_samples']}, "
                   f"mean_proba={row['mean_proba']:.3f}, "
                   f"distinctive: {row['distinctive_features']}")

    return cluster_df


def _report_header(metadata: dict) -> str:
    """Generate report header section."""
    return f"""# Interpretability Report

**{metadata['model_name']} + {metadata['feature_set']} | Threshold: {metadata['optimal_threshold']:.2f} | G-Mean: {metadata['test_gmean']:.1%}**"""


def _report_misclassification(fn_df: pd.DataFrame, n_total_failures: int = 21) -> str:
    """Generate misclassification analysis section."""
    if len(fn_df) == 0:
        return "\n\n## False Negatives\n\nNone in test set."

    top3 = fn_df.head(3)[['test_index', 'predicted_proba', 'distance_to_threshold']]
    top3.columns = ['test_idx', 'prob', 'gap']
    table = top3.to_markdown(index=False)

    return f"""

## False Negatives

{len(fn_df)}/{n_total_failures} failures missed. SHAP waterfalls: `reports/figures/misclassification/`

{table}"""


def _report_residuals(residual_analysis: dict) -> str:
    """Generate residual distribution analysis section."""
    overall = residual_analysis['overall']
    interp = residual_analysis['interpretation']

    if interp['pass_bias'] == interp['fail_bias']:
        calibration = interp['pass_bias']
    else:
        calibration = f"passes {interp['pass_bias']}, failures {interp['fail_bias']}"

    return f"""

## Calibration

Both classes {calibration} (mean residual {overall['mean']:.2f}). Consider Platt scaling."""


def _report_missingness(missingness_df: pd.DataFrame) -> str:
    """Generate missingness performance section (skip if empty)."""
    return ""  # Omit from concise report


def _report_clusters(cluster_df: pd.DataFrame) -> str:
    """Generate FN cluster analysis section."""
    if len(cluster_df) == 0:
        return ""

    rows = []
    for _, row in cluster_df.iterrows():
        rows.append(f"| {row['cluster']} | {row['n_samples']} | {row['distinctive_features']} |")

    table = "| Cluster | n | Key Features |\n|--------:|--:|--------------|\n" + "\n".join(rows)

    return f"""

## FN Clusters

{table}"""


def _report_recommendations() -> str:
    """Generate footer."""
    return """

---

*Generated by pipelines/analyze.py*
"""


def generate_analysis_report(
    fn_df: pd.DataFrame,
    residual_analysis: dict,
    missingness_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    metadata: dict,
    n_total_failures: int = 21
) -> None:
    """Generate markdown summary report of all analyses."""
    sections = [
        _report_header(metadata),
        _report_misclassification(fn_df, n_total_failures),
        _report_residuals(residual_analysis),
        _report_missingness(missingness_df),
        _report_clusters(cluster_df),
        _report_recommendations(),
    ]
    report = '\n'.join(sections)

    with open(REPORTS_DIR / 'interpretability_report.md', 'w') as f:
        f.write(report)

    logger.info("Analysis report saved to reports/interpretability_report.md")


def main():
    parser = argparse.ArgumentParser(description='SECOM interpretability analysis')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--n-fn-examples', type=int, default=5, help='Number of FN waterfall plots')
    args = parser.parse_args()

    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    ensure_directories()

    # Load artifacts
    model, threshold_info, metadata = load_production_artifacts()
    threshold = threshold_info['optimal_threshold']
    feature_set = threshold_info['feature_set']

    # Load data
    y_train, y_test = load_labels()
    X_train, X_test = load_features(feature_set)
    feature_names = load_feature_names(feature_set, X_test.shape[1])

    # Generate predictions
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    logger.info(f"Loaded model: {metadata['model_name']}, threshold={threshold:.3f}")
    logger.info(f"Test set: {len(y_test)} samples, {y_test.sum()} failures")

    # Run analyses
    fn_df = analyze_misclassifications(
        model, X_test, y_test, y_pred, y_proba,
        feature_names, threshold, n_examples=args.n_fn_examples
    )

    residual_analysis = analyze_residuals(y_test, y_proba, y_pred)

    missingness_df = analyze_missingness_performance(X_test, y_test, y_pred, y_proba)

    cluster_df = analyze_fn_clusters(X_test, y_test, y_pred, y_proba, feature_names)

    # Generate summary report
    generate_analysis_report(
        fn_df, residual_analysis, missingness_df, cluster_df, metadata,
        n_total_failures=int(y_test.sum())
    )

    logger.info("Analysis complete. See reports/interpretability_report.md")


if __name__ == '__main__':
    main()
