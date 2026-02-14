#!/usr/bin/env python
"""
SECOM Data Preprocessing Pipeline

Reproducible preprocessing for SECOM yield prediction project.
Extracts preprocessing logic from EDA notebook for production use.

METHODOLOGY NOTES:
- Train/test split happens BEFORE any feature selection to prevent data leakage
- VIF reduction uses hierarchical clustering to handle multicollinearity
- Imputation happens BEFORE scaling (correct order)
- L1-penalized LogisticRegression for classification-appropriate feature selection

Usage:
    python -m pipelines.preprocess
    python -m pipelines.preprocess --verbose
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import (
    DATA_DIR, SECOM_RAW_DIR, MODELS_DIR,
    RANDOM_STATE, setup_logging, compute_class_ratio, ensure_directories
)
from src.data.transformers import (
    HighMissingRemover, ZeroVarianceRemover, VIFReducer, DataFrameWrapper
)

logger = logging.getLogger('secom')


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw SECOM dataset with validation.

    Raises:
        FileNotFoundError: If raw data files are missing
        ValueError: If data format is unexpected (wrong dimensions, etc.)
    """
    features_path = SECOM_RAW_DIR / 'secom.data'
    labels_path = SECOM_RAW_DIR / 'secom_labels.data'

    # Validate raw files exist
    if not features_path.exists():
        raise FileNotFoundError(
            f"Raw features file not found: {features_path}\n"
            "Download SECOM dataset from UCI ML Repository and place in data/raw/secom/"
        )
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Raw labels file not found: {labels_path}\n"
            "Download SECOM dataset from UCI ML Repository and place in data/raw/secom/"
        )

    # Load features (590 sensor readings)
    try:
        X = pd.read_csv(features_path, sep=' ', header=None, index_col=False)
    except pd.errors.EmptyDataError as e:
        raise ValueError(f"Features file is empty: {features_path}") from e
    except pd.errors.ParserError as e:
        raise ValueError(f"Features file has invalid format: {e}") from e

    # Load labels (binary: originally -1 = pass, 1 = fail)
    try:
        labels_df = pd.read_csv(labels_path, sep=' ', header=None, names=['label', 'timestamp'])
    except pd.errors.EmptyDataError as e:
        raise ValueError(f"Labels file is empty: {labels_path}") from e
    except pd.errors.ParserError as e:
        raise ValueError(f"Labels file has invalid format: {e}") from e

    # Validate dimensions
    if X.empty:
        raise ValueError(f"Features file has no data rows: {features_path}")
    if labels_df.empty:
        raise ValueError(f"Labels file has no data rows: {labels_path}")

    if len(X) != len(labels_df):
        raise ValueError(
            f"Row count mismatch: features has {len(X)} rows, "
            f"labels has {len(labels_df)} rows"
        )

    # Validate expected column counts (SECOM should have 590 features)
    expected_features = 590
    if X.shape[1] != expected_features:
        logger.warning(
            f"Unexpected feature count: expected {expected_features}, got {X.shape[1]}. "
            f"This may indicate a different dataset version."
        )

    # Validate labels are in expected format
    y = labels_df[['label']].copy()
    unique_labels = set(y['label'].dropna().unique())
    expected_labels = {-1, 1}
    if not unique_labels.issubset(expected_labels | {-1.0, 1.0}):
        raise ValueError(
            f"Unexpected label values: {unique_labels}. "
            f"Expected: {expected_labels}"
        )

    y['target'] = y['label'].map({-1: 0, 1: 1})

    # Replace NaN indicator (-9999 or similar) with actual NaN
    X = X.replace({-9999.0: np.nan, -999.0: np.nan})

    # Create feature names
    X.columns = [f'feature_{i}' for i in range(X.shape[1])]

    missing_rate = X.isna().sum().sum() / (X.shape[0] * X.shape[1]) * 100
    logger.info(f"Raw data: {X.shape[0]} samples, {X.shape[1]} features, {missing_rate:.1f}% missing")

    return X, y


def create_train_test_split(
    X: pd.DataFrame,
    y: pd.DataFrame,
    test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create stratified train/test split."""
    y_binary = y['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=test_size, random_state=RANDOM_STATE, stratify=y_binary
    )

    # Compute actual class imbalance ratio for logging (DRY: use centralized function)
    imbalance_ratio = compute_class_ratio(y_train)

    logger.info(f"Split: train={len(X_train)} test={len(X_test)} "
                f"(fail_rate={y_train.mean():.1%}, ratio={imbalance_ratio:.1f}:1)")

    return X_train, X_test, y_train, y_test


def create_pipelines(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    verbose: bool = False
) -> tuple[dict[str, Pipeline], dict[str, np.ndarray]]:
    """
    Create all preprocessing pipelines.

    NOTE: Input data is already cleaned (no NaN, no zero-variance, VIF-reduced).
    Pipelines only need to scale and apply feature selection.

    METHODOLOGY:
    - Data arrives clean from shared preprocessing (HighMissing + ZeroVar + Impute + VIF)
    - L1FeatureSelector uses LogisticRegression (classification-appropriate)
    - PCA retains 95% variance
    """
    pipelines = {}
    transformed_data = {}

    # =========================================================================
    # PIPELINE 1: BASIC (scale only - keep all VIF-reduced features)
    # =========================================================================
    logger.info("[Pipeline: BASIC] Starting - scale only (all features)")
    pipeline_basic = Pipeline([
        ('scale', StandardScaler())
    ])

    X_train_basic = pipeline_basic.fit_transform(X_train, y_train)
    pipelines['basic'] = pipeline_basic
    transformed_data['basic'] = X_train_basic
    logger.info(f"[Pipeline: BASIC] Complete - {X_train_basic.shape[1]} features")

    # =========================================================================
    # PIPELINE 2: LASSO (scale + LassoCV feature selection)
    # =========================================================================
    logger.info("[Pipeline: LASSO] Starting - LassoCV feature selection")

    # Simple approach: LassoCV auto-tunes alpha, SelectFromModel picks non-zero coefs
    lasso_selector = SelectFromModel(
        LassoCV(cv=5, random_state=RANDOM_STATE, max_iter=10000),
        threshold=1e-10  # Keep features with any non-zero coefficient
    )

    pipeline_lasso = Pipeline([
        ('scale', StandardScaler()),
        ('select_features', lasso_selector)
    ])

    X_train_lasso = pipeline_lasso.fit_transform(X_train, y_train)
    pipelines['lasso'] = pipeline_lasso
    transformed_data['lasso'] = X_train_lasso

    # Get selected feature info
    lasso_model = pipeline_lasso.named_steps['select_features'].estimator_
    feature_mask = pipeline_lasso.named_steps['select_features'].get_support()
    selected_features = X_train.columns[feature_mask].tolist()
    best_alpha = lasso_model.alpha_

    logger.info(f"[Pipeline: LASSO] Complete - {len(selected_features)} features (alpha={best_alpha:.6f})")

    if verbose:
        logger.debug(f"LASSO selected features: {selected_features}")

    # =========================================================================
    # PIPELINE 3: PCA (scale + dimensionality reduction)
    # =========================================================================
    logger.info("[Pipeline: PCA] Starting - 95% variance retention")
    pipeline_pca = Pipeline([
        ('scale', StandardScaler()),
        ('pca', PCA(n_components=0.95, svd_solver='full'))
    ])

    X_train_pca = pipeline_pca.fit_transform(X_train, y_train)
    pipelines['pca'] = pipeline_pca
    transformed_data['pca'] = X_train_pca

    n_components = pipeline_pca.named_steps['pca'].n_components_
    logger.info(f"[Pipeline: PCA] Complete - {n_components} components")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    logger.info(f"Pipeline summary: basic={X_train_basic.shape[1]}, l1={len(selected_features)}, pca={n_components}")

    return pipelines, transformed_data


def save_artifacts(
    pipelines: dict[str, Pipeline],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    vif_features: list[str] | None = None,
    verbose: bool = False
) -> None:
    """Save all preprocessing artifacts."""
    # Ensure directories exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save labels
    y_train.to_csv(DATA_DIR / 'y_train.csv', index=False)
    y_test.to_csv(DATA_DIR / 'y_test.csv', index=False)

    # Transform and save each feature set
    for name, pipeline in pipelines.items():
        X_train_transformed = pipeline.transform(X_train)
        X_test_transformed = pipeline.transform(X_test)

        # Verify no NaN in output (handle both numpy arrays and DataFrames)
        train_nan_count = np.isnan(X_train_transformed).sum()
        test_nan_count = np.isnan(X_test_transformed).sum()
        # Convert to scalar if it's a Series (from DataFrame input)
        if hasattr(train_nan_count, 'sum'):
            train_nan_count = train_nan_count.sum()
        if hasattr(test_nan_count, 'sum'):
            test_nan_count = test_nan_count.sum()
        if train_nan_count > 0 or test_nan_count > 0:
            logger.warning(f"Pipeline '{name}' produced NaN values: train={train_nan_count}, test={test_nan_count}")

        # Save transformed features
        if name == 'basic':
            np.save(DATA_DIR / 'X_train_all_features.npy', X_train_transformed)
            np.save(DATA_DIR / 'X_test_all_features.npy', X_test_transformed)
        elif name == 'lasso':
            np.save(DATA_DIR / 'X_train_lasso.npy', X_train_transformed)
            np.save(DATA_DIR / 'X_test_lasso.npy', X_test_transformed)

            # Save selected feature names (SelectFromModel uses get_support())
            feature_mask = pipeline.named_steps['select_features'].get_support()
            selected_features = X_train.columns[feature_mask].tolist()
            with open(DATA_DIR / 'lasso_selected_features.txt', 'w') as f:
                for feature in selected_features:
                    f.write(f"{feature}\n")
        elif name == 'pca':
            np.save(DATA_DIR / 'X_train_pca_raw.npy', X_train_transformed)
            np.save(DATA_DIR / 'X_test_pca_raw.npy', X_test_transformed)

        # Save pipeline
        joblib.dump(pipeline, MODELS_DIR / f'preprocessing_pipeline_{name}.pkl')

    logger.info(f"Saved: {DATA_DIR} (6 .npy, 2 .csv), {MODELS_DIR} (3 .pkl)")


def main():
    parser = argparse.ArgumentParser(description='SECOM data preprocessing')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()

    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    ensure_directories()

    # Load raw data
    X, y = load_raw_data()

    # CRITICAL: Split BEFORE any feature selection to prevent data leakage
    X_train, X_test, y_train, y_test = create_train_test_split(X, y)

    # =========================================================================
    # SHARED PREPROCESSING (all pipelines receive this data)
    # Steps: HighMissing -> ZeroVariance -> Impute -> VIF Reduction
    # =========================================================================
    logger.info("[Shared] Starting preprocessing...")

    # Step 1: Remove features with >50% missing
    missing_remover = HighMissingRemover(threshold=0.50, name='shared')
    X_train_clean = missing_remover.fit_transform(X_train)
    X_test_clean = missing_remover.transform(X_test)

    # Step 2: Remove zero-variance features
    zv_remover = ZeroVarianceRemover(name='shared')
    X_train_clean = zv_remover.fit_transform(X_train_clean)
    X_test_clean = zv_remover.transform(X_test_clean)

    # Step 3: Impute remaining missing values (median)
    imputer = DataFrameWrapper(SimpleImputer(strategy='median'))
    X_train_clean = imputer.fit_transform(X_train_clean)
    X_test_clean = imputer.transform(X_test_clean)
    logger.info(f"  [shared] Imputation: {X_train_clean.shape[1]} features (no NaN)")

    # Step 4: VIF reduction (on clean, complete data)
    vif_reducer = VIFReducer(vif_threshold=10.0, cluster_distance=0.10)
    X_train_vif = vif_reducer.fit_transform(X_train_clean, y_train)
    X_test_vif = vif_reducer.transform(X_test_clean)

    # Save VIF-selected features for reproducibility
    vif_features = vif_reducer.features_to_keep_
    with open(DATA_DIR / 'vif_selected_features.txt', 'w') as f:
        for feature in vif_features:
            f.write(f"{feature}\n")

    logger.info(f"[Shared] Complete: {len(vif_features)} features after cleaning + VIF reduction")

    # Create preprocessing pipelines (using cleaned + VIF-reduced data)
    pipelines, _ = create_pipelines(X_train_vif, y_train, verbose=args.verbose)

    # Save all artifacts
    save_artifacts(pipelines, X_train_vif, X_test_vif, y_train, y_test, vif_features, verbose=args.verbose)


if __name__ == '__main__':
    main()
