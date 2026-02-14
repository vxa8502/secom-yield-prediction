"""
Production artifact loading utilities for Streamlit dashboard.
Author: Victoria A.

Provides functions to dynamically load production model and metadata
instead of hardcoding values.

Uses @st.cache_resource for Streamlit-native caching of expensive
objects (models, pipelines). This integrates with Streamlit's lifecycle
and provides better memory management than functools.lru_cache.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator

# Centralized path setup
from streamlit_app import setup_project_path
setup_project_path()

from src.config import PRODUCTION_ARTIFACTS, DATA_DIR, REPORTS_DIR

logger = logging.getLogger('secom')


def get_classifier(model: BaseEstimator) -> BaseEstimator:
    """
    Extract classifier from sklearn pipeline or return model directly.

    Args:
        model: sklearn model or Pipeline

    Returns:
        The classifier component
    """
    if hasattr(model, 'named_steps'):
        return model.named_steps.get('classifier', model)
    return model


@st.cache_resource
def load_production_model() -> BaseEstimator | None:
    """
    Load production model from disk.

    Uses @st.cache_resource to cache the model across Streamlit reruns,
    avoiding repeated disk I/O and model deserialization.

    Returns:
        sklearn pipeline or None if not found
    """
    model_path = PRODUCTION_ARTIFACTS['model']
    if not model_path.exists():
        return None

    return joblib.load(model_path)


@st.cache_data
def load_production_metadata() -> dict[str, Any]:
    """
    Load production model metadata.

    Uses @st.cache_data for JSON metadata (serializable, immutable).

    Returns:
        dict with model configuration or empty dict if not found
    """
    metadata_path = PRODUCTION_ARTIFACTS['metadata']
    if not metadata_path.exists():
        return {}

    with open(metadata_path, 'r') as f:
        return json.load(f)


@st.cache_data
def load_threshold_config() -> dict[str, float]:
    """
    Load threshold configuration.

    Returns:
        dict with threshold info or default values
    """
    threshold_path = PRODUCTION_ARTIFACTS['threshold']
    if not threshold_path.exists():
        return {'optimal_threshold': 0.5}

    with open(threshold_path, 'r') as f:
        return json.load(f)


@st.cache_data
def load_lasso_features() -> list[str]:
    """
    Load LASSO selected feature names.

    Returns:
        list of feature names or empty list if not found
    """
    features_path = PRODUCTION_ARTIFACTS['lasso_features']
    if not features_path.exists():
        return []

    with open(features_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


@st.cache_resource
def load_preprocessing_pipeline() -> BaseEstimator | None:
    """
    Load LASSO preprocessing pipeline.

    Uses @st.cache_resource for sklearn pipeline (non-serializable).

    Returns:
        sklearn pipeline or None if not found
    """
    pipeline_path = PRODUCTION_ARTIFACTS.get('preprocessing_pipeline')
    if pipeline_path and pipeline_path.exists():
        return joblib.load(pipeline_path)
    return None


def get_model_display_info():
    """
    Get model information for UI display.

    Returns:
        dict with display-ready metrics and model info
    """
    metadata = load_production_metadata()
    threshold_config = load_threshold_config()
    lasso_features = load_lasso_features()

    # Check if model is available
    model_available = load_production_model() is not None

    if not model_available or not metadata:
        return {
            'available': False,
            'message': 'Production model not trained. Run `make pipeline` first.',
            'model_name': 'Not available',
            'cv_gmean': 0.0,
            'test_gmean': 0.0,
            'sensitivity': 0.0,
            'specificity': 0.0,
            'threshold': 0.5,
            'n_features': 0,
            'feature_set': 'unknown',
            'sampling_strategy': 'unknown',
        }

    return {
        'available': True,
        'message': None,
        'model_name': metadata.get('model_name', 'Unknown'),
        'cv_gmean': metadata.get('cv_gmean', 0.0),
        'test_gmean': metadata.get('test_gmean', 0.0),
        'test_auc_roc': metadata.get('test_auc_roc', 0.0),
        'sensitivity': metadata.get('cv_sensitivity', 0.0),
        'specificity': metadata.get('cv_specificity', 0.0),
        'threshold': threshold_config.get('optimal_threshold', 0.5),
        'n_features': metadata.get('n_features', len(lasso_features)),
        'feature_set': metadata.get('feature_set', 'unknown'),
        'sampling_strategy': metadata.get('sampling_strategy', 'unknown'),
        'lasso_features': lasso_features,
        'training_time': metadata.get('training_time_seconds', 0.0),
        'timestamp': metadata.get('timestamp', 'Unknown'),
    }


def predict_single(features_dict, threshold=None):
    """
    Make prediction for a single wafer.

    Args:
        features_dict: dict of feature name -> value
        threshold: Classification threshold (default: use production threshold)

    Returns:
        dict with prediction, probability, confidence, class_label

    Raises:
        RuntimeError: If production model not available
        ValueError: If required features are missing from input dict
    """
    model = load_production_model()
    if model is None:
        raise RuntimeError("Production model not available. Run `make pipeline` first.")

    threshold_config = load_threshold_config()
    if threshold is None:
        threshold = threshold_config.get('optimal_threshold', 0.5)

    lasso_features = load_lasso_features()

    # Validate required features exist in input
    missing_features = set(lasso_features) - set(features_dict.keys())
    if missing_features:
        raise ValueError(
            f"Missing required features: {sorted(missing_features)}. "
            f"Expected keys: {lasso_features}"
        )

    # Build feature array in correct order
    feature_values = [features_dict[f] for f in lasso_features]
    X = np.array(feature_values).reshape(1, -1)

    # Get prediction
    probabilities = model.predict_proba(X)[0]
    prob_fail = probabilities[1]

    prediction = int(prob_fail >= threshold)
    confidence = max(probabilities)

    return {
        'prediction': prediction,
        'probability': prob_fail,
        'confidence': confidence,
        'class_label': 'FAIL' if prediction == 1 else 'PASS',
        'threshold_used': threshold,
    }


def predict_batch(features_df, threshold=None):
    """
    Make predictions for multiple wafers.

    Args:
        features_df: DataFrame with LASSO feature columns
        threshold: Classification threshold (default: use production threshold)

    Returns:
        DataFrame with predictions

    Raises:
        RuntimeError: If production model not available
        ValueError: If required features are missing from input DataFrame
    """
    model = load_production_model()
    if model is None:
        raise RuntimeError("Production model not available. Run `make pipeline` first.")

    threshold_config = load_threshold_config()
    if threshold is None:
        threshold = threshold_config.get('optimal_threshold', 0.5)

    lasso_features = load_lasso_features()

    # Validate required features exist in input
    missing_features = set(lasso_features) - set(features_df.columns)
    if missing_features:
        raise ValueError(
            f"Missing required features in uploaded data: {sorted(missing_features)}. "
            f"Expected columns: {lasso_features}"
        )

    # Ensure columns are in correct order
    X = features_df[lasso_features].values

    # Get predictions
    probabilities = model.predict_proba(X)
    prob_fail = probabilities[:, 1]
    predictions = (prob_fail >= threshold).astype(int)
    confidence = np.max(probabilities, axis=1)

    return pd.DataFrame({
        'prediction': predictions,
        'probability': prob_fail,
        'confidence': confidence,
        'class_label': ['FAIL' if p == 1 else 'PASS' for p in predictions],
    })


@st.cache_data
def load_test_data() -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """
    Load test data arrays (X_test and y_test).

    Centralized loader to avoid code duplication across pages.
    Uses @st.cache_data for efficient caching.

    Returns:
        Tuple of (X_test, y_test) arrays, or (None, None) if not available
    """
    X_test_path = DATA_DIR / "X_test_lasso.npy"
    y_test_path = DATA_DIR / "y_test.csv"

    if not X_test_path.exists() or not y_test_path.exists():
        return None, None

    X_test = np.load(X_test_path)
    y_test = pd.read_csv(y_test_path).values.ravel()

    return X_test, y_test


def load_test_sample(n_samples=5):
    """
    Load sample rows from test set for example predictions.

    Returns:
        DataFrame with test samples or None if not available
    """
    lasso_features = load_lasso_features()
    if not lasso_features:
        return None

    X_test, y_test = load_test_data()
    if X_test is None:
        return None

    # Create DataFrame with feature names
    df = pd.DataFrame(X_test, columns=lasso_features)
    df['actual_label'] = y_test

    # Sample balanced examples
    pass_samples = df[df['actual_label'] == 0].head(n_samples // 2 + 1)
    fail_samples = df[df['actual_label'] == 1].head(n_samples // 2 + 1)

    return pd.concat([pass_samples, fail_samples]).head(n_samples)


@st.cache_data
def get_feature_importance():
    """
    Get feature importance from production model or saved SHAP importance.

    Uses @st.cache_data to avoid recomputing on every page navigation.

    Priority:
    1. Model's coef_ (linear models)
    2. Model's feature_importances_ (tree-based)
    3. Saved SHAP feature importance CSV (SVM, other models)

    Returns:
        DataFrame with feature importance or None if extraction fails
    """
    model = load_production_model()
    lasso_features = load_lasso_features()

    # Try to extract from model directly
    if model is not None and lasso_features:
        try:
            classifier = get_classifier(model)

            if hasattr(classifier, 'coef_'):
                importance = np.abs(classifier.coef_[0])
                df = pd.DataFrame({
                    'feature': lasso_features,
                    'importance': importance
                })
                return df.sort_values('importance', ascending=False)
            elif hasattr(classifier, 'feature_importances_'):
                importance = classifier.feature_importances_
                df = pd.DataFrame({
                    'feature': lasso_features,
                    'importance': importance
                })
                return df.sort_values('importance', ascending=False)
        except Exception as e:
            logger.debug(f"Could not extract from model: {e}")

    # Fallback to saved SHAP importance (works for SVM, etc.)
    shap_path = REPORTS_DIR / 'shap_feature_importance.csv'
    if shap_path.exists():
        try:
            df = pd.read_csv(shap_path)
            if 'feature' in df.columns and 'mean_abs_shap' in df.columns:
                df = df.rename(columns={'mean_abs_shap': 'importance'})
            return df.sort_values('importance', ascending=False)
        except Exception as e:
            logger.warning(f"Failed to load SHAP importance: {e}")

    logger.debug("No feature importance available")
    return None


def extract_shap_values(shap_result) -> np.ndarray:
    """
    Extract raw SHAP values from SHAP Explanation object.

    SHAP library returns different formats depending on explainer type.
    This helper normalizes access to the underlying values array.

    Args:
        shap_result: SHAP Explanation object or raw numpy array

    Returns:
        numpy array of SHAP values
    """
    if hasattr(shap_result, 'values'):
        return shap_result.values
    return shap_result


def compute_single_prediction_shap(
    features_dict: dict[str, float]
) -> pd.DataFrame | None:
    """
    Compute SHAP contributions for a single prediction.

    This is a lightweight SHAP computation for real-time predictions.
    Returns the contribution of each feature to the prediction.

    Args:
        features_dict: Dictionary of feature name -> value

    Returns:
        DataFrame with 'feature' and 'contribution' columns,
        sorted by absolute contribution. Returns None if SHAP
        computation fails or model doesn't support it.
    """
    model = load_production_model()
    lasso_features = load_lasso_features()

    if model is None or not lasso_features:
        return None

    try:
        import shap

        # Get classifier from pipeline
        classifier = get_classifier(model)

        # Build feature array
        X = np.array([[features_dict[f] for f in lasso_features]])

        # Use TreeExplainer for tree-based models, LinearExplainer for linear
        if hasattr(classifier, 'feature_importances_'):
            # Tree-based model
            explainer = shap.TreeExplainer(classifier)
        elif hasattr(classifier, 'coef_'):
            # Linear model
            explainer = shap.LinearExplainer(classifier, X)
        else:
            # Fallback - use coef_ or feature_importances_ directly
            return _compute_importance_fallback(classifier, lasso_features, X)

        # Compute SHAP values for single sample
        shap_values = explainer.shap_values(X)

        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            # Binary classification - use class 1 (failure) contributions
            contributions = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        elif hasattr(shap_values, 'values'):
            contributions = shap_values.values[0]
        else:
            contributions = shap_values[0]

        # Build result DataFrame
        df = pd.DataFrame({
            'feature': lasso_features,
            'contribution': contributions
        })
        df['abs_contribution'] = df['contribution'].abs()
        df = df.sort_values('abs_contribution', ascending=False)

        return df[['feature', 'contribution']]

    except Exception as e:
        logger.debug(f"SHAP computation failed: {e}")
        return None


def _compute_importance_fallback(
    classifier,
    feature_names: list[str],
    X: np.ndarray
) -> pd.DataFrame | None:
    """
    Fallback: compute pseudo-contributions using model coefficients.

    For linear models: contribution = coef * feature_value
    For tree models: uses feature_importances_ (not instance-specific)

    This is less accurate than SHAP but provides some insight.
    """
    if hasattr(classifier, 'coef_'):
        # Linear model: contribution = coef * value
        coefs = classifier.coef_.flatten()
        contributions = coefs * X.flatten()
    elif hasattr(classifier, 'feature_importances_'):
        # Tree model: use global importance (not instance-specific)
        contributions = classifier.feature_importances_
    else:
        return None

    # Validate lengths match before creating DataFrame
    if len(feature_names) != len(contributions):
        logger.debug(
            f"Feature count mismatch: {len(feature_names)} names vs "
            f"{len(contributions)} contributions"
        )
        return None

    return pd.DataFrame({
        'feature': feature_names,
        'contribution': contributions
    })


# =============================================================================
# HEALTH CHECK UTILITIES
# =============================================================================

@st.cache_data
def load_latency_benchmark() -> dict[str, Any] | None:
    """
    Load latency benchmark data if available.

    Returns:
        dict with latency metrics or None if benchmark not found
    """
    latency_path = REPORTS_DIR / 'latency_benchmark.csv'
    if not latency_path.exists():
        return None

    df = pd.read_csv(latency_path)
    if df.empty or 'mean_per_sample_latency_ms' not in df.columns:
        return None

    return df.iloc[0].to_dict()


def get_system_health() -> dict[str, Any]:
    """
    Get comprehensive system health status for the dashboard.

    Checks availability of:
    - Production model
    - Training/test data
    - LASSO features
    - Threshold configuration

    Returns:
        dict with health status for each component and overall status
    """
    health = {
        'model': {'available': False, 'message': 'Not loaded'},
        'train_data': {'available': False, 'message': 'Not found'},
        'test_data': {'available': False, 'message': 'Not found'},
        'features': {'available': False, 'message': 'Not found', 'count': 0},
        'threshold': {'available': False, 'message': 'Using default'},
        'overall': 'unhealthy',
        'ready_for_prediction': False,
    }

    # Check model
    model = load_production_model()
    if model is not None:
        health['model'] = {'available': True, 'message': 'Loaded'}

    # Check training data
    X_train_path = DATA_DIR / "X_train_lasso.npy"
    if X_train_path.exists():
        health['train_data'] = {'available': True, 'message': 'Available'}

    # Check test data
    X_test, y_test = load_test_data()
    if X_test is not None:
        health['test_data'] = {
            'available': True,
            'message': f'{len(X_test)} samples'
        }

    # Check features
    features = load_lasso_features()
    if features:
        health['features'] = {
            'available': True,
            'message': f'{len(features)} features',
            'count': len(features)
        }

    # Check threshold
    threshold_config = load_threshold_config()
    if threshold_config.get('optimal_threshold', 0.5) != 0.5:
        health['threshold'] = {
            'available': True,
            'message': f"{threshold_config['optimal_threshold']:.3f}"
        }

    # Determine overall health
    critical_components = ['model', 'features']
    all_critical_available = all(
        health[comp]['available'] for comp in critical_components
    )

    if all_critical_available:
        health['overall'] = 'healthy'
        health['ready_for_prediction'] = True
    elif health['model']['available']:
        health['overall'] = 'degraded'
        health['ready_for_prediction'] = False
    else:
        health['overall'] = 'unhealthy'
        health['ready_for_prediction'] = False

    return health


# =============================================================================
# REQUEST LOGGING UTILITIES
# =============================================================================

def log_prediction_request(
    prediction: int,
    probability: float,
    threshold: float,
    input_method: str = 'unknown',
    n_features: int = 0
) -> None:
    """
    Log a prediction request for debugging and analytics.

    Logs are written to the 'secom' logger at INFO level.
    In production, these can be aggregated for monitoring.

    Args:
        prediction: Model prediction (0=PASS, 1=FAIL)
        probability: Failure probability
        threshold: Classification threshold used
        input_method: How input was provided (manual, example, csv)
        n_features: Number of features in input
    """
    from datetime import datetime

    log_entry = {
        'event': 'prediction_request',
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'prediction': 'FAIL' if prediction == 1 else 'PASS',
        'probability': round(probability, 4),
        'threshold': round(threshold, 4),
        'input_method': input_method,
        'n_features': n_features,
        'above_threshold': probability >= threshold,
    }

    logger.info(f"Prediction: {log_entry}")
