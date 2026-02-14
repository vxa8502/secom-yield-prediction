"""
SHAP-based model explainability utilities.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from numpy.typing import NDArray
from sklearn.base import BaseEstimator

from ..config import RANDOM_STATE, SHAP_MAX_DISPLAY
from .plots import finalize_figure

logger = logging.getLogger('secom')

# Cache configuration
MAX_EXPLAINER_CACHE_SIZE = 5  # Limit to 5 explainers (SHAP explainers can be memory-intensive)

# Module-level LRU cache for SHAP explainers (improves dashboard latency)
# Key: (model_id, model_class_name) to avoid stale cache if model is GC'd and new one reuses memory
# Uses OrderedDict for LRU eviction - most recently used items are moved to end
_explainer_cache: OrderedDict[tuple[int, str], shap.Explainer] = OrderedDict()


def clear_explainer_cache() -> None:
    """Clear the SHAP explainer cache to free memory."""
    n_entries = len(_explainer_cache)
    _explainer_cache.clear()
    logger.info(f"SHAP explainer cache cleared ({n_entries} entries removed)")


def get_cache_info() -> dict[str, Any]:
    """
    Get information about the current SHAP explainer cache state.

    Useful for monitoring cache utilization in production dashboards.

    Returns:
        Dict with cache_size, max_size, and cached_model_types
    """
    return {
        'cache_size': len(_explainer_cache),
        'max_size': MAX_EXPLAINER_CACHE_SIZE,
        'cached_model_types': [key[1] for key in _explainer_cache.keys()],
        'utilization_pct': len(_explainer_cache) / MAX_EXPLAINER_CACHE_SIZE * 100
    }


def _get_cache_key(model: Any) -> tuple[int, str]:
    """Generate robust cache key using model id and class name."""
    return (id(model), type(model).__name__)


def get_shap_base_value(explainer: shap.Explainer) -> float:
    """
    Extract base value from SHAP explainer for waterfall plots.

    Handles different explainer types that store expected_value differently
    (scalar, list, or array).

    Args:
        explainer: SHAP explainer instance

    Returns:
        Base value as float (expected model output on background data)
    """
    if not hasattr(explainer, 'expected_value'):
        return 0.0

    base_value = explainer.expected_value

    if isinstance(base_value, (list, np.ndarray)):
        # Binary classification: index 1 is positive class probability
        return float(base_value[1]) if len(base_value) > 1 else float(base_value[0])

    return float(base_value)


def get_sample_feature_contributions(
    shap_values: NDArray,
    feature_values: NDArray,
    feature_names: list[str],
    sample_idx: int = 0,
    top_n: int = 5
) -> pd.DataFrame:
    """
    Extract feature contributions for a single sample, sorted by importance.

    Centralizes the pattern of creating a DataFrame with SHAP values,
    feature values, and absolute SHAP for sorting.

    Args:
        shap_values: SHAP values array (n_samples, n_features)
        feature_values: Feature values array (n_samples, n_features)
        feature_names: List of feature names
        sample_idx: Index of sample to analyze (default: 0)
        top_n: Number of top features to return (default: 5, use -1 for all)

    Returns:
        DataFrame with columns: feature, shap_value, feature_value, abs_shap
        Sorted by absolute SHAP value descending
    """
    contributions = pd.DataFrame({
        'feature': feature_names,
        'shap_value': shap_values[sample_idx],
        'feature_value': feature_values[sample_idx]
    })
    contributions['abs_shap'] = np.abs(contributions['shap_value'])
    contributions = contributions.sort_values('abs_shap', ascending=False)

    if top_n > 0:
        return contributions.head(top_n)
    return contributions


def get_cached_explainer(
    model: Any,
    X_background: np.ndarray,
    model_type: str = 'auto',
    max_background_samples: int = 100
) -> shap.Explainer:
    """
    Get or create a cached SHAP explainer for the given model.

    For production dashboards with repeated explanations, this avoids
    recreating the explainer on every request.

    Uses (id, class_name) as cache key to avoid returning stale explainer
    if a model is garbage collected and new model reuses the same memory address.

    Implements LRU eviction with MAX_EXPLAINER_CACHE_SIZE limit to prevent
    unbounded memory growth in long-running dashboard sessions.

    Args:
        model: Trained model
        X_background: Background dataset for explainer
        model_type: 'auto', 'tree', 'linear', or 'kernel'
        max_background_samples: Max samples for background (KernelExplainer)

    Returns:
        Cached or newly created SHAP explainer
    """
    cache_key = _get_cache_key(model)

    if cache_key in _explainer_cache:
        # Move to end (mark as recently used)
        _explainer_cache.move_to_end(cache_key)
        logger.debug("Using cached SHAP explainer")
        return _explainer_cache[cache_key]

    # Evict oldest entry if cache is at capacity
    if len(_explainer_cache) >= MAX_EXPLAINER_CACHE_SIZE:
        evicted_key, _ = _explainer_cache.popitem(last=False)
        logger.debug(f"Evicted oldest SHAP explainer from cache (id={evicted_key[0]})")

    explainer = create_shap_explainer(model, X_background, model_type, max_background_samples)
    _explainer_cache[cache_key] = explainer
    logger.info(f"Created and cached SHAP explainer for {type(model).__name__} "
                f"(cache size: {len(_explainer_cache)}/{MAX_EXPLAINER_CACHE_SIZE})")

    return explainer


def create_shap_explainer(
    model: BaseEstimator,
    X_background: NDArray,
    model_type: Literal['auto', 'tree', 'linear', 'kernel'] = 'auto',
    max_background_samples: int = 100
) -> shap.Explainer:
    """
    Create appropriate SHAP explainer based on model type.

    Args:
        model: Trained sklearn-compatible model
        X_background: Background dataset for explainer
        model_type: Explainer type ('auto' detects from model)
        max_background_samples: Max samples for background (KernelExplainer)

    Returns:
        SHAP explainer instance
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from xgboost import XGBClassifier

    if len(X_background) > max_background_samples:
        # Use isolated RNG to avoid mutating global random state
        rng = np.random.default_rng(RANDOM_STATE)
        indices = rng.choice(len(X_background), max_background_samples, replace=False)
        X_background_sample = X_background[indices]
    else:
        X_background_sample = X_background

    if model_type == 'auto':
        if isinstance(model, (XGBClassifier, RandomForestClassifier)):
            model_type = 'tree'
        elif isinstance(model, LogisticRegression):
            model_type = 'linear'
        elif isinstance(model, SVC):
            model_type = 'kernel'
        else:
            model_type = 'kernel'
            logger.warning(f"Unknown model type {type(model)}, using KernelExplainer")

    if model_type == 'tree':
        logger.debug("Using TreeExplainer")
        explainer = shap.TreeExplainer(model)

    elif model_type == 'linear':
        logger.debug("Using LinearExplainer")
        explainer = shap.LinearExplainer(model, X_background_sample)

    elif model_type == 'kernel':
        logger.debug(f"Using KernelExplainer (n={len(X_background_sample)} background samples)")

        if not hasattr(model, 'predict_proba'):
            raise ValueError(
                f"Model {type(model).__name__} does not support predict_proba. "
                "KernelExplainer requires probability predictions."
            )

        def model_predict(X: NDArray) -> NDArray:
            return model.predict_proba(X)[:, 1]

        explainer = shap.KernelExplainer(model_predict, X_background_sample)

    else:
        raise ValueError(f"Invalid model_type: {model_type}")

    return explainer


def compute_shap_values(
    explainer: shap.Explainer,
    X_explain: NDArray,
    model_type: str = 'auto',
    max_samples: int | None = None
) -> tuple[NDArray, NDArray | None]:
    """
    Compute SHAP values for given samples.

    Args:
        explainer: SHAP explainer instance
        X_explain: Samples to explain
        model_type: Model type (for logging)
        max_samples: Max samples to compute (subsamples if exceeded)

    Returns:
        Tuple of (shap_values array, indices used or None if all samples)
    """
    if max_samples and len(X_explain) > max_samples:
        # Use isolated RNG to avoid mutating global random state
        rng = np.random.default_rng(RANDOM_STATE)
        indices = rng.choice(len(X_explain), max_samples, replace=False)
        X_explain_sample = X_explain[indices]
    else:
        X_explain_sample = X_explain
        indices = None

    start_time = time.time()

    if isinstance(explainer, shap.TreeExplainer):
        shap_values = explainer.shap_values(X_explain_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

    elif isinstance(explainer, shap.LinearExplainer):
        shap_values = explainer.shap_values(X_explain_sample)

    elif isinstance(explainer, shap.KernelExplainer):
        shap_values = explainer.shap_values(X_explain_sample, silent=True)

    else:
        shap_values = explainer(X_explain_sample)

    elapsed = time.time() - start_time
    logger.debug(f"SHAP computed for {len(X_explain_sample)} samples in {elapsed:.1f}s")

    return shap_values, indices


def plot_shap_summary(
    shap_values: NDArray,
    X_explain: NDArray,
    feature_names: list[str] | None = None,
    plot_type: Literal['bar', 'beeswarm'] = 'bar',
    max_display: int = SHAP_MAX_DISPLAY,
    save_path: str | Path | None = None
) -> None:
    """
    Plot SHAP summary (global feature importance).

    Args:
        shap_values: SHAP values array
        X_explain: Feature values for the explained samples
        feature_names: List of feature names
        plot_type: 'bar' for importance bar chart, 'beeswarm' for distribution
        max_display: Maximum features to display
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=(10, 6))

    if plot_type == 'bar':
        shap.summary_plot(
            shap_values, X_explain,
            feature_names=feature_names,
            plot_type='bar',
            max_display=max_display,
            show=False
        )
        plt.title("Feature Importance (Mean |SHAP|)", fontsize=14, fontweight='bold')

    elif plot_type == 'beeswarm':
        shap.summary_plot(
            shap_values, X_explain,
            feature_names=feature_names,
            max_display=max_display,
            show=False
        )
        plt.title("SHAP Summary", fontsize=14, fontweight='bold')

    plt.tight_layout()
    finalize_figure(save_path)


def plot_shap_waterfall(
    shap_values: NDArray | shap.Explanation,
    X_explain: NDArray,
    sample_idx: int,
    feature_names: list[str] | None = None,
    base_value: float | None = None,
    max_display: int = SHAP_MAX_DISPLAY,
    save_path: str | Path | None = None
) -> None:
    """
    Plot SHAP waterfall for individual prediction explanation.

    Args:
        shap_values: SHAP values array or Explanation object
        X_explain: Feature values for the explained samples
        sample_idx: Index of sample to explain
        feature_names: List of feature names
        base_value: Expected value (model output on background)
        max_display: Maximum features to display
        save_path: Optional path to save the figure
    """
    if not isinstance(shap_values, shap.Explanation):
        if base_value is None:
            base_value = 0.0

        shap_exp = shap.Explanation(
            values=shap_values[sample_idx],
            base_values=base_value,
            data=X_explain[sample_idx],
            feature_names=feature_names
        )
    else:
        shap_exp = shap_values[sample_idx]

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_exp, max_display=max_display, show=False)
    plt.title(f"SHAP Waterfall (Sample {sample_idx})", fontsize=14, fontweight='bold')
    plt.tight_layout()
    finalize_figure(save_path)


def plot_shap_dependence(
    shap_values: NDArray,
    X_explain: NDArray,
    feature_idx: int | str,
    feature_names: list[str] | None = None,
    interaction_idx: int | str = 'auto',
    save_path: str | Path | None = None
) -> None:
    """
    Plot SHAP dependence (how feature value affects SHAP value).

    Args:
        shap_values: SHAP values array
        X_explain: Feature values for the explained samples
        feature_idx: Index or name of feature to plot
        feature_names: List of feature names
        interaction_idx: Feature to color by ('auto' selects strongest interaction)
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=(10, 6))

    shap.dependence_plot(
        feature_idx, shap_values, X_explain,
        feature_names=feature_names,
        interaction_index=interaction_idx,
        show=False
    )

    if isinstance(feature_idx, int) and feature_names:
        feature_name = feature_names[feature_idx]
    else:
        feature_name = feature_idx

    plt.title(f"SHAP Dependence: {feature_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    finalize_figure(save_path)


def get_top_shap_features(
    shap_values: NDArray,
    feature_names: list[str] | None = None,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Extract top features by mean absolute SHAP value.

    Args:
        shap_values: SHAP values from compute_shap_values()
        feature_names: List of feature names
        top_n: Number of top features to return

    Returns:
        DataFrame with columns: feature, mean_abs_shap, rank
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(mean_abs_shap))]
    elif len(feature_names) != len(mean_abs_shap):
        raise ValueError(
            f"feature_names length ({len(feature_names)}) must match "
            f"number of features ({len(mean_abs_shap)})"
        )

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap,
        'rank': range(1, len(mean_abs_shap) + 1)
    })

    importance_df = importance_df.sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
    importance_df['rank'] = range(1, len(importance_df) + 1)

    return importance_df.head(top_n)


def explain_prediction_shap(
    model: BaseEstimator,
    X_background: NDArray,
    X_single: NDArray,
    feature_names: list[str] | None = None,
    model_type: Literal['auto', 'tree', 'linear', 'kernel'] = 'auto',
    save_path: str | Path | None = None,
    use_cache: bool = True
) -> dict[str, Any]:
    """
    End-to-end SHAP explanation for a single prediction.

    Convenience function that creates explainer, computes SHAP, and plots waterfall.
    Useful for production deployment to explain individual wafer predictions.

    Args:
        model: Trained sklearn-compatible model
        X_background: Background dataset (training set sample)
        X_single: Single sample to explain (1D or 2D array)
        feature_names: List of feature names
        model_type: 'auto', 'tree', 'linear', or 'kernel'
        save_path: Path to save waterfall plot
        use_cache: Use cached explainer for repeated calls (default: True)

    Returns:
        dict with keys: shap_values, prediction, probability, top_features
    """
    # Ensure X_single is 2D
    if X_single.ndim == 1:
        X_single = X_single.reshape(1, -1)

    # Validate feature_names length if provided
    if feature_names is not None and len(feature_names) != X_single.shape[1]:
        raise ValueError(
            f"feature_names length ({len(feature_names)}) must match "
            f"number of features ({X_single.shape[1]})"
        )

    # Get explainer (cached for repeated dashboard calls)
    if use_cache:
        explainer = get_cached_explainer(model, X_background, model_type=model_type)
    else:
        explainer = create_shap_explainer(model, X_background, model_type=model_type)

    # Compute SHAP values
    shap_values, _ = compute_shap_values(explainer, X_single, model_type=model_type)

    # Get prediction
    prediction = model.predict(X_single)[0]
    probability = model.predict_proba(X_single)[0, 1]

    # Plot waterfall
    plot_shap_waterfall(shap_values, X_single, sample_idx=0, feature_names=feature_names, save_path=save_path)

    # Extract top features using centralized helper (DRY)
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X_single.shape[1])]

    top_features = get_sample_feature_contributions(
        shap_values, X_single, feature_names, sample_idx=0, top_n=5
    )

    return {
        'shap_values': shap_values[0],
        'prediction': int(prediction),
        'probability': float(probability),
        'top_features': top_features
    }
