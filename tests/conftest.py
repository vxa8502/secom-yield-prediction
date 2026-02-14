"""
Shared pytest fixtures for SECOM yield prediction tests.

Provides common test data fixtures to eliminate duplication across test files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import RANDOM_STATE


# =============================================================================
# BASIC ARRAY FIXTURES
# =============================================================================

@pytest.fixture
def y_binary_balanced() -> np.ndarray:
    """Balanced binary labels (3 each class)."""
    return np.array([0, 0, 0, 1, 1, 1])


@pytest.fixture
def y_binary_imbalanced() -> np.ndarray:
    """Imbalanced binary labels (~10% positive)."""
    np.random.seed(RANDOM_STATE)
    y = np.array([0] * 90 + [1] * 10)
    np.random.shuffle(y)
    return y


@pytest.fixture
def y_proba_perfect() -> np.ndarray:
    """Perfect probability predictions (for balanced labels)."""
    return np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])


@pytest.fixture
def y_proba_random() -> np.ndarray:
    """Random probability predictions."""
    np.random.seed(RANDOM_STATE)
    return np.random.rand(6)


# =============================================================================
# FEATURE ARRAY FIXTURES
# =============================================================================

@pytest.fixture
def X_small() -> np.ndarray:
    """Small feature array (6 samples, 5 features)."""
    np.random.seed(RANDOM_STATE)
    return np.random.randn(6, 5)


@pytest.fixture
def X_medium() -> np.ndarray:
    """Medium feature array (100 samples, 10 features)."""
    np.random.seed(RANDOM_STATE)
    return np.random.randn(100, 10)


@pytest.fixture
def X_with_predictive_signal() -> tuple[np.ndarray, np.ndarray]:
    """Features with predictive signal for classification tests."""
    np.random.seed(RANDOM_STATE)
    n_samples = 100
    X = np.random.randn(n_samples, 5)
    # First feature is predictive
    y = (X[:, 0] > 0).astype(int)
    return X, y


# =============================================================================
# DATAFRAME FIXTURES
# =============================================================================

@pytest.fixture
def df_with_missing() -> pd.DataFrame:
    """DataFrame with missing values (~5% missing)."""
    np.random.seed(RANDOM_STATE)
    n_samples = 100
    n_features = 10

    data = np.random.randn(n_samples, n_features)
    data[np.random.rand(*data.shape) < 0.05] = np.nan

    return pd.DataFrame(
        data,
        columns=[f"feature_{i}" for i in range(n_features)]
    )


@pytest.fixture
def df_with_high_correlation() -> pd.DataFrame:
    """DataFrame with highly correlated features."""
    np.random.seed(RANDOM_STATE)
    n_samples = 100

    base = np.random.randn(n_samples)
    return pd.DataFrame({
        "feature_0": base,
        "feature_1": base * 0.99 + np.random.randn(n_samples) * 0.01,
        "feature_2": base * 0.98 + np.random.randn(n_samples) * 0.02,
        "feature_3": np.random.randn(n_samples),
        "feature_4": np.random.randn(n_samples),
    })


@pytest.fixture
def df_with_zero_variance() -> pd.DataFrame:
    """DataFrame with zero variance features."""
    np.random.seed(RANDOM_STATE)
    n_samples = 100

    return pd.DataFrame({
        "feature_0": np.random.randn(n_samples),
        "feature_1": np.ones(n_samples),  # Zero variance
        "feature_2": np.random.randn(n_samples),
        "feature_3": np.zeros(n_samples),  # Zero variance
    })


# =============================================================================
# MODEL FIXTURES
# =============================================================================

@pytest.fixture
def fitted_logreg(X_with_predictive_signal) -> LogisticRegression:
    """Fitted LogisticRegression model."""
    X, y = X_with_predictive_signal
    model = LogisticRegression(random_state=RANDOM_STATE, max_iter=200)
    model.fit(X, y)
    return model


@pytest.fixture
def fitted_rf(X_with_predictive_signal) -> RandomForestClassifier:
    """Fitted RandomForestClassifier model."""
    X, y = X_with_predictive_signal
    model = RandomForestClassifier(n_estimators=10, random_state=RANDOM_STATE)
    model.fit(X, y)
    return model


# =============================================================================
# PIPELINE CONFIGURATION FIXTURES
# =============================================================================

@pytest.fixture
def model_config_logreg() -> dict:
    """Configuration dict for LogisticRegression."""
    return {
        "model": "LogReg",
        "feature_set": "lasso",
        "sampling_strategy": "native",
        "param_C": 1.0,
        "param_l1_ratio": 0.5,
    }


@pytest.fixture
def model_config_svm() -> dict:
    """Configuration dict for SVM."""
    return {
        "model": "SVM",
        "feature_set": "lasso",
        "sampling_strategy": "native",
        "param_C": 1.0,
        "param_gamma": 0.1,
    }


@pytest.fixture
def model_config_xgboost() -> dict:
    """Configuration dict for XGBoost."""
    return {
        "model": "XGBoost",
        "feature_set": "lasso",
        "sampling_strategy": "native",
        "param_n_estimators": 50,
        "param_max_depth": 3,
        "param_learning_rate": 0.1,
    }


@pytest.fixture
def model_config_rf() -> dict:
    """Configuration dict for RandomForest."""
    return {
        "model": "RandomForest",
        "feature_set": "lasso",
        "sampling_strategy": "native",
        "param_n_estimators": 50,
        "param_max_depth": 5,
    }


# =============================================================================
# MOCK FEATURE SETS FIXTURE
# =============================================================================

@pytest.fixture
def mock_feature_sets() -> dict:
    """Mock feature sets for tuning tests."""
    np.random.seed(RANDOM_STATE)
    return {
        "lasso": (np.random.randn(80, 6), np.random.randn(20, 6)),
        "pca": (np.random.randn(80, 10), np.random.randn(20, 10)),
        "all": (np.random.randn(80, 30), np.random.randn(20, 30)),
    }


@pytest.fixture
def mock_labels() -> tuple[np.ndarray, np.ndarray]:
    """Mock labels with class imbalance."""
    np.random.seed(RANDOM_STATE)
    y_train = np.array([0] * 72 + [1] * 8)
    y_test = np.array([0] * 18 + [1] * 2)
    np.random.shuffle(y_train)
    np.random.shuffle(y_test)
    return y_train, y_test


# =============================================================================
# EDGE CASE FIXTURES
# =============================================================================

@pytest.fixture
def X_tiny() -> np.ndarray:
    """Tiny feature array (3 samples, 2 features) - below CV fold minimum."""
    np.random.seed(RANDOM_STATE)
    return np.random.randn(3, 2)


@pytest.fixture
def y_single_class() -> np.ndarray:
    """Labels with only one class (all zeros)."""
    return np.array([0, 0, 0, 0, 0, 0])


@pytest.fixture
def y_all_positive() -> np.ndarray:
    """Labels with only positive class (all ones)."""
    return np.array([1, 1, 1, 1, 1, 1])


@pytest.fixture
def df_all_missing() -> pd.DataFrame:
    """DataFrame with 100% missing values in some columns."""
    np.random.seed(RANDOM_STATE)
    n_samples = 20

    return pd.DataFrame({
        "feature_0": np.random.randn(n_samples),
        "feature_1": [np.nan] * n_samples,  # 100% missing
        "feature_2": np.random.randn(n_samples),
        "feature_3": [np.nan] * n_samples,  # 100% missing
    })


@pytest.fixture
def X_empty() -> np.ndarray:
    """Empty feature array (0 samples)."""
    return np.empty((0, 5))


@pytest.fixture
def y_empty() -> np.ndarray:
    """Empty label array (0 samples)."""
    return np.array([])


@pytest.fixture
def X_single_sample() -> np.ndarray:
    """Single sample feature array."""
    np.random.seed(RANDOM_STATE)
    return np.random.randn(1, 5)


@pytest.fixture
def X_nan_heavy() -> np.ndarray:
    """Feature array with many NaN values (~50%)."""
    np.random.seed(RANDOM_STATE)
    X = np.random.randn(20, 5)
    X[np.random.rand(*X.shape) < 0.5] = np.nan
    return X
