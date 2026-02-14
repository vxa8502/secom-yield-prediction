"""
Data loading utilities for SECOM yield prediction project
Author: Victoria A.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..config import DATA_DIR, FEATURE_SETS

logger = logging.getLogger('secom')

FeatureSetName = Literal['lasso', 'pca', 'all']


def load_labels() -> tuple[NDArray, NDArray]:
    """
    Load train and test labels.

    Returns:
        Tuple of (y_train, y_test) arrays

    Raises:
        FileNotFoundError: If label files don't exist (run preprocessing first)
    """
    train_path = DATA_DIR / 'y_train.csv'
    test_path = DATA_DIR / 'y_test.csv'

    if not train_path.exists():
        raise FileNotFoundError(
            f"Labels not found: {train_path}\n"
            "Run `make preprocess` or `make pipeline` first."
        )

    y_train = pd.read_csv(train_path).values.ravel()
    y_test = pd.read_csv(test_path).values.ravel()

    logger.info(f"Loaded labels: train={len(y_train)}, test={len(y_test)}, "
                f"fail_rate={y_train.mean():.1%}")
    return y_train, y_test


def load_features(feature_set: FeatureSetName = 'lasso') -> tuple[NDArray, NDArray]:
    """
    Load preprocessed feature arrays for specified feature set.

    Args:
        feature_set: One of 'lasso', 'pca', or 'all'

    Returns:
        Tuple of (X_train, X_test) arrays

    Raises:
        ValueError: If feature_set is not valid or loaded data is malformed
        FileNotFoundError: If feature files don't exist (run preprocessing first)
        IOError: If files cannot be loaded
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"Invalid feature_set '{feature_set}'. "
            f"Valid options: {list(FEATURE_SETS.keys())}"
        )

    config = FEATURE_SETS[feature_set]
    train_path = DATA_DIR / config['train_file']
    test_path = DATA_DIR / config['test_file']

    if not train_path.exists():
        raise FileNotFoundError(
            f"Features not found: {train_path}\n"
            "Run `make preprocess` or `make pipeline` first."
        )

    # Load with error handling
    try:
        X_train = np.load(train_path)
        X_test = np.load(test_path)
    except (OSError, ValueError) as e:
        raise IOError(f"Failed to load feature arrays: {e}") from e

    # Validate array dimensions
    if X_train.ndim != 2:
        raise ValueError(
            f"Expected 2D train array, got shape {X_train.shape} (ndim={X_train.ndim})"
        )
    if X_test.ndim != 2:
        raise ValueError(
            f"Expected 2D test array, got shape {X_test.shape} (ndim={X_test.ndim})"
        )

    # Validate feature dimension match
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: train has {X_train.shape[1]} features, "
            f"test has {X_test.shape[1]} features"
        )

    # Check for NaN values (warning, not error - may be intentional)
    train_nan_count = np.isnan(X_train).sum()
    test_nan_count = np.isnan(X_test).sum()
    if train_nan_count > 0 or test_nan_count > 0:
        logger.warning(
            f"NaN values detected in {feature_set}: "
            f"train={train_nan_count}, test={test_nan_count}"
        )

    logger.info(f"Loaded features ({feature_set}): train={X_train.shape}, test={X_test.shape}")
    return X_train, X_test


def load_all_feature_sets() -> dict[str, tuple[NDArray, NDArray]]:
    """
    Load all three feature sets at once.

    Returns:
        Dict mapping feature set name to (X_train, X_test) tuple
    """
    logger.info("Loading all feature sets...")
    return {fs: load_features(fs) for fs in ['lasso', 'pca', 'all']}
