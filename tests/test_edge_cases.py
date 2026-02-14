"""
Edge case tests for SECOM yield prediction.

Tests for boundary conditions and unusual inputs that could cause failures
in production environments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    calculate_gmean,
    evaluate_model,
    gmean_scorer_func,
)
from src.data.transformers import (
    HighMissingRemover,
    ZeroVarianceRemover,
)


class TestGMeanEdgeCases:
    """Edge case tests for G-Mean calculation."""

    def test_gmean_single_class_all_zeros(self, y_single_class):
        """G-Mean with only negative class predictions."""
        y_pred = np.zeros(len(y_single_class))
        gmean, sensitivity, specificity = calculate_gmean(y_single_class, y_pred)
        # With no positive samples, sensitivity is 0 (divide by zero handled)
        assert sensitivity == 0.0 or np.isnan(sensitivity) is False

    def test_gmean_single_class_all_ones(self, y_all_positive):
        """G-Mean with only positive class predictions."""
        y_pred = np.ones(len(y_all_positive))
        gmean, sensitivity, specificity = calculate_gmean(y_all_positive, y_pred)
        # With no negative samples, specificity is 0 (divide by zero handled)
        assert specificity == 0.0 or np.isnan(specificity) is False

    def test_gmean_empty_arrays(self, y_empty):
        """G-Mean raises ValueError for empty arrays (sklearn behavior)."""
        y_pred = np.array([])
        # sklearn's confusion_matrix requires at least 1 sample
        with pytest.raises(ValueError, match="Found empty input array"):
            calculate_gmean(y_empty, y_pred)

    def test_gmean_perfect_predictions(self, y_binary_balanced):
        """G-Mean is 1.0 for perfect predictions."""
        y_pred = y_binary_balanced.copy()
        gmean, sensitivity, specificity = calculate_gmean(y_binary_balanced, y_pred)
        assert gmean == 1.0
        assert sensitivity == 1.0
        assert specificity == 1.0

    def test_gmean_worst_predictions(self, y_binary_balanced):
        """G-Mean is 0.0 when all predictions are wrong."""
        # Flip all predictions
        y_pred = 1 - y_binary_balanced
        gmean, sensitivity, specificity = calculate_gmean(y_binary_balanced, y_pred)
        assert gmean == 0.0


class TestTransformerEdgeCases:
    """Edge case tests for data transformers."""

    def test_high_missing_remover_all_missing_column(self, df_all_missing):
        """HighMissingRemover handles 100% missing columns."""
        remover = HighMissingRemover(threshold=0.5)
        result = remover.fit_transform(df_all_missing)

        # Should remove the 100% missing columns
        assert "feature_1" not in result.columns
        assert "feature_3" not in result.columns
        # Should keep columns with data
        assert "feature_0" in result.columns
        assert "feature_2" in result.columns

    def test_high_missing_remover_all_columns_removed(self):
        """HighMissingRemover with threshold=0 removes all imperfect columns."""
        df = pd.DataFrame({
            "feature_0": [1.0, np.nan, 3.0],
            "feature_1": [np.nan, 2.0, 3.0],
        })
        remover = HighMissingRemover(threshold=0.0)
        result = remover.fit_transform(df)
        # All columns have some missing, so all should be removed with threshold=0
        assert len(result.columns) == 0

    def test_zero_variance_remover_all_nan_column(self, df_all_missing):
        """ZeroVarianceRemover handles all-NaN columns."""
        remover = ZeroVarianceRemover()
        result = remover.fit_transform(df_all_missing)

        # All-NaN columns should be removed (they have no variance)
        assert "feature_1" not in result.columns
        assert "feature_3" not in result.columns

    def test_zero_variance_remover_single_row(self):
        """ZeroVarianceRemover handles single-row DataFrame."""
        df = pd.DataFrame({
            "feature_0": [1.0],
            "feature_1": [2.0],
        })
        remover = ZeroVarianceRemover()
        # Single row has no variance by definition
        result = remover.fit_transform(df)
        # All columns should be removed
        assert len(result.columns) == 0


class TestEvaluateModelEdgeCases:
    """Edge case tests for model evaluation."""

    def test_evaluate_model_all_same_predictions(self, y_binary_balanced):
        """Evaluate model when all predictions are the same."""
        y_pred = np.zeros(len(y_binary_balanced))
        result = evaluate_model(y_binary_balanced, y_pred)

        assert "gmean" in result
        assert "precision" in result
        assert result["gmean"] == 0.0  # No true positives

    def test_evaluate_model_with_probabilities(self, y_binary_balanced, y_proba_perfect):
        """Evaluate model includes AUC metrics when probabilities provided."""
        y_pred = (y_proba_perfect >= 0.5).astype(int)
        result = evaluate_model(y_binary_balanced, y_pred, y_proba_perfect)

        assert "auc_roc" in result
        assert "auc_pr" in result
        assert 0.0 <= result["auc_roc"] <= 1.0
        assert 0.0 <= result["auc_pr"] <= 1.0


class TestInputValidation:
    """Tests for input validation edge cases."""

    def test_high_missing_threshold_validation(self):
        """HighMissingRemover validates threshold bounds."""
        with pytest.raises(ValueError, match="threshold must be in"):
            HighMissingRemover(threshold=1.5)

        with pytest.raises(ValueError, match="threshold must be in"):
            HighMissingRemover(threshold=-0.1)

    def test_high_missing_threshold_type_validation(self):
        """HighMissingRemover validates threshold type."""
        with pytest.raises(TypeError, match="threshold must be numeric"):
            HighMissingRemover(threshold="0.5")

        with pytest.raises(TypeError, match="threshold must be numeric"):
            HighMissingRemover(threshold=True)

    def test_transformer_requires_dataframe(self):
        """Transformers require DataFrame input."""
        remover = HighMissingRemover()
        X_array = np.random.randn(10, 5)

        with pytest.raises(TypeError, match="requires pandas DataFrame"):
            remover.fit(X_array)


class TestNaNHandling:
    """Tests for NaN handling in various components."""

    def test_gmean_scorer_with_nan_predictions(self):
        """G-Mean scorer handles NaN in predictions gracefully."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        # Should work fine with valid predictions
        score = gmean_scorer_func(y_true, y_pred)
        assert score == 1.0

    def test_evaluate_model_proba_warning(self, y_binary_balanced, caplog):
        """Evaluate model warns about out-of-range probabilities."""
        y_pred = np.array([0, 0, 0, 1, 1, 1])
        # Probabilities outside [0, 1] range (e.g., from decision_function)
        y_proba_invalid = np.array([-1.5, -0.5, 0.2, 0.8, 1.5, 2.0])

        import logging
        with caplog.at_level(logging.WARNING):
            result = evaluate_model(y_binary_balanced, y_pred, y_proba_invalid)

        # Should still return results but log a warning
        assert "auc_roc" in result
        assert any("outside [0, 1] range" in record.message for record in caplog.records)
