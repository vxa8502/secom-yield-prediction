"""
Custom scikit-learn transformers for preprocessing

These transformers are used in preprocessing pipelines and need to be
importable when loading pickled pipelines.

METHODOLOGY NOTES:
- L1FeatureSelector uses LogisticRegressionCV (classification-appropriate)
  instead of regression LASSO
- ZeroVarianceRemover handles NaN values correctly
- All transformers preserve DataFrame structure where possible
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LassoCV, Lasso, LogisticRegressionCV
from statsmodels.stats.outliers_influence import variance_inflation_factor

from ..config import RANDOM_STATE

logger = logging.getLogger('secom')


def _validate_dataframe(X: Any, transformer_name: str) -> None:
    """Validate input is a pandas DataFrame."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            f"{transformer_name} requires pandas DataFrame, "
            f"got {type(X).__name__}"
        )


class HighMissingRemover(BaseEstimator, TransformerMixin):
    """
    Remove features with missing values above threshold.

    Args:
        threshold: Maximum allowed missing rate (default 0.50 = 50%)
        name: Pipeline name for logging context

    Raises:
        TypeError: If input is not a pandas DataFrame
        ValueError: If threshold is not in [0, 1]
    """

    def __init__(self, threshold: float = 0.50, name: str = ""):
        if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
            raise TypeError(f"threshold must be numeric, got {type(threshold).__name__}")
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        self.threshold = float(threshold)
        self.name = name
        self.features_to_keep_: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: ArrayLike | None = None) -> "HighMissingRemover":
        _validate_dataframe(X, "HighMissingRemover")
        missing_pct = X.isna().sum() / len(X)
        self.features_to_keep_ = X.columns[missing_pct <= self.threshold].tolist()
        removed = len(X.columns) - len(self.features_to_keep_)
        prefix = f"  [{self.name}] " if self.name else "  "
        logger.info(f"{prefix}HighMissingRemover (>{self.threshold:.0%}): {len(self.features_to_keep_)}/{len(X.columns)} features (removed {removed})")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.features_to_keep_]

    def get_feature_names_out(self, input_features: ArrayLike | None = None) -> NDArray:
        """Return feature names for sklearn 1.1+ compatibility."""
        return np.array(self.features_to_keep_)


class VIFReducer(BaseEstimator, TransformerMixin):
    """
    Reduce multicollinearity using VIF-based hierarchical clustering.

    Strategy:
    1. Calculate VIF for all features
    2. Identify high-VIF features (VIF > threshold)
    3. Cluster high-VIF features by correlation similarity
    4. Select best feature from each cluster (highest target correlation)
    5. Keep low-VIF features + selected representatives

    Args:
        vif_threshold: Features with VIF above this are candidates for removal (default 10)
        cluster_distance: Distance threshold for hierarchical clustering (default 0.10)
            Distance = 1 - |correlation|, so 0.10 means features with |r| >= 0.90 cluster together

    Raises:
        TypeError: If input is not a pandas DataFrame
    """

    def __init__(self, vif_threshold: float = 10.0, cluster_distance: float = 0.10):
        self.vif_threshold = vif_threshold
        self.cluster_distance = cluster_distance
        self.features_to_keep_: list[str] | None = None
        self.vif_data_: pd.DataFrame | None = None
        self.n_high_vif_: int = 0
        self.n_clusters_: int = 0

    def _calculate_vif(self, X: pd.DataFrame) -> pd.DataFrame:
        """Calculate VIF for each feature. Expects clean data with no NaN."""
        if X.isna().any().any():
            raise ValueError("VIFReducer requires clean data with no NaN values. Run imputation first.")
        vif_data = pd.DataFrame({
            "feature": X.columns,
            "vif": [variance_inflation_factor(X.values, i)
                    for i in range(X.shape[1])]
        })
        return vif_data.sort_values('vif', ascending=False)

    def fit(self, X: pd.DataFrame, y: ArrayLike | None = None) -> "VIFReducer":
        _validate_dataframe(X, "VIFReducer")

        logger.info(f"  [shared] VIF: calculating for {len(X.columns)} features...")

        # Calculate VIF
        self.vif_data_ = self._calculate_vif(X)

        # Identify high-VIF features
        high_vif = self.vif_data_[self.vif_data_['vif'] > self.vif_threshold]
        high_vif_features = high_vif['feature'].tolist()
        self.n_high_vif_ = len(high_vif_features)

        logger.info(f"  [shared] VIF: {self.n_high_vif_} features with VIF > {self.vif_threshold}")

        if self.n_high_vif_ == 0:
            # No high-VIF features, keep all
            self.features_to_keep_ = X.columns.tolist()
            self.n_clusters_ = 0
            return self

        # Prepare high-VIF features for clustering (data is already clean, no fillna needed)
        X_high_vif = X[high_vif_features]
        non_constant = X_high_vif.loc[:, X_high_vif.std() > 0]

        if len(non_constant.columns) < 2:
            # Not enough features to cluster
            self.features_to_keep_ = X.columns.tolist()
            self.n_clusters_ = 0
            return self

        # Compute correlation-based distance matrix
        corr = non_constant.corr().abs().fillna(0.0)
        distance_matrix = (1 - corr).clip(lower=0.0)
        condensed_distance = squareform(distance_matrix, checks=False)

        # Hierarchical clustering
        Z = linkage(condensed_distance, method='complete')
        cluster_labels = fcluster(Z, t=self.cluster_distance, criterion='distance')

        clusters = pd.DataFrame({
            "feature": non_constant.columns,
            "cluster": cluster_labels
        })
        self.n_clusters_ = len(clusters['cluster'].unique())

        # Select best feature from each cluster (highest target correlation)
        if y is not None:
            y_series = pd.Series(y).reset_index(drop=True)
            target_corr = X_high_vif.corrwith(y_series).abs()
        else:
            # Fallback: use feature with lowest VIF in cluster
            target_corr = pd.Series(
                {f: -self.vif_data_[self.vif_data_['feature'] == f]['vif'].values[0]
                 for f in high_vif_features}
            )

        selected_from_clusters = (
            clusters
            .assign(score=lambda df: df['feature'].map(target_corr))
            .sort_values(['cluster', 'score'], ascending=[True, False])
            .groupby('cluster')
            .first()['feature']
            .tolist()
        )

        # Final feature set: low-VIF + selected representatives
        features_to_remove = set(high_vif_features) - set(selected_from_clusters)
        self.features_to_keep_ = [f for f in X.columns if f not in features_to_remove]

        removed = len(X.columns) - len(self.features_to_keep_)
        logger.info(f"  [shared] VIF: {len(self.features_to_keep_)}/{len(X.columns)} features "
                    f"({self.n_clusters_} clusters, removed {removed})")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.features_to_keep_]

    def get_feature_names_out(self, input_features: ArrayLike | None = None) -> NDArray:
        """Return feature names for sklearn 1.1+ compatibility."""
        return np.array(self.features_to_keep_)


class ZeroVarianceRemover(BaseEstimator, TransformerMixin):
    """
    Remove features with zero variance.

    FIXED: Handles NaN values correctly by computing variance on non-null values.
    A feature is considered zero-variance if all non-null values are identical.

    Args:
        name: Pipeline name for logging context

    Raises:
        TypeError: If input is not a pandas DataFrame
    """

    def __init__(self, name: str = ""):
        self.name = name
        self.features_to_keep_: list[str] | None = None
        self.all_nan_features_: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: ArrayLike | None = None) -> "ZeroVarianceRemover":
        _validate_dataframe(X, "ZeroVarianceRemover")

        features_to_keep = []
        all_nan_features = []

        for col in X.columns:
            col_data = X[col].dropna()
            if len(col_data) == 0:
                all_nan_features.append(col)
                continue
            if col_data.nunique() > 1:
                features_to_keep.append(col)

        self.features_to_keep_ = features_to_keep
        self.all_nan_features_ = all_nan_features
        removed = len(X.columns) - len(self.features_to_keep_)
        prefix = f"  [{self.name}] " if self.name else "  "

        if all_nan_features:
            logger.warning(f"{prefix}ZeroVarianceRemover: {len(all_nan_features)} features are all NaN")
        if removed > 0:
            logger.info(f"{prefix}ZeroVarianceRemover: removed {removed} constant/empty features")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.features_to_keep_]

    def get_feature_names_out(self, input_features: ArrayLike | None = None) -> NDArray:
        """Return feature names for sklearn 1.1+ compatibility."""
        return np.array(self.features_to_keep_)


class CorrelationSelector(BaseEstimator, TransformerMixin):
    """
    Select features based on correlation with target.

    Note: This is a simple filter method. For classification,
    consider using L1FeatureSelector instead.
    """

    def __init__(self, threshold: float = 0.10):
        if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
            raise TypeError(f"threshold must be numeric, got {type(threshold).__name__}")
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        self.threshold = float(threshold)
        self.selected_features_: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CorrelationSelector":
        # Compute correlation with target
        df = pd.concat([X, y.reset_index(drop=True)], axis=1)
        target_corr = df.corr().abs()[y.name].drop(y.name)
        self.selected_features_ = target_corr[target_corr > self.threshold].index.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.selected_features_]

    def get_feature_names_out(self, input_features: ArrayLike | None = None) -> NDArray:
        """Return feature names for sklearn 1.1+ compatibility."""
        return np.array(self.selected_features_)


class L1FeatureSelector(BaseEstimator, TransformerMixin):
    """
    L1-penalized feature selection using LogisticRegressionCV.

    This is the correct approach for binary classification - uses logistic regression
    with L1 penalty to select features based on classification boundary, not regression.

    Args:
        cv: Number of cross-validation folds for selecting regularization strength
        max_iter: Maximum iterations for solver convergence
        Cs: Regularization strengths to try. Default includes weak regularization
            (large C values) to ensure features are selected.
    """

    # Default Cs range: includes weak regularization (large C) to avoid zeroing all coefficients
    DEFAULT_CS = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]

    def __init__(
        self,
        cv: int = 5,
        max_iter: int = 5000,
        Cs: list[float] | None = None
    ):
        # Type validation for integer parameters
        for name, value in [('cv', cv), ('max_iter', max_iter)]:
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be int, got {type(value).__name__}")
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")

        self.cv = cv
        self.max_iter = max_iter
        self.Cs = Cs if Cs is not None else self.DEFAULT_CS
        self.selected_features_: list[str] | None = None
        self.model_: LogisticRegressionCV | None = None
        self.best_C_: float | None = None
        self.feature_importances_: NDArray | None = None

    def fit(self, X: pd.DataFrame, y: ArrayLike) -> "L1FeatureSelector":
        """
        Fit L1-penalized logistic regression and select non-zero features.
        """
        logger.info(f"  L1FeatureSelector: fitting on {X.shape[1]} features with Cs={self.Cs}")

        # Use LogisticRegressionCV with L1 penalty
        self.model_ = LogisticRegressionCV(
            penalty='l1',
            solver='saga',
            cv=self.cv,
            Cs=self.Cs,
            max_iter=self.max_iter,
            random_state=RANDOM_STATE,
            class_weight='balanced',  # Handle class imbalance
            n_jobs=-1
        )

        self.model_.fit(X, y)
        self.best_C_ = self.model_.C_[0]

        # Select features with non-zero coefficients
        coef = self.model_.coef_.ravel()
        non_zero_mask = coef != 0
        self.selected_features_ = X.columns[non_zero_mask].tolist()
        self.feature_importances_ = np.abs(coef)

        n_selected = len(self.selected_features_)
        logger.info(f"  L1FeatureSelector: selected {n_selected} features at C={self.best_C_:.4f}")

        if n_selected == 0:
            raise ValueError(
                f"L1 selected 0 features even with weak regularization (C={self.best_C_}). "
                f"This indicates a data issue. Check that features have predictive signal."
            )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if len(self.selected_features_) == 0:
            raise ValueError(
                "L1 selected 0 features. Try increasing Cs or reducing regularization."
            )
        return X[self.selected_features_]

    def get_feature_names_out(self, input_features: ArrayLike | None = None) -> NDArray:
        """Return feature names for sklearn 1.1+ compatibility."""
        return np.array(self.selected_features_)


class LassoSelector(BaseEstimator, TransformerMixin):
    """
    DEPRECATED: Use L1FeatureSelector instead.

    LASSO-based feature selection using regression.
    Kept for backward compatibility with existing pickled pipelines.

    Note: This uses regression LASSO which is suboptimal for classification.
    For new code, use L1FeatureSelector which uses LogisticRegressionCV.
    """

    def __init__(
        self,
        alpha: float | None = None,
        cv: int = 5,
        alphas: ArrayLike | None = None
    ):
        self.alpha = alpha
        self.cv = cv
        self.alphas = alphas
        self.selected_features_: list[str] | None = None
        self.lasso_: Lasso | LassoCV | None = None

    def fit(self, X: pd.DataFrame, y: ArrayLike) -> "LassoSelector":
        import warnings
        warnings.warn(
            "LassoSelector uses regression LASSO for classification. "
            "Consider using L1FeatureSelector instead.",
            DeprecationWarning
        )

        # Use LassoCV if alpha not specified
        if self.alpha is None:
            self.lasso_ = LassoCV(
                cv=self.cv,
                alphas=self.alphas,
                random_state=RANDOM_STATE,
                max_iter=10000
            )
        else:
            self.lasso_ = Lasso(alpha=self.alpha, random_state=RANDOM_STATE, max_iter=10000)

        # Fit LASSO
        self.lasso_.fit(X, y)

        # Select features with non-zero coefficients
        non_zero_mask = self.lasso_.coef_ != 0
        self.selected_features_ = X.columns[non_zero_mask].tolist()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if len(self.selected_features_) == 0:
            raise ValueError("LASSO selected 0 features. Try reducing alpha.")
        return X[self.selected_features_]

    def get_feature_names_out(self, input_features: ArrayLike | None = None) -> NDArray:
        """Return feature names for sklearn 1.1+ compatibility."""
        return np.array(self.selected_features_)


class DataFrameWrapper(BaseEstimator, TransformerMixin):
    """
    Wrap sklearn transformers to preserve DataFrame structure.

    Many sklearn transformers (StandardScaler, SimpleImputer) return numpy arrays.
    This wrapper ensures the output remains a DataFrame with column names preserved.
    """

    def __init__(self, transformer: BaseEstimator):
        self.transformer = transformer
        self.feature_names_: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: ArrayLike | None = None) -> "DataFrameWrapper":
        self.feature_names_ = X.columns.tolist()
        self.transformer.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_transformed = self.transformer.transform(X)
        return pd.DataFrame(X_transformed, columns=self.feature_names_, index=X.index)

    def get_feature_names_out(self, input_features: ArrayLike | None = None) -> NDArray:
        """Return feature names for sklearn 1.1+ compatibility."""
        return np.array(self.feature_names_)
