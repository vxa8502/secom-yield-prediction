"""
Unit tests for preprocessing transformers.
"""

import numpy as np
import pandas as pd
import pickle
import tempfile
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.data.transformers import (
    HighMissingRemover,
    ZeroVarianceRemover,
    DataFrameWrapper,
)


class TestHighMissingRemover:
    """Tests for HighMissingRemover transformer."""

    def test_removes_high_missing_columns(self):
        """Test that columns with >threshold missing are removed."""
        # Create DataFrame with different missing rates
        df = pd.DataFrame({
            'low_missing': [1.0, 2.0, 3.0, 4.0, 5.0],  # 0% missing
            'medium_missing': [1.0, np.nan, 3.0, 4.0, 5.0],  # 20% missing
            'high_missing': [1.0, np.nan, np.nan, np.nan, 5.0],  # 60% missing
        })

        transformer = HighMissingRemover(threshold=0.50)
        transformer.fit(df)
        result = transformer.transform(df)

        assert 'low_missing' in result.columns
        assert 'medium_missing' in result.columns
        assert 'high_missing' not in result.columns

    def test_keeps_all_if_below_threshold(self):
        """Test that all columns kept if below threshold."""
        df = pd.DataFrame({
            'col1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'col2': [1.0, 2.0, 3.0, np.nan, 5.0],  # 20% missing
        })

        transformer = HighMissingRemover(threshold=0.50)
        transformer.fit(df)
        result = transformer.transform(df)

        assert len(result.columns) == 2

    def test_stores_features_to_keep(self):
        """Test that features_to_keep_ is set after fit."""
        df = pd.DataFrame({
            'keep': [1.0, 2.0, 3.0],
            'remove': [np.nan, np.nan, 3.0],  # 67% missing
        })

        transformer = HighMissingRemover(threshold=0.50)
        transformer.fit(df)

        assert transformer.features_to_keep_ == ['keep']


class TestZeroVarianceRemover:
    """Tests for ZeroVarianceRemover transformer."""

    def test_removes_constant_columns(self):
        """Test that constant columns are removed."""
        df = pd.DataFrame({
            'varying': [1.0, 2.0, 3.0, 4.0, 5.0],
            'constant': [1.0, 1.0, 1.0, 1.0, 1.0],
        })

        transformer = ZeroVarianceRemover()
        transformer.fit(df)
        result = transformer.transform(df)

        assert 'varying' in result.columns
        assert 'constant' not in result.columns

    def test_keeps_varying_columns(self):
        """Test that varying columns are kept."""
        df = pd.DataFrame({
            'col1': [1.0, 2.0, 3.0],
            'col2': [4.0, 5.0, 6.0],
        })

        transformer = ZeroVarianceRemover()
        transformer.fit(df)
        result = transformer.transform(df)

        assert len(result.columns) == 2


class TestDataFrameWrapper:
    """Tests for DataFrameWrapper transformer."""

    def test_preserves_dataframe_output(self):
        """Test that output is DataFrame with correct columns."""
        df = pd.DataFrame({
            'col1': [1.0, 2.0, 3.0],
            'col2': [4.0, 5.0, 6.0],
        })

        transformer = DataFrameWrapper(StandardScaler())
        transformer.fit(df)
        result = transformer.transform(df)

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ['col1', 'col2']

    def test_applies_inner_transformer(self):
        """Test that inner transformer is applied."""
        df = pd.DataFrame({
            'col1': [0.0, 10.0, 20.0],
        })

        transformer = DataFrameWrapper(StandardScaler())
        transformer.fit(df)
        result = transformer.transform(df)

        # StandardScaler should center and scale
        # Note: pandas std() uses ddof=1 by default, sklearn uses ddof=0
        assert np.isclose(result['col1'].mean(), 0.0, atol=1e-10)
        assert np.isclose(result['col1'].std(ddof=0), 1.0, atol=1e-10)


class TestPipelinePicklability:
    """Tests for pipeline serialization."""

    def test_high_missing_remover_picklable(self):
        """Test that HighMissingRemover can be pickled."""
        df = pd.DataFrame({
            'col1': [1.0, 2.0, 3.0],
            'col2': [np.nan, np.nan, 3.0],
        })

        transformer = HighMissingRemover(threshold=0.50)
        transformer.fit(df)

        # Pickle and unpickle
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            pickle.dump(transformer, f)
            f.seek(0)
            loaded = pickle.load(open(f.name, 'rb'))

        assert loaded.features_to_keep_ == transformer.features_to_keep_

    def test_zero_variance_remover_picklable(self):
        """Test that ZeroVarianceRemover can be pickled."""
        df = pd.DataFrame({
            'varying': [1.0, 2.0, 3.0],
            'constant': [1.0, 1.0, 1.0],
        })

        transformer = ZeroVarianceRemover()
        transformer.fit(df)

        # Pickle and unpickle
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            pickle.dump(transformer, f)
            f.seek(0)
            loaded = pickle.load(open(f.name, 'rb'))

        assert loaded.features_to_keep_ == transformer.features_to_keep_

    def test_full_pipeline_picklable(self):
        """Test that full preprocessing pipeline can be pickled."""
        df = pd.DataFrame({
            'col1': [1.0, 2.0, np.nan, 4.0],
            'col2': [5.0, 5.0, 5.0, 5.0],  # constant
            'col3': [1.0, np.nan, np.nan, 4.0],  # 50% missing
        })

        pipeline = Pipeline([
            ('remove_zero_variance', ZeroVarianceRemover()),
            ('impute', DataFrameWrapper(SimpleImputer(strategy='median'))),
        ])

        pipeline.fit(df)
        result1 = pipeline.transform(df)

        # Pickle and unpickle
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            pickle.dump(pipeline, f)
            f.seek(0)
            loaded = pickle.load(open(f.name, 'rb'))

        result2 = loaded.transform(df)

        # Results should be identical
        np.testing.assert_array_almost_equal(result1.values, result2.values)
