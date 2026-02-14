"""
Unit tests for model factory.
"""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline

from src.models.factory import build_model, build_pipeline, get_sampler
from src.models.registry import SUPPORTED_MODELS, MODEL_CLASSES


class TestBuildModel:
    """Tests for build_model function."""

    def test_builds_logreg(self):
        """Test building LogisticRegression from config."""
        config = {
            'model': 'LogReg',
            'param_C': 0.5,
            'param_l1_ratio': 0.3,
            'param_max_iter': 1000,
        }

        model = build_model(config)

        assert isinstance(model, LogisticRegression)
        assert model.C == 0.5
        assert model.max_iter == 1000

    def test_builds_randomforest(self):
        """Test building RandomForestClassifier from config."""
        config = {
            'model': 'RandomForest',
            'param_n_estimators': 200,
            'param_max_depth': 10,
        }

        model = build_model(config)

        assert isinstance(model, RandomForestClassifier)
        assert model.n_estimators == 200
        assert model.max_depth == 10

    def test_builds_xgboost(self):
        """Test building XGBClassifier from config."""
        config = {
            'model': 'XGBoost',
            'param_n_estimators': 100,
            'param_learning_rate': 0.1,
        }

        model = build_model(config)

        assert isinstance(model, XGBClassifier)
        assert model.n_estimators == 100
        assert model.learning_rate == 0.1

    def test_builds_svm(self):
        """Test building SVC from config."""
        config = {
            'model': 'SVM',
            'param_C': 1.0,
            'param_kernel': 'rbf',
        }

        model = build_model(config)

        assert isinstance(model, SVC)
        assert model.C == 1.0
        assert model.kernel == 'rbf'
        assert model.probability == True  # Should always be True

    def test_class_balancing_applied(self):
        """Test that class balancing params are applied when requested."""
        config = {'model': 'LogReg', 'param_C': 1.0}

        model_balanced = build_model(config, with_class_balancing=True)
        model_unbalanced = build_model(config, with_class_balancing=False)

        assert model_balanced.class_weight == 'balanced'
        assert model_unbalanced.class_weight is None

    def test_all_supported_models(self):
        """Test that all supported models can be built."""
        for model_name in SUPPORTED_MODELS:
            config = {'model': model_name}
            model = build_model(config)

            assert isinstance(model, MODEL_CLASSES[model_name])


class TestBuildPipeline:
    """Tests for build_pipeline function."""

    def test_builds_pipeline_no_sampling(self):
        """Test building pipeline without sampling."""
        config = {
            'model': 'LogReg',
            'sampling_strategy': 'native',
            'param_C': 1.0,
        }

        pipeline = build_pipeline(config)

        assert isinstance(pipeline, Pipeline)
        assert 'classifier' in pipeline.named_steps

    def test_builds_pipeline_with_smote(self):
        """Test building pipeline with SMOTE sampling."""
        config = {
            'model': 'LogReg',
            'sampling_strategy': 'smote',
            'param_C': 1.0,
        }

        pipeline = build_pipeline(config)

        assert isinstance(pipeline, ImbPipeline)
        assert 'sampler' in pipeline.named_steps
        assert 'classifier' in pipeline.named_steps

    def test_builds_pipeline_with_adasyn(self):
        """Test building pipeline with ADASYN sampling."""
        config = {
            'model': 'LogReg',
            'sampling_strategy': 'adasyn',
            'param_C': 1.0,
        }

        pipeline = build_pipeline(config)

        assert isinstance(pipeline, ImbPipeline)
        assert 'sampler' in pipeline.named_steps

    def test_sampling_override(self):
        """Test that sampling_strategy parameter overrides config."""
        config = {
            'model': 'LogReg',
            'sampling_strategy': 'smote',
            'param_C': 1.0,
        }

        # Override to native
        pipeline = build_pipeline(config, sampling_strategy='native')

        assert isinstance(pipeline, Pipeline)
        assert 'sampler' not in pipeline.named_steps

    def test_invalid_sampling_raises(self):
        """Test that invalid sampling strategy raises error."""
        config = {
            'model': 'LogReg',
            'sampling_strategy': 'invalid_strategy',
            'param_C': 1.0,
        }

        with pytest.raises(ValueError, match="Unknown sampling_strategy"):
            build_pipeline(config)


class TestGetSampler:
    """Tests for get_sampler function."""

    def test_native_returns_none(self):
        """Test that 'native' strategy returns None (uses model's built-in balancing)."""
        sampler = get_sampler('native')
        assert sampler is None

    def test_smote_returns_smote(self):
        """Test that 'smote' returns SMOTE sampler."""
        from imblearn.over_sampling import SMOTE

        sampler = get_sampler('smote')
        assert isinstance(sampler, SMOTE)

    def test_adasyn_returns_adasyn(self):
        """Test that 'adasyn' returns ADASYN sampler."""
        from imblearn.over_sampling import ADASYN

        sampler = get_sampler('adasyn')
        assert isinstance(sampler, ADASYN)

    def test_invalid_raises(self):
        """Test that invalid strategy raises error."""
        with pytest.raises(ValueError):
            get_sampler('invalid')


class TestPipelineFitPredict:
    """Integration tests for pipeline fit/predict."""

    @pytest.fixture
    def sample_data(self):
        """Create sample imbalanced data."""
        np.random.seed(42)
        # 100 samples, 10 features, 10% positive class
        X = np.random.randn(100, 10)
        y = np.array([1] * 10 + [0] * 90)
        np.random.shuffle(y)
        return X, y

    def test_pipeline_fits_and_predicts(self, sample_data):
        """Test that pipeline can fit and predict."""
        X, y = sample_data

        config = {
            'model': 'LogReg',
            'sampling_strategy': 'native',
            'param_C': 1.0,
            'param_max_iter': 1000,
        }

        pipeline = build_pipeline(config)
        pipeline.fit(X, y)

        predictions = pipeline.predict(X)
        probabilities = pipeline.predict_proba(X)

        assert len(predictions) == len(y)
        assert probabilities.shape == (len(y), 2)

    def test_smote_pipeline_fits(self, sample_data):
        """Test that SMOTE pipeline can fit."""
        X, y = sample_data

        config = {
            'model': 'RandomForest',
            'sampling_strategy': 'smote',
            'param_n_estimators': 10,
            'param_max_depth': 5,
        }

        pipeline = build_pipeline(config)
        pipeline.fit(X, y)

        predictions = pipeline.predict(X)
        assert len(predictions) == len(y)
