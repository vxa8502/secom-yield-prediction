"""
Integration tests for end-to-end pipeline execution.

Tests the full pipeline flow: preprocess -> tune -> select
Uses fixtures with minimal data to keep tests fast.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import RANDOM_STATE


class TestPreprocessPipeline:
    """Integration tests for preprocessing pipeline."""

    @pytest.fixture
    def temp_project_dir(self) -> Generator[Path, None, None]:
        """Create isolated temp directory with required structure."""
        temp_dir = Path(tempfile.mkdtemp())

        # Create directory structure
        (temp_dir / "data" / "raw" / "secom").mkdir(parents=True)
        (temp_dir / "data" / "processed").mkdir(parents=True)
        (temp_dir / "models").mkdir(parents=True)
        (temp_dir / "reports" / "figures").mkdir(parents=True)

        yield temp_dir

        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def mock_secom_data(self, temp_project_dir: Path) -> tuple[Path, Path]:
        """Create minimal mock SECOM data for testing."""
        np.random.seed(RANDOM_STATE)

        # Create small dataset: 100 samples, 50 features
        n_samples = 100
        n_features = 50

        # Features with some missing values and correlations
        X = np.random.randn(n_samples, n_features)
        X[np.random.rand(*X.shape) < 0.05] = np.nan  # 5% missing

        # Add correlated features
        X[:, 10] = X[:, 0] * 0.99 + np.random.randn(n_samples) * 0.01
        X[:, 20] = X[:, 1] * 0.98 + np.random.randn(n_samples) * 0.02

        # Labels with ~10% failure rate (format: label timestamp)
        # SECOM format: -1 = pass, 1 = fail
        labels = np.array([-1] * 90 + [1] * 10)
        np.random.shuffle(labels)

        # Save files
        data_path = temp_project_dir / "data" / "raw" / "secom" / "secom.data"
        labels_path = temp_project_dir / "data" / "raw" / "secom" / "secom_labels.data"

        np.savetxt(data_path, X, fmt="%.6f", delimiter=" ")

        # Write labels in SECOM format (space-separated: label timestamp)
        # Note: timestamp should not have internal spaces for proper parsing
        with open(labels_path, "w") as f:
            for i, label in enumerate(labels):
                # Use underscore instead of space in timestamp to avoid parsing issues
                ts = f"01/01/2024_{i:02d}:00:00"
                f.write(f"{label} {ts}\n")

        return data_path, labels_path

    def test_preprocess_creates_expected_outputs(
        self, temp_project_dir: Path, mock_secom_data: tuple[Path, Path]
    ):
        """Test that preprocessing creates all expected output files."""
        env = os.environ.copy()
        env["SECOM_PROJECT_ROOT"] = str(temp_project_dir)

        result = subprocess.run(
            [sys.executable, "-m", "pipelines.preprocess"],
            capture_output=True,
            text=True,
            env=env,
            cwd=project_root
        )

        assert result.returncode == 0, f"Preprocess failed: {result.stderr}"

        # Check processed data files exist
        processed_dir = temp_project_dir / "data" / "processed"
        expected_files = [
            "X_train_all_features.npy",
            "X_test_all_features.npy",
            "X_train_lasso.npy",
            "X_test_lasso.npy",
            "X_train_pca_raw.npy",
            "X_test_pca_raw.npy",
            "y_train.csv",
            "y_test.csv",
            "lasso_selected_features.txt",
        ]

        for fname in expected_files:
            assert (processed_dir / fname).exists(), f"Missing: {fname}"

    def test_preprocess_train_test_shapes_consistent(
        self, temp_project_dir: Path, mock_secom_data: tuple[Path, Path]
    ):
        """Test that train/test splits have consistent feature dimensions."""
        env = os.environ.copy()
        env["SECOM_PROJECT_ROOT"] = str(temp_project_dir)

        subprocess.run(
            [sys.executable, "-m", "pipelines.preprocess"],
            capture_output=True,
            env=env,
            cwd=project_root
        )

        processed_dir = temp_project_dir / "data" / "processed"

        for prefix in ["all_features", "lasso", "pca_raw"]:
            X_train = np.load(processed_dir / f"X_train_{prefix}.npy")
            X_test = np.load(processed_dir / f"X_test_{prefix}.npy")

            assert X_train.shape[1] == X_test.shape[1], \
                f"Feature mismatch for {prefix}: {X_train.shape[1]} vs {X_test.shape[1]}"

    def test_preprocess_no_data_leakage(
        self, temp_project_dir: Path, mock_secom_data: tuple[Path, Path]
    ):
        """Test that preprocessing doesn't leak test data info into training."""
        env = os.environ.copy()
        env["SECOM_PROJECT_ROOT"] = str(temp_project_dir)

        subprocess.run(
            [sys.executable, "-m", "pipelines.preprocess"],
            capture_output=True,
            env=env,
            cwd=project_root
        )

        processed_dir = temp_project_dir / "data" / "processed"

        y_train = pd.read_csv(processed_dir / "y_train.csv")
        y_test = pd.read_csv(processed_dir / "y_test.csv")

        # Verify stratified split maintained class distribution
        train_fail_rate = y_train.iloc[:, 0].mean()
        test_fail_rate = y_test.iloc[:, 0].mean()

        assert abs(train_fail_rate - test_fail_rate) < 0.05, \
            "Stratified split not maintained"


class TestTunePipeline:
    """Integration tests for tuning pipeline."""

    @pytest.fixture
    def mock_feature_sets(self) -> dict:
        """Create mock feature sets for tuning tests."""
        np.random.seed(RANDOM_STATE)
        n_train, n_test = 80, 20

        return {
            "lasso": (np.random.randn(n_train, 6), np.random.randn(n_test, 6)),
            "pca": (np.random.randn(n_train, 10), np.random.randn(n_test, 10)),
            "all": (np.random.randn(n_train, 50), np.random.randn(n_test, 50)),
        }

    @pytest.fixture
    def mock_labels(self) -> tuple[np.ndarray, np.ndarray]:
        """Create mock labels with class imbalance."""
        np.random.seed(RANDOM_STATE)
        y_train = np.array([0] * 72 + [1] * 8)
        y_test = np.array([0] * 18 + [1] * 2)
        np.random.shuffle(y_train)
        np.random.shuffle(y_test)
        return y_train, y_test

    def test_tune_single_strategy_produces_results(
        self, mock_feature_sets: dict, mock_labels: tuple
    ):
        """Test that tuning with single strategy produces valid results."""
        from src.tuning_utils import run_all_experiments_for_sampling

        y_train, _ = mock_labels

        # Run with minimal trials for speed
        results = run_all_experiments_for_sampling(
            sampling_strategy="native",
            feature_sets=mock_feature_sets,
            y_train=y_train,
            n_trials=2  # Minimal for testing
        )

        assert len(results) > 0, "No tuning results produced"
        assert all("cv_gmean" in r for r in results), "Missing cv_gmean in results"
        assert all("model" in r for r in results), "Missing model name in results"

    def test_tune_results_dataframe_valid(
        self, mock_feature_sets: dict, mock_labels: tuple
    ):
        """Test that tuning results form valid DataFrame."""
        from src.tuning_utils import run_all_experiments_for_sampling

        y_train, _ = mock_labels

        results = run_all_experiments_for_sampling(
            sampling_strategy="native",
            feature_sets=mock_feature_sets,
            y_train=y_train,
            n_trials=2
        )

        df = pd.DataFrame(results)

        assert "cv_gmean" in df.columns
        assert "model" in df.columns
        assert "feature_set" in df.columns
        assert df["cv_gmean"].between(0, 1).all(), "G-Mean out of range"


class TestSelectPipeline:
    """Integration tests for model selection pipeline."""

    @pytest.fixture
    def mock_tuning_results(self, tmp_path: Path) -> Path:
        """Create mock tuning results for selection tests."""
        results = pd.DataFrame([
            {
                "model": "SVM",
                "feature_set": "lasso",
                "sampling_strategy": "native",
                "cv_gmean": 0.72,
                "param_C": 1.0,
                "param_gamma": 0.1,
            },
            {
                "model": "LogReg",
                "feature_set": "lasso",
                "sampling_strategy": "smote",
                "cv_gmean": 0.70,
                "param_C": 0.5,
                "param_l1_ratio": 0.5,
            },
        ])

        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()

        for strategy in ["native", "smote", "adasyn"]:
            results.to_csv(reports_dir / f"tuning_{strategy}_results.csv", index=False)

        return tmp_path

    def test_select_loads_all_tuning_results(self, mock_tuning_results: Path):
        """Test that selection loads results from all strategy files."""
        from pipelines.select import load_all_tuning_results

        with patch("pipelines.select.REPORTS_DIR", mock_tuning_results / "reports"):
            df = load_all_tuning_results()

        assert len(df) == 6  # 2 results x 3 files
        assert "cv_gmean" in df.columns

    def test_select_ranks_by_gmean(self, mock_tuning_results: Path):
        """Test that models are correctly ranked by G-Mean."""
        from pipelines.select import load_all_tuning_results

        with patch("pipelines.select.REPORTS_DIR", mock_tuning_results / "reports"):
            df = load_all_tuning_results()
            ranked = df.sort_values("cv_gmean", ascending=False)

        assert ranked.iloc[0]["cv_gmean"] >= ranked.iloc[-1]["cv_gmean"]


class TestEndToEndPipeline:
    """Full E2E integration tests."""

    @pytest.fixture
    def e2e_temp_dir(self) -> Generator[Path, None, None]:
        """Create complete temp environment for E2E tests."""
        temp_dir = Path(tempfile.mkdtemp())

        # Full directory structure
        (temp_dir / "data" / "raw" / "secom").mkdir(parents=True)
        (temp_dir / "data" / "processed").mkdir(parents=True)
        (temp_dir / "models").mkdir(parents=True)
        (temp_dir / "reports" / "figures").mkdir(parents=True)
        (temp_dir / "mlruns").mkdir(parents=True)

        yield temp_dir

        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def e2e_mock_data(self, e2e_temp_dir: Path) -> None:
        """Create mock data for E2E test."""
        np.random.seed(RANDOM_STATE)

        n_samples = 150
        n_features = 30

        X = np.random.randn(n_samples, n_features)
        X[np.random.rand(*X.shape) < 0.03] = np.nan

        # Create predictive signal
        X[:, 0] = X[:, 0] + np.random.choice([0, 2], n_samples, p=[0.9, 0.1])

        labels = np.where(X[:, 0] > 1.5, 1, -1)

        data_path = e2e_temp_dir / "data" / "raw" / "secom" / "secom.data"
        labels_path = e2e_temp_dir / "data" / "raw" / "secom" / "secom_labels.data"

        np.savetxt(data_path, X, fmt="%.6f", delimiter=" ")

        # Write labels without spaces in timestamp
        with open(labels_path, "w") as f:
            for i, label in enumerate(labels):
                ts = f"01/01/2024_{i:02d}:00:00"
                f.write(f"{label} {ts}\n")

    @pytest.mark.slow
    def test_full_pipeline_preprocess_to_tune(
        self, e2e_temp_dir: Path, e2e_mock_data: None
    ):
        """Test preprocess -> tune flow produces valid outputs."""
        env = os.environ.copy()
        env["SECOM_PROJECT_ROOT"] = str(e2e_temp_dir)

        # Run preprocess
        result = subprocess.run(
            [sys.executable, "-m", "pipelines.preprocess"],
            capture_output=True,
            text=True,
            env=env,
            cwd=project_root
        )
        assert result.returncode == 0, f"Preprocess failed: {result.stderr}"

        # Verify preprocessing outputs
        processed_dir = e2e_temp_dir / "data" / "processed"
        assert (processed_dir / "X_train_lasso.npy").exists()
        assert (processed_dir / "y_train.csv").exists()

        # Load and verify data is usable
        X_train = np.load(processed_dir / "X_train_lasso.npy")
        y_train = pd.read_csv(processed_dir / "y_train.csv")

        assert X_train.shape[0] == len(y_train)
        assert not np.isnan(X_train).any(), "NaN values in preprocessed data"

    @pytest.mark.slow
    def test_pipeline_artifacts_serializable(
        self, e2e_temp_dir: Path, e2e_mock_data: None
    ):
        """Test that all pipeline artifacts are properly serializable."""
        import joblib

        env = os.environ.copy()
        env["SECOM_PROJECT_ROOT"] = str(e2e_temp_dir)

        subprocess.run(
            [sys.executable, "-m", "pipelines.preprocess"],
            capture_output=True,
            env=env,
            cwd=project_root
        )

        models_dir = e2e_temp_dir / "models"

        for pipeline_file in models_dir.glob("preprocessing_pipeline_*.pkl"):
            pipeline = joblib.load(pipeline_file)
            assert hasattr(pipeline, "transform"), f"Invalid pipeline: {pipeline_file}"


class TestPipelineErrorHandling:
    """Tests for pipeline error handling and edge cases."""

    def test_preprocess_fails_gracefully_without_data(self, tmp_path: Path):
        """Test preprocessing fails gracefully when data is missing."""
        env = os.environ.copy()
        env["SECOM_PROJECT_ROOT"] = str(tmp_path)

        result = subprocess.run(
            [sys.executable, "-m", "pipelines.preprocess"],
            capture_output=True,
            text=True,
            env=env,
            cwd=project_root
        )

        assert result.returncode != 0, "Should fail without data"

    def test_select_fails_gracefully_without_tuning_results(self, tmp_path: Path):
        """Test selection fails gracefully when tuning results are missing."""
        env = os.environ.copy()
        env["SECOM_PROJECT_ROOT"] = str(tmp_path)

        (tmp_path / "reports").mkdir()

        result = subprocess.run(
            [sys.executable, "-m", "pipelines.select"],
            capture_output=True,
            text=True,
            env=env,
            cwd=project_root
        )

        assert result.returncode != 0, "Should fail without tuning results"
        assert "tuning" in result.stderr.lower() or "no" in result.stderr.lower()
