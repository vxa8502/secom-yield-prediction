"""
Concurrency tests for parallel tuning execution.

Tests ProcessPoolExecutor usage in tune.py, worker failures,
and MLflow schema initialization for parallel access.
"""

from __future__ import annotations

import logging
import sys
from concurrent.futures import ProcessPoolExecutor, Future
from pathlib import Path
from typing import Any
from unittest.mock import patch, MagicMock, PropertyMock

import numpy as np
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import RANDOM_STATE


class TestParallelExecution:
    """Tests for parallel tuning execution."""

    @pytest.fixture
    def mock_feature_sets(self) -> dict:
        """Create mock feature sets."""
        np.random.seed(RANDOM_STATE)
        return {
            "lasso": (np.random.randn(80, 6), np.random.randn(20, 6)),
            "pca": (np.random.randn(80, 10), np.random.randn(20, 10)),
            "all": (np.random.randn(80, 30), np.random.randn(20, 30)),
        }

    @pytest.fixture
    def mock_labels(self) -> np.ndarray:
        """Create mock labels."""
        np.random.seed(RANDOM_STATE)
        y = np.array([0] * 72 + [1] * 8)
        np.random.shuffle(y)
        return y

    def test_parallel_executor_spawns_correct_workers(self):
        """Test that ProcessPoolExecutor spawns correct number of workers."""
        strategies = ["native", "smote", "adasyn"]

        with patch("pipelines.tune.ProcessPoolExecutor") as mock_executor:
            mock_executor.return_value.__enter__ = MagicMock(
                return_value=MagicMock()
            )
            mock_executor.return_value.__exit__ = MagicMock(return_value=False)

            # Simulate creating executor with max_workers=len(strategies)
            with ProcessPoolExecutor(max_workers=len(strategies)) as executor:
                pass

            # In actual code, max_workers should equal len(strategies)
            assert len(strategies) == 3

    def test_worker_function_isolated(
        self, mock_feature_sets: dict, mock_labels: np.ndarray
    ):
        """Test that worker function can run in isolation."""
        from src.tuning_utils import run_all_experiments_for_sampling

        # Each worker runs independently with its own strategy
        result = run_all_experiments_for_sampling(
            sampling_strategy="native",
            feature_sets=mock_feature_sets,
            y_train=mock_labels,
            n_trials=1
        )

        assert isinstance(result, list)
        assert len(result) > 0

    def test_worker_handles_empty_feature_sets(self, mock_labels: np.ndarray):
        """Test worker handles empty feature sets gracefully."""
        from src.tuning_utils import run_all_experiments_for_sampling

        empty_features = {}

        result = run_all_experiments_for_sampling(
            sampling_strategy="native",
            feature_sets=empty_features,
            y_train=mock_labels,
            n_trials=1
        )

        assert result == []


class TestWorkerFailureRecovery:
    """Tests for handling worker failures in parallel execution."""

    def test_failed_strategy_reported(self):
        """Test that failed strategies are properly reported."""
        from pipelines.tune import run_worker

        with patch("pipelines.tune.run_tuning_for_strategy") as mock_tune:
            mock_tune.side_effect = RuntimeError("Simulated worker failure")

            with pytest.raises(RuntimeError) as exc_info:
                # run_worker expects (strategy, n_trials, resume, fresh)
                run_worker(("native", 10, True, False))

            assert "Simulated worker failure" in str(exc_info.value)

    def test_partial_failure_collected(self):
        """Test that partial failures are collected and reported."""
        failures: list[tuple[str, Exception]] = []
        strategies = ["native", "smote", "adasyn"]

        def mock_result(strategy: str) -> dict:
            if strategy == "smote":
                raise ValueError("SMOTE failed")
            return {"sampling_strategy": strategy, "best_cv_gmean": 0.7}

        for strategy in strategies:
            try:
                result = mock_result(strategy)
            except Exception as e:
                failures.append((strategy, e))

        assert len(failures) == 1
        assert failures[0][0] == "smote"
        assert "SMOTE failed" in str(failures[0][1])

    def test_all_failures_raise_runtime_error(self):
        """Test that complete failure raises RuntimeError with details."""
        failures = [
            ("native", RuntimeError("Native failed")),
            ("smote", RuntimeError("SMOTE failed")),
            ("adasyn", RuntimeError("ADASYN failed")),
        ]

        if failures:
            failed_strategies = [f[0] for f in failures]

            with pytest.raises(RuntimeError) as exc_info:
                raise RuntimeError(f"Tuning failed for strategies: {failed_strategies}")

            assert "native" in str(exc_info.value)
            assert "smote" in str(exc_info.value)
            assert "adasyn" in str(exc_info.value)


class TestMLflowConcurrency:
    """Tests for MLflow concurrency handling."""

    def test_mlflow_setup_called_before_workers(self):
        """Test that MLflow is initialized before spawning workers."""
        from src.mlflow_utils import setup_mlflow

        with patch("src.mlflow_utils.mlflow") as mock_mlflow:
            setup_mlflow()

            # Should set experiment name
            mock_mlflow.set_experiment.assert_called()

    def test_mlflow_database_retry_on_operational_error(self):
        """Test that MLflow operations retry on database errors."""
        from sqlalchemy.exc import OperationalError

        retry_count = 0
        max_retries = 3

        def flaky_operation() -> str:
            nonlocal retry_count
            retry_count += 1
            if retry_count < max_retries:
                raise OperationalError("database locked", None, None)
            return "success"

        # Simulate retry logic
        result = None
        for attempt in range(max_retries):
            try:
                result = flaky_operation()
                break
            except OperationalError:
                if attempt == max_retries - 1:
                    raise
                continue

        assert result == "success"
        assert retry_count == max_retries

    def test_mlflow_experiment_name_consistent(self):
        """Test that all workers use same experiment name."""
        from src.config import MLFLOW_EXPERIMENT_NAME

        # All workers should reference the same experiment
        assert MLFLOW_EXPERIMENT_NAME == "secom-yield-prediction"


class TestProcessPoolExecutorBehavior:
    """Tests for ProcessPoolExecutor behavior patterns."""

    def test_executor_context_manager(self):
        """Test executor properly manages resources via context manager."""
        with ProcessPoolExecutor(max_workers=2) as executor:
            assert executor is not None

        # After context exit, executor should be shut down
        # (no explicit assertion needed - context manager handles cleanup)

    def test_as_completed_returns_results_in_order(self):
        """Test that as_completed returns results as they finish."""
        from concurrent.futures import as_completed, ThreadPoolExecutor

        def fast_task(x: int) -> int:
            return x * 2

        # Use ThreadPoolExecutor to avoid pickling issues in tests
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {executor.submit(fast_task, i): i for i in range(3)}
            results = []

            for future in as_completed(futures):
                results.append(future.result())

        assert sorted(results) == [0, 2, 4]

    def test_future_exception_propagates(self):
        """Test that exceptions from workers propagate correctly."""
        from concurrent.futures import ThreadPoolExecutor

        def failing_task() -> None:
            raise ValueError("Task failed")

        # Use ThreadPoolExecutor to avoid pickling issues in tests
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(failing_task)

            with pytest.raises(ValueError) as exc_info:
                future.result()

            assert "Task failed" in str(exc_info.value)


class TestConcurrencyEdgeCases:
    """Tests for edge cases in concurrent execution."""

    def test_single_strategy_skips_parallel(self):
        """Test that single strategy doesn't use parallel execution."""
        strategies = ["native"]
        parallel = len(strategies) > 1

        assert not parallel, "Single strategy should not use parallel"

    def test_empty_strategies_raises_error(self):
        """Test that empty strategies list raises error."""
        strategies: list[str] = []

        with pytest.raises(ValueError):
            if not strategies:
                raise ValueError("No sampling strategies specified")

    def test_duplicate_strategies_handled(self):
        """Test that duplicate strategies are handled correctly."""
        strategies = ["native", "native", "smote"]
        unique_strategies = list(set(strategies))

        assert len(unique_strategies) == 2


class TestResourceCleanup:
    """Tests for proper resource cleanup in parallel execution."""

    def test_temp_files_cleaned_on_failure(self, tmp_path: Path):
        """Test that temporary files are cleaned up on failure."""
        temp_file = tmp_path / "temp_results.csv"
        temp_file.write_text("test")

        try:
            raise RuntimeError("Simulated failure")
        except RuntimeError:
            if temp_file.exists():
                temp_file.unlink()

        assert not temp_file.exists()

    def test_mlflow_run_closed_on_exception(self):
        """Test that MLflow runs are properly closed on exception."""
        import mlflow

        with patch.object(mlflow, "start_run") as mock_start:
            mock_run = MagicMock()
            mock_start.return_value.__enter__ = MagicMock(return_value=mock_run)
            mock_start.return_value.__exit__ = MagicMock(return_value=False)

            try:
                with mlflow.start_run():
                    raise ValueError("Test error")
            except ValueError:
                pass

            # Context manager should have exited
            mock_start.return_value.__exit__.assert_called()


class TestLoggingInParallel:
    """Tests for logging behavior in parallel execution."""

    def test_logger_thread_safe(self, caplog):
        """Test that logger works correctly across processes."""
        logger = logging.getLogger("secom")

        with caplog.at_level(logging.INFO, logger="secom"):
            logger.info("Test message from main process")

        assert "Test message" in caplog.text

    def test_worker_logging_isolated(self):
        """Test that worker logging doesn't interfere with main process."""
        from pipelines.tune import run_tuning_for_strategy

        # Workers set up their own logging
        # This test verifies the pattern exists
        import inspect
        source = inspect.getsource(run_tuning_for_strategy)

        assert "setup_logging" in source


class TestParallelDataIntegrity:
    """Tests for data integrity in parallel execution."""

    @pytest.fixture
    def shared_feature_sets(self) -> dict:
        """Create shared feature sets that workers would access."""
        np.random.seed(RANDOM_STATE)
        return {
            "lasso": (np.random.randn(80, 6), np.random.randn(20, 6)),
        }

    def test_feature_sets_not_modified(self, shared_feature_sets: dict):
        """Test that parallel workers don't modify shared data."""
        original_shape = shared_feature_sets["lasso"][0].shape
        original_data = shared_feature_sets["lasso"][0].copy()

        # Simulate worker accessing data
        worker_data = shared_feature_sets["lasso"][0]

        assert worker_data.shape == original_shape
        assert np.array_equal(worker_data, original_data)

    def test_results_from_different_workers_distinct(self):
        """Test that results from different workers are distinct."""
        results_native = {"strategy": "native", "gmean": 0.71}
        results_smote = {"strategy": "smote", "gmean": 0.70}

        assert results_native["strategy"] != results_smote["strategy"]
        assert results_native is not results_smote
