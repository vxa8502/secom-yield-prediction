"""
Configuration constants for SECOM yield prediction project
Author: Victoria A.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Generator, Literal


def load_json(path: Path | str) -> dict[str, Any]:
    """
    Load JSON file with consistent error handling.

    Args:
        path: Path to JSON file

    Returns:
        Parsed JSON as dict

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    path = Path(path)
    with open(path) as f:
        return json.load(f)


# Random seed for reproducibility
RANDOM_STATE = 42

# =============================================================================
# PREPROCESSING THRESHOLDS (centralized to avoid magic numbers)
# =============================================================================
# Rationale documented for each threshold

# Features with missing rate above this are removed
# 50% chosen to balance data retention vs. imputation reliability
MISSING_THRESHOLD_STRICT = 0.50

# Relaxed threshold for basic preprocessing (retains more features)
MISSING_THRESHOLD_RELAXED = 0.90

# Correlation threshold for removing redundant features
# 0.95 removes near-duplicates while preserving distinct signals
CORRELATION_THRESHOLD = 0.95

# Default classification threshold (should be optimized per model)
DEFAULT_CLASSIFICATION_THRESHOLD = 0.5

# Cross-validation folds (5-fold standard for small datasets)
CV_FOLDS = 5

# Sampling strategies available for tuning
SAMPLING_STRATEGIES = ('native', 'smote', 'adasyn')

# Type alias for sampling strategy (centralized for DRY type hints across modules)
SamplingStrategy = Literal['native', 'smote', 'adasyn']

# Threshold optimization parameters
THRESHOLD_RANGE = (0.01, 0.99)
THRESHOLD_STEP = 0.01

# Cost-sensitive classification
# In manufacturing, missed defects (FN) cost more than false alarms (FP)
DEFAULT_COST_RATIO = 1.0  # Standard G-Mean (equal weights)
COST_PROFILES = {
    'balanced': 1.0,              # Equal FN/FP cost
    'manufacturing_typical': 5.0, # FN costs 5x more than FP
    'manufacturing_critical': 10.0,  # FN costs 10x more (safety-critical)
}

# Resampling minimum sample requirements
# SMOTE needs at least k_neighbors + 1 minority samples (default k=5, so min=2 for k=1)
MIN_SAMPLES_SMOTE = 2
# ADASYN needs at least n_neighbors + 1 samples (default n_neighbors=5)
MIN_SAMPLES_ADASYN = 6

# Libraries to suppress verbose logging
_SUPPRESS_LIBRARIES = (
    ('mlflow', logging.WARNING),
    ('alembic', logging.WARNING),
    ('optuna', logging.WARNING),
)


def _apply_log_suppression() -> None:
    """Apply log level suppression to noisy libraries."""
    for lib_name, level in _SUPPRESS_LIBRARIES:
        logging.getLogger(lib_name).setLevel(level)


def setup_logging(level: int = logging.INFO, force: bool = False) -> logging.Logger:
    """
    Configure project logging with proper handler management.

    Unlike logging.basicConfig(), this function properly handles repeated calls
    by clearing existing handlers first when force=True.

    Args:
        level: Logging level (default: INFO)
        force: If True, remove existing handlers before adding new ones.
               Useful in tests or when reconfiguring logging.

    Returns:
        Configured 'secom' logger instance
    """
    logger = logging.getLogger('secom')

    # Remove existing handlers if force=True or if no handlers exist
    if force or not logger.handlers:
        # Clear existing handlers to prevent duplicates
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        logger.setLevel(level)

        # Create console handler with formatter
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
        logger.addHandler(console_handler)

        # Prevent propagation to root logger (avoids duplicate messages)
        logger.propagate = False

    _apply_log_suppression()
    return logger


# =============================================================================
# STRUCTURED LOGGING (JSON format for log aggregation)
# =============================================================================

@dataclass
class LogRecord:
    """Structured log record for JSON serialization."""
    timestamp: str
    level: str
    message: str
    logger: str = "secom"
    module: str | None = None
    function: str | None = None
    line: int | None = None
    duration_ms: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        data = {k: v for k, v in asdict(self).items() if v is not None}
        if not data.get("metrics"):
            data.pop("metrics", None)
        if not data.get("context"):
            data.pop("context", None)
        return json.dumps(data)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_record = LogRecord(
            timestamp=datetime.utcnow().isoformat() + "Z",
            level=record.levelname,
            message=record.getMessage(),
            logger=record.name,
            module=record.module,
            function=record.funcName,
            line=record.lineno,
        )

        # Extract custom fields from record
        if hasattr(record, "duration_ms"):
            log_record.duration_ms = record.duration_ms
        if hasattr(record, "metrics"):
            log_record.metrics = record.metrics
        if hasattr(record, "context"):
            log_record.context = record.context

        return log_record.to_json()


def setup_json_logging(
    level: int = logging.INFO,
    log_file: Path | None = None,
    console: bool = True
) -> logging.Logger:
    """
    Configure structured JSON logging.

    Args:
        level: Logging level (default: INFO)
        log_file: Optional file path for JSON logs
        console: Whether to also log to console in JSON format

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("secom")
    logger.setLevel(level)
    logger.handlers.clear()

    json_formatter = JSONFormatter()

    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(json_formatter)
        logger.addHandler(console_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(json_formatter)
        logger.addHandler(file_handler)

    _apply_log_suppression()
    return logger


@contextmanager
def log_execution_time(
    logger: logging.Logger,
    operation: str,
    level: int = logging.INFO,
    extra_context: dict[str, Any] | None = None
) -> Generator[dict[str, Any], None, None]:
    """
    Context manager for logging operation execution time.

    Args:
        logger: Logger instance
        operation: Name of the operation being timed
        level: Log level for the completion message
        extra_context: Additional context to include in log

    Yields:
        Dict for collecting metrics during execution

    Example:
        with log_execution_time(logger, "model_training") as metrics:
            model.fit(X, y)
            metrics["n_samples"] = len(X)
    """
    metrics: dict[str, Any] = {}
    context = extra_context or {}
    start_time = time.perf_counter()

    try:
        yield metrics
    finally:
        duration_ms = (time.perf_counter() - start_time) * 1000

        record = logging.LogRecord(
            name=logger.name,
            level=level,
            pathname="",
            lineno=0,
            msg=f"{operation} completed",
            args=(),
            exc_info=None,
        )
        record.duration_ms = duration_ms
        record.metrics = metrics
        record.context = {"operation": operation, **context}

        logger.handle(record)


def log_metric(
    logger: logging.Logger,
    metric_name: str,
    value: float,
    step: int | None = None,
    context: dict[str, Any] | None = None,
    level: int = logging.INFO
) -> None:
    """
    Log a single metric value in structured format.

    Args:
        logger: Logger instance
        metric_name: Name of the metric
        value: Metric value
        step: Optional step/epoch number
        context: Additional context
        level: Log level
    """
    metrics = {metric_name: value}
    if step is not None:
        metrics["step"] = step

    record = logging.LogRecord(
        name=logger.name,
        level=level,
        pathname="",
        lineno=0,
        msg=f"metric:{metric_name}={value}",
        args=(),
        exc_info=None,
    )
    record.metrics = metrics
    record.context = context or {}

    logger.handle(record)


def log_pipeline_step(
    logger: logging.Logger,
    step_name: str,
    status: str,
    duration_ms: float | None = None,
    metrics: dict[str, Any] | None = None,
    level: int = logging.INFO
) -> None:
    """
    Log a pipeline step execution.

    Args:
        logger: Logger instance
        step_name: Name of the pipeline step
        status: Status (started, completed, failed)
        duration_ms: Execution time in milliseconds
        metrics: Step-specific metrics
        level: Log level
    """
    record = logging.LogRecord(
        name=logger.name,
        level=level,
        pathname="",
        lineno=0,
        msg=f"pipeline:{step_name}:{status}",
        args=(),
        exc_info=None,
    )
    record.duration_ms = duration_ms
    record.metrics = metrics or {}
    record.context = {"step": step_name, "status": status}

    logger.handle(record)


# =============================================================================
# PROJECT PATHS
# =============================================================================
# Support SECOM_PROJECT_ROOT env var for container/CI deployments
PROJECT_ROOT = Path(os.environ.get('SECOM_PROJECT_ROOT', Path(__file__).parent.parent))
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
SECOM_RAW_DIR = RAW_DATA_DIR / "secom"  # UCI SECOM dataset location
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# NOTE: Directories are NOT created at import time to avoid side effects.
# Call ensure_directories() explicitly in main entry points.


def ensure_directories(create: bool = True) -> dict[str, Path]:
    """
    Ensure required project directories exist and are accessible.

    Call this function explicitly in main entry points (pipelines, CLI)
    rather than at module import time to avoid side effects during testing.

    Args:
        create: If True, create missing directories. If False, only validate.

    Returns:
        Dict mapping directory names to Path objects.

    Raises:
        RuntimeError: If directory creation fails (permissions, disk full, etc.)
    """
    directories = {
        'data': DATA_DIR,
        'raw_data': RAW_DATA_DIR,
        'models': MODELS_DIR,
        'reports': REPORTS_DIR,
        'figures': FIGURES_DIR,
    }

    if not create:
        return directories

    for name, path in directories.items():
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise RuntimeError(
                f"Cannot create {name} directory at {path}: {e}"
            ) from e

    return directories


# Dataset constants
# Default class imbalance ratio (pass:fail) - used as fallback
# For accurate results, compute dynamically using compute_class_ratio()
DEFAULT_CLASS_IMBALANCE_RATIO = 14


def compute_class_ratio(y: Any) -> float:
    """
    Compute class imbalance ratio from labels.

    Args:
        y: Binary labels array (0=pass, 1=fail)

    Returns:
        Ratio of majority to minority class (e.g., 14.0 means 14:1)
    """
    import numpy as np
    y_arr = np.asarray(y)
    n_positive = (y_arr == 1).sum()
    n_negative = (y_arr == 0).sum()

    if n_positive == 0:
        return float(DEFAULT_CLASS_IMBALANCE_RATIO)

    return n_negative / n_positive


# Backward compatibility alias
CLASS_IMBALANCE_RATIO = DEFAULT_CLASS_IMBALANCE_RATIO

# =============================================================================
# BRAND COLOR PALETTE (immutable to prevent runtime modification)
# =============================================================================
COLORS = MappingProxyType({
    'primary': '#1428A0',      # Deep blue (brand)
    'secondary': '#2596be',    # Light blue (accents)
    'pass': '#00C1B0',         # Teal (success/pass)
    'fail': '#EA580C',         # Deep orange (fail - high visibility)
    'warning': '#F59E0B',      # Amber (warning - softer)
    'neutral': '#757575',      # Gray
})

# Visualization configuration (immutable)
VIZ_CONFIG = MappingProxyType({
    'dpi': 400,
    'style': 'seaborn-v0_8-darkgrid',
    'palette': 'colorblind',
    'context': 'notebook',

    # Font sizes
    'title_fontsize': 14,
    'label_fontsize': 12,
    'tick_fontsize': 10,

    # Brand color palette (references COLORS)
    'primary': COLORS['primary'],
    'secondary': COLORS['secondary'],
    'pass_color': COLORS['pass'],
    'fail_color': COLORS['fail'],
    'neutral': COLORS['neutral'],

    # Curve colors (primary for main, secondary for comparison)
    'roc_color': COLORS['primary'],
    'pr_color': COLORS['primary'],
    'threshold_color': COLORS['primary'],
    'optimal_marker': COLORS['fail'],

    # Heatmap colormap (custom blue gradient)
    'heatmap_cmap': 'Blues',

    # Figure sizes (width, height) for Streamlit dashboard
    'figsize_wide': (10, 6),
    'figsize_medium': (8, 5),
    'figsize_square': (6, 5),
    'figsize_small': (6, 4),

    # Histogram bins
    'hist_bins': 30,
    'prob_bins': 25,
})

# Production artifact paths
PRODUCTION_ARTIFACTS = {
    'model': MODELS_DIR / 'production_model.pkl',
    'threshold': MODELS_DIR / 'production_threshold.json',
    'metadata': MODELS_DIR / 'production_model_metadata.json',
    'lasso_features': DATA_DIR / 'lasso_selected_features.txt',
    'preprocessing_pipeline': MODELS_DIR / 'preprocessing_pipeline_lasso.pkl',
}

# MLflow configuration
MLFLOW_EXPERIMENT_NAME = "secom-yield-prediction"

# Optuna configuration
# SQLite storage for persistence and crash recovery
# Studies can be resumed if tuning is interrupted
OPTUNA_STORAGE_PATH = PROJECT_ROOT / "optuna_studies.db"
OPTUNA_N_TRIALS_DEFAULT = 100

# SHAP configuration
SHAP_MAX_DISPLAY = 15  # Max features to display in SHAP plots

# Model validation thresholds (for CI gate)
MODEL_VALIDATION = {
    'min_cv_gmean': 0.65,   # Fail CI if below this
    'warn_cv_gmean': 0.70,  # Warn if below this
    'max_regression': 0.02, # Max allowed drop from previous model
}

# Feature set configurations
FEATURE_SETS = {
    'lasso': {
        'train_file': 'X_train_lasso.npy',
        'test_file': 'X_test_lasso.npy',
        'description': 'LASSO feature selection'
    },
    'pca': {
        'train_file': 'X_train_pca_raw.npy',
        'test_file': 'X_test_pca_raw.npy',
        'description': 'PCA dimensionality reduction'
    },
    'all': {
        'train_file': 'X_train_all_features.npy',
        'test_file': 'X_test_all_features.npy',
        'description': 'All features'
    }
}
