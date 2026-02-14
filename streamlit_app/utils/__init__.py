"""
Utility functions for SECOM dashboard.

- styling: PAGE_CONFIG, STYLE_CSS, apply_styling, show_figure, TITLE_STYLE
- config: COLORS, FEATURE_DESCRIPTIONS, get_threshold_options
- artifact_loader: Dynamic loading of production model and artifacts
"""

from .styling import PAGE_CONFIG, STYLE_CSS, apply_styling, show_figure, TITLE_STYLE
from .config import COLORS, FEATURE_DESCRIPTIONS, get_threshold_options
from .artifact_loader import (
    load_production_model,
    load_production_metadata,
    load_threshold_config,
    load_lasso_features,
    load_test_data,
    get_model_display_info,
    predict_single,
    predict_batch,
    load_test_sample,
    get_feature_importance,
    extract_shap_values,
    get_system_health,
    log_prediction_request,
)
