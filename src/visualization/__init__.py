"""
Visualization utilities.
"""

from .plots import (
    finalize_figure,
    compare_feature_distributions,
    plot_calibration_curve,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_confusion_matrix,
    plot_threshold_curve,
)
from .shap_plots import (
    create_shap_explainer,
    get_cached_explainer,
    clear_explainer_cache,
    get_shap_base_value,
    get_sample_feature_contributions,
    compute_shap_values,
    plot_shap_summary,
    plot_shap_waterfall,
    plot_shap_dependence,
    get_top_shap_features,
    explain_prediction_shap,
)

__all__ = [
    'finalize_figure',
    'compare_feature_distributions',
    'plot_calibration_curve',
    'plot_roc_curve',
    'plot_precision_recall_curve',
    'plot_confusion_matrix',
    'plot_threshold_curve',
    'create_shap_explainer',
    'get_cached_explainer',
    'clear_explainer_cache',
    'get_shap_base_value',
    'get_sample_feature_contributions',
    'compute_shap_values',
    'plot_shap_summary',
    'plot_shap_waterfall',
    'plot_shap_dependence',
    'get_top_shap_features',
    'explain_prediction_shap',
]
