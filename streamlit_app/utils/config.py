"""
Dashboard Configuration
Centralized configuration for Streamlit dashboard constants.

NOTE: Model-specific values (threshold, metrics, features) are loaded dynamically
from production artifacts via artifact_loader.py - NOT hardcoded here.
CSS styling is in styling.py.
"""

# Centralized path setup
from streamlit_app import setup_project_path
setup_project_path()

# Import colors from single source of truth
from src.config import COLORS

# Feature descriptions for UI display
# Human-readable labels for anonymized SECOM features
FEATURE_DESCRIPTIONS = {
    "feature_22": "Process temperature measurement",
    "feature_60": "Pressure sensor reading",
    "feature_65": "Chemical concentration monitor",
    "feature_104": "Plasma density measurement",
    "feature_130": "Deposition rate sensor",
    "feature_349": "Vacuum level indicator",
    "feature_511": "RF power monitor"
}


def get_threshold_options(production_threshold: float) -> dict[str, float]:
    """
    Generate threshold options for different operational priorities.

    Classification threshold controls the tradeoff between false alarms (FP)
    and missed defects (FN). Lower thresholds catch more failures but increase
    false alarms; higher thresholds reduce false alarms but risk missing defects.

    Args:
        production_threshold: Optimal threshold from CV-based tuning (G-Mean maximized)

    Returns:
        dict with three threshold options:
            - conservative: 70% of optimal, prioritizes catching failures
            - default: Production optimal threshold (balanced G-Mean)
            - aggressive: 150% of optimal, prioritizes reducing false alarms
    """
    return {
        "conservative": max(0.01, production_threshold * 0.7),
        "default": production_threshold,
        "aggressive": min(0.99, production_threshold * 1.5),
    }
