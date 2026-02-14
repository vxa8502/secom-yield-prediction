"""
SECOM Defect Prediction Dashboard Package

Provides centralized path setup and shared utilities for Streamlit pages.
"""

import sys
from pathlib import Path

__version__ = "1.0.0"

# Centralized project root - computed once, reused everywhere
PROJECT_ROOT = Path(__file__).parent.parent


def setup_project_path() -> Path:
    """
    Add project root to sys.path for module imports.

    Call this once at the top of each Streamlit page to enable
    imports from src/ without repeating path manipulation.

    Returns:
        Project root path
    """
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    return PROJECT_ROOT
