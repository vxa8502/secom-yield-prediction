"""
SECOM Defect Prediction Dashboard
Interactive dashboard for semiconductor manufacturing yield prediction
"""

from __future__ import annotations

# Bootstrap: Add project root to sys.path before any streamlit_app imports
import sys
from pathlib import Path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from streamlit_app.utils.styling import PAGE_CONFIG, apply_styling, show_figure, TITLE_STYLE
from streamlit_app.utils.artifact_loader import (
    get_model_display_info,
    load_lasso_features,
    load_latency_benchmark,
    load_production_model,
    load_threshold_config,
    load_test_data,
    get_system_health,
)
from src.config import COLORS, VIZ_CONFIG
from src.evaluation import unpack_confusion_matrix

# Page configuration
st.set_page_config(**PAGE_CONFIG)

# Custom CSS
apply_styling(st)


# =============================================================================
# SIDEBAR: System Health Indicator
# =============================================================================
def render_health_indicator() -> None:
    """Render system health status in sidebar."""
    health = get_system_health()

    with st.sidebar:
        st.subheader("System Status")

        # Overall status indicator
        status_colors = {
            'healthy': COLORS['pass'],
            'degraded': COLORS['warning'],
            'unhealthy': COLORS['fail'],
        }
        status_icons = {
            'healthy': 'OK',
            'degraded': '!',
            'unhealthy': 'X',
        }

        overall = health['overall']
        st.markdown(
            f"<div style='padding: 0.5rem; border-radius: 0.5rem; "
            f"background-color: {status_colors[overall]}20; "
            f"border-left: 4px solid {status_colors[overall]};'>"
            f"<strong>[{status_icons[overall]}] {overall.upper()}</strong>"
            f"</div>",
            unsafe_allow_html=True
        )

        st.caption("Component Status:")

        # Component status list
        components = [
            ('Model', health['model']),
            ('Features', health['features']),
            ('Test Data', health['test_data']),
            ('Threshold', health['threshold']),
        ]

        for name, status in components:
            icon = "[+]" if status['available'] else "[-]"
            color = COLORS['pass'] if status['available'] else COLORS['neutral']
            st.markdown(
                f"<span style='color: {color};'>{icon}</span> "
                f"**{name}:** {status['message']}",
                unsafe_allow_html=True
            )

        # Latency metric (if benchmark available)
        latency_data = load_latency_benchmark()
        if latency_data is not None:
            st.markdown("")
            st.metric(
                "Inference Latency",
                f"{latency_data['mean_per_sample_latency_ms']:.2f} ms/sample",
                help="Average prediction time per wafer"
            )

        st.markdown("---")


# Render health indicator in sidebar
render_health_indicator()


@st.cache_data
def compute_confusion_matrix():
    """Compute confusion matrix from test predictions."""
    model = load_production_model()
    threshold_config = load_threshold_config()

    if model is None:
        return None

    X_test, y_test = load_test_data()
    if X_test is None:
        return None

    threshold = threshold_config.get('optimal_threshold', 0.5)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = unpack_confusion_matrix(y_test, predictions)

    return {
        'matrix': np.array([[tn, fp], [fn, tp]]),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'total': int(tn + fp + fn + tp),
    }


@st.cache_data
def compute_prediction_distribution():
    """Compute prediction probability distribution for historical analysis."""
    model = load_production_model()
    threshold_config = load_threshold_config()

    if model is None:
        return None

    X_test, y_test = load_test_data()
    if X_test is None:
        return None

    threshold = threshold_config.get('optimal_threshold', 0.5)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        'probabilities': probabilities,
        'actuals': y_test,
        'threshold': threshold,
        'n_samples': len(y_test),
    }


# Load model info dynamically
model_info = get_model_display_info()
lasso_features = load_lasso_features()

# Main page content
st.title("Semiconductor Defect Prediction System")

# Hero section - 30-second story
if model_info['available']:
    st.markdown("Catch defects before they cost you.")
else:
    st.markdown("Train a model to enable predictions. Run `make pipeline` to get started.")

st.markdown("---")

# Check if model is available
if not model_info['available']:
    st.markdown(f"*{model_info['message']}*")

# Project overview
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Project Overview")

    if model_info['available']:
        cv_gmean_pct = model_info['cv_gmean'] * 100
        sensitivity_pct = model_info['sensitivity'] * 100 if model_info['sensitivity'] else 0
        specificity_pct = model_info['specificity'] * 100 if model_info['specificity'] else 0

        st.markdown(f"""
        This system predicts semiconductor wafer failures using sensor data from the
        manufacturing line. The production model achieves **{cv_gmean_pct:.1f}% G-Mean** on cross-validation
        with balanced sensitivity and specificity.

        **Key Features:**
        - Real-time defect prediction from 590 sensor measurements
        - SHAP-based explanations for process engineer decision support
        - Feature importance analysis identifying critical process parameters
        - Calibrated probabilities for threshold-based quality gates
        """)

        st.subheader("Model Architecture")
        st.markdown(f"""
        **Production Model:** {model_info['feature_set'].upper()} Feature Selection + {model_info['sampling_strategy'].upper()} + {model_info['model_name']}
        - **Feature Reduction:** 590 -> {model_info['n_features']} critical sensors via {model_info['feature_set'].upper()}
        - **Imbalance Handling:** {model_info['sampling_strategy'].upper()} (6.6% failure rate)
        - **Optimal Threshold:** {model_info['threshold']:.3f}
        """)
    else:
        st.markdown("""
        This system predicts semiconductor wafer failures using sensor data from the
        manufacturing line.

        **To get started:**
        1. Run the preprocessing pipeline: `make preprocess`
        2. Run hyperparameter tuning: `make tune`
        3. Select production model: `make select`
        4. Or run full pipeline: `make pipeline`
        """)

with col2:
    st.header("Key Metrics")

    if model_info['available']:
        # G-Mean (primary metric)
        st.metric(
            label="CV G-Mean",
            value=f"{model_info['cv_gmean']*100:.1f}%",
            help="Geometric mean of sensitivity and specificity (balanced metric for imbalanced data)"
        )

        # AUC-ROC
        if model_info.get('test_auc_roc'):
            st.metric(
                label="Test AUC-ROC",
                value=f"{model_info['test_auc_roc']*100:.1f}%",
                help="Area under ROC curve on held-out test set"
            )

        # Test G-Mean
        if model_info.get('test_gmean'):
            st.metric(
                label="Test G-Mean",
                value=f"{model_info['test_gmean']*100:.1f}%",
                help="G-Mean on held-out test set (n=313)"
            )

        st.metric(
            label="Optimal Threshold",
            value=f"{model_info['threshold']:.3f}",
            help="Decision boundary optimized via CV"
        )
    else:
        st.metric(label="Model Status", value="Not Trained")

st.markdown("---")

# Confusion Matrix Section (Sofia's requirement: front and center)
if model_info['available']:
    st.header("Model Performance: Confusion Matrix")

    st.markdown("""
    The confusion matrix shows prediction outcomes on the held-out test set.

    **False negatives (missed defects) are costly** in semiconductor manufacturing.
    """)

    cm_data = compute_confusion_matrix()

    if cm_data is not None:
        col1, col2 = st.columns([1.5, 1])

        with col1:
            # Confusion matrix with custom colors highlighting FN
            fig, ax = plt.subplots(figsize=VIZ_CONFIG['figsize_square'])

            cm = cm_data['matrix']

            # Create cell colors manually
            # TN, FP = light blue shades; FN = orange (critical); TP = teal
            cell_colors = np.array([
                [COLORS['primary'] + '30', COLORS['primary'] + '50'],  # TN, FP (blue shades)
                [COLORS['fail'], COLORS['pass']]  # FN (orange!), TP (teal)
            ])

            # Draw colored rectangles
            for i in range(2):
                for j in range(2):
                    color = cell_colors[i, j]
                    rect = plt.Rectangle((j, i), 1, 1, fill=True, color=color, alpha=0.7)
                    ax.add_patch(rect)

                    # Add text labels
                    label_map = [['TN', 'FP'], ['FN', 'TP']]
                    text_color = 'white' if (i == 1) else '#333'
                    ax.text(j + 0.5, i + 0.5, f"{label_map[i][j]}\n{cm[i,j]}",
                           ha='center', va='center', fontsize=16, fontweight='bold',
                           color=text_color)

            ax.set_xlim(0, 2)
            ax.set_ylim(0, 2)
            ax.set_xticks([0.5, 1.5])
            ax.set_yticks([0.5, 1.5])
            ax.set_xticklabels(['Predicted PASS', 'Predicted FAIL'], fontsize=11)
            ax.set_yticklabels(['Actual PASS', 'Actual FAIL'], fontsize=11)
            ax.invert_yaxis()
            ax.set_title('Test Set Confusion Matrix', **TITLE_STYLE)
            ax.set_xlabel('Predicted Label', fontsize=VIZ_CONFIG['label_fontsize'])
            ax.set_ylabel('Actual Label', fontsize=VIZ_CONFIG['label_fontsize'])

            # Remove spines
            for spine in ax.spines.values():
                spine.set_visible(False)

            show_figure(fig)

        with col2:
            st.subheader("Interpretation")

            total = cm_data['total']
            fn = cm_data['fn']
            fp = cm_data['fp']
            tp = cm_data['tp']
            tn = cm_data['tn']

            # Calculate rates
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0

            st.metric("Test Samples", total)
            st.metric("Sensitivity (Recall)", f"{sensitivity*100:.1f}%",
                     help="Of actual FAILs, how many did we catch?")
            st.metric("Specificity", f"{specificity*100:.1f}%",
                     help="Of actual PASSes, how many did we correctly identify?")

        # Error analysis row (below confusion matrix)
        fn = cm_data['fn']
        fp = cm_data['fp']

        err_col1, err_col2 = st.columns(2)

        with err_col1:
            if fn > 0:
                st.markdown(f"""
                **{fn} False Negatives (Missed Defects)**

                These wafers failed but were predicted to pass.
                In production, missed defects proceed downstream.
                """)
            else:
                st.markdown("No false negatives on test set.")

        with err_col2:
            if fp > 0:
                st.markdown(f"""
                **{fp} False Positives (False Alarms)**

                These wafers passed but were flagged for inspection.
                """)

    st.markdown("---")

# Historical Prediction Distribution (Marcus's requirement)
if model_info['available']:
    st.header("Prediction Distribution Analysis")

    st.markdown("""
    How would this model perform across the historical test set?
    This shows the distribution of predicted failure probabilities.
    """)

    dist_data = compute_prediction_distribution()

    if dist_data is not None:
        col1, col2 = st.columns([2, 1])

        with col1:
            fig, ax = plt.subplots(figsize=VIZ_CONFIG['figsize_wide'])

            # Separate by actual class
            pass_probs = dist_data['probabilities'][dist_data['actuals'] == 0]
            fail_probs = dist_data['probabilities'][dist_data['actuals'] == 1]

            # Histogram
            bins = np.linspace(0, 1, VIZ_CONFIG['prob_bins'])
            ax.hist(pass_probs, bins=bins, alpha=0.6, label=f'Actual PASS (n={len(pass_probs)})',
                   color=COLORS['pass'], edgecolor='black', linewidth=0.5)
            ax.hist(fail_probs, bins=bins, alpha=0.6, label=f'Actual FAIL (n={len(fail_probs)})',
                   color=COLORS['fail'], edgecolor='black', linewidth=0.5)

            # Threshold line
            ax.axvline(x=dist_data['threshold'], color=COLORS['primary'],
                      linestyle='--', linewidth=2, label=f"Threshold ({dist_data['threshold']:.2f})")

            ax.set_xlabel('Predicted Failure Probability', fontsize=VIZ_CONFIG['label_fontsize'])
            ax.set_ylabel('Count', fontsize=VIZ_CONFIG['label_fontsize'])
            ax.set_title('Prediction Distribution by Actual Outcome', **TITLE_STYLE)
            ax.legend(loc='upper right')
            ax.set_xlim(0, 1)

            show_figure(fig)

        with col2:
            st.subheader("What This Shows")

            st.markdown(f"""
            **Ideal separation:**
            - PASS wafers cluster near 0
            - FAIL wafers cluster near 1
            - Clear gap at threshold

            **Current threshold: {dist_data['threshold']:.2f}**

            Samples above threshold are
            flagged for inspection.
            """)

            # Calculate overlap metrics
            false_alarm_risk = (pass_probs >= dist_data['threshold']).mean() * 100
            miss_risk = (fail_probs < dist_data['threshold']).mean() * 100

            st.metric("False Alarm Rate", f"{false_alarm_risk:.1f}%",
                     help="% of PASS wafers incorrectly flagged")
            st.metric("Miss Rate", f"{miss_risk:.1f}%",
                     help="% of FAIL wafers incorrectly passed")

    st.markdown("---")

# Technical approach
st.header("Technical Approach")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Data Pipeline")
    st.markdown("""
    1. **Missing Data:** KNN imputation for 4.5% missing values
    2. **Feature Selection:** LASSO identifies critical sensors
    3. **Scaling:** StandardScaler for logistic regression
    4. **Train/Test Split:** Stratified 80/20 split
    5. **Cross-Validation:** 5-fold stratified CV
    """)

with col2:
    st.subheader("Model Selection")
    st.markdown("""
    **36 Baseline Experiments:**
    - 4 models x 3 feature sets x 3 sampling strategies

    **Hyperparameter Tuning:**
    - Optuna optimization (100 trials/experiment)

    **Selection Criteria:**
    - CV G-Mean @ optimal threshold
    """)

with col3:
    st.subheader("Production Features")
    st.markdown("""
    **Observability:**
    - Feature drift detection (KS test)
    - Probability calibration curves
    - Prediction latency benchmarks

    **Explainability:**
    - SHAP feature importance
    - Individual prediction waterfall plots
    """)

st.markdown("---")

# Known Limitations Section (Sofia's requirement)
st.header("Known Limitations")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Dataset Constraints")
    st.markdown("""
    **Small Sample Size:**
    - 1,567 total records (1,254 train / 313 test)
    - Only 104 failure cases (6.6% of data)
    - Limited statistical power for rare failure modes

    **Anonymized Features:**
    - 590 sensors labeled feature_0 through feature_589
    - No physical interpretation possible
    - Cannot map findings to specific equipment

    **No Temporal Information:**
    - Snapshot data without timestamps
    - Cannot detect time-based patterns
    - No lot-to-lot tracking capability
    """)

with col2:
    st.subheader("Model Coverage Gaps")
    st.markdown("""
    **What This Model Does NOT Capture:**

    - **Equipment Drift:** No modeling of sensor calibration changes over time
    - **Lot-to-Lot Variation:** Cannot account for batch effects
    - **Recipe Changes:** Model assumes static process parameters
    - **Environmental Factors:** Temperature, humidity not included
    - **Operator Effects:** No human factor variables

    **Recommendation:** Retrain model quarterly or when process changes occur.
    """)

st.markdown("---")

# LASSO selected features
if lasso_features:
    st.header(f"Critical Process Parameters ({len(lasso_features)} LASSO Features)")

    st.markdown(f"""
    The model identified **{len(lasso_features)} critical sensors** from 590 total features.
    These parameters provide the strongest signal for defect prediction:
    """)

    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("**Selected Features:**")
        for feature in lasso_features:
            st.markdown(f"- **{feature}**")

    with col2:
        st.markdown("""
        **Production Tip:**

        Monitor these features closely
        for distribution drift. Changes
        in these sensor readings may
        indicate process issues.
        """)

st.markdown("---")

# Navigation guide
st.header("Dashboard Navigation")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Available Pages")
    st.markdown("""
    1. **Overview** (this page) - Project summary and metrics
    2. **Prediction** - Interactive wafer quality prediction
    3. **Exploration** - Feature distributions and correlations
    4. **Explainability** - SHAP analysis and feature importance
    """)

with col2:
    st.subheader("Getting Started")
    st.markdown("""
    **For Process Engineers:**
    - Use the Prediction page to assess wafer quality
    - Check Explainability to understand why predictions were made

    **For Data Scientists:**
    - Review model architecture and metrics above
    - Explore feature importance in Explainability page
    """)

st.markdown("---")
st.caption("Built with Streamlit | SECOM UCI Dataset | Production-Ready ML Pipeline")
