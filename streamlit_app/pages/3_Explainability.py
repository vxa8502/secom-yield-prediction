"""
Explainability Page
SHAP analysis and feature importance visualization
"""

import hashlib
import time

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from streamlit_app.utils.artifact_loader import (
    get_model_display_info,
    load_lasso_features,
    get_feature_importance,
    load_production_model,
    load_production_metadata,
    extract_shap_values,
    get_classifier,
)
from streamlit_app.utils.styling import PAGE_CONFIG, apply_styling, show_figure, TITLE_STYLE
from src.config import COLORS, DATA_DIR, SHAP_MAX_DISPLAY, VIZ_CONFIG

# SHAP computation settings
SHAP_COOLDOWN_SECONDS = 5  # Minimum time between SHAP computations

st.set_page_config(**PAGE_CONFIG)
apply_styling(st)

st.title("Model Explainability & Feature Importance")
st.markdown("---")

# Check model availability
model_info = get_model_display_info()
lasso_features = load_lasso_features()

if not model_info['available']:
    st.markdown(f"*{model_info['message']}*")


@st.cache_data
def load_background_data(n_samples=50):
    """Load background data for SHAP computation."""
    X_train_path = DATA_DIR / "X_train_lasso.npy"
    if not X_train_path.exists():
        return None

    X_train = np.load(X_train_path)
    if len(X_train) > n_samples:
        indices = np.random.choice(len(X_train), n_samples, replace=False)
        return X_train[indices]
    return X_train


@st.cache_data
def load_test_samples(n_samples=20):
    """Load test samples for SHAP explanation."""
    X_test_path = DATA_DIR / "X_test_lasso.npy"
    y_test_path = DATA_DIR / "y_test.csv"

    if not X_test_path.exists() or not y_test_path.exists():
        return None, None

    X_test = np.load(X_test_path)
    y_test = pd.read_csv(y_test_path).values.ravel()

    if len(X_test) > n_samples:
        indices = np.random.choice(len(X_test), n_samples, replace=False)
        return X_test[indices], y_test[indices]
    return X_test, y_test


def get_model_type(model):
    """Determine model type for SHAP explainer selection."""
    if model is None:
        return None

    classifier = get_classifier(model)
    class_name = classifier.__class__.__name__.lower()

    if 'xgb' in class_name or 'randomforest' in class_name or 'tree' in class_name:
        return 'tree'
    elif 'logistic' in class_name or 'linear' in class_name:
        return 'linear'
    else:
        return 'kernel'  # Fallback for SVM and others


def compute_shap_values(model, X_background, X_explain, model_type):
    """Compute SHAP values with appropriate explainer."""
    import shap

    classifier = get_classifier(model)

    if model_type == 'tree':
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer(X_explain)
    elif model_type == 'linear':
        explainer = shap.LinearExplainer(classifier, X_background)
        shap_values = explainer(X_explain)
    else:
        # KernelExplainer for SVM and other models
        def model_predict(X):
            return classifier.predict_proba(X)[:, 1]

        explainer = shap.KernelExplainer(model_predict, X_background)
        shap_values = explainer(X_explain)

    return shap_values, explainer


def get_model_fingerprint() -> str:
    """
    Generate a fingerprint for the current production model.

    Used for cache invalidation - if model changes, SHAP cache is cleared.

    Returns:
        Hash string based on model metadata (timestamp + model name)
    """
    metadata = load_production_metadata()
    if not metadata:
        return "no_model"

    # Create fingerprint from timestamp and model name
    fingerprint_data = f"{metadata.get('timestamp', '')}-{metadata.get('model_name', '')}"
    return hashlib.md5(fingerprint_data.encode()).hexdigest()[:12]


def init_shap_session_state() -> None:
    """
    Initialize SHAP-related session state with cache invalidation tracking.

    Clears cached SHAP values if:
    - Model fingerprint has changed (model was retrained)
    - Session state was never initialized
    """
    current_fingerprint = get_model_fingerprint()

    # Check if we need to invalidate cache
    if 'shap_model_fingerprint' not in st.session_state:
        # First initialization
        st.session_state.shap_model_fingerprint = current_fingerprint
        st.session_state.shap_values = None
        st.session_state.shap_X_explain = None
        st.session_state.shap_y_explain = None
        st.session_state.shap_computed = False
        st.session_state.shap_n_samples = None
        st.session_state.shap_last_compute_time = 0

    elif st.session_state.shap_model_fingerprint != current_fingerprint:
        # Model changed - invalidate cache
        st.session_state.shap_model_fingerprint = current_fingerprint
        st.session_state.shap_values = None
        st.session_state.shap_X_explain = None
        st.session_state.shap_y_explain = None
        st.session_state.shap_computed = False
        st.session_state.shap_n_samples = None
        st.session_state.shap_last_compute_time = 0


def clear_shap_cache() -> None:
    """Clear SHAP cache manually (user-triggered)."""
    st.session_state.shap_values = None
    st.session_state.shap_X_explain = None
    st.session_state.shap_y_explain = None
    st.session_state.shap_computed = False
    st.session_state.shap_n_samples = None


def can_compute_shap() -> tuple[bool, float]:
    """
    Check if SHAP computation is allowed (rate limiting).

    Returns:
        Tuple of (allowed: bool, wait_seconds: float)
    """
    last_compute = st.session_state.get('shap_last_compute_time', 0)
    elapsed = time.time() - last_compute
    if elapsed < SHAP_COOLDOWN_SECONDS:
        return False, SHAP_COOLDOWN_SECONDS - elapsed
    return True, 0


# Overview
st.markdown("""
This page provides insights into how the model makes predictions and which features
are most important for defect detection. Understanding model behavior is critical for
process engineers to trust and act on predictions.
""")

# Feature Importance Section
st.header("Feature Importance Analysis")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Model Coefficients / Feature Weights")

    importance_df = get_feature_importance()

    if importance_df is not None and len(importance_df) > 0:
        # Create bar chart
        fig, ax = plt.subplots(figsize=VIZ_CONFIG['figsize_wide'])
        bars = ax.barh(importance_df["feature"], importance_df["importance"])

        # Color bars
        for bar in bars:
            bar.set_color(COLORS["primary"])

        ax.set_xlabel("Absolute Coefficient Value / Importance", fontsize=VIZ_CONFIG['label_fontsize'])
        ax.set_ylabel("Feature", fontsize=VIZ_CONFIG['label_fontsize'])
        ax.set_title(f"Feature Importance ({model_info['model_name']})", **TITLE_STYLE)
        ax.invert_yaxis()

        show_figure(fig)

        # Show importance table
        st.subheader("Importance Values")
        st.dataframe(importance_df, use_container_width=True)
    else:
        st.markdown("*Feature importance not available. Model may not be trained yet.*")

with col2:
    st.subheader("Key Insights")

    if importance_df is not None and len(importance_df) > 0:
        top_features = importance_df.head(3)['feature'].tolist()

        st.markdown(f"""
        **Top {min(3, len(importance_df))} Most Important Features:**
        """)
        for i, feat in enumerate(top_features, 1):
            st.markdown(f"{i}. {feat}")

        st.markdown("""
        These features show the strongest
        predictive signal and should be
        monitored closely in production.
        """)
    else:
        st.markdown("Train a model to see feature importance.")

    st.markdown("""
    **For Process Engineers:**

    If predictions seem incorrect,
    first check if these top features
    have unusual values or
    measurement errors.
    """)

st.markdown("---")

# SHAP Explanation Section
st.header("SHAP (SHapley Additive exPlanations)")

st.markdown("""
SHAP values explain individual predictions by quantifying each feature's contribution.

**How to Read SHAP Values:**
- **Positive SHAP value:** Feature pushes prediction toward FAIL
- **Negative SHAP value:** Feature pushes prediction toward PASS
- **Magnitude:** Shows how strongly the feature influenced the prediction
""")

if model_info['available']:
    # Initialize session state with cache invalidation
    init_shap_session_state()

    # SHAP computation controls
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        n_samples = st.slider(
            "Number of samples to explain",
            min_value=5,
            max_value=50,
            value=20,
            step=5,
            help="More samples = better global view but slower computation"
        )

    with col2:
        # Check rate limiting
        can_compute, wait_time = can_compute_shap()

        # Check if we can reuse cached values
        cached_n_samples = st.session_state.get('shap_n_samples')
        cache_valid = (
            st.session_state.shap_computed and
            cached_n_samples == n_samples
        )

        if cache_valid:
            button_label = "Recompute SHAP"
            button_help = f"SHAP values cached for {n_samples} samples. Click to recompute."
        else:
            button_label = "Compute SHAP Values"
            button_help = None

        compute_button = st.button(
            button_label,
            use_container_width=True,
            type="primary",
            disabled=not can_compute,
            help=button_help if can_compute else f"Please wait {wait_time:.0f}s"
        )

    with col3:
        # Clear cache button
        if st.session_state.shap_computed:
            if st.button("Clear Cache", use_container_width=True):
                clear_shap_cache()
                st.rerun()

    # Show cache status
    if st.session_state.shap_computed and cached_n_samples:
        st.caption(f"Cached: {cached_n_samples} samples | Model: {st.session_state.shap_model_fingerprint}")

    # Compute SHAP values
    if compute_button and can_compute:
        with st.spinner("Computing SHAP values (this may take a moment)..."):
            try:
                import shap

                model = load_production_model()
                model_type = get_model_type(model)

                X_background = load_background_data(n_samples=50)
                X_explain, y_explain = load_test_samples(n_samples=n_samples)

                if X_background is None or X_explain is None:
                    st.error("Training/test data not available. Run preprocessing first.")
                else:
                    shap_values, explainer = compute_shap_values(
                        model, X_background, X_explain, model_type
                    )

                    # Store in session state with metadata
                    st.session_state.shap_values = shap_values
                    st.session_state.shap_X_explain = X_explain
                    st.session_state.shap_y_explain = y_explain
                    st.session_state.shap_computed = True
                    st.session_state.shap_model_type = model_type
                    st.session_state.shap_n_samples = n_samples
                    st.session_state.shap_last_compute_time = time.time()

                    st.success(f"SHAP values computed for {len(X_explain)} samples using {model_type.upper()} explainer.")

            except Exception as e:
                st.error(f"SHAP computation failed: {str(e)}")
                st.exception(e)

    # Display SHAP visualizations if computed
    if st.session_state.shap_computed and st.session_state.shap_values is not None:
        import shap

        shap_values = st.session_state.shap_values
        X_explain = st.session_state.shap_X_explain

        st.markdown("---")

        # SHAP visualization tabs
        shap_tab1, shap_tab2, shap_tab3 = st.tabs([
            "Summary Plot",
            "Feature Impact",
            "Individual Explanation"
        ])

        with shap_tab1:
            st.subheader("SHAP Summary Plot")
            st.markdown("""
            Each point represents one sample. Position on x-axis shows impact on prediction.
            Color shows feature value (red=high, blue=low).
            """)

            # Create summary plot
            fig, ax = plt.subplots(figsize=VIZ_CONFIG['figsize_wide'])

            # Extract raw values from SHAP result
            values = extract_shap_values(shap_values)

            # Beeswarm-style summary
            shap.summary_plot(
                values,
                X_explain,
                feature_names=lasso_features,
                max_display=SHAP_MAX_DISPLAY,
                show=False
            )

            show_figure(plt.gcf())

        with shap_tab2:
            st.subheader("Mean Absolute SHAP Values")
            st.markdown("Average impact of each feature across all explained samples.")

            values = extract_shap_values(shap_values)
            mean_abs_shap = np.abs(values).mean(axis=0)

            shap_importance = pd.DataFrame({
                'feature': lasso_features,
                'mean_shap': mean_abs_shap
            }).sort_values('mean_shap', ascending=True)

            fig, ax = plt.subplots(figsize=VIZ_CONFIG['figsize_wide'])
            bars = ax.barh(shap_importance['feature'], shap_importance['mean_shap'])

            for bar in bars:
                bar.set_color(COLORS["secondary"])

            ax.set_xlabel("Mean |SHAP Value|", fontsize=VIZ_CONFIG['label_fontsize'])
            ax.set_ylabel("Feature", fontsize=VIZ_CONFIG['label_fontsize'])
            ax.set_title("Feature Impact (SHAP)", **TITLE_STYLE)

            show_figure(fig)

            # Display table
            st.dataframe(
                shap_importance.sort_values('mean_shap', ascending=False).reset_index(drop=True),
                use_container_width=True
            )

        with shap_tab3:
            st.subheader("Individual Prediction Explanation")

            y_explain = st.session_state.get('shap_y_explain', None)

            # Sample selector
            sample_idx = st.selectbox(
                "Select sample to explain",
                range(len(X_explain)),
                format_func=lambda i: f"Sample {i+1} (Actual: {'FAIL' if y_explain is not None and y_explain[i] == 1 else 'PASS'})"
            )

            st.markdown("**Waterfall Plot:** Shows how each feature contributes to this prediction")

            # Waterfall plot for selected sample
            fig, ax = plt.subplots(figsize=VIZ_CONFIG['figsize_wide'])

            shap.plots.waterfall(shap_values[sample_idx], max_display=SHAP_MAX_DISPLAY, show=False)

            show_figure(plt.gcf())

            # Feature values for this sample
            st.markdown("**Feature Values for This Sample:**")
            sample_df = pd.DataFrame({
                'Feature': lasso_features,
                'Value': X_explain[sample_idx],
                'SHAP': extract_shap_values(shap_values)[sample_idx]
            })
            sample_df['Impact'] = sample_df['SHAP'].apply(
                lambda x: 'Toward FAIL' if x > 0 else 'Toward PASS'
            )
            st.dataframe(sample_df, use_container_width=True)

    else:
        st.info("Click 'Compute SHAP Values' to generate explanations. Results are cached per model version.")

else:
    st.markdown("*Train a model first to enable SHAP analysis.*")

st.markdown("---")

# LASSO Feature Selection
st.header("LASSO Feature Selection")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Dimensionality Reduction")

    n_features = len(lasso_features) if lasso_features else 0

    st.markdown(f"""
    The model uses LASSO (L1 regularization) to select the most predictive features
    from the original 590 sensor measurements.

    **Selection Process:**
    1. Train LASSO with L1 penalty
    2. Identify non-zero coefficients
    3. Keep only features with significant impact
    4. Result: **590 -> {n_features} features** ({(1 - n_features/590)*100:.1f}% reduction)

    **Benefits:**
    - Reduced computational cost
    - Improved interpretability
    - Reduced overfitting
    - Faster inference (critical for production)
    """)

with col2:
    st.subheader("Selected Features")

    if lasso_features:
        feature_table = pd.DataFrame({
            "Feature": lasso_features,
            "Index": range(1, len(lasso_features) + 1),
        })
        st.dataframe(feature_table, use_container_width=True)
    else:
        st.markdown("*No features selected. Run preprocessing first.*")

st.markdown("---")

# Model Performance Summary
st.header("Model Performance Summary")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Production Model")

    if model_info['available']:
        st.markdown(f"""
        **Architecture:** {model_info['feature_set'].upper()} + {model_info['sampling_strategy'].upper()} + {model_info['model_name']}

        **Performance:**
        - CV G-Mean: {model_info['cv_gmean']*100:.2f}%
        - Test G-Mean: {model_info.get('test_gmean', 0)*100:.2f}%
        - Optimal Threshold: {model_info['threshold']:.3f}

        **Strengths:**
        - Optimized via Optuna (100 trials)
        - Threshold tuned on CV predictions
        - No data leakage
        """)
    else:
        st.markdown("*Model not trained yet.*")

with col2:
    st.subheader("Model Selection Process")

    st.markdown("""
    **36 Experiments Evaluated:**
    - 4 models (LogReg, RandomForest, XGBoost, SVM)
    - 3 feature sets (All, PCA, LASSO)
    - 3 sampling strategies (None, SMOTE, ADASYN)

    **Selection Criteria:**
    - Primary: CV G-Mean @ optimal threshold
    - Secondary: Stability (low CV std)

    **Why G-Mean?**
    - Balanced metric for imbalanced data
    - Equally weights sensitivity and specificity
    - More robust than accuracy for 14:1 imbalance
    """)

st.markdown("---")

# Actionable Insights
st.header("Actionable Insights for Production")

col1, col2 = st.columns(2)

with col1:
    st.subheader("For Process Engineers")
    st.markdown("""
    1. **Monitor top features closely**
       - Watch for drift in important sensors

    2. **Investigate low-confidence predictions**
       - When confidence is low (<60%)
       - Check sensor calibration

    3. **Use feature values for root cause**
       - Identify which features caused failure
       - Trace back to process step
    """)

with col2:
    st.subheader("For Data Scientists")
    st.markdown("""
    1. **Retrain triggers**
       - Feature drift detected (KS test)
       - Performance degradation

    2. **Model updates**
       - Recalibrate threshold quarterly
       - A/B test new architectures

    3. **Monitoring**
       - Track prediction distribution
       - Monitor latency
    """)

st.markdown("---")
st.caption("Explainability Framework: SHAP + Feature Importance + Cross-Model Validation")
