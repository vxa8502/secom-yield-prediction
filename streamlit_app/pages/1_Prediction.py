"""
Prediction Page
Interactive wafer quality prediction interface with side-by-side layout
"""

import streamlit as st
import pandas as pd
import numpy as np

from streamlit_app.utils.artifact_loader import (
    load_lasso_features,
    load_threshold_config,
    predict_single,
    load_test_sample,
    get_model_display_info,
    log_prediction_request,
    compute_single_prediction_shap,
)
from streamlit_app.utils.styling import PAGE_CONFIG, apply_styling
from streamlit_app.utils.config import get_threshold_options
from streamlit_app.utils.recommendations import (
    generate_recommendations,
    SEVERITY_CONFIG,
)
from src.config import COLORS

# Input validation constants
MAX_CSV_SIZE_MB = 10
FEATURE_VALUE_MIN = -1e6
FEATURE_VALUE_MAX = 1e6

st.set_page_config(**PAGE_CONFIG)
apply_styling(st)

# Custom CSS for better styling
st.markdown("""
<style>
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 1rem;
    }
    .result-card-fail {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
    }
    .result-card-pass {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .result-label {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    .metric-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border-left: 4px solid #667eea;
    }
    .section-header {
        color: #1e3a5f;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .input-section {
        background: #fafbfc;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e1e4e8;
    }
</style>
""", unsafe_allow_html=True)

st.title("Wafer Quality Prediction")

# Check model availability
model_info = get_model_display_info()
if not model_info['available']:
    st.markdown(f"*{model_info['message']}*")
    st.stop()

# Load configuration
lasso_features = load_lasso_features()
threshold_config = load_threshold_config()
default_threshold = threshold_config.get('optimal_threshold', 0.5)

# Initialize session state
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None

# Full-width header row
header_col1, header_col2 = st.columns([1, 1], gap="large")
with header_col1:
    st.markdown('<p class="section-header">Sensor Input</p>', unsafe_allow_html=True)
with header_col2:
    st.markdown('<p class="section-header">Prediction Results</p>', unsafe_allow_html=True)

# Input method selector (full width of left side conceptually, but in column)
input_col, results_col = st.columns([1, 1], gap="large")

# --- LEFT COLUMN: Input Form ---
with input_col:
    input_method = st.radio(
        "Input Method",
        ["Manual Entry", "Example Values", "Upload CSV"],
        horizontal=True,
        label_visibility="collapsed"
    )

    feature_values = {}
    submitted = False
    threshold = default_threshold

    if input_method == "Manual Entry":
        with st.form("prediction_form"):
            st.caption(f"Enter values for {len(lasso_features)} features")

            n_features = len(lasso_features)
            mid = (n_features + 1) // 2
            col1, col2 = st.columns(2)

            with col1:
                for feature in lasso_features[:mid]:
                    feature_values[feature] = st.number_input(
                        label=f"{feature}",
                        value=0.0,
                        min_value=FEATURE_VALUE_MIN,
                        max_value=FEATURE_VALUE_MAX,
                        format="%.4f",
                        key=f"input_{feature}"
                    )

            with col2:
                for feature in lasso_features[mid:]:
                    feature_values[feature] = st.number_input(
                        label=f"{feature}",
                        value=0.0,
                        min_value=FEATURE_VALUE_MIN,
                        max_value=FEATURE_VALUE_MAX,
                        format="%.4f",
                        key=f"input_{feature}"
                    )

            threshold_options = get_threshold_options(default_threshold)
            threshold_option = st.selectbox(
                "Threshold",
                [
                    f"Default ({threshold_options['default']:.3f})",
                    f"Conservative ({threshold_options['conservative']:.3f})",
                    f"Aggressive ({threshold_options['aggressive']:.3f})"
                ],
            )

            if "Default" in threshold_option:
                threshold = threshold_options['default']
            elif "Conservative" in threshold_option:
                threshold = threshold_options['conservative']
            else:
                threshold = threshold_options['aggressive']

            submitted = st.form_submit_button("Predict Quality", use_container_width=True, type="primary")

    elif input_method == "Example Values":
        test_samples = load_test_sample()  # Load all test samples

        if test_samples is not None:
            example_type = st.selectbox(
                "Select Example",
                ["Failing Wafer (from test set)", "Passing Wafer (from test set)"],
                label_visibility="collapsed"
            )

            if "Passing" in example_type:
                sample_row = test_samples[test_samples['actual_label'] == 0].sample(1).iloc[0]
            else:
                sample_row = test_samples[test_samples['actual_label'] == 1].sample(1).iloc[0]

            feature_values = {f: sample_row[f] for f in lasso_features}
        else:
            example_type = st.selectbox(
                "Select Example",
                ["Typical Values (zeros)", "Random Values"]
            )
            if "zeros" in example_type.lower():
                feature_values = {f: 0.0 for f in lasso_features}
            else:
                np.random.seed(42)
                feature_values = {f: np.random.randn() for f in lasso_features}

        # Styled dataframe
        df_display = pd.DataFrame({
            'Feature': list(feature_values.keys()),
            'Value': [f"{v:.4f}" for v in feature_values.values()]
        })
        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            height=250
        )

        submitted = st.button("Predict Quality", use_container_width=True, type="primary")

    else:  # Upload CSV
        st.caption(f"Max size: {MAX_CSV_SIZE_MB} MB")

        uploaded_file = st.file_uploader(
            "Upload CSV",
            type=["csv"],
            label_visibility="collapsed"
        )

        if uploaded_file is not None:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            if file_size_mb > MAX_CSV_SIZE_MB:
                st.error(f"File too large ({file_size_mb:.1f} MB)")
            else:
                try:
                    df = pd.read_csv(uploaded_file)
                    missing_features = set(lasso_features) - set(df.columns)

                    if missing_features:
                        st.error(f"Missing: {missing_features}")
                    else:
                        st.dataframe(df[lasso_features].head(), height=150, hide_index=True)
                        feature_values = df[lasso_features].iloc[0].to_dict()
                        submitted = st.button("Predict Quality", use_container_width=True, type="primary")

                except Exception as e:
                    st.error(f"Error: {e}")

# --- Process prediction ---
if submitted and feature_values:
    try:
        result = predict_single(feature_values, threshold=threshold)

        log_prediction_request(
            prediction=result['prediction'],
            probability=result['probability'],
            threshold=result['threshold_used'],
            input_method=input_method.lower().replace(" ", "_"),
            n_features=len(feature_values)
        )

        shap_contributions = compute_single_prediction_shap(feature_values)
        recommendations = generate_recommendations(
            prediction=result['prediction'],
            probability=result['probability'],
            threshold=result['threshold_used'],
            confidence=result['confidence'],
            shap_contributions=shap_contributions
        )

        st.session_state.prediction_result = result
        st.session_state.recommendations = recommendations

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        st.session_state.prediction_result = None

# --- LEFT COLUMN: Result Card (under input) ---
with input_col:
    result = st.session_state.prediction_result
    if result is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        card_class = "result-card-fail" if result["prediction"] == 1 else "result-card-pass"
        st.markdown(f"""
        <div class="result-card {card_class}">
            <p class="result-label">{result['class_label']}</p>
        </div>
        """, unsafe_allow_html=True)

# --- RIGHT COLUMN: Metrics & Recommendations ---
with results_col:
    result = st.session_state.prediction_result
    recommendations = st.session_state.recommendations

    if result is not None:
        # Accent color based on prediction
        accent_color = "#eb3349" if result["prediction"] == 1 else "#11998e"

        # Gap to align with input content below
        st.markdown("<div style='height: 2.5rem;'></div>", unsafe_allow_html=True)

        # Metrics in styled containers with matching accent
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; text-align: center; border-left: 4px solid {accent_color};">
                <div style="color: #666; font-size: 0.85rem;">Failure Probability</div>
                <div style="font-size: 1.8rem; font-weight: 700; color: #1e3a5f;">{result['probability']:.1%}</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; text-align: center; border-left: 4px solid {accent_color};">
                <div style="color: #666; font-size: 0.85rem;">Confidence</div>
                <div style="font-size: 1.8rem; font-weight: 700; color: #1e3a5f;">{result['confidence']:.1%}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Recommendations
        severity_cfg = SEVERITY_CONFIG[recommendations.severity]
        st.markdown(f"""
        **Severity:** <span style='background-color: {severity_cfg['color']}; color: white;
        padding: 3px 10px; border-radius: 4px; font-weight: 600;'>{severity_cfg['label']}</span>
        &nbsp;&nbsp; **Confidence:** {recommendations.confidence_tier.upper()}
        """, unsafe_allow_html=True)

        st.markdown(f"#### {recommendations.primary_action}")
        st.info(recommendations.process_guidance)

        if recommendations.secondary_actions:
            actions_html = "<br>".join([f"&bull; {action}" for action in recommendations.secondary_actions])
            st.markdown(f"""
            <div style="margin-top: 0.5rem;">
                <strong>Additional Actions:</strong><br>
                <span style="line-height: 1.4;">{actions_html}</span>
            </div>
            """, unsafe_allow_html=True)

        if recommendations.feature_insights:
            with st.expander("Feature Insights (SHAP)"):
                for insight in recommendations.feature_insights:
                    icon = "+" if insight['direction'] == 'increasing' else "-"
                    st.markdown(f"- **{insight['feature']}**: {icon}{insight['contribution']:.4f}")

        st.caption(f"Threshold: {result['threshold_used']:.3f}")

    else:
        st.markdown("""
        <div style="height: 2.5rem;"></div>
        <div style="background: #f0f2f6; padding: 2rem; border-radius: 8px; text-align: center;">
            <p style="color: #666; margin: 0;">Select an example or enter values, then click <b>Predict Quality</b></p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.caption(f"Model: {model_info['model_name']} | CV G-Mean: {model_info['cv_gmean']*100:.1f}%")
