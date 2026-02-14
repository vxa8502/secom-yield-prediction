"""
Prediction Page
Interactive wafer quality prediction interface
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

st.title("Wafer Quality Prediction")
st.markdown("---")

# Check model availability
model_info = get_model_display_info()
if not model_info['available']:
    st.markdown(f"*{model_info['message']}*")
    st.stop()

# Load configuration
lasso_features = load_lasso_features()
threshold_config = load_threshold_config()
default_threshold = threshold_config.get('optimal_threshold', 0.5)

# Instructions
st.markdown(f"""
**Instructions:** Enter sensor readings for the {len(lasso_features)} critical process parameters.
The model will predict whether the wafer will pass or fail quality inspection.
""")

# Input method selection
input_method = st.radio(
    "Input Method",
    ["Manual Entry", "Example Values", "Upload CSV"],
    horizontal=True
)

# Initialize feature values
feature_values = {}

if input_method == "Manual Entry":
    st.subheader("Enter Sensor Readings")

    # Create input form
    with st.form("prediction_form"):
        st.caption(f"Valid range: {FEATURE_VALUE_MIN:.0e} to {FEATURE_VALUE_MAX:.0e}")

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
                    format="%.6f",
                    help=f"Sensor reading for {feature}",
                    key=f"input_{feature}"
                )

        with col2:
            for feature in lasso_features[mid:]:
                feature_values[feature] = st.number_input(
                    label=f"{feature}",
                    value=0.0,
                    min_value=FEATURE_VALUE_MIN,
                    max_value=FEATURE_VALUE_MAX,
                    format="%.6f",
                    help=f"Sensor reading for {feature}",
                    key=f"input_{feature}"
                )

        # Threshold selection using centralized config
        st.markdown("---")
        threshold_options = get_threshold_options(default_threshold)
        threshold_option = st.selectbox(
            "Classification Threshold",
            [
                f"Default ({threshold_options['default']:.3f}) - Balanced",
                f"Conservative ({threshold_options['conservative']:.3f}) - Catch more defects",
                f"Aggressive ({threshold_options['aggressive']:.3f}) - Fewer false alarms"
            ],
            help="Default: Production threshold | Conservative: Fewer missed defects | Aggressive: Fewer false alarms"
        )

        if "Default" in threshold_option:
            threshold = threshold_options['default']
        elif "Conservative" in threshold_option:
            threshold = threshold_options['conservative']
        else:
            threshold = threshold_options['aggressive']

        # Submit button
        submitted = st.form_submit_button("Predict Quality", use_container_width=True)

elif input_method == "Example Values":
    st.subheader("Example Sensor Readings")

    # Load test samples
    test_samples = load_test_sample(n_samples=6)

    if test_samples is not None:
        example_type = st.selectbox(
            "Select Example",
            ["Passing Wafer (from test set)", "Failing Wafer (from test set)"]
        )

        if "Passing" in example_type:
            sample_row = test_samples[test_samples['actual_label'] == 0].iloc[0]
        else:
            sample_row = test_samples[test_samples['actual_label'] == 1].iloc[0]

        feature_values = {f: sample_row[f] for f in lasso_features}

    else:
        # Fallback to synthetic examples
        example_type = st.selectbox(
            "Select Example",
            ["Typical Values (zeros)", "Random Values"]
        )

        if "zeros" in example_type.lower():
            feature_values = {f: 0.0 for f in lasso_features}
        else:
            np.random.seed(42)
            feature_values = {f: np.random.randn() for f in lasso_features}

    # Display values
    st.dataframe(
        pd.DataFrame([feature_values]).T.rename(columns={0: "Value"}),
        use_container_width=True
    )

    threshold = default_threshold
    submitted = st.button("Predict Quality", use_container_width=True)

else:  # Upload CSV
    st.subheader("Upload Sensor Data")

    st.caption(f"Maximum file size: {MAX_CSV_SIZE_MB} MB")

    uploaded_file = st.file_uploader(
        "Upload CSV file with sensor readings",
        type=["csv"],
        help=f"CSV must contain columns: {', '.join(lasso_features[:3])}..."
    )

    if uploaded_file is not None:
        # File size validation
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > MAX_CSV_SIZE_MB:
            st.error(f"File too large ({file_size_mb:.1f} MB). Maximum allowed: {MAX_CSV_SIZE_MB} MB")
            submitted = False
        else:
            try:
                df = pd.read_csv(uploaded_file)

                # Check for required features
                missing_features = set(lasso_features) - set(df.columns)
                if missing_features:
                    st.error(f"Missing required features: {missing_features}")
                    submitted = False
                else:
                    # Validate numeric content for required columns
                    feature_df = df[lasso_features]
                    non_numeric = feature_df.apply(lambda col: ~pd.to_numeric(col, errors='coerce').notna()).any()
                    invalid_cols = non_numeric[non_numeric].index.tolist()

                    if invalid_cols:
                        st.error(f"Non-numeric values found in columns: {invalid_cols}")
                        submitted = False
                    else:
                        # Check for out-of-range values
                        out_of_range = (
                            (feature_df < FEATURE_VALUE_MIN) | (feature_df > FEATURE_VALUE_MAX)
                        ).any()
                        oor_cols = out_of_range[out_of_range].index.tolist()

                        if oor_cols:
                            st.warning(f"Values outside expected range in: {oor_cols}")

                        st.write("Uploaded data preview:")
                        st.dataframe(df.head())
                        st.markdown(f"All {len(lasso_features)} required features found. {len(df)} row(s) uploaded.")

                        feature_values = df[lasso_features].iloc[0].to_dict()
                        threshold = default_threshold
                        submitted = st.button("Predict Quality", use_container_width=True)

            except pd.errors.ParserError as e:
                st.error(f"Failed to parse CSV: {e}")
                submitted = False
            except Exception as e:
                st.error(f"Error processing file: {e}")
                submitted = False
    else:
        submitted = False

# Make prediction
if submitted and feature_values:
    with st.spinner("Running prediction..."):
        try:
            # Get prediction
            result = predict_single(feature_values, threshold=threshold)

            # Log the prediction request
            log_prediction_request(
                prediction=result['prediction'],
                probability=result['probability'],
                threshold=result['threshold_used'],
                input_method=input_method.lower().replace(" ", "_"),
                n_features=len(feature_values)
            )

            st.markdown("---")
            st.header("Prediction Results")

            # Display results in columns
            col1, col2, col3 = st.columns(3)

            with col1:
                prediction_color = COLORS["fail"] if result["prediction"] == 1 else COLORS["pass"]
                st.markdown(
                    f"<h2 style='text-align: center; color: {prediction_color};'>{result['class_label']}</h2>",
                    unsafe_allow_html=True
                )

            with col2:
                st.metric(
                    label="Failure Probability",
                    value=f"{result['probability']:.2%}",
                    delta=None
                )

            with col3:
                st.metric(
                    label="Confidence",
                    value=f"{result['confidence']:.2%}",
                    delta=None
                )

            # Generate actionable recommendations
            shap_contributions = compute_single_prediction_shap(feature_values)
            recommendations = generate_recommendations(
                prediction=result['prediction'],
                probability=result['probability'],
                threshold=result['threshold_used'],
                confidence=result['confidence'],
                shap_contributions=shap_contributions
            )

            # Display severity badge and recommendations
            st.markdown("---")
            st.subheader("Actionable Recommendations")

            severity_cfg = SEVERITY_CONFIG[recommendations.severity]
            severity_badge = f"<span style='background-color: {severity_cfg['color']}; color: white; padding: 4px 12px; border-radius: 4px; font-weight: bold;'>{severity_cfg['label']}</span>"

            st.markdown(f"**Severity:** {severity_badge}", unsafe_allow_html=True)
            st.markdown(f"**Confidence:** {recommendations.confidence_tier.upper()}")

            # Primary action
            st.markdown(f"### {recommendations.primary_action}")

            # Process guidance
            st.info(recommendations.process_guidance)

            # Secondary actions
            if recommendations.secondary_actions:
                st.markdown("**Additional Actions:**")
                for action in recommendations.secondary_actions:
                    st.markdown(f"- {action}")

            # Feature insights (expandable)
            if recommendations.feature_insights:
                with st.expander("Feature Insights (SHAP Analysis)"):
                    for insight in recommendations.feature_insights:
                        direction_icon = "+" if insight['direction'] == 'increasing' else "-"
                        st.markdown(
                            f"- **{insight['feature']}**: {direction_icon}{insight['contribution']:.4f} "
                            f"({insight['description']})"
                        )

            # Threshold info
            st.caption(f"Threshold: {result['threshold_used']:.3f} | Probability: {result['probability']:.2%}")

            # Probability gauge
            st.markdown("---")
            st.subheader("Probability Distribution")

            prob_data = pd.DataFrame({
                "Outcome": ["PASS", "FAIL"],
                "Probability": [1 - result["probability"], result["probability"]]
            })

            st.bar_chart(prob_data.set_index("Outcome"))

            # Feature values summary
            st.markdown("---")
            st.subheader("Input Feature Summary")

            feature_summary = pd.DataFrame({
                "Feature": lasso_features,
                "Value": [feature_values[f] for f in lasso_features],
            })

            st.dataframe(feature_summary, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.exception(e)

st.markdown("---")
st.caption(f"Production Model: {model_info['model_name']} | CV G-Mean: {model_info['cv_gmean']*100:.2f}%")
