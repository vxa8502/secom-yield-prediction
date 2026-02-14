"""
Exploration Page
Feature distributions, correlations, and data quality analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from streamlit_app.utils.artifact_loader import (
    load_lasso_features,
)
from streamlit_app.utils.styling import PAGE_CONFIG, apply_styling, show_figure, TITLE_STYLE
from src.config import COLORS, DATA_DIR, VIZ_CONFIG

st.set_page_config(**PAGE_CONFIG)
apply_styling(st)

st.title("Data Exploration")
st.markdown("---")


@st.cache_data
def load_training_data():
    """Load training data for exploration."""
    lasso_features = load_lasso_features()
    if not lasso_features:
        return None, None, None

    X_train_path = DATA_DIR / "X_train_lasso.npy"
    y_train_path = DATA_DIR / "y_train.csv"

    if not X_train_path.exists() or not y_train_path.exists():
        return None, None, None

    X_train = np.load(X_train_path)
    y_train = pd.read_csv(y_train_path).values.ravel()

    df = pd.DataFrame(X_train, columns=lasso_features)
    df["target"] = y_train

    return df, lasso_features, y_train


@st.cache_data
def compute_correlations(df, features):
    """Compute feature correlations."""
    return df[features].corr()


# Load data
df, lasso_features, y_train = load_training_data()

if df is None:
    st.markdown("*Training data not available. Run `make pipeline` first to preprocess data.*")
    st.stop()

# Overview
st.markdown(f"""
**Dataset Overview:**
- Training samples: {len(df):,}
- Selected features: {len(lasso_features)} (from 590 original sensors)
- Class distribution: {(y_train == 0).sum()} PASS / {(y_train == 1).sum()} FAIL
""")

# Navigation tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Class Distribution",
    "Feature Distributions",
    "Feature Correlations",
    "Feature Statistics"
])

# Tab 1: Class Distribution
with tab1:
    st.header("Class Distribution")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Training Set Balance")

        n_pass = (y_train == 0).sum()
        n_fail = (y_train == 1).sum()
        imbalance_ratio = n_pass / n_fail if n_fail > 0 else 0

        fig, ax = plt.subplots(figsize=VIZ_CONFIG['figsize_small'])
        bars = ax.bar(
            ["PASS", "FAIL"],
            [n_pass, n_fail],
            color=[COLORS["pass"], COLORS["fail"]],
            edgecolor="black",
            linewidth=1.2
        )

        for bar, count in zip(bars, [n_pass, n_fail]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 10,
                f"{count:,}",
                ha="center",
                va="bottom",
                fontsize=VIZ_CONFIG['label_fontsize'],
                fontweight="bold"
            )

        ax.set_ylabel("Count", fontsize=VIZ_CONFIG['label_fontsize'])
        ax.set_title("Class Distribution (Training Set)", **TITLE_STYLE)
        ax.set_ylim(0, n_pass * 1.15)
        show_figure(fig)

    with col2:
        st.subheader("Imbalance Analysis")

        st.metric("Imbalance Ratio", f"{imbalance_ratio:.1f}:1")
        st.metric("Minority Class (FAIL)", f"{n_fail / len(y_train) * 100:.1f}%")

        st.markdown(f"""
        **Implications:**
        - Severe class imbalance ({imbalance_ratio:.0f}:1 ratio)
        - Naive classifier achieves {n_pass / len(y_train) * 100:.1f}% accuracy
        - G-Mean metric used to balance sensitivity/specificity
        - SMOTE/ADASYN applied during training to handle imbalance

        **Why This Matters:**
        - Minority class (FAIL) is the class of interest
        - Missing a defect is costlier than a false alarm
        - Model optimized for balanced detection
        """)

# Tab 2: Feature Distributions
with tab2:
    st.header("Feature Distributions")

    st.markdown("""
    Examine the distribution of each selected feature. Features with unusual distributions
    (heavy tails, multimodality) may require special attention during monitoring.
    """)

    # Feature selector
    selected_feature = st.selectbox(
        "Select Feature",
        lasso_features,
        help="Choose a feature to visualize its distribution"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Histogram by Class")

        fig, ax = plt.subplots(figsize=VIZ_CONFIG['figsize_medium'])

        pass_data = df[df["target"] == 0][selected_feature]
        fail_data = df[df["target"] == 1][selected_feature]

        ax.hist(
            pass_data,
            bins=VIZ_CONFIG['hist_bins'],
            alpha=0.6,
            label=f"PASS (n={len(pass_data)})",
            color=COLORS["pass"],
            edgecolor="black",
            linewidth=0.5
        )
        ax.hist(
            fail_data,
            bins=VIZ_CONFIG['hist_bins'],
            alpha=0.6,
            label=f"FAIL (n={len(fail_data)})",
            color=COLORS["fail"],
            edgecolor="black",
            linewidth=0.5
        )

        ax.set_xlabel(selected_feature, fontsize=VIZ_CONFIG['label_fontsize'])
        ax.set_ylabel("Frequency", fontsize=VIZ_CONFIG['label_fontsize'])
        ax.set_title(f"Distribution of {selected_feature}", **TITLE_STYLE)
        ax.legend()
        show_figure(fig)

    with col2:
        st.subheader("Box Plot by Class")

        fig, ax = plt.subplots(figsize=VIZ_CONFIG['figsize_medium'])

        box_data = [
            df[df["target"] == 0][selected_feature],
            df[df["target"] == 1][selected_feature]
        ]

        bp = ax.boxplot(
            box_data,
            labels=["PASS", "FAIL"],
            patch_artist=True,
            medianprops={"color": "black", "linewidth": 2}
        )

        bp["boxes"][0].set_facecolor(COLORS["pass"])
        bp["boxes"][1].set_facecolor(COLORS["fail"])

        ax.set_ylabel(selected_feature, fontsize=VIZ_CONFIG['label_fontsize'])
        ax.set_title(f"Box Plot: {selected_feature}", **TITLE_STYLE)
        show_figure(fig)

    # Statistics table
    st.subheader("Feature Statistics by Class")

    stats_pass = df[df["target"] == 0][selected_feature].describe()
    stats_fail = df[df["target"] == 1][selected_feature].describe()

    stats_df = pd.DataFrame({
        "Statistic": ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"],
        "PASS": stats_pass.values,
        "FAIL": stats_fail.values
    })

    st.dataframe(stats_df, use_container_width=True)

# Tab 3: Feature Correlations
with tab3:
    st.header("Feature Correlations")

    st.markdown("""
    Correlation heatmap shows relationships between selected features.
    High correlations may indicate redundant information or underlying physical relationships.
    """)

    # Compute correlations
    corr_matrix = compute_correlations(df, lasso_features)

    # Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True if len(lasso_features) <= 10 else False,
        fmt=".2f",
        cmap=VIZ_CONFIG['heatmap_cmap'],
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": 0.8, "label": "Correlation"}
    )

    ax.set_title("Feature Correlation Matrix", **TITLE_STYLE)
    show_figure(fig)

    # High correlations table
    st.subheader("Highest Feature Correlations")

    # Extract upper triangle pairs
    corr_pairs = []
    for i in range(len(lasso_features)):
        for j in range(i + 1, len(lasso_features)):
            corr_pairs.append({
                "Feature 1": lasso_features[i],
                "Feature 2": lasso_features[j],
                "Correlation": corr_matrix.iloc[i, j]
            })

    corr_df = pd.DataFrame(corr_pairs)
    corr_df["Abs Correlation"] = corr_df["Correlation"].abs()
    corr_df = corr_df.sort_values("Abs Correlation", ascending=False).head(10)
    corr_df = corr_df.drop(columns=["Abs Correlation"])

    st.dataframe(corr_df, use_container_width=True)

    # Correlation with target
    st.subheader("Correlation with Target (Defect)")

    target_corr = df[lasso_features].corrwith(df["target"]).sort_values(key=abs, ascending=False)

    fig, ax = plt.subplots(figsize=VIZ_CONFIG['figsize_wide'])

    colors = [COLORS["fail"] if c > 0 else COLORS["pass"] for c in target_corr.values]

    ax.barh(target_corr.index, target_corr.values, color=colors)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel("Correlation with Target", fontsize=VIZ_CONFIG['label_fontsize'])
    ax.set_title("Feature-Target Correlation", **TITLE_STYLE)
    ax.invert_yaxis()
    show_figure(fig)

    st.markdown("""
    **Interpretation:**
    - Positive correlation: Higher values associated with FAIL
    - Negative correlation: Higher values associated with PASS
    - Features with stronger correlations have higher predictive power
    """)

# Tab 4: Feature Statistics
with tab4:
    st.header("Feature Statistics Summary")

    st.markdown("""
    Comprehensive statistics for all selected features. Use this to identify
    features with unusual ranges, high variance, or potential data quality issues.
    """)

    # Compute statistics
    stats_all = df[lasso_features].describe().T
    stats_all["range"] = stats_all["max"] - stats_all["min"]
    stats_all["cv"] = stats_all["std"] / stats_all["mean"].abs()  # Coefficient of variation
    stats_all = stats_all.round(4)

    st.dataframe(stats_all, use_container_width=True)

    # Summary metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Features", len(lasso_features))

    with col2:
        high_cv = (stats_all["cv"].abs() > 1).sum()
        st.metric("High Variance Features", high_cv, help="Features with CV > 1")

    with col3:
        st.metric("Feature Reduction", f"{(1 - len(lasso_features) / 590) * 100:.1f}%")

    # Download option
    st.markdown("---")
    st.subheader("Export Data")

    csv_data = stats_all.to_csv()
    st.download_button(
        label="Download Feature Statistics (CSV)",
        data=csv_data,
        file_name="feature_statistics.csv",
        mime="text/csv"
    )

st.markdown("---")
st.caption("Data Exploration: Feature distributions, correlations, and statistics for selected LASSO features")
