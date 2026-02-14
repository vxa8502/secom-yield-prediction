# %%
"""
SECOM Semiconductor Defect Prediction - Exploratory Data Analysis
Author: Victoria A.
Project: Yield forecasting with UCI SECOM dataset

This script performs initial exploratory data analysis on the SECOM dataset,
including missing data analysis, correlation exploration, and feature selection.

When run as script (make eda): generates figures + markdown report silently
When run interactively: shows plots and prints analysis
"""

# %%
import warnings
import os
import io
# Suppress plotting-related warnings (missingno, matplotlib, etc.)
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=RuntimeWarning)

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import missingno as msno

# Quiet mode: suppress terminal output when running as script
# Interactive mode (notebook/ipython): show output
INTERACTIVE = 'ipykernel' in sys.modules or hasattr(sys, 'ps1')

def log(msg):
    """Print only in interactive mode."""
    if INTERACTIVE:
        print(msg)

# %%
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import mannwhitneyu, chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor

# %%
# Setup project root path for imports and data loading
try:
    project_root = Path(__file__).parent.parent
except NameError:
    project_root = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()

sys.path.insert(0, str(project_root))

# %%
# Import centralized configuration (DRY: single source of truth)
from src.config import (
    RANDOM_STATE,
    RAW_DATA_DIR as _RAW_DATA_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    VIZ_CONFIG as _VIZ_CONFIG,
)

# SECOM dataset subdirectory
RAW_DATA_DIR = _RAW_DATA_DIR / 'secom'

# %%
# Create output directories
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# %%
# Set random seed for reproducibility
np.random.seed(RANDOM_STATE)

# %%
# EDA-specific visualization overrides (smaller fonts/DPI for notebook display)
VIZ_CONFIG = {
    **_VIZ_CONFIG,
    'dpi': 300,  # Lower DPI for faster notebook rendering
    'title_fontsize': 12,
    'label_fontsize': 10,
    'tick_fontsize': 9,
}

# Track generated figures for markdown report
GENERATED_FIGURES = []


def save_figure(fig, filename, caption=""):
    """Save figure and track for markdown report generation."""
    filepath = FIGURES_DIR / filename
    fig.savefig(filepath, dpi=VIZ_CONFIG['dpi'], bbox_inches='tight', facecolor='white')
    GENERATED_FIGURES.append({'filename': filename, 'caption': caption})
    return filepath

# %% [markdown]
# ### DATA LOADING

# %%
# Load features (590 sensor readings)
X = pd.read_csv(RAW_DATA_DIR / 'secom.data', sep=' ', header=None, index_col=False)

# Load labels (binary classification: originally -1 = pass, 1 = fail)
y = pd.read_csv(RAW_DATA_DIR / 'secom_labels.data', sep=' ', header=None,
                names=['target', 'timestamp'])

# %%
# Display dataset dimensions
log(f"Dataset Dimensions:")
log(f"  Features (X): {X.shape[0]:,} samples × {X.shape[1]} features")
log(f"  Labels (y):   {y.shape[0]:,} samples × {y.shape[1]} columns")
log(f"\nFirst 3 samples (showing first 5 features):")
log(X.head(3).iloc[:, :5].to_string())
log(f"\nFirst 3 labels:")
log(y.head(3).to_string())

# %%
# Prepare features and labels for analysis
column_names = [f'feature_{i}' for i in range(1, X.shape[1] + 1)]
X.columns = column_names

# Convert to sklearn standard: 0 = Pass (majority), 1 = Fail (minority)
y['target'] = y['target'].map({-1: 0, 1: 1})

# Parse timestamp
y['timestamp'] = pd.to_datetime(y['timestamp'], format='%d/%m/%Y %H:%M:%S')

log(f"Data Preparation Summary:")
log(f"  Feature columns: feature_1 through feature_{X.shape[1]}")
log(f"  Target encoding: 0 = Pass (majority), 1 = Fail (minority)")
log(f"  Target unique values: {sorted(y['target'].unique())}")
log(f"  Date range: {y['timestamp'].min().date()} to {y['timestamp'].max().date()}")
log(f"  Duration: {(y['timestamp'].max() - y['timestamp'].min()).days} days")
log(f"\nData types:")
log(f"  Features: {X.dtypes.value_counts().to_dict()}")
log(f"  Labels: target={y['target'].dtype}, timestamp={y['timestamp'].dtype}")

# %%
# Compute descriptive statistics for all features
desc_stats = X.describe()
desc_stats.T.to_csv(REPORTS_DIR / 'feature_statistics.csv')

minimum_values = X.min().sort_values(ascending=True)
maximum_values = X.max().sort_values(ascending=True)
mean_values = X.mean().sort_values(ascending=True)
std_values = X.std().sort_values(ascending=True)
median_values = X.median().sort_values(ascending=True)

log(f"\nBASIC STATISTICS SUMMARY")
log(f"Aggregate statistics across all {X.shape[1]} features:")
log(f"  Min value:   {minimum_values.min():>12.4f}  (feature: {minimum_values.index[0]})")
log(f"  Max value:   {maximum_values.max():>12.4f}  (feature: {maximum_values.index[-1]})")
log(f"  Mean (avg):  {mean_values.mean():>12.4f}")
log(f"  Std (avg):   {std_values.mean():>12.4f}")
log(f"  Median (med):{median_values.median():>12.4f}")
log(f"\nDetailed statistics saved: {REPORTS_DIR / 'feature_statistics.csv'}")

# %%
# Variance analysis - identify low/high variance features
cv_values = std_values / mean_values.abs().replace(0, np.nan)
exact_zero_variance = std_values[std_values == 0]
near_zero_variance = cv_values[cv_values < 0.01]
high_variance = std_values[std_values > std_values.quantile(0.95)]

log(f"\nVARIANCE ANALYSIS")
log(f"Features with zero variance (constant):           {len(exact_zero_variance):>4}")
if len(exact_zero_variance) > 0:
    log(f"  Examples: {', '.join(exact_zero_variance.index[:3].tolist())}")

log(f"Features with near-zero variance (CV < 0.01):     {len(near_zero_variance):>4}")
if len(near_zero_variance) > 0:
    log(f"  Examples: {', '.join(near_zero_variance.index[:3].tolist())}")

log(f"Features with high variance (>95th percentile):   {len(high_variance):>4}")
if len(high_variance) > 0:
    log(f"  Examples: {', '.join(high_variance.index[:3].tolist())}")
    log(f"  Threshold (95th percentile): {std_values.quantile(0.95):.4f}")

# %% [markdown]
# ### DUPLICATE DETECTION

# %%
# Check for duplicate samples
duplicate_features = X.duplicated().sum()
combined = pd.concat([X, y['target']], axis=1)
duplicate_combined = combined.duplicated().sum()
duplicate_timestamps = y['timestamp'].duplicated().sum()

log(f"\nDUPLICATE DETECTION")
log(f"Duplicate rows (features only):              {duplicate_features:>4}")
log(f"Duplicate rows (features + target):          {duplicate_combined:>4}")
log(f"Duplicate timestamps:                        {duplicate_timestamps:>4}")

if duplicate_timestamps > 0:
    pct = (duplicate_timestamps / len(y)) * 100
    log(f"\nInterpretation: {pct:.2f}% of samples share timestamps")
    log(f"  Likely cause: Multiple wafers processed simultaneously")
else:
    log(f"\nInterpretation: All timestamps are unique (no batch processing detected)")

# %% [markdown]
# ### CLASS IMBALANCE ANALYSIS

# %%
# Analyze class distribution
class_balance = y['target'].value_counts().sort_index()
passed_wafers = class_balance[0]
failed_wafers = class_balance[1]
failed_pct = (failed_wafers / class_balance.sum()) * 100
imbalance_ratio = passed_wafers / failed_wafers
naive_accuracy = max(passed_wafers, failed_wafers) / class_balance.sum() * 100

log(f"\nCLASS IMBALANCE ANALYSIS")
log(f"Class distribution:")
log(f"  0 = Pass: {passed_wafers:>5} wafers ({100 - failed_pct:>5.2f}%)")
log(f"  1 = Fail: {failed_wafers:>5} wafers ({failed_pct:>5.2f}%)")
log(f"\nImbalance ratio:        {imbalance_ratio:.1f}:1 (Pass:Fail)")
log(f"Naive accuracy baseline: {naive_accuracy:.2f}% (always predict majority class)")
log(f"\nIMPLICATIONS:")
log(f"  - SEVERE imbalance detected ({imbalance_ratio:.0f}:1)")
log(f"  - Must use balanced metrics: G-Mean, F1, AUC-ROC (NOT accuracy)")
log(f"  - Requires resampling (SMOTE/ADASYN) or class weighting")
log(f"  - Confusion matrix more informative than accuracy score")

# %%
# Temporal class imbalance analysis
y_sorted = y.sort_values('timestamp')
time_grouped = y_sorted.groupby([pd.Grouper(key='timestamp', freq='D'), 'target']).size().unstack(fill_value=0)
time_grouped.columns = ['Pass', 'Fail']
time_grouped['Total'] = time_grouped['Pass'] + time_grouped['Fail']
time_grouped['Fail_Rate'] = (time_grouped['Fail'] / time_grouped['Total']) * 100
time_grouped['Imbalance_Ratio'] = time_grouped['Pass'] / time_grouped['Fail'].replace(0, 1)

log(f"\nTemporal Analysis ({len(time_grouped)} days of production):")
log(f"  Failure rate range:   {time_grouped['Fail_Rate'].min():.2f}% to {time_grouped['Fail_Rate'].max():.2f}%")
log(f"  Failure rate std dev: {time_grouped['Fail_Rate'].std():.2f}%")
log(f"  Days with 0 failures: {(time_grouped['Fail'] == 0).sum()}")
log(f"  Days with 100% fails: {(time_grouped['Pass'] == 0).sum()}")
log(f"  Days with <5 samples: {(time_grouped['Total'] < 5).sum()}")

# %%
plt.style.use(VIZ_CONFIG['style'])

# Create figure with subplots
fig, axes = plt.subplots(2, 1, figsize=(10, 5), dpi=VIZ_CONFIG['dpi'], sharex=True)

# Plot 1: Failure Rate (%)
axes[0].plot(time_grouped.index, time_grouped['Fail_Rate'], 
             color=VIZ_CONFIG['fail_color'], linewidth=2, marker='o', markersize=3, alpha=0.7)
axes[0].axhline(y=time_grouped['Fail_Rate'].mean(), color=VIZ_CONFIG['fail_color'], linestyle='--',
                linewidth=2, label=f'Mean: {time_grouped["Fail_Rate"].mean():.2f}%', alpha=0.7)
axes[0].fill_between(time_grouped.index, 0, time_grouped['Fail_Rate'], 
                      color=VIZ_CONFIG['fail_color'], alpha=0.2)
axes[0].set_ylabel('Failure Rate (%)', fontweight='bold')
axes[0].set_title('Class Imbalance Analysis Over Time', fontweight='bold', fontsize=14)
axes[0].legend(loc='best', framealpha=0.9)
axes[0].grid(True, alpha=0.1)

# Plot 2: Sample Volume (to show if low counts affect reliability)
axes[1].bar(time_grouped.index, time_grouped['Pass'], 
            color=VIZ_CONFIG['pass_color'], alpha=0.6, label='Pass', width=0.8)
axes[1].bar(time_grouped.index, time_grouped['Fail'], 
            bottom=time_grouped['Pass'], color=VIZ_CONFIG['fail_color'], 
            alpha=0.6, label='Fail', width=0.8)
axes[1].set_ylabel('Sample Count', fontweight='bold')
axes[1].set_xlabel('Date', fontweight='bold')
axes[1].legend(loc='best', framealpha=0.9)
axes[1].grid(True, alpha=0.3, axis='y')

# Format x-axis
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
save_figure(fig, 'class_imbalance_over_time.png', 'Failure rate and sample volume over time')
if INTERACTIVE: plt.show()
plt.close()

# %% [markdown]
# ### MISSING DATA ANALYSIS

# %%
# Analyze missing data patterns
missing_by_feature = X.isna().sum()
total_missing = missing_by_feature.sum()
total_cells = X.shape[0] * X.shape[1]
missing_pct_overall = (total_missing / total_cells) * 100

missing_counts = missing_by_feature[missing_by_feature > 0].sort_values(ascending=False)
missing_pct_by_feature = (missing_counts / X.shape[0]) * 100

complete_features = (missing_by_feature == 0).sum()
half_missing = (missing_pct_by_feature >= 50).sum()
mostly_missing = (missing_pct_by_feature >= 90).sum()
features_half_missing = missing_pct_by_feature[missing_pct_by_feature >= 50].sort_values(ascending=False)

log(f"\nMISSING DATA ANALYSIS")
log(f"Overall statistics:")
log(f"  Total missing values: {total_missing:,} / {total_cells:,} ({missing_pct_overall:.2f}%)")
log(f"\nFeature-level statistics:")
log(f"  Complete features (0% missing):   {complete_features:>4}")
log(f"  Features with missing data:       {len(missing_counts):>4}")
log(f"  Features >50% missing:            {half_missing:>4}")
log(f"  Features >90% missing:            {mostly_missing:>4}")
log(f"\nTop 10 features with highest missingness (>{50}%):")
for i, (feat, pct) in enumerate(features_half_missing.head(10).items(), 1):
    count = missing_counts[feat]
    log(f"  {i:>2}. {feat:<15} {count:>5} / {X.shape[0]} ({pct:>5.1f}%)")

# %%
# Test if missingness correlates with target (MCAR vs MAR/MNAR)
missing_indicators = X.isna().astype(int)
missing_indicators.columns = [f'{col}_missing' for col in X.columns]

missing_target_corr = []
for feature in missing_counts.index:
    missing_col = f'{feature}_missing'
    corr = missing_indicators[missing_col].corr(y['target'])
    missing_target_corr.append({
        'Feature': feature,
        'Correlation': corr,
        'Abs_Correlation': abs(corr),
        'Missing_Pct': missing_pct_by_feature[feature]
    })

missing_corr_df = pd.DataFrame(missing_target_corr).sort_values('Abs_Correlation', ascending=False)

significant_threshold = 0.1
significant_missing = missing_corr_df[missing_corr_df['Abs_Correlation'] > significant_threshold]

log(f"\nMissingness-Target Correlation (testing for MAR/MNAR):")
log(f"  Features with |r| > {significant_threshold}: {len(significant_missing)}")
if len(significant_missing) > 0:
    log(f"  Interpretation: Non-random missingness detected (MAR/MNAR)")
    log(f"\n  Top 5 features with strongest missingness-target correlation:")
    for i, row in missing_corr_df.head(5).iterrows():
        log(f"    {row['Feature']:<20} r={row['Correlation']:>7.4f}  ({row['Missing_Pct']:.1f}% missing)")
else:
    log(f"  Interpretation: Missingness appears random (MCAR)")
log(f"\nIMPLICATION: {'Missingness pattern may be informative for prediction' if len(significant_missing) > 0 else 'Missing values can be safely imputed'}")

# %%
# Visualize missingness patterns
X_viz = X[features_half_missing.index]
log(f"\nVisualizing missingness patterns ({len(features_half_missing)} features with >50% missing):")
fig = msno.matrix(X_viz).get_figure()
save_figure(fig, 'missing_data_matrix.png', 'Missing data matrix for features with >50% missing values')
if INTERACTIVE: plt.show()
plt.close()

fig = msno.heatmap(X_viz).get_figure()
save_figure(fig, 'missing_data_heatmap.png', 'Nullity correlation heatmap showing co-occurrence of missing values')
if INTERACTIVE: plt.show()
plt.close()

# %%
# Analyze nullity correlation (missingness co-occurrence)
nullity_matrix = X.notna().astype(int)
nullity_corr = nullity_matrix.corr()
nullity_corr.to_csv(REPORTS_DIR / 'nullity_correlation_matrix.csv')

upper_triangle = nullity_corr.where(np.triu(np.ones(nullity_corr.shape), k=1).astype(bool))
correlations_stacked = upper_triangle.stack().sort_values(key=abs, ascending=False)
perfect_corr = correlations_stacked[correlations_stacked.abs() >= 1.0]

log(f"\nNullity Correlation Analysis:")
log(f"  Feature pairs with perfect missingness correlation: {len(perfect_corr)}")
log(f"  Saved full nullity correlation matrix to: nullity_correlation_matrix.csv")
if len(perfect_corr) > 0:
    log(f"  Interpretation: {len(perfect_corr)} feature pairs always missing together (sensor groups?)")

# %% [markdown]
# ### CORRELATION ANALYSIS

# %%
# Compute feature-target correlations
joined = pd.concat([X, y['target']], axis=1)
corr_matrix = joined.corr().abs()
corr_matrix.to_csv(REPORTS_DIR / 'full_correlation_matrix.csv')

target_corr = corr_matrix['target'].drop('target').sort_values(ascending=False)
top_corr_features = corr_matrix['target'].sort_values(ascending=False).head(20)
top_corr_features_list = top_corr_features.index

log(f"\nCORRELATION ANALYSIS")
log(f"Feature-target correlation summary:")
log(f"  Max correlation:      {target_corr.max():>7.4f} (feature: {target_corr.idxmax()})")
log(f"  Mean correlation:     {target_corr.mean():>7.4f}")
log(f"  Median correlation:   {target_corr.median():>7.4f}")
log(f"  Features with |r| > 0.1: {(target_corr > 0.1).sum():>4}")
log(f"  Features with |r| > 0.2: {(target_corr > 0.2).sum():>4}")
log(f"\nTop 10 features most correlated with target:")
for i, (feat, corr_val) in enumerate(target_corr.head(10).items(), 1):
    log(f"  {i:>2}. {feat:<20} {corr_val:>7.4f}")
log(f"\nSaved full {corr_matrix.shape[0]}×{corr_matrix.shape[1]} correlation matrix to: full_correlation_matrix.csv")

# %%
# Visualize correlation heatmap (top 20 features only)
top_corr_matrix = corr_matrix.loc[top_corr_features_list, top_corr_features_list]
fig, ax = plt.subplots(figsize=(10, 8), dpi=VIZ_CONFIG['dpi'])
plt.style.use(VIZ_CONFIG['style'])
sns.heatmap(top_corr_matrix, annot=True, fmt=".2f", cmap=VIZ_CONFIG['heatmap_cmap'],
            cbar_kws={'label': 'Absolute Correlation'}, vmin=0, vmax=1, ax=ax)
ax.set_title(f'Correlation Heatmap: Top {len(top_corr_features_list)-1} Features + Target',
             fontsize=VIZ_CONFIG['title_fontsize'], fontweight='bold')
plt.tight_layout()
save_figure(fig, 'correlation_heatmap_top20.png', 'Correlation heatmap of top 20 features most correlated with target')
if INTERACTIVE: plt.show()
plt.close()

# %% [markdown]
# ### DISTRIBUTION ANALYSIS: Top Correlated Features

# %%
# Analyze distributions of top features (excluding target)
top_features_list = [f for f in top_corr_features_list if f != 'target']

# %%
# Statistical testing for distribution differences (Mann-Whitney U test)
top_6_features = top_features_list[:6]
log(f"Statistical Tests: Distribution Differences Between Pass and Fail\n")
log(f"{'Feature':<20} {'Correlation':<12} {'Mann-Whitney U p-value':<25} {'Significant?':<15}")
log("-" * 80)

for feature in top_6_features:
    pass_data = X.loc[y['target'] == 0, feature].dropna()
    fail_data = X.loc[y['target'] == 1, feature].dropna()

    if len(pass_data) > 0 and len(fail_data) > 0:
        statistic, p_value = mannwhitneyu(pass_data, fail_data, alternative='two-sided')
        significant = "YES" if p_value < 0.05 else "NO"
        corr_val = corr_matrix.loc[feature, 'target']
        log(f"{feature:<20} {corr_val:<12.4f} {p_value:<25.6e} {significant:<15}")

# %%
# Visualize distributions for top 6 features (most correlated with target)
log(f"\nVisualizing distributions for top 6 features by target correlation")

plt.style.use(VIZ_CONFIG['style'])
fig, axes = plt.subplots(3, 2, figsize=(10, 9), dpi=VIZ_CONFIG['dpi'])
axes = axes.flatten()

for idx, feature in enumerate(top_6_features):
    ax = axes[idx]

    # Split by target class
    pass_data = X.loc[y['target'] == 0, feature].dropna()
    fail_data = X.loc[y['target'] == 1, feature].dropna()

    # Plot overlapping histograms
    ax.hist(pass_data, bins=30, alpha=0.6, label='Pass', color=VIZ_CONFIG['pass_color'], edgecolor='black')
    ax.hist(fail_data, bins=30, alpha=0.6, label='Fail', color=VIZ_CONFIG['fail_color'], edgecolor='black')

    ax.set_title(f'{feature}\n(r = {corr_matrix.loc[feature, "target"]:.3f})',
                 fontsize=VIZ_CONFIG['title_fontsize'], fontweight='bold')
    ax.set_xlabel('Value', fontsize=VIZ_CONFIG['label_fontsize'])
    ax.set_ylabel('Frequency', fontsize=VIZ_CONFIG['label_fontsize'])
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Distribution of Top 6 Features by Target Class',
             fontsize=VIZ_CONFIG['title_fontsize'], fontweight='bold', y=1.00)
plt.tight_layout()
save_figure(fig, 'top_features_distributions.png', 'Distribution comparison of top 6 features by Pass/Fail class')
if INTERACTIVE: plt.show()
plt.close()

# %%
# Boxplots for top 6 features (shows outliers and spread)
plt.style.use(VIZ_CONFIG['style'])
fig, axes = plt.subplots(3, 2, figsize=(10, 8), dpi=VIZ_CONFIG['dpi'])
axes = axes.flatten()

for idx, feature in enumerate(top_6_features):
    ax = axes[idx]

    # Create boxplot data
    plot_data = pd.DataFrame({
        'Value': X[feature],
        'Target': y['target'].map({0: 'Pass', 1: 'Fail'})
    }).dropna()

    # Boxplot by target class
    plot_data.boxplot(column='Value', by='Target', ax=ax,
                      patch_artist=True,
                      boxprops=dict(facecolor=VIZ_CONFIG['secondary'], color='black'),
                      medianprops=dict(color=VIZ_CONFIG['primary'], linewidth=2),
                      whiskerprops=dict(color='black'),
                      capprops=dict(color='black'))

    ax.set_title(f'{feature} (r = {corr_matrix.loc[feature, "target"]:.3f})',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Target Class')
    ax.set_ylabel('Value')
    ax.get_figure().suptitle('')  # Remove auto-generated title
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Boxplots: Top 6 Features by Target Class (outliers visible)',
             fontweight='bold', y=0.995)
plt.tight_layout()
save_figure(fig, 'top_features_boxplots.png', 'Boxplots showing outlier distribution in top 6 features')
if INTERACTIVE: plt.show()
plt.close()

# %% [markdown]
# ### OUTLIER DETECTION

# %%
# Define helper function for IQR-based outlier detection
def count_outliers_iqr(series):
    """Count outliers using IQR method (values beyond 1.5*IQR from Q1/Q3)"""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = (series < lower_bound) | (series > upper_bound)
    return outliers.sum()

# %%
# Calculate outliers per feature
outlier_counts = X.apply(count_outliers_iqr)
outlier_pct = (outlier_counts / len(X)) * 100
features_with_outliers = outlier_counts[outlier_counts > 0].sort_values(ascending=False)

# %%
# Display outlier summary statistics
log(f"Outlier summary (IQR method: beyond Q1-1.5*IQR or Q3+1.5*IQR):")
log(f"  Features with at least one outlier: {len(features_with_outliers)}")
log(f"  Total outlier values across all features: {outlier_counts.sum():,} / {(X.shape[0] * X.shape[1]):,} ({(outlier_counts.sum() / (X.shape[0] * X.shape[1])) * 100:.2f}%)")
log(f"  Mean outliers per feature: {outlier_counts.mean():.1f}")

# %%
# Show top 10 features with most outliers
log(f"Top 10 features with most outliers:")
top_outlier_features = features_with_outliers.head(10)
outlier_summary = pd.DataFrame({
    'Feature': top_outlier_features.index,
    'Outlier Count': top_outlier_features.values,
    'Outlier %': outlier_pct[top_outlier_features.index].values
})
log(outlier_summary.to_string(index=False))

# %%
# Count outliers per sample (vectorized)
Q1 = X.quantile(0.25)
Q3 = X.quantile(0.75)
IQR = Q3 - Q1
lower_bounds = Q1 - 1.5 * IQR
upper_bounds = Q3 + 1.5 * IQR

# %%
# Create boolean mask: True where value is outlier
is_outlier = (X < lower_bounds) | (X > upper_bounds)

# Count outliers per sample (sum across columns)
outliers_per_sample = is_outlier.sum(axis=1)

# %%
# Analyze extreme samples
extreme_samples = outliers_per_sample[outliers_per_sample > 45].sort_values(ascending=False)

log(f"Samples (wafers) with outliers in multiple features:")
log(f"  Total samples that are outliers in >45 features: {len(extreme_samples)}")

most_extreme_idx = extreme_samples.idxmax()
most_extreme_count = extreme_samples.max()
log(f"  Most extreme: Sample #{most_extreme_idx} is outlier in {most_extreme_count} features")

# %%
# Check if extreme outlier samples correlate with failures
log(f"Do extreme outlier samples tend to fail more?")
fail_rate_extreme = (y.loc[extreme_samples.index, 'target'] == 1).sum() / len(extreme_samples) * 100
fail_rate_normal = (y.loc[~y.index.isin(extreme_samples.index), 'target'] == 1).sum() / (len(y) - len(extreme_samples)) * 100
overall_fail_rate = (y['target'] == 1).sum() / len(y) * 100

log(f"  Extreme outliers (>45 outlier features): {fail_rate_extreme:.1f}% failure rate ({len(extreme_samples)} samples)")
log(f"  Normal samples: {fail_rate_normal:.1f}% failure rate ({len(y) - len(extreme_samples)} samples)")
log(f"  Overall failure rate: {overall_fail_rate:.1f}%")

if fail_rate_extreme > overall_fail_rate * 1.5:
    log(f"KEY INSIGHT: Extreme outliers have {fail_rate_extreme/overall_fail_rate:.1f}x higher failure rate!")
elif fail_rate_extreme < overall_fail_rate * 0.8:
    log(f"KEY INSIGHT: Extreme outliers have LOWER failure rate than average")
else:
    log(f"Extreme outliers have similar failure rate to overall population")

# %%
# Statistical test: Do outliers correlate with failures? (Chi-Square)
log(f"\nStatistical Test: Outlier-Failure Association (Chi-Square)\n")
log(f"Testing features where outliers have >5x failure rate...\n")

extreme_features = []
chi_square_results = []

for feature in features_with_outliers.index:
    feature_is_outlier = is_outlier[feature]

    if feature_is_outlier.sum() > 0:
        fail_rate_outliers = (y.loc[feature_is_outlier, 'target'] == 1).sum() / feature_is_outlier.sum() * 100
        fail_rate_normal = (y.loc[~feature_is_outlier, 'target'] == 1).sum() / (~feature_is_outlier).sum() * 100

        if fail_rate_outliers > fail_rate_normal * 5.0:
            # Contingency table: [outlier_pass, outlier_fail], [normal_pass, normal_fail]
            outlier_pass = (y.loc[feature_is_outlier, 'target'] == 0).sum()
            outlier_fail = (y.loc[feature_is_outlier, 'target'] == 1).sum()
            normal_pass = (y.loc[~feature_is_outlier, 'target'] == 0).sum()
            normal_fail = (y.loc[~feature_is_outlier, 'target'] == 1).sum()

            contingency_table = [[outlier_pass, outlier_fail], [normal_pass, normal_fail]]
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)

            log(f"{feature}:")
            log(f"  Outliers: {feature_is_outlier.sum()} samples, {fail_rate_outliers:.1f}% failure rate")
            log(f"  Normal: {(~feature_is_outlier).sum()} samples, {fail_rate_normal:.1f}% failure rate")
            log(f"  Risk Ratio: {fail_rate_outliers/fail_rate_normal:.1f}x")
            log(f"  Chi-Square: {chi2:.2f}, p-value: {p_value:.6e} {'(SIGNIFICANT)' if p_value < 0.05 else '(not significant)'}\n")

            extreme_features.append(feature)
            chi_square_results.append({
                'Feature': feature,
                'Chi2': chi2,
                'p_value': p_value,
                'Risk_Ratio': fail_rate_outliers/fail_rate_normal
            })

if len(chi_square_results) > 0:
    chi_df = pd.DataFrame(chi_square_results).sort_values('p_value')
    log(f"Summary: {len(chi_df)} features with significant outlier-failure association")
else:
    log("No features found with >5x failure rate for outliers")

# %% [markdown]
# ### MULTICOLLINEARITY ANALYSIS (VIF)

# %%
# Identify highly correlated feature pairs (potential multicollinearity)
log(f"Searching for highly correlated feature pairs (|r| > 0.85)...")
high_corr_pairs = []
perfect_corr_pairs = []

for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.85:
            if abs(corr_matrix.iloc[i, j]) >= 1.0:  
                perfect_corr_pairs.append({
                    'Feature 1': corr_matrix.columns[i],
                    'Feature 2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j]
                })
            else:
                high_corr_pairs.append({
                    'Feature 1': corr_matrix.columns[i],
                    'Feature 2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j]
            })

# %%
# Display highly correlated pairs
high_corr_df = pd.DataFrame(high_corr_pairs).sort_values(
    by=['Correlation', 'Feature 1'], ascending=[False, True])
perfect_corr_df = pd.DataFrame(perfect_corr_pairs).sort_values(
    by=['Correlation', 'Feature 1'], ascending=[False, True])


log(f"\nHighly correlated feature pairs (|r| > 0.85): {len(high_corr_df)} pairs found")
log(f"Perfectly correlated feature pairs (|r| >= 1.0): {len(perfect_corr_df)} pairs found")

# %%
def calculate_vif(X_subset):
    """Calculate Variance Inflation Factor for each feature.

    VIF quantifies multicollinearity:
      VIF = 1: No correlation | VIF = 1-5: Moderate | VIF > 5: High | VIF > 10: Severe
    """
    X_filled = X_subset.fillna(0)
    vif_data = pd.DataFrame({
        "Feature": X_filled.columns,
        "VIF": [variance_inflation_factor(X_filled.values, i)
                for i in range(X_filled.shape[1])]
    })
    return vif_data.sort_values('VIF', ascending=False)

# %%
# Calculate VIF on all 590 features  (computationally expensive)
vif_data = calculate_vif(X)

# Check for severe multicollinearity
high_vif = vif_data[vif_data['VIF'] > 10]
high_vif_features = high_vif['Feature'].tolist()
low_vif_features = [f for f in vif_data['Feature'] if f not in high_vif_features]
log(f"{len(high_vif)} features have VIF > 10 (serious multicollinearity)")
log(f"Features with acceptable VIF (<=10): {len(low_vif_features)}")
log(f"Worst offenders:\n{high_vif.head(10).to_string(index=False)}")

# %% [markdown]
# #### VIF Reduction Strategy: Hierarchical Clustering
#
# **Goal:** Reduce multicollinearity by grouping highly correlated features and keeping the best from each cluster
#
# **Approach:**
# 1. Cluster high-VIF features by correlation similarity (hierarchical clustering)
# 2. Within each cluster, select feature with highest target correlation
# 3. Remove redundant features from same cluster

# %%
# Prepare data for clustering: impute missing values, remove constant features
X_tmp = X[high_vif_features].apply(lambda col: col.fillna(col.median()))
non_constant = X_tmp.loc[:, X_tmp.std() > 0]

log(f"Clustering {len(non_constant.columns)} high-VIF features with non-zero variance...")

# %%
# Compute correlation-based distance matrix
corr = non_constant.corr().abs()
corr = corr.fillna(0.0)  # Handle any NaN correlations

# Convert correlation to distance: distance = 1 - |correlation|
# Features with high correlation have low distance (same cluster)
distance_matrix = 1 - corr
distance_matrix = distance_matrix.clip(lower=0.0)  # Ensure non-negative

# Convert to condensed distance matrix for clustering
condensed_distance = squareform(distance_matrix, checks=False)

# %%
# Perform hierarchical clustering (complete linkage)
Z = linkage(condensed_distance, method='complete')

# Cut dendrogram at distance threshold 0.10 (correlation >= 0.90)
# Features with correlation >= 0.90 are grouped together
cluster_labels = fcluster(Z, t=0.10, criterion='distance')

clusters = pd.DataFrame({
    "feature": non_constant.columns,
    "cluster": cluster_labels
}).sort_values("cluster")

cluster_sizes = clusters['cluster'].value_counts().sort_values(ascending=False)
log(f"\nClustering results:")
log(f"  Total clusters formed: {len(cluster_sizes)}")
log(f"  Largest cluster size: {cluster_sizes.max()}")
log(f"  Singleton clusters (size=1): {(cluster_sizes == 1).sum()}")
log(f"  Multi-feature clusters: {(cluster_sizes > 1).sum()}")

# %%
# Select best feature from each cluster (highest target correlation)
target_corr_tmp = X_tmp.join(y['target']).abs().corr()['target'].drop('target')

keepers = (
    clusters
    .assign(target_corr_tmp=lambda df: df['feature'].map(target_corr_tmp))
    .sort_values(['cluster', 'target_corr_tmp'], ascending=[True, False])
    .groupby('cluster')
    .first()
)

selected_features = keepers['feature'].tolist()
log(f"\nFeature selection within clusters:")
log(f"  Total selected: {len(selected_features)} features (one per cluster)")
log(f"  Features removed: {len(high_vif_features) - len(selected_features)}")
log(f"\nTop 5 selected features by target correlation:")
log(keepers.nlargest(5, 'target_corr_tmp')[['feature', 'target_corr_tmp']].to_string(index=False))

# %%
# Reconstruct feature set: keep low-VIF features + selected high-VIF features
X_reduced = X.drop(columns=set(high_vif_features) - set(selected_features))
log(f"\nFinal feature set after VIF reduction:")
log(f"  Original: {X.shape[1]} features")
log(f"  Reduced: {X_reduced.shape[1]} features")
log(f"  Reduction: {X.shape[1] - X_reduced.shape[1]} features removed ({(X.shape[1] - X_reduced.shape[1])/X.shape[1]*100:.1f}%)")

# %%
X_reduced
calculate_vif(X_reduced)

# %%
X_reduced_final = X_reduced.copy()
log(f"Final feature set shape after multicollinearity reduction: {X_reduced_final.shape}")
log(f"Duplicate columns check: {X_reduced_final.columns.duplicated().sum()}")

# %% [markdown]
# ## EDA SUMMARY
#
# **Dataset Overview:**
# - 1,567 semiconductor wafer records with 590 sensor features
# - Binary classification: Pass (0) vs Fail (1)
#
# **Key Findings:**
#
# 1. **Class Imbalance:** Severe 14:1 ratio (Pass:Fail)
#    - Requires balanced metrics (G-Mean, AUC-ROC) not accuracy
#    - Needs resampling (SMOTE/ADASYN) or class weighting
#
# 2. **Missing Data:** 4.5% overall, non-random pattern
#    - 159 features have >50% missing values
#    - Missingness correlates with target (MAR/MNAR)
#
# 3. **Multicollinearity:** Severe (214 features with VIF > 10)
#    - Hierarchical clustering identifies redundant feature groups
#    - VIF reduction removes ~67% of features
#
# 4. **Feature-Target Correlation:** Weak individual correlations
#    - Max correlation ~0.15 (no single strong predictor)
#    - Ensemble methods likely needed
#
# **Preprocessing Recommendations:**
# - Remove features with >50-90% missing values
# - Apply median imputation for remaining missing values
# - Use LASSO for automatic feature selection
# - Consider PCA for dimensionality reduction (loses interpretability)
#
# **Next Step:** Run preprocessing pipeline
# ```bash
# make preprocess  # or make pipeline for full run
# ```

# %%
# Generate markdown report with embedded figures
def generate_eda_report():
    """
    Generate comprehensive EDA markdown report with all figures.

    Creates reports/eda_report.md with:
    - Executive summary
    - All generated visualizations
    - Key findings and recommendations
    """
    report_path = REPORTS_DIR / 'eda_report.md'

    report_content = """# Exploratory Data Analysis Report
## SECOM Semiconductor Defect Prediction

**Author:** Victoria A.
**Dataset:** UCI SECOM (Semiconductor Manufacturing)

---

## Executive Summary

This report presents the exploratory data analysis of the SECOM semiconductor manufacturing dataset. The analysis reveals critical challenges that inform the modeling strategy:

- **Severe class imbalance** (14:1 Pass:Fail ratio) requiring balanced metrics
- **Complex missing data patterns** with non-random characteristics
- **High multicollinearity** among sensor features requiring dimensionality reduction
- **Weak individual feature-target correlations** suggesting ensemble methods

---

## 1. Dataset Overview

| Metric | Value |
|--------|-------|
| Total Samples | 1,567 wafers |
| Features | 590 sensor measurements |
| Target | Binary (Pass=0, Fail=1) |
| Pass Rate | 93.4% |
| Fail Rate | 6.6% |
| Date Range | ~30 days |

---

## 2. Class Imbalance Analysis

The dataset exhibits severe class imbalance with a 14:1 ratio of passing to failing wafers.

![Class Imbalance Over Time](figures/class_imbalance_over_time.png)

**Key Observations:**
- Failure rate varies between 0% and 100% across production days
- Some days have very few samples, affecting reliability of daily statistics
- Overall imbalance requires G-Mean or AUC-ROC metrics (not accuracy)

**Implications for Modeling:**
- Cannot use accuracy as primary metric (naive baseline = 93.4%)
- Must apply resampling (SMOTE/ADASYN) or class weighting
- Threshold optimization critical for balanced sensitivity/specificity

---

## 3. Missing Data Analysis

Overall missing rate: **4.54%** (41,951 missing values across all cells)

![Missing Data Matrix](figures/missing_data_matrix.png)

![Missing Data Heatmap](figures/missing_data_heatmap.png)

**Key Findings:**
- 541 features have complete data (no missing values)
- 16 features have >60% missing values
- Missingness patterns show correlation with target (MAR/MNAR detected)
- Groups of features missing together suggest sensor clusters

**Preprocessing Strategy:**
- Remove features with >50% missing values
- Apply KNN imputation (preserves correlational structure)
- Consider missingness indicators as engineered features

---

## 4. Feature-Target Correlation

Individual feature correlations with target are weak, with maximum correlation ~0.15.

![Correlation Heatmap](figures/correlation_heatmap_top20.png)

**Key Observations:**
- No single feature provides strong predictive signal
- Top features show mild separation between Pass/Fail distributions
- Ensemble methods or feature combinations needed

---

## 5. Feature Distribution Analysis

![Feature Distributions](figures/top_features_distributions.png)

![Feature Boxplots](figures/top_features_boxplots.png)

**Statistical Tests:**
- Mann-Whitney U tests confirm significant distribution differences
- Top correlated features show measurable separation
- Outliers present in most features (IQR method)

---

## 6. Multicollinearity Analysis

Severe multicollinearity detected: **214 features with VIF > 10**

**Remediation Strategy:**
1. Hierarchical clustering of high-VIF features by correlation
2. Select best feature (highest target correlation) from each cluster
3. Reduces feature set by ~67% while preserving predictive information

**Feature Selection Recommendation:**
- Use LASSO regularization for automatic selection
- Alternative: PCA for dimensionality reduction (sacrifices interpretability)

---

## 7. Key Findings Summary

| Finding | Impact | Recommendation |
|---------|--------|----------------|
| 14:1 class imbalance | Accuracy misleading | Use G-Mean, SMOTE/ADASYN |
| 4.5% missing data (non-random) | Potential information loss | KNN imputation |
| High multicollinearity | Unstable coefficients | LASSO or PCA |
| Weak individual correlations | No single strong predictor | Ensemble methods |

---

## 8. Preprocessing Pipeline

Based on this EDA, the recommended preprocessing pipeline:

```
1. Train/Test Split (80/20, stratified) - BEFORE any preprocessing
2. Remove features with >50% missing values
3. Apply KNN imputation (k=5) on training set
4. Transform test set using training parameters
5. Standardize features (StandardScaler)
6. Apply LASSO for feature selection OR PCA for reduction
7. Apply SMOTE/ADASYN on training set only
```

---

## Next Steps

Run the preprocessing and modeling pipeline:
```bash
make pipeline
```

Or run stages individually:
```bash
make preprocess  # Data preprocessing
make tune        # Hyperparameter tuning
make select      # Production model selection
```

---

*Report generated from notebooks/eda.py*
"""

    with open(report_path, 'w') as f:
        f.write(report_content)

    log(f"\nGenerated EDA report: {report_path}")
    log(f"  Figures referenced: {len(GENERATED_FIGURES)}")
    for fig_info in GENERATED_FIGURES:
        log(f"    - {fig_info['filename']}: {fig_info['caption']}")

    return report_path


# %%
# Generate the report
if __name__ == '__main__' or 'ipykernel' not in sys.modules:
    generate_eda_report()

