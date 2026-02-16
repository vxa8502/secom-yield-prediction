# Data Card: SECOM Semiconductor Manufacturing Dataset

This data card documents the dataset used for the SECOM yield prediction model, following best practices for ML dataset documentation.

---

## Dataset Overview

| Attribute | Value |
|-----------|-------|
| **Name** | SECOM (Semiconductor Manufacturing) |
| **Source** | UCI Machine Learning Repository |
| **URL** | https://archive.ics.uci.edu/ml/datasets/SECOM |
| **Domain** | Semiconductor manufacturing quality control |
| **Task** | Binary classification (Pass/Fail) |
| **Samples** | 1,567 wafers |
| **Features** | 590 anonymized sensor measurements |
| **Collection Period** | ~30 days (dates anonymized) |

---

## Data Provenance

### Original Collection

- Collected from a real semiconductor manufacturing line
- Features represent FDC (Fault Detection and Classification) sensor readings
- Each row is one wafer's sensor snapshot at end-of-line testing
- Labels determined by physical quality inspection

### Dataset Publication

- Published by McCann & Johnston (2008)
- Part of UCI ML Repository since 2008
- Widely used benchmark for imbalanced classification research

### Our Processing

- Downloaded from UCI repository (secom.data, secom_labels.data)
- No modifications to raw data values
- Preprocessing applied after train/test split to prevent leakage

---

## Data Characteristics

### Class Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| Pass (0) | 1,463 | 93.4% |
| Fail (-1/1) | 104 | 6.6% |

**Imbalance Ratio:** 14:1 (Pass:Fail)

### Missing Data

| Metric | Value |
|--------|-------|
| Total missing cells | 41,951 |
| Overall missing rate | 4.54% |
| Features with 0% missing | 541 |
| Features with >60% missing | 16 |

**Missing Data Pattern:** Not completely random (MCAR violated). Some features have correlated missingness, suggesting sensor cluster failures or process-specific data collection gaps.

### Feature Characteristics

- All features are continuous (float)
- Features are anonymized (no semantic labels)
- High multicollinearity: 214 features with VIF > 10
- Weak individual correlations with target (max ~0.15)

---

## Preprocessing Decisions

### 1. Train/Test Split (First Step)

**Decision:** 80/20 stratified split performed BEFORE any preprocessing

**Rationale:** Prevents data leakage. Many academic implementations incorrectly apply feature selection or imputation before splitting, leading to inflated performance estimates.

**Implementation:** `train_test_split(stratify=y, test_size=0.2, random_state=42)`

### 2. Missing Value Removal

**Decision:** Remove features with >50% missing values

**Rationale:**
- Features with majority missing provide unreliable signal
- KNN imputation becomes unstable with very sparse features
- Threshold of 50% balances data retention vs. quality

**Result:** ~31 features removed

### 3. Missing Value Imputation

**Decision:** KNN imputation with k=5

**Rationale:**
- Preserves correlational structure better than mean imputation
- Sensor measurements often correlate (physical process relationships)
- k=5 provides smoothing without over-reliance on single neighbors

**Alternative Rejected:** Mean imputation (ignores feature correlations)

### 4. High Correlation Removal

**Decision:** Remove features with >0.95 pairwise correlation

**Rationale:**
- Near-duplicate features add no information
- Reduce dimensionality without losing signal
- 0.95 threshold conservative to preserve distinct measurements

**Result:** Reduces feature redundancy by ~20%

### 5. Feature Selection (LASSO)

**Decision:** L1-regularized logistic regression for feature selection

**Rationale:**
- LASSO naturally selects sparse feature sets
- 590 features too many for interpretability
- Selected features have demonstrated predictive value

**Result:** 8 features selected (feature_21, feature_59, feature_64, feature_75, feature_103, feature_114, feature_129, feature_510)

### 6. Standardization

**Decision:** StandardScaler (z-score normalization)

**Rationale:**
- SVM requires scaled features for RBF kernel
- Ensures no single feature dominates distance calculations
- Applied after train/test split (fit on train, transform both)

---

## Known Limitations

### Data Quality Issues

1. **Anonymized Features:** Cannot validate if selected features are physically meaningful
2. **Unknown Temporal Ordering:** Cannot detect equipment drift or time-based patterns
3. **Single Fab Source:** Results may not generalize to other manufacturing lines
4. **30-Day Window:** Does not capture seasonal or long-term process variation

### Population Biases

1. **Selection Bias:** Only wafers that reached end-of-line testing included; early-stage failures not represented
2. **Survivorship Bias:** Catastrophic failures may be underrepresented if they damaged equipment
3. **Label Noise:** Binary pass/fail may oversimplify (some failures worse than others)

### Missing Data Concerns

1. **MNAR Possibility:** Missingness may correlate with failure modes (sensors fail when process fails)
2. **Imputation Artifacts:** KNN imputation creates synthetic values that may not reflect reality
3. **Cluster Missingness:** Groups of features missing together suggest unmeasured factors

---

## Features Removed and Why

### High Missingness Removal (>50% missing)

Approximately 31 features removed due to excessive missing values. These features cannot be reliably imputed and would introduce noise.

### High Correlation Removal (>0.95 correlation)

Features removed if they were near-duplicates of other features. The feature with higher target correlation was retained from each correlated pair.

### LASSO Zero-Coefficient Features

582 features received zero coefficients from L1 regularization, indicating they provided no additional predictive power beyond the 8 selected features.

**Selected Features (ranked by SHAP importance):**
| Rank | Feature | Mean SHAP | Reason for Selection |
|------|---------|-----------|----------------------|
| 1 | feature_59 | 0.0597 | Highest impact on predictions |
| 2 | feature_129 | 0.0480 | Non-zero LASSO coefficient |
| 3 | feature_103 | 0.0448 | Non-zero LASSO coefficient |
| 4 | feature_21 | 0.0378 | Non-zero LASSO coefficient |
| 5 | feature_64 | 0.0357 | Non-zero LASSO coefficient |
| 6 | feature_510 | 0.0215 | Non-zero LASSO coefficient |
| 7 | feature_75 | 0.0196 | Non-zero LASSO coefficient |
| 8 | feature_114 | 0.0036 | Lowest impact but still selected |

Semantic interpretation not possible due to anonymization.

---

## Ethical Considerations

### Potential Harms

- **False Negatives:** Missed defects reach customers, causing product failures
- **False Positives:** Unnecessary scrapping of good wafers, economic waste
- **Automation Bias:** Over-reliance on model predictions may reduce human oversight

### Mitigation Strategies

1. Model deployed as decision support, not autonomous decision-maker
2. Threshold tuned to balance FN/FP based on business cost analysis
3. SHAP explanations enable human review of flagged wafers
4. Regular retraining recommended (quarterly or on process change)

### Fairness Considerations

Not applicable in traditional sense (no protected demographic groups). However:
- Model should perform consistently across different process conditions
- Missingness-stratified analysis checks for performance gaps

---

## Recommended Use

### Appropriate Uses

- Defect prediction during semiconductor manufacturing
- Process engineer decision support for yield improvement
- Benchmark dataset for imbalanced classification research
- Educational demonstration of ML for manufacturing

### Inappropriate Uses

- Direct deployment without domain expert validation
- Generalization to different semiconductor processes without retraining
- Real-time control systems without additional safety checks
- Replacement for physical quality inspection

---

## Maintenance

### Versioning

- Raw data: secom.data (MD5: available from UCI)
- Processed data: X_train_lasso.npy, X_test_lasso.npy, etc.
- Processing code: pipelines/preprocess.py (version controlled)

### Update Frequency

Dataset is static (historical). Model should be retrained:
- When new process data becomes available
- After equipment or recipe changes
- If prediction accuracy degrades in production

---

## References

1. McCann, M., & Johnston, A. (2008). SECOM Dataset. UCI Machine Learning Repository.

2. Park, S., et al. (2024). Effect of data preprocessing on the classification of the SECOM dataset. Sensors.

3. Gebru, T., et al. (2021). Datasheets for Datasets. Communications of the ACM.
