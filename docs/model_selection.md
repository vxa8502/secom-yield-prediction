# Model Selection Rationale

This document explains the technical decisions behind the SECOM yield prediction model architecture.

---

## Why Tree-Based Models Over LSTM

### The Data Characteristics

The SECOM dataset presents **tabular sensor snapshots**, not time series:
- Each row represents a single wafer's sensor readings at one point in time
- No temporal dependencies between rows (wafers are independent)
- 590 features from correlated process parameters
- 1,567 samples total (small for deep learning)

### Evidence Against Sequential Models

**1. Tabular Data Performance (Grinsztajn et al., NeurIPS 2022)**

"Why do tree-based models still outperform deep learning on tabular data?" established that:
- Tree-based models (XGBoost, Random Forest) remain state-of-the-art for medium-sized tabular data
- Deep learning excels when data has spatial/temporal structure (images, sequences)
- Gradient boosting handles irrelevant features and missing values robustly

**2. Sample Size Constraints**

- SECOM: 1,567 samples with 590 features
- LSTM/Transformer architectures typically require 10,000+ samples
- Overfitting risk is severe with deep architectures on small tabular datasets

**3. Missing Value Handling**

- XGBoost handles missing values natively (learns optimal split directions)
- LSTM requires imputation, potentially introducing bias
- Our 4.5% missing rate would require preprocessing for neural approaches

**4. Training Efficiency**

Published SECOM benchmarks show:
- XGBoost: ~3 seconds training time
- LSTM approaches: ~10 seconds training time (3.2x slower)
- For Optuna optimization (100+ trials), this difference compounds significantly

### When LSTM Would Be Appropriate

LSTM/sequential models would be preferred if:
- Data represented wafer trajectories over multiple process steps
- Predicting time-to-failure rather than pass/fail classification
- Strong temporal autocorrelation existed in sensor readings
- Dataset size exceeded 10,000+ samples

For SECOM's snapshot-based quality classification, tree-based methods are the correct choice.

---

## Why SVM Over XGBoost (Final Selection)

### Experimental Results

| Model | Feature Set | Sampling | CV G-Mean |
|-------|-------------|----------|-----------|
| **SVM (RBF)** | LASSO | ADASYN | **71.0%** |
| LogReg | LASSO | native | 69.7% |
| LogReg | LASSO | ADASYN | 69.3% |
| SVM (linear) | LASSO | native | 69.0% |

SVM with RBF kernel and ADASYN resampling achieved the highest CV G-Mean (71.0%) on LASSO-selected features (8 features).

### Why SVM Performed Best

**1. Small Feature Space After LASSO**
- LASSO reduced 590 features to 8 critical parameters
- In low-dimensional spaces, SVM's kernel trick effectively captures nonlinear boundaries
- XGBoost's ensemble approach provides less advantage with few features

**2. Class Imbalance Handling**
- SVM combined with ADASYN resampling outperformed native class weighting
- ADASYN focuses on hard-to-learn minority samples near decision boundary
- Native handling competitive but ADASYN provided 2% G-Mean improvement

**3. Margin Maximization**
- SVM's maximum margin principle provides good generalization
- Robust to small perturbations in feature values
- Less prone to overfitting than deep trees on small datasets

### XGBoost Remains Valid

XGBoost showed strong performance (68.5% CV G-Mean) and offers advantages:
- Native SHAP support (TreeExplainer is faster than KernelExplainer)
- Better scalability if feature set grows
- Handles missing values without imputation

For deployment with more features or larger datasets, XGBoost would be reconsidered.

---

## Why G-Mean Over Accuracy

### The Class Imbalance Problem

SECOM exhibits 14:1 class imbalance:
- 1,463 passes (93.4%)
- 104 failures (6.6%)

A naive model predicting "pass" for everything achieves **93.4% accuracy** while detecting **zero defects**.

### G-Mean Definition

G-Mean = sqrt(Sensitivity x Specificity)

Where:
- Sensitivity (Recall) = TP / (TP + FN) = Defect detection rate
- Specificity = TN / (TN + FP) = Correct pass rate

### Why G-Mean Is Appropriate

**1. Balanced Penalty for Both Error Types**
- G-Mean punishes models that sacrifice minority class detection
- A model with 100% specificity but 0% sensitivity scores G-Mean = 0
- Forces optimization to balance both classes

**2. Manufacturing Context**
- False negatives (missed defects) = Defective wafers reach customers
- False positives (false alarms) = Unnecessary rework, reduced throughput
- Both matter; neither should be ignored

**3. Threshold Independence**
- Unlike F1-score, G-Mean doesn't favor precision over recall
- Appropriate when positive/negative misclassification costs are similar

### Alternative Metrics Considered

| Metric | Formula | Issue for SECOM |
|--------|---------|-----------------|
| Accuracy | (TP+TN)/Total | Dominated by majority class |
| F1-Score | 2*P*R/(P+R) | Ignores true negatives |
| AUC-ROC | Area under ROC | Threshold-independent (less actionable) |
| AUC-PR | Area under PR curve | Better for imbalanced, but less interpretable |

G-Mean was selected as the primary optimization target, with AUC-ROC/AUC-PR tracked for monitoring.

---

## Why Threshold Optimization

### Default Threshold Problem

Sklearn's `predict()` uses threshold=0.5 by default. For imbalanced data:
- Probability scores skew toward majority class
- 0.5 threshold may classify all samples as "pass"
- Without resampling, optimal threshold is often much lower (0.05-0.20)
- With ADASYN resampling, probability distributions are rebalanced and optimal threshold is closer to 0.5

### Our Approach

1. Train model with cross-validation
2. Generate out-of-fold probability predictions
3. Search thresholds from 0.01 to 0.99 in 0.01 steps
4. Select threshold maximizing CV G-Mean

**Result:** Optimal threshold = 0.54 (close to default, due to ADASYN balancing)

### Why CV-Based Threshold Tuning

- Uses cross_val_predict to get out-of-fold predictions
- Prevents data leakage (threshold not tuned on test set)
- Represents realistic deployment performance

---

## What We Tried and Rejected

### Models Rejected

| Model | Issue |
|-------|-------|
| LSTM | Inappropriate for tabular data; insufficient samples |
| Deep MLP | Overfitting on 1,567 samples; no improvement over SVM |
| Naive Bayes | Assumes feature independence; violated by correlated sensors |
| KNN | Poor performance with high dimensionality |

### Preprocessing Rejected

| Approach | Issue |
|----------|-------|
| PCA-only | Lower G-Mean than LASSO; sacrificed interpretability |
| Mean imputation | Worse than KNN imputation for correlated features |
| No feature selection | 590 features caused overfitting; slower training |

### Sampling Strategies Tested

| Strategy | Result |
|----------|--------|
| ADASYN | Best for SVM (71.0% G-Mean) |
| Native (class_weight) | Competitive (68.8-69.7% G-Mean) |
| SMOTE | Terminated after 60 hours (computationally infeasible for SVM + large features) |

ADASYN with SVM (RBF kernel) proved most effective for this dataset.

---

## Sampling Strategy Comparison (Final Results)

| Model | Features | Sampling | Kernel | CV G-Mean | Threshold |
|-------|----------|----------|--------|-----------|-----------|
| SVM   | LASSO (8) | ADASYN  | RBF    | **71.0%** | 0.54 |
| SVM   | LASSO (8) | native  | linear | 68.8%     | 0.08 |
| SVM   | LASSO (8) | SMOTE   | -      | N/A (terminated) | - |

### Analysis Notes
- SMOTE tuning terminated after 60 hours (computationally infeasible for SVM + PCA/all features)
- 24 of 36 experiments completed; LASSO feature set completed for all samplers
- ADASYN outperformed native class weighting by ~2% G-Mean
- Park et al. (2024) benchmark: 72.95% G-Mean with SVM+ADASYN+MaxAbs
- Our result: **72.2% Test G-Mean** (99% of benchmark)

---

## References

1. Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? NeurIPS 2022.

2. Park, S., et al. (2024). Effect of data preprocessing on the classification of the SECOM dataset. Sensors.

3. He, H., & Garcia, E. A. (2009). Learning from imbalanced data. IEEE TKDE.
