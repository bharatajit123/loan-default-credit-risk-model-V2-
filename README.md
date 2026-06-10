# Loan Default Prediction Model (V2)

## 📋 Project Overview

This project builds and compares multiple machine learning models to predict loan defaults. The analysis uses a dataset of 255,347 loan records with 18 features to identify borrowers at risk of default, enabling proactive risk management and better lending decisions.

---

## 📊 Dataset Summary

### Dataset Characteristics
- **Total Records:** 255,347 loans
- **Features:** 18 columns (16 features + 1 ID + 1 target)
- **Target Variable:** `Default` (binary: 0 = No Default, 1 = Default)
- **Data Quality:** No missing values
- **Class Distribution:** 
  - No Default: 225,694 (88.39%)
  - Default: 29,653 (11.61%)
  - ⚠️ **Note:** Imbalanced dataset - consider class weighting in modeling

### Feature Categories

#### Numeric Features (9)
| Feature | Range | Mean | Std Dev |
|---------|-------|------|---------|
| Age | 18-69 | 43.5 | 15.0 |
| Income | $15K-$150K | $82.5K | $39K |
| LoanAmount | $5K-$250K | $127.6K | $70.8K |
| CreditScore | 300-849 | 574 | 159 |
| MonthsEmployed | 0-119 | 59.5 | 34.6 |
| NumCreditLines | 1-4 | 2.5 | 1.1 |
| InterestRate | 2%-25% | 13.49% | 6.64% |
| LoanTerm | 12-60 months | 36 | 17 |
| DTIRatio | 0.1-0.9 | 0.5 | 0.23 |

#### Categorical Features (7)
- **Education:** Bachelor's, High School, Master's, PhD (balanced ~64K each)
- **EmploymentType:** Full-time, Part-time, Unemployed, Self-employed (balanced ~64K each)
- **MaritalStatus:** Married, Divorced, Single (balanced ~85K each)
- **LoanPurpose:** Business, Home, Education, Other, Auto (balanced ~51K each)
- **HasMortgage:** Yes/No (balanced ~50/50)
- **HasDependents:** Yes/No (balanced ~50/50)
- **HasCoSigner:** Yes/No (balanced ~50/50)

---

## 🎯 Key Findings from EDA

### Top Predictors of Default (Correlation Analysis)
1. **Age** (-0.168) - Younger borrowers default more frequently
2. **Income** (-0.099) - Lower income correlates with higher default risk
3. **MonthsEmployed** (-0.097) - Shorter employment history = higher risk
4. **InterestRate** (+0.131) - Higher rates indicate higher-risk borrowers
5. **LoanAmount** (+0.087) - Larger loans slightly increase default risk

### Risk Profile Summary
- **High-Risk Borrowers:** Young (<35), low income (<$50K), unemployed/self-employed, high DTI (>0.7)
- **Low-Risk Borrowers:** Mature (>50), high income (>$100K), full-time employed, co-signer present

---

## 🚀 Data Split & Preprocessing

### Train-Test Split
- **Training Set:** 204,277 samples (80%)
- **Test Set:** 51,070 samples (20%)
- **Stratification:** Maintains class distribution in both sets
- **Encoding:** All categorical variables label-encoded

### Class Distribution Maintained
```
Training Set:  180,555 non-defaults (88.39%) | 23,722 defaults (11.61%)
Test Set:      45,139 non-defaults (88.39%) | 5,931 defaults (11.61%)
```

---

## 📈 Model Performance Comparison

### Overall Results (Test Set)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Gradient Boosting** ⭐ | 88.62% | 57.79% | 7.57% | 0.1339 | **0.7560** |
| **Random Forest** | 88.57% | 63.71% | 3.76% | 0.0710 | 0.7464 |
| **Logistic Regression** | 88.50% | 61.86% | 2.46% | 0.0473 | 0.7466 |
| **Naive Bayes** | 88.42% | 61.90% | 0.66% | 0.0130 | 0.7435 |
| **Decision Tree** | 85.72% | 25.63% | 12.06% | 0.1640 | 0.6074 |

### 🏆 Best Models by Metric
- **Best ROC-AUC:** Gradient Boosting (0.7560) ✓ Best discrimination
- **Best Accuracy:** Gradient Boosting (88.62%)
- **Best Precision:** Random Forest (63.71%) ✓ Fewest false alarms
- **Best Recall:** Decision Tree (12.06%) - but poor precision
- **Best Balance:** Gradient Boosting - highest ROC-AUC with decent recall

---

## 🎚️ Threshold Analysis (Logistic Regression)

### Threshold Performance Trade-offs

| Threshold | Accuracy | Precision | Recall | F1-Score | Defaults Flagged |
|-----------|----------|-----------|--------|----------|------------------|
| **0.50** | 88.50% | 61.86% | 2.46% | 0.047 | 236 (0.46%) |
| **0.30** | 87.27% | 40.61% | 20.86% | 0.276 | 3,046 (5.96%) |
| **0.20** | 81.83% | 30.20% | 43.06% | 0.355 | 8,458 (16.56%) ⭐ |
| **0.15** | 74.76% | 24.96% | 58.51% | 0.350 | 13,900 (27.22%) |
| **0.10** | 61.45% | 19.82% | 76.14% | 0.315 | 22,787 (44.62%) |

### Recommended Threshold: **0.20**
- ✅ Catches 43% of defaults (vs 2.5% at default 0.5)
- ✅ Maintains 81.83% accuracy
- ✅ Reasonable false positive rate (30% precision)
- ✅ Manageable for manual review (8,458 flagged)

---

## 🎯 Feature Importance (Logistic Regression)

### Top 10 Most Influential Features

| Rank | Feature | Coefficient | Direction | Impact |
|------|---------|-------------|-----------|--------|
| 1 | EmploymentType | +0.1124 | ↑ | More defaults with unemployment/self-employment |
| 2 | Education | -0.0823 | ↓ | Higher education = lower risk |
| 3 | HasCoSigner | -0.0806 | ↓ | Co-signer reduces default risk |
| 4 | HasDependents | -0.0778 | ↓ | Dependents reduce default risk |
| 5 | NumCreditLines | +0.0662 | ↑ | More credit lines = higher risk |
| 6 | InterestRate | +0.0599 | ↑ | Higher rates = higher risk |
| 7 | HasMortgage | -0.0501 | ↓ | Mortgage ownership = lower risk |
| 8 | Age | -0.0387 | ↓ | Older borrowers = lower risk |
| 9 | LoanPurpose | -0.0367 | ↓ | Purpose matters (business/auto safer) |
| 10 | MaritalStatus | -0.0312 | ↓ | Married status = lower risk |

---

## 📁 Directory Structure

```
Loan-Default-Model(V2)/
├── data/
│   └── loans.csv                          # Main dataset (255K records)
├── models/
│   ├── logistic_regression_model.pkl      # Saved model files
│   ├── random_forest_model.pkl
│   ├── gradient_boosting_model.pkl
│   └── model_comparison.json              # Model metadata
├── notebooks/
│   ├── 01_EDA.ipynb                       # Exploratory Data Analysis
│   ├── 02_model_training.ipynb            # Model development
│   └── 03_threshold_optimization.ipynb    # Threshold tuning
├── reports/
│   ├── accuracy_comparison.png            # Model accuracy bar chart
│   ├── metrics_heatmap.png                # Performance metrics heatmap
│   ├── roc_curves_comparison.png          # ROC curves overlay
│   ├── precision_recall_f1_comparison.png # Detailed metrics
│   ├── confusion_matrices_all_models.png  # Confusion matrices
│   ├── all_metrics_comparison.png         # Comprehensive comparison
│   ├── feature_importance.png             # Feature coefficients
│   ├── threshold_analysis.png             # Threshold performance
│   ├── probability_distribution.png       # Model calibration
│   ├── model_report.md                    # Full analysis report
│   └── ANALYSIS_SUMMARY.md                # Quick reference guide
├── src/
│   ├── preprocess.py                      # Data preprocessing utilities
│   ├── model.py                           # Model training functions
│   ├── evaluate.py                        # Evaluation metrics
│   └── utils.py                           # Helper functions
├── requirements.txt                       # Python dependencies
├── README.md                              # This file
└── DEPLOYMENT_GUIDE.md                    # Production deployment
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

```bash
# Clone or navigate to project
cd Loan-Default-Model\(V2\)

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import pandas, sklearn, matplotlib; print('All packages installed!')"
```

---

## 💻 Usage

### Quick Start - Training Models

```python
from src.model import train_all_models
from src.preprocess import load_and_preprocess

# Load and preprocess data
X_train, X_test, y_train, y_test = load_and_preprocess('data/loans.csv')

# Train all models
models = train_all_models(X_train, y_train)

# Evaluate
from src.evaluate import evaluate_models
results = evaluate_models(models, X_test, y_test)
```

### Making Predictions

```python
from src.model import load_model
import pandas as pd

# Load trained model
model = load_model('models/gradient_boosting_model.pkl')

# Prepare new loan data
new_loan = pd.DataFrame({
    'Age': [35],
    'Income': [60000],
    'LoanAmount': [100000],
    # ... other features
})

# Get prediction
default_probability = model.predict_proba(new_loan)[0, 1]
print(f"Default Risk: {default_probability:.2%}")

# Apply threshold
is_risky = default_probability > 0.20
print(f"Flagged for Review: {is_risky}")
```

### Threshold Tuning

```python
from src.evaluate import find_optimal_threshold

# Find best threshold for your use case
optimal_threshold = find_optimal_threshold(
    y_test, 
    y_pred_proba,
    metric='f1',  # or 'precision', 'recall', 'balanced_accuracy'
    beta=1.0
)
```

---

## 📊 Visualization Guide

### Key Visualizations in `reports/`

1. **accuracy_comparison.png** - Quick accuracy overview across all models
2. **metrics_heatmap.png** - Comprehensive color-coded performance matrix
3. **roc_curves_comparison.png** - Discrimination ability comparison (larger AUC = better)
4. **roc_curves_comparison.png** - See which model best separates defaults
5. **confusion_matrices_all_models.png** - True positives/negatives for each model
6. **feature_importance.png** - Which features matter most for predictions

---

## 🎓 Model Recommendations

### For Production Deployment
**Use: Gradient Boosting Classifier**
- ✅ Highest ROC-AUC (0.7560) - best discrimination
- ✅ Balanced performance (88.62% accuracy)
- ✅ Catches 7.57% of defaults (vs 2.46% at default threshold)
- ✅ Handles non-linear relationships well
- ⚠️ More complex, slower predictions

### For Interpretability
**Use: Logistic Regression**
- ✅ Simple, fast, interpretable coefficients
- ✅ Good ROC-AUC (0.7466) - comparable to Gradient Boosting
- ✅ Easy to explain to business stakeholders
- ✅ Fast inference on large batches
- ⚠️ Assumes linear relationships

### For Risk-Averse Lending
**Use: Random Forest**
- ✅ Highest precision (63.71%) - fewer false positives
- ✅ Strong accuracy (88.57%)
- ✅ Robust to outliers and feature scaling
- ⚠️ Less recall for defaults

---

## ⚠️ Important Considerations

### Class Imbalance
- Dataset has 88.4% non-defaults vs 11.6% defaults
- Default threshold of 0.5 catches only 2.46% of actual defaults
- **Solution:** Lower threshold to 0.15-0.20 for better recall

### Model Limitations
- Models trained on historical data - may not capture market shifts
- External factors (economy, policy) not captured in features
- Regular retraining recommended (quarterly or semi-annually)

### Recommended Next Steps
1. **Feature Engineering:** Create interaction features (e.g., Age × Income)
2. **Class Balancing:** Try SMOTE or class weighting
3. **Ensemble Methods:** Stack multiple models for better performance
4. **Threshold Optimization:** Adjust based on business cost-benefit analysis
5. **A/B Testing:** Test in production with control group
6. **Monitoring:** Track model drift and performance over time

---

## 📈 Performance Metrics Explained

- **Accuracy:** Overall correctness (88.5%) - biased toward majority class
- **Precision:** Of predicted defaults, how many are true defaults (61.8%)
- **Recall:** Of actual defaults, how many we catch (2.46%) - **KEY METRIC**
- **F1-Score:** Harmonic mean of precision & recall (0.047)
- **ROC-AUC:** Discrimination ability (0.7466) - **BEST OVERALL METRIC**

**For Loan Defaults: Recall is critical!** Missing defaults is costly.

---

## 🔗 Related Files

- **ANALYSIS_SUMMARY.md** - Quick reference guide
- **model_comparison.json** - Detailed metrics for all models
- **DEPLOYMENT_GUIDE.md** - Production deployment steps

---

## 👨‍💻 Model Development Timeline

- **Phase 1:** Exploratory Data Analysis (EDA)
- **Phase 2:** Data Preprocessing & Feature Engineering
- **Phase 3:** Model Training (5 algorithms)
- **Phase 4:** Threshold Optimization
- **Phase 5:** Visualization & Documentation
- **Phase 6:** Model Comparison & Selection

---

## 📞 Contact & Questions

For questions about this project, refer to the detailed analysis reports in `reports/` directory.

---

**Last Updated:** June 10, 2026  
**Dataset Size:** 255,347 records | **Models Compared:** 5 | **Best Model:** Gradient Boosting (ROC-AUC: 0.7560)
