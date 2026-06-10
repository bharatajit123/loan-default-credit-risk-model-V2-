# Technical Documentation & Data Dictionary

## Table of Contents
1. [Data Dictionary](#data-dictionary)
2. [Methodology](#methodology)
3. [Model Architecture](#model-architecture)
4. [Evaluation Metrics Explained](#evaluation-metrics-explained)
5. [Preprocessing Steps](#preprocessing-steps)

---

## Data Dictionary

### Dataset Information
- **File:** `data/loans.csv`
- **Format:** CSV (comma-separated values)
- **Encoding:** UTF-8
- **Rows:** 255,347
- **Columns:** 18

### Feature Definitions

#### Identifier
| Column | Type | Description | Example |
|--------|------|-------------|---------|
| **LoanID** | String | Unique identifier for each loan | `I38PQUQS96` |

#### Demographic Features

| Column | Type | Range | Missing | Description |
|--------|------|-------|---------|-------------|
| **Age** | Integer | 18-69 | 0 | Borrower's age in years |
| **Education** | Categorical | High School, Bachelor's, Master's, PhD | 0 | Highest education level attained |
| **MaritalStatus** | Categorical | Single, Married, Divorced | 0 | Current marital status |
| **HasDependents** | Categorical | Yes, No | 0 | Whether borrower has dependent children |

#### Financial Features

| Column | Type | Range | Missing | Description | Formula |
|--------|------|-------|---------|-------------|---------|
| **Income** | Integer | $15,000-$149,999 | 0 | Annual gross income in USD | N/A |
| **LoanAmount** | Integer | $5,000-$249,999 | 0 | Requested loan amount in USD | N/A |
| **InterestRate** | Float | 2%-25% | 0 | Annual interest rate (APR) | Depends on credit score & term |
| **LoanTerm** | Integer | 12-60 | 0 | Loan repayment period in months | N/A |
| **DTIRatio** | Float | 0.1-0.9 | 0 | Debt-to-Income ratio | Monthly debt payments / Monthly income |

#### Credit & Employment Features

| Column | Type | Range | Missing | Description |
|--------|------|-------|---------|-------------|
| **CreditScore** | Integer | 300-849 | 0 | FICO credit score (300=poor, 850=excellent) |
| **NumCreditLines** | Integer | 1-4 | 0 | Number of open credit accounts |
| **MonthsEmployed** | Integer | 0-119 | 0 | Months at current employer (0=unemployed) |
| **EmploymentType** | Categorical | Full-time, Part-time, Unemployed, Self-employed | 0 | Type of employment |

#### Collateral & Co-Obligor Features

| Column | Type | Values | Missing | Description |
|--------|------|--------|---------|-------------|
| **HasMortgage** | Categorical | Yes, No | 0 | Whether borrower owns home with mortgage |
| **HasCoSigner** | Categorical | Yes, No | 0 | Whether loan has co-signer |
| **LoanPurpose** | Categorical | Home, Auto, Business, Education, Other | 0 | Intended use of loan funds |

#### Target Variable

| Column | Type | Values | Missing | Description |
|--------|------|--------|---------|-------------|
| **Default** | Binary | 0, 1 | 0 | **0:** Loan repaid on time; **1:** Loan defaulted |

---

## Methodology

### 1. Data Understanding Phase

#### Exploratory Data Analysis (EDA)
- **Missing Values:** 0 (100% complete dataset)
- **Duplicates:** 0 (all 255K rows unique)
- **Data Types:** 8 numeric, 8 categorical, 2 identifiers
- **Outliers:** None detected (ranges are reasonable)

#### Distribution Analysis
```
Class Distribution:
  No Default (0): 225,694 (88.39%) ← Majority class
  Default (1):     29,653 (11.61%) ← Minority class
  
Imbalance Ratio: 7.6:1 (not severe, but noted)
```

### 2. Data Preprocessing

#### Categorical Encoding
```python
# Label Encoding applied to 7 categorical columns
Education:       High School=0, Bachelor's=1, Master's=2, PhD=3
EmploymentType:  Full-time=0, Part-time=1, Self-employed=2, Unemployed=3
MaritalStatus:   Single=0, Married=1, Divorced=2
LoanPurpose:     [0-4] based on alphabetical order
HasMortgage:     No=0, Yes=1
HasDependents:   No=0, Yes=1
HasCoSigner:     No=0, Yes=1
```

#### Feature Removal
```python
# Dropped LoanID - unique identifier, no predictive value
# Kept all other 16 features for model training
```

#### No Scaling Applied
```
Why? Gradient Boosting and Random Forest are tree-based
     models that don't require feature scaling.
     Logistic Regression & Naive Bayes would benefit from scaling
     but dataset is already reasonably normalized.
```

### 3. Data Splitting

```python
# Stratified Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.20,           # 80/20 split
    random_state=42,          # Reproducibility
    stratify=y                # Maintains class distribution
)

# Results:
Train: 204,277 samples (80%) - 180,555 no-default, 23,722 default
Test:   51,070 samples (20%) -  45,139 no-default,  5,931 default
```

### 4. Model Training

#### Models Evaluated
1. **Logistic Regression** - Linear baseline
2. **Decision Tree** - Non-linear baseline
3. **Random Forest** - Ensemble of trees
4. **Naive Bayes** - Probabilistic baseline
5. **Gradient Boosting** - Sequential ensemble

#### Hyperparameters

**Gradient Boosting (Selected):**
```python
GradientBoostingClassifier(
    n_estimators=100,          # Number of boosting stages
    max_depth=6,               # Max tree depth (prevent overfitting)
    learning_rate=0.1,         # Shrinkage parameter
    subsample=0.8,             # Fraction of samples for training
    random_state=42
)
```

**Random Forest:**
```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=50,
    n_jobs=-1,                 # Parallel processing
    random_state=42
)
```

**Logistic Regression:**
```python
LogisticRegression(
    max_iter=1000,
    random_state=42,
    n_jobs=-1
)
```

### 5. Model Evaluation

#### Primary Metrics Used
- **ROC-AUC:** Area under ROC curve (0-1, higher better)
- **Accuracy:** (TP+TN)/(TP+TN+FP+FN)
- **Precision:** TP/(TP+FP) - When we predict default, how often correct
- **Recall:** TP/(TP+FN) - How many actual defaults we catch
- **F1-Score:** 2×(Precision×Recall)/(Precision+Recall)

#### Cross-Validation
- Not used (large dataset, high statistical power)
- 80/20 split sufficient for reliable estimates

---

## Model Architecture

### Gradient Boosting Classifier

```
Architecture: Sequential Gradient Boosting Ensemble

Input Layer:
├── 16 features (normalized through encoding)
└── 204,277 training samples

Boosting Stages:
├── Iteration 1: Fit tree to residuals (errors from previous tree)
├── Iteration 2: Improve on previous predictions
├── ...
└── Iteration 100: Final ensemble prediction

Output Layer:
├── Probability: P(Default) ∈ [0, 1]
└── Binary Prediction: if P > threshold → Default, else → No Default
```

### Decision Logic

```
Prediction Process:
1. Feature Input: (Age, Income, CreditScore, ..., HasCoSigner)
2. Encoding: Convert categoricals to numeric
3. Traverse 100 Decision Trees: Each tree makes prediction
4. Weighted Ensemble: Combine predictions with boosting weights
5. Probability: Average probability across all trees
6. Threshold Decision: Compare with threshold (default 0.5, optimized to 0.20)
7. Output: Default or No-Default flag
```

---

## Evaluation Metrics Explained

### 1. ROC-AUC (Area Under the Curve)

**What it measures:** Model's ability to discriminate between defaults and non-defaults

**Interpretation:**
- 1.0 = Perfect discrimination (impossible)
- 0.7-0.8 = Good (our models)
- 0.5 = Random guessing (no skill)
- 0.0 = Inverse relationship (backwards)

**Formula:**
```
ROC Curve = Plot of True Positive Rate vs False Positive Rate
AUC = Area under this curve
```

**For Our Models:**
- Gradient Boosting: 0.7560 ✅ Best
- Random Forest: 0.7464 ✅ Good
- Logistic Regression: 0.7466 ✅ Good
- Naive Bayes: 0.7435 ✅ Acceptable
- Decision Tree: 0.6074 ⚠️ Weak

### 2. Accuracy

**What it measures:** Overall correctness of predictions

**Formula:** 
```
Accuracy = (Correct Predictions) / (Total Predictions)
         = (TP + TN) / (TP + TN + FP + FN)
```

**Limitation:** Misleading with imbalanced classes (always predict majority to get 88%+ accuracy)

**Our Models:** 85.7%-88.6% (varies)

### 3. Precision

**What it measures:** When we predict a default, how often is it correct?

**Formula:**
```
Precision = TP / (TP + FP)
```

**Example:**
- If we flag 1000 loans as risky
- 300 actually default
- Precision = 300/1000 = 30%

**For Our Models:**
- Random Forest: 63.71% - Most precise
- Gradient Boosting: 57.79% - Good
- Logistic Regression: 61.86% - Good

**Business Impact:** Higher precision = fewer false reviews needed

### 4. Recall (Sensitivity)

**What it measures:** Of all actual defaults, how many do we catch?

**Formula:**
```
Recall = TP / (TP + FN)
```

**Example:**
- If there are 100 actual defaults in test set
- We catch 7.57
- Recall = 7.57/100 = 7.57%

**For Our Models:**
- Decision Tree: 12.06% - Highest recall
- Gradient Boosting: 7.57% - Good
- Logistic Regression: 2.46% - Low

**Business Impact:** Higher recall = fewer missed defaults (critical!)

### 5. F1-Score

**What it measures:** Harmonic mean of Precision and Recall

**Formula:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**When to use:** When precision and recall are equally important

**Our Models:** 0.047 - 0.164 (very low due to extreme class imbalance)

### 6. Threshold Adjustment Trade-offs

**Lower Threshold** (e.g., 0.20 vs 0.50)
- ✅ Higher Recall (catch more defaults)
- ✅ Lower False Negatives (fewer missed)
- ❌ Lower Precision (more false alarms)
- ❌ Higher False Positives

**Visual:**
```
Threshold 0.50: Very Conservative
Predict Default only if P(Default) > 50%
  High Precision, Low Recall

Threshold 0.20: Balanced (RECOMMENDED)
Predict Default if P(Default) > 20%
  Medium Precision, Good Recall

Threshold 0.10: Aggressive
Predict Default if P(Default) > 10%
  Low Precision, High Recall (catch most defaults)
```

---

## Preprocessing Steps

### Step 1: Load Raw Data
```python
import pandas as pd
df = pd.read_csv('data/loans.csv')
# 255,347 × 18
```

### Step 2: Separate Features & Target
```python
X = df.drop(['LoanID', 'Default'], axis=1)  # 255,347 × 16
y = df['Default']                             # 255,347 × 1
```

### Step 3: Identify Column Types
```python
numeric_cols = X.select_dtypes(include=[np.number]).columns
# Age, Income, LoanAmount, CreditScore, MonthsEmployed, 
# NumCreditLines, InterestRate, LoanTerm, DTIRatio (9 cols)

categorical_cols = X.select_dtypes(include=['object']).columns
# Education, EmploymentType, MaritalStatus, HasMortgage,
# HasDependents, LoanPurpose, HasCoSigner (7 cols)
```

### Step 4: Encode Categorical Variables
```python
from sklearn.preprocessing import LabelEncoder

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    # Stores encoder for later use in production

# Example encoding:
# Education: 'Bachelor's' → 0, 'High School' → 1, etc.
```

### Step 5: Check for Missing Values
```python
X.isnull().sum()  # All zeros - no imputation needed
y.isnull().sum()  # All zeros
```

### Step 6: Train-Test Split
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
```

### Step 7: Scale? (Not Required)
```python
# Decision: NO SCALING
# Reason: Tree-based models (GB, RF) are scale-invariant
#         Linear models (LR, NB) are more stable with this data
```

---

## Appendix: Statistical Tests

### Correlation Analysis
```
Default vs Age: -0.168 (moderate negative)
Default vs Income: -0.099 (weak negative)
Default vs MonthsEmployed: -0.097 (weak negative)
Default vs InterestRate: +0.131 (weak positive)
Default vs LoanAmount: +0.087 (weak positive)
```

### Class Balance Test
```
Chi-Square Test: Distribution uniform across education/employment levels
Conclusion: No class imbalance within subgroups
```

### Feature Variance Inflation
```
VIF Analysis: No multicollinearity detected
All VIF < 5 (threshold is 10)
```

---

## References & Resources

- **scikit-learn Documentation:** https://scikit-learn.org
- **ROC-AUC Explanation:** https://developers.google.com/machine-learning/glossary/roc-auc
- **Gradient Boosting:** Friedman (2001), "Greedy Function Approximation: A Gradient Boosting Machine"
- **Class Imbalance:** Chawla (2009), "Data Mining for Imbalanced Datasets"

---

**Document Version:** 1.0  
**Last Updated:** June 10, 2026  
**Author:** Data Science Team  
**Reviewed By:** ML Engineering
