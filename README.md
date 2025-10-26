# Loan Default Prediction using MLP

## Overview
This project builds a machine learning pipeline for **Loan Default Prediction** using a **Multi-Layer Perceptron (MLP)** model implemented in **PyTorch**.  
The dataset undergoes extensive preprocessing, exploratory data analysis (EDA), feature encoding, and normalization before training.  
The model is evaluated using **AUC** (Area Under the ROC Curve) and **F1-score** to assess its classification performance.

---

## Project Workflow

### 1. Data Preprocessing
The dataset contains various borrower, loan, and credit history features. The preprocessing steps include:

- **Subsampling** the dataset to make processing manageable.
- **Handling missing values** through imputation where necessary.
- **Dropping columns** with more than **80% null values**.
- **Creating a `region` column** by mapping each U.S. state to one of the following regions:
  - North
  - South
  - East
  - West
  - Southeast
  - Southwest
  - Northeast
  - Midwest
- **Target Variable Creation**: 
  - Loans labeled as `Fully Paid` are assigned **0**.
  - All other loan statuses (e.g., `Charged Off`, `Late`, etc.) are assigned **1**.

---

### 2. Exploratory Data Analysis (EDA)

EDA was performed to understand feature distributions, correlations, and relationships with the target variable.

Key visualizations include:
- **Histplots** for numerical variables (e.g., `loan_amnt`, `int_rate`, `annual_inc`, etc.).
- **Countplots** for categorical variables like `loan_status`, `home_ownership`, and `purpose`.
- **Seaborn pairplots** to visualize relationships between key numeric variables such as **FICO scores** and loan amount.
- **Boxplots** and **barplots** to explore the effect of categorical variables on loan status.
- Computation of **median FICO score** from `fico_range_low` and `fico_range_high`.

These analyses helped in feature selection, understanding class imbalance, and outlier detection.

---

### 3. Feature Engineering

- **Target Encoding** for categorical variables where applicable.
- **One-Hot Encoding** using `pd.get_dummies()` for binary/multi-class categorical columns.
- **Pairing categorical variables** wherever meaningful to create combined interaction features.
- **Categorical Variable Classification**:
  - **Binary variables:** e.g., `pymnt_plan_y`, `hardship_flag_Y`, etc.
  - **Multi-class variables:** e.g., `grade`, `sub_grade`, `home_ownership`, `verification_status`, `purpose`, `addr_state`.
- **Date Handling**: Date columns like `issue_d`, `earliest_cr_line`, `last_pymnt_d`, and `last_credit_pull_d` were properly formatted as datetime objects and excluded from scaling.

---

### 4. Train-Test Split

- The dataset is split into:
  - **Training set:** 80%
  - **Test set:** 20%
- `X_train`, `X_test`, `y_train`, and `y_test` are prepared with consistent columns after encoding.

---

### 5. Feature Scaling

- Numerical columns were standardized using **`StandardScaler`**.
- Only numeric columns (excluding date and categorical ones) were scaled.
- The same scaling parameters (mean, std) were applied to both training and test data.

---

### 6. Model Architecture: Multi-Layer Perceptron (MLP)

The MLP model is a fully connected neural network implemented in PyTorch with:
- Input layer matching the number of features
- Two hidden layers with ReLU activation
- Output layer using a Sigmoid activation for binary classification

**Loss Function:** `BCELoss`  
**Optimizer:** `Adam`  
**Metrics:** AUC and F1-score

---

### 7. Model Training and Evaluation

The model was trained using:
- Batch-wise training on the training data
- Validation on the test data after training

Evaluation metrics:
- **AUC (Area Under ROC Curve):** Measures separability between default and non-default classes.
- **F1-score:** Balances precision and recall, suitable for imbalanced datasets.

