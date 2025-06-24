# Insurance Prediction Script Documentation

This document provides a detailed overview of the functionalities implemented in `insurance_prediction.py`. The script is designed for robust predictive modeling, evaluation, and interpretation of insurance risk analytics, supporting both regression and classification tasks. Below is a breakdown of the main features and methods available in the `InsurancePredictionModel` class.

---

## 1. Data Loading and Initialization
- **Class:** `InsurancePredictionModel`
- **Initialization:**
  - Takes a file path as input and loads the data (expects a `.txt` file with `|`-separated values).
  - Data is loaded into a pandas DataFrame for further analysis.

---

## 2. Data Preparation
- **handle_missing_data(drop_threshold=0.3)**
  - Drops columns with more than `drop_threshold` (fraction) missing values.
  - Imputes missing values: categorical columns with mode, numeric columns with median.
- **check_missing_values()**
  - Prints the count of missing values for each column with missing data.
- **feature_engineering()**
  - Creates new features such as `ClaimFrequency`, `Margin`, and `PremiumPerVehicle`.
- **encode_categorical(encoding='onehot', drop_first=True, onehot_cols=None)**
  - Encodes categorical columns using one-hot or label encoding.
  - Only one-hot encodes columns specified in `onehot_cols`.

---

## 3. Data Splitting
- **train_test_split(target, test_size=0.2, random_state=42)**
  - Splits the data into train and test sets for modeling.

---

## 4. Model Training
- **fit_linear_regression(X_train, y_train)**
  - Trains a Linear Regression model.
- **fit_random_forest(X_train, y_train, n_estimators=100, random_state=42)**
  - Trains a Random Forest Regressor.
- **fit_xgboost(X_train, y_train, n_estimators=100, random_state=42)**
  - Trains an XGBoost Regressor.

---

## 5. Model Evaluation
- **evaluate_regression(model, X_test, y_test)**
  - Evaluates regression models using RMSE, R2, and Accuracy (%).
- **evaluate_classification(model, X_test, y_test, average='binary')**
  - Evaluates classification models using accuracy, precision, recall, and F1-score.
- **evaluate_model(model, X_test, y_test, task='auto', average='binary')**
  - Automatically selects regression or classification evaluation based on the target variable.

---

## 6. Feature Importance
- **feature_importance(model, feature_names=None, top_n=20)**
  - Extracts and returns feature importances for tree-based models or coefficients for linear models.
  - Returns a sorted DataFrame of the top features.

---

## 7. Model Interpretation
- **interpret_with_shap(model, X, feature_names=None, plot_type='summary', sample_size=100)**
  - Uses SHAP to interpret model predictions globally (summary plot) or locally (force plot).
- **interpret_with_lime(model, X, mode='regression', feature_names=None, sample_idx=0)**
  - Uses LIME to interpret local predictions, robust to notebook display issues, and ensures white background for plots.

---

## 8. Model Comparison
- **compare_models(results_dict, task='regression', metrics=None, plot_library='matplotlib')**
  - Compares the performance of multiple models.
  - Accepts a dictionary of model names to their evaluation results.
  - Displays a table and bar plots for each metric using matplotlib or Plotly.

---

## Usage Example
```python
from scripts.insurance_prediction import InsurancePredictionModel

# Initialize and prepare data
model = InsurancePredictionModel('../data/MachineLearningRating_v3.txt')
model.handle_missing_data()
model.feature_engineering()
model.encode_categorical(onehot_cols=['Province', 'CoverType'])

# For regression (e.g., claim severity prediction)
X_train, X_test, y_train, y_test = model.train_test_split(target='TotalClaims')

# Linear Regression
lr = model.fit_linear_regression(X_train, y_train)
lr_eval = model.evaluate_model(lr, X_test, y_test, task='regression')
print('Linear Regression:', lr_eval)

# Random Forest
rf = model.fit_random_forest(X_train, y_train)
rf_eval = model.evaluate_model(rf, X_test, y_test, task='regression')
print('Random Forest:', rf_eval)

# XGBoost
xgb = model.fit_xgboost(X_train, y_train)
xgb_eval = model.evaluate_model(xgb, X_test, y_test, task='regression')
print('XGBoost:', xgb_eval)

# Compare models
results = {
    'Linear Regression': lr_eval,
    'Random Forest': rf_eval,
    'XGBoost': xgb_eval
}
model.compare_models(results, task='regression')

# Feature importance (example for Random Forest)
fi = model.feature_importance(rf, feature_names=X_train.columns)
print(fi)

# SHAP interpretation (global)
model.interpret_with_shap(rf, X_test, feature_names=X_test.columns, plot_type='summary')

# LIME interpretation (local)
model.interpret_with_lime(rf, X_test, mode='regression', feature_names=X_test.columns, sample_idx=0)
```

---

## Notes
- The script is modular and each method can be called independently.
- Designed for use in Jupyter notebooks or as a standalone script.
- Handles missing values, encoding, and feature engineering robustly.
- Produces both static and interactive plots for model evaluation and interpretation.
- Supports both regression and classification workflows for insurance analytics.
