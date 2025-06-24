# Insurance Analysis Script Documentation

This document provides a detailed overview of the functionalities implemented in `insurance_analysis.py`. The script is designed for comprehensive Exploratory Data Analysis (EDA) and data cleaning of insurance datasets, particularly those in the `MachineLearningRating_v3.txt` file. Below is a breakdown of the main features and methods available in the `InsuranceAnalysis` class.

---

## 1. Data Loading and Initialization
- **Class:** `InsuranceAnalysis`
- **Initialization:**
  - Takes a file path as input and loads the data (expects a `.txt` file with `|`-separated values).
  - Data is loaded into a pandas DataFrame for further analysis.

---

## 2. Data Summarization
- **summarize_data()**
  - Prints the shape of the data and descriptive statistics for numerical columns.
  - Displays data types for all columns and checks if they are appropriate (e.g., categorical, datetime).
- **check_data_types()**
  - Prints the data type of each column in the dataset.

---

## 3. Data Quality Assessment
- **check_missing_values()**
  - Identifies columns with missing values and prints the count of missing values per column.
- **check_duplicates()**
  - Checks for duplicate rows and prints the number of duplicates.
  - Calls `view_duplicates()` to display duplicate rows if any.
- **view_duplicates()**
  - Prints duplicate rows in the dataset, if present.

---

## 4. Data Cleaning
- **clean_data()**
  - Converts date columns (`VehicleIntroDate`, `TransactionMonth`) to datetime format.
  - Drops columns with more than 30% missing values.
  - Imputes missing values:
    - Categorical columns: filled with mode.
    - Numerical columns: filled with median.
  - Removes duplicate rows.
  - Returns the cleaned DataFrame.
- **convert_to_datetime(column_name)**
  - Converts a specified column to datetime format, handling errors gracefully.
- **export_cleaned_data(output_path)**
  - Exports the cleaned DataFrame to a specified CSV file.

---

## 5. Univariate Analysis
- **univariate_analysis(save_plots=False, output_dir=None, num_cols=None, cat_cols=None)**
  - Plots histograms for up to 5 numerical columns and bar charts for up to 5 categorical columns.
  - Plots can be displayed or saved to a directory.
  - Columns can be specified or automatically selected.

---

## 6. Bivariate and Multivariate Analysis
- **bivariate_multivariate_analysis(save_plots=False, output_dir=None)**
  - Explores relationships between monthly changes in `TotalPremium` and `TotalClaims` as a function of `PostalCode`.
  - Generates scatter plots and correlation matrices for these features.
  - Plots can be displayed or saved.

---

## 7. Data Comparison Across Geography
- **compare_trends_over_geography(save_plots=False, output_dir=None)**
  - Compares changes in insurance cover type, premium, and auto make across geographic columns (e.g., Country, Province, PostalCode).
  - Provides interactive visualizations using dropdowns for different geographic levels.
  - Visualizes trends over time and distributions by geography.

---

## 8. Outlier Detection
- **outlier_detection(save_plots=False, output_dir=None, num_cols=None)**
  - Uses box plots to detect outliers in up to 5 numerical columns (or user-specified columns).
  - Plots can be displayed or saved.

---

## 9. Creative Insight Plots
- **creative_insight_plots(save_plots=False, output_dir=None)**
  - Produces three creative and insightful plots:
    1. Interactive box plot of `TotalPremium` by `CoverType` (using Plotly).
    2. Correlation heatmap of all numerical features.
    3. Interactive bar plot of the top 10 provinces by average `TotalClaims`.
  - Plots can be displayed or saved as HTML/PNG files.

---

## 10. Visualization
- The script uses `matplotlib`, `seaborn`, and `plotly` for static and interactive visualizations.
- Interactive widgets (ipywidgets) are used for dynamic exploration in Jupyter environments.

---

## Usage Example
```python
from scripts.insurance_analysis import InsuranceAnalysis
analyzer = InsuranceAnalysis('../data/MachineLearningRating_v3.txt')
analyzer.summarize_data()
analyzer.check_missing_values()
analyzer.clean_data()
analyzer.univariate_analysis()
analyzer.bivariate_multivariate_analysis()
analyzer.compare_trends_over_geography()
analyzer.outlier_detection()
analyzer.creative_insight_plots()
```

---

## Notes
- The script is modular and each method can be called independently.
- Designed for use in Jupyter notebooks or as a standalone script.
- Handles missing values, data types, and duplicates robustly.
- Produces both static and interactive plots for comprehensive EDA.

## Information About The Data
- Columns about the insurance policy
    - UnderwrittenCoverID
    - PolicyID
    - TransactionMonth
- Columns about the client
    - IsVATRegistered
    - Citizenship
    - LegalType
    - Title
    - Language
    - Bank
    - AccountType
    - MaritalStatus
    - Gender

- Columns about the client location
    - Country
    - Province
    - PostalCode
    - MainCrestaZone
    - SubCrestaZone

- Columns about the car insured
    - ItemType
    - Mmcode
    - VehicleType
    - RegistrationYear
    - make
    - Model
    - Cylinders
    - Cubiccapacity
    - Kilowatts
    - Bodytype
    - NumberOfDoors
    - VehicleIntroDate
    - CustomValueEstimate
    - AlarmImmobiliser
    - TrackingDevice
    - CapitalOutstanding
    - NewVehicle
    - WrittenOff
    - Rebuilt
    - Converted
    - CrossBorder
    - NumberOfVehiclesInFleet

- Columns about the plan
    - SumInsured
    - TermFrequency
    - CalculatedPremiumPerTerm
    - ExcessSelected
    - CoverCategory
    - CoverType
    - CoverGroup
    - Section
    - Product
    - StatutoryClass
    - StatutoryRiskType

- Columns about the plan
    - TotalPremium
    - TotalClaims

---

# Hypothesis Testing Script Documentation

This section documents the functionalities implemented in `hypothesis_testing.py`. The script provides modular functions for statistical hypothesis testing on insurance risk drivers, supporting A/B and group comparisons for business analytics and segmentation.

---

## Purpose
- Statistically validate or reject key business hypotheses about risk drivers (e.g., province, zip code, gender) using cleaned insurance data.
- Quantify risk using:
  - **Claim Frequency:** Proportion of policies with at least one claim
  - **Claim Severity:** Average amount of a claim, given a claim occurred
  - **Margin:** Difference between TotalPremium and TotalClaims

---

## Main Functions

### 1. `compute_metrics(df: pd.DataFrame) -> pd.DataFrame`
- Adds the following columns to the DataFrame:
  - `ClaimFrequency`: 1 if `TotalClaims` > 0, else 0
  - `NumClaims`: 1 if `TotalClaims` > 0, else 0 (fallback if explicit count not available)
  - `ClaimSeverity`: `TotalClaims` / `NumClaims` (if `NumClaims` > 0, else 0)
  - `Margin`: `TotalPremium` - `TotalClaims`
- **Usage:**
```python
from scripts.hypothesis_testing import compute_metrics
df = compute_metrics(df)
```

### 2. `segment_data(df, feature, group_a, group_b) -> Tuple[DataFrame, DataFrame]`
- Splits the DataFrame into two groups based on the values of a categorical feature.
- **Usage:**
```python
group_a_df, group_b_df = segment_data(df, 'Province', 'Gauteng', 'Western Cape')
```

### 3. `t_test_metric(group_a, group_b, metric)`
- Performs an independent t-test for a numerical metric between two groups.
- Returns test statistic, p-value, and group means.
- **Usage:**
```python
result = t_test_metric(group_a_df, group_b_df, 'ClaimSeverity')
```

### 4. `z_test_proportion(group_a, group_b, metric)`
- Performs a z-test for proportions (e.g., claim frequency) between two groups.
- Returns test statistic, p-value, and group proportions.
- **Usage:**
```python
result = z_test_proportion(group_a_df, group_b_df, 'ClaimFrequency')
```

### 5. `chi2_test_categorical(df, feature, target)`
- Performs a chi-squared test for independence between a categorical feature and a binary target.
- Returns chi2 statistic, p-value, degrees of freedom, expected counts, and contingency table.
- **Usage:**
```python
result = chi2_test_categorical(df, 'Province', 'ClaimFrequency')
```

### 6. `analyze_and_report(result, test_type, alpha=0.05)`
- Interprets the result of a statistical test and returns a human-readable conclusion based on the p-value and significance level.
- **Usage:**
```python
conclusion = analyze_and_report(result, 't-test')
```

### 7. `test_hypothesis(df, feature, group_a, group_b, metric, test_type='t-test', alpha=0.05)`
- General workflow for hypothesis testing between two groups on a metric.
- Selects and runs the appropriate test (`t-test`, `z-test`, or `chi2`), and returns a dictionary with test results and conclusion.
- **Usage:**
```python
result = test_hypothesis(df, 'Gender', 'Male', 'Female', 'ClaimFrequency', test_type='z-test')
print(result['conclusion'])
```

---

## Example Workflow
```python
from scripts.hypothesis_testing import compute_metrics, test_hypothesis
import pandas as pd

df = pd.read_csv('../data/insurance_cleaned.csv')
df = compute_metrics(df)

# Test for risk difference between two provinces
result = test_hypothesis(df, 'Province', 'Gauteng', 'Western Cape', 'ClaimFrequency', test_type='z-test')
print(result['conclusion'])
```

---

## Notes
- All functions are modular and can be used independently or in a notebook workflow.
- Designed for use in Jupyter notebooks or as part of automated analytics scripts.
- Supports both numerical and categorical hypothesis testing relevant to insurance risk segmentation.
- Results are suitable for business reporting and strategy development.
