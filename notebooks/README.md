# Notebooks Documentation

This document provides an overview and usage guide for the Jupyter notebooks in the `notebooks` directory. These notebooks are designed for exploratory data analysis (EDA), hypothesis testing, and business reporting for insurance risk analytics.

---

## 1. `insurance_eda.ipynb`
**Purpose:**
- Perform comprehensive exploratory data analysis (EDA) on the insurance dataset.
- Visualize distributions, relationships, and trends in the data.
- Identify data quality issues, outliers, and key features for further analysis.

**Typical Structure:**
1. **Introduction and Objectives**
2. **Data Loading and Cleaning**
3. **Univariate Analysis** (histograms, bar charts)
4. **Bivariate/Multivariate Analysis** (scatter plots, correlation matrices)
5. **Geographical and Temporal Trends**
6. **Outlier Detection**
7. **Creative/Insightful Visualizations**
8. **Summary of Findings**

**Usage:**
- Run all cells sequentially to reproduce the EDA workflow.
- Modify or extend analysis as needed for new data or business questions.

---

## 2. `hypothesis.ipynb`
**Purpose:**
- Statistically validate or reject key business hypotheses about insurance risk drivers.
- Use modular functions from `scripts/hypothesis_testing.py` to perform A/B and group comparisons.
- Document and visualize the results of hypothesis tests for business reporting.

**Typical Structure:**
1. **Introduction and Objectives**
2. **Data Loading and Metric Computation**
3. **Hypothesis Statements**
4. **Statistical Testing** (e.g., t-test, z-test, chi-squared)
5. **Result Interpretation and Visualization**
6. **Summary Table of Conclusions**
7. **Business Implications**

**Usage:**
- Update the data path if using a new dataset.
- Adjust the hypotheses or features under test as business needs evolve.
- Use the summary table for quick reference in presentations or reports.

---

## 3. `__init__.py`
- This file is present to make the `notebooks` directory a Python package if needed for imports.
- It does not contain analysis code.

---

## General Notes
- All notebooks use an XML-based cell format for compatibility with VS Code and automated tools.
- Notebooks are modular and can be extended or reused for new analyses.
- For best results, run notebooks in order and ensure all dependencies are installed (see `requirements.txt`).
- Visualizations use `matplotlib`, `seaborn`, and `plotly` for both static and interactive analysis.

---

## Example Workflow
1. Start with `insurance_eda.ipynb` to understand the data and identify key features.
2. Proceed to `hypothesis.ipynb` to test specific business hypotheses and inform segmentation strategies.
3. Use findings to guide further modeling or reporting.
