import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
from typing import Tuple, Dict, Any

'''
A/B Hypothesis Testing
For this analysis, "risk" will be quantified by two metrics: Claim Frequency (proportion of policies with at least one claim) and Claim Severity (the average amount of a claim, given a claim occurred). "Margin" is defined as (TotalPremium - TotalClaims).

Accept or reject the following Null Hypotheses: 
H₀:There are no risk differences across provinces 
H₀:There are no risk differences between zip codes 
H₀:There are no significant margin (profit) difference between zip codes 
H₀:There are not significant risk difference between Women and Men
'''

def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute claim frequency, claim severity, and margin for each row.
    - Claim Frequency: 1 if TotalClaims > 0 else 0
    - Claim Severity: TotalClaims / NumClaims (if NumClaims > 0)
    - Margin: TotalPremium - TotalClaims
    """
    df = df.copy()
    df['ClaimFrequency'] = (df['TotalClaims'] > 0).astype(int)
    df['NumClaims'] = df['TotalClaims'].apply(lambda x: 1 if x > 0 else 0)  # fallback if no explicit count
    df['ClaimSeverity'] = df.apply(lambda row: row['TotalClaims'] / row['NumClaims'] if row['NumClaims'] > 0 else 0, axis=1)
    df['Margin'] = df['TotalPremium'] - df['TotalClaims']
    return df


def segment_data(df: pd.DataFrame, feature: str, group_a, group_b) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Segment data into two groups based on the feature and values for A and B.
    """
    group_a_df = df[df[feature] == group_a]
    group_b_df = df[df[feature] == group_b]
    return group_a_df, group_b_df


def t_test_metric(group_a, group_b, metric: str) -> Dict[str, Any]:
    """
    Perform independent t-test for a given metric between two groups.
    """
    a = group_a[metric].dropna()
    b = group_b[metric].dropna()
    stat, p = stats.ttest_ind(a, b, equal_var=False)
    return {'statistic': stat, 'p_value': p, 'group_a_mean': a.mean(), 'group_b_mean': b.mean()}


def chi2_test_categorical(df: pd.DataFrame, feature: str, target: str) -> Dict[str, Any]:
    """
    Perform chi-squared test for independence between a categorical feature and a binary target.
    """
    contingency = pd.crosstab(df[feature], df[target])
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    return {'chi2': chi2, 'p_value': p, 'dof': dof, 'expected': expected, 'contingency': contingency}


def z_test_proportion(group_a, group_b, metric: str) -> Dict[str, Any]:
    """
    Perform z-test for proportions (e.g., claim frequency) between two groups.
    """
    count = np.array([group_a[metric].sum(), group_b[metric].sum()])
    nobs = np.array([len(group_a), len(group_b)])
    stat, p = proportions_ztest(count, nobs)
    return {'statistic': stat, 'p_value': p, 'group_a_prop': count[0]/nobs[0], 'group_b_prop': count[1]/nobs[1]}


def analyze_and_report(result: Dict[str, Any], test_type: str, alpha: float = 0.05) -> str:
    """
    Analyze test result and return interpretation string.
    """
    p = result.get('p_value', 1)
    if p < alpha:
        conclusion = f"Reject the null hypothesis (p-value={p:.4f} < {alpha}). Statistically significant difference detected by {test_type}."
    else:
        conclusion = f"Fail to reject the null hypothesis (p-value={p:.4f} >= {alpha}). No statistically significant difference detected by {test_type}."
    return conclusion

# Example workflow function for a single hypothesis
def test_hypothesis(df: pd.DataFrame, feature: str, group_a, group_b, metric: str, test_type: str = 't-test', alpha: float = 0.05) -> Dict[str, Any]:
    """
    General workflow for hypothesis testing between two groups on a metric.
    """
    group_a_df, group_b_df = segment_data(df, feature, group_a, group_b)
    if test_type == 't-test':
        result = t_test_metric(group_a_df, group_b_df, metric)
    elif test_type == 'z-test':
        result = z_test_proportion(group_a_df, group_b_df, metric)
    elif test_type == 'chi2':
        result = chi2_test_categorical(df[df[feature].isin([group_a, group_b])], feature, metric)
    else:
        raise ValueError('Unsupported test_type')
    result['conclusion'] = analyze_and_report(result, test_type, alpha)
    return result