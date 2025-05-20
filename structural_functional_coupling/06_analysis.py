import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Read and prepare the data
# ------------------------------------------------------------------------------
data = pd.read_csv("/home/rachel/Desktop/data/clean_data_all.csv")

# Extract numeric id 
data['id'] = data['id'].astype(str).str.replace("sub-", "", regex=True).astype(float)

# Create cohort variable
data['cohort'] = np.where(data['id'] > 5000, "bbhi", "bbhi_senior")
print(data.head(5))

# Pivot data from wide to long 
pattern = re.compile(r"(.+)_(\d+)$") # Identify columns with timepoints
columns_to_pivot = [col for col in data.columns if pattern.match(col)]

# Get unique base variable names
base_variables = set()
for col in columns_to_pivot:
    match = pattern.match(col)
    if match:
        base_variables.add(match.group(1))

# Identify the non-timepoint columns 
id_columns = [col for col in data.columns if col not in columns_to_pivot]

# Create an empty list to store dataframes for each timepoint
timepoint_dfs = []

# For each timepoint, create a dataframe with all variables for that timepoint
timepoints = sorted(set([int(pattern.match(col).group(2)) for col in columns_to_pivot if pattern.match(col)]))

for tp in timepoints:
    # Create a new dataframe with just the non-timepoint columns
    tp_data = data[id_columns].copy()
    
    # Add a timepoint column
    tp_data['timepoint'] = tp
    
    # For each base variable, add its value at the current timepoint
    for base_var in base_variables:
        timepoint_col = f"{base_var}_{tp}"
        if timepoint_col in data.columns:
            tp_data[base_var] = data[timepoint_col]
    
    # Add this to the list
    timepoint_dfs.append(tp_data)

# Combine all timepoint dataframes
long_data = pd.concat(timepoint_dfs, ignore_index=True)

# Prepare to run stats
# ------------------------------------------------------------------------------
def run_models(df, independent_vars, dependent_vars, timepoint, model_type="aov"):
    """Runs a linear model or ANOVA for the given variables and returns 
    the p-values and F-values.

    Args:
        df: dataframe containing the data
        independent_vars: list of columns to treat as grouping or factor
        dependent_vars: list of dependent variables (columns) to loop through
        timepoint: timepoint 
        model_type: "aov" for ANOVA or "lm" for linear model using statsmodels
    """
    all_results = []

    for ivar in independent_vars:
        for dvar in dependent_vars:
            if timepoint == 1 or timepoint == None:
                formula_str = f'{dvar} ~ C({ivar}) + age_1 + YoE + sex'
            else: # e.g. timepoint == 2
                formula_str = f'{dvar} ~ C({ivar}) + age_2 + YoE + sex'

            if dvar not in df.columns:
                continue

            try:
                # Fit model
                if model_type == "aov":
                    # Check whether using anova_lm is best
                    lm_fit = smf.ols(formula_str, data=df).fit()
                    aov_table = anova_lm(lm_fit, typ=2)

                    # Extract row associated with ivar
                    if f'C({ivar})' in aov_table.index:
                        F_val = aov_table.loc[f'C({ivar})', 'F']
                        p_val = aov_table.loc[f'C({ivar})', 'PR(>F)']
                    else:
                        continue

                elif model_type == "lm":
                    lm_fit = smf.ols(formula_str, data=df).fit()
                    relevant_coefs = [coef for coef in lm_fit.params.index if coef.startswith(f'C({ivar})')]
                    if len(relevant_coefs) == 0:
                        continue

                    # For simplicity, just extract p-value of the first level comparison
                    p_val = lm_fit.pvalues[relevant_coefs[0]]
                    t_val = lm_fit.tvalues[relevant_coefs[0]]
                    F_val = t_val**2

                sig = (p_val < 0.05)
                all_results.append((dvar, ivar, F_val, p_val, sig))

            except Exception:
                continue

    results_df = pd.DataFrame(all_results, columns=['dependent_var', 'independent_var', 'F_value', 'p_value', 'significant'])

    # Sort by p-value
    results_df = results_df.sort_values('p_value')

    # Apply FDR correction across all tests
    from statsmodels.stats.multitest import multipletests
    if len(results_df) > 0:
        rejected, fdr_p, _, _ = multipletests(results_df['p_value'], method='fdr_bh')
        results_df['fdr_p_value'] = fdr_p
        results_df['fdr_significant'] = rejected
    
    return results_df

# Test model
vars_long_sfc = [
    "sfc_all_slopes",
    "sfc_Default_slopes",
    "sfc_Frontoparietal_slopes",
    "sfc_VentralAttention_slopes",
    "sfc_DorsalAttention_slopes"
]

lm_results = run_models(data, vars_long_sfc, "memory_slopes", timepoint=None, model_type="lm")
print(lm_results)