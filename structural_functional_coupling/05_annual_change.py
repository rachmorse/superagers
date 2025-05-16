import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

def calculate_annual_change(data, n_timepoints):
    """
    Calculate slopes for all variables over time using age as the predictor.
    
    Args:
        data (pd.DataFrame): DataFrame containing the data
        n_timepoints (int): Number of timepoints in the data
        
    Returns:
        pd.DataFrame: DataFrame with added slope columns for each variable
    """
    # Create a copy of the input dataframe to avoid modifying the original
    result_df = data.copy()
    
    # Get age columns
    age_cols = [f"age.{i}" for i in range(1, n_timepoints+1)]
    
    # Get all variable columns except 'id' and age columns
    variable_cols = {}
    for col in data.columns:
        if col != 'id' and not col.startswith('age.'):
            # Extract the base variable name (before the dot)
            if '.' in col:
                var_name = col.split('.')[0]
                timepoint = int(col.split('.')[1])
                if var_name not in variable_cols:
                    variable_cols[var_name] = []
                variable_cols[var_name].append(col)
    
    # For each row in the dataframe
    for i in range(len(data)):
        # Get age data for this subject
        age_data = data.iloc[i][age_cols].values
        
        # For each variable
        for var_name, var_cols in variable_cols.items():
            # Only process variables that have timepoints
            if len(var_cols) > 1:
                # Get variable data for this subject
                var_data = data.iloc[i][var_cols].values
                
                # Get non-NA indices
                valid_indices = ~np.isnan(var_data) & ~np.isnan(age_data)
                
                if np.sum(valid_indices) > 1:
                    # Calculate slope using linear regression
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        age_data[valid_indices], 
                        var_data[valid_indices]
                    )
                    
                    # Store the slope
                    result_df.loc[i, f"{var_name}_slopes"] = slope
                    
                    # Calculate time difference
                    valid_ages = age_data[valid_indices]
                    result_df.loc[i, f"{var_name}_time"] = np.max(valid_ages) - np.min(valid_ages)
                else:
                    result_df.loc[i, f"{var_name}_slopes"] = np.nan
                    result_df.loc[i, f"{var_name}_time"] = np.nan
    
    return result_df

def main():
    # Define parameters
    ses = "ses-01" 
    
    # Base directories
    base_dir = Path("/home/rachel/Desktop/schaefer_analysis/structure_function_coupling")
    metric_data_tp1 = base_dir / f"ses-01/network_metrics/network_connectivity_metrics_ses-01.csv"
    metric_data_tp2 = base_dir / f"ses-02/network_metrics/network_connectivity_metrics_ses-02.csv"

    # Merge the two dfs by id and keep id only if it is in both
    metric_data_tp1 = pd.read_csv(metric_data_tp1)
    metric_data_tp2 = pd.read_csv(metric_data_tp2)
    metric_data = pd.merge(metric_data_tp1, metric_data_tp2, on='id', suffixes=('_1', '_2'))

    result = calculate_annual_change(metric_data, 2)

    # Save the result
    result.to_csv('data_with_slopes.csv', index=False)

if __name__ == "__main__":
    main()

