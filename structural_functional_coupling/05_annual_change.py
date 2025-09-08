import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import re

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
    age_cols = [f"age_{i}" for i in range(1, n_timepoints+1)]
    
    # Get all variable columns except 'id' and age columns
    variable_cols = {}
    for col in data.columns:
        if col != 'id' and not col.startswith('age_'):
            # Extract the base variable name (before the _)
            if '_' in col:
                parts = col.split('_')
                var_name = '_'.join(parts[:-1])
                if parts[-1].isdigit():
                    # timepoint = int(parts[-1])
                    # Only add columns with integer timepoints
                    if var_name not in variable_cols:
                        variable_cols[var_name] = []
                    variable_cols[var_name].append(col)
    
    # For each row in the dataframe
    for i in range(len(data)):
        # Get age data for this subject
        age_data = pd.to_numeric(data.iloc[i][age_cols], errors='coerce').values
        # For each variable
        for var_name, var_cols in variable_cols.items():
            # Only process variables that have timepoints
            if len(var_cols) > 1:
                # Get variable data for this subject
                var_data = pd.to_numeric(data.iloc[i][var_cols], errors='coerce').values        
                
                # Get non-NA indices
                valid_indices = ~np.isnan(var_data) & ~np.isnan(age_data)
                
                if np.sum(valid_indices) > 1:
                    # Calculate slope using linear regression
                    lr_result = stats.linregress(
                        age_data[valid_indices],
                        var_data[valid_indices]
                    )

                    # Store the slope
                    result_df.loc[i, f"{var_name}_slopes"] = lr_result.slope
                    
                    # Calculate time difference
                    valid_ages = age_data[valid_indices]
                    result_df.loc[i, f"fu_time"] = np.max(valid_ages) - np.min(valid_ages)
                else:
                    result_df.loc[i, f"{var_name}_slopes"] = np.nan
    
    print("Slope calculation complete.")

    return result_df

def main():
    # Base directories
    base_dir = Path("/home/rachel/Desktop/schaefer_analysis/structure_function_coupling")
    metric_data_tp1 = base_dir / f"ses-01/network_metrics/network_connectivity_metrics_ses-01.csv"
    metric_data_tp2 = base_dir / f"ses-02/network_metrics/network_connectivity_metrics_ses-02.csv"
    age_dir = Path("/home/rachel/Desktop/data")

    # Merge the two dfs by id and keep id only if it is in both
    metric_data_tp1 = pd.read_csv(metric_data_tp1)
    metric_data_tp2 = pd.read_csv(metric_data_tp2)
    metric_data_tp1.rename(columns={'subject': 'id'}, inplace=True)
    metric_data_tp2.rename(columns={'subject': 'id'}, inplace=True)
    metric_data = pd.merge(metric_data_tp1, metric_data_tp2, on='id', how='outer', suffixes=('_1', '_2'))

    # Merge in the age data
    age_data = pd.read_csv(age_dir / "maintainer_superager_data.csv")
    age_data.columns = [re.sub(r"^w(\d)_(.*)", r"\2_\1", col) for col in age_data.columns] # Rename the columns
    age_data['id'] = 'sub-' + age_data['id'].astype(str) # Add 'sub-' to the id
    age_data_filt = age_data[['id', 'age_1', 'age_2']] # Keep only the relevant columns
    metric_data = pd.merge(metric_data, age_data_filt, on='id', how='left')

    result = calculate_annual_change(metric_data, 2)

    # Save the result
    result.to_csv(age_dir / 'superager_data_slopes.csv', index=False)

    # Merge the result with the original data to get a clean output
    merged_data = pd.merge(age_data, result, on=['id', 'age_1', 'age_2'], how='left')

    # Filter participants whose follow-up time is less than 1.5 years
    merged_data = merged_data[merged_data['fu_time'] > 1.8]
    print(merged_data.sample(5))
    merged_data.to_csv(age_dir / 'clean_data_all.csv', index=False)

    # Print summary statistics
    print(f"Number of participants: {len(merged_data)}")
    print(f"Age at timepoint 1: {merged_data['age_1'].mean():.2f} ± {merged_data['age_1'].std():.2f}")
    print(f"Percentage female: {merged_data['sex'].value_counts(normalize=True).get('female', 0) * 100:.2f}%")
    print(f"Mean follow-up time: {merged_data['fu_time'].mean():.2f} ± {merged_data['fu_time'].std():.2f} years")
    print(f"Number of superagers: {merged_data['superager'].sum()}")

if __name__ == "__main__":
    main()

