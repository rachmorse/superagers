import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from typing import Union
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

# This script looks at differences in connectivity matrices (currently 
# longitudinal) between groups. 

def main(
        connectivity_tp1: Union[str, Path], 
        connectivity_tp2: Union[str, Path], 
        superager_file: Union[str, Path], 
        out_file: Union[str, Path]
       ):
    """
    1) Reads two CSVs with connectivity data for two time points (TP1 and TP2).
    2) Reads a CSV with superager status info (id, superager, maintainer, w1_age, w2_age).
    3) Calculates the change in connectivity rates between TP1 and TP2 for each subject.
    4) Merges the change data with superager info, filtering out any missing subjects.
    5) Splits data into 'superagers' vs. 'non-superagers'.
    6) Runs an independent t-test on each ROI-to-ROI column.
    7) Corrects for multiple comparisons using FDR.
    8) Saves a CSV of only the edges that differ significantly (FDR < 0.05).
    """
    
    # Load both CSVs into DataFrames
    df_tp1 = pd.read_csv(connectivity_tp1) # each row is a subject, columns = edges
    df_tp2 = pd.read_csv(connectivity_tp2)
    df_superager = pd.read_csv(superager_file)

    # Make 'id' columns match in both DataFrames (e.g., add "sub-" prefix in superager file)
    df_tp1['id'] = df_tp1['id'].astype(str)
    df_tp2['id'] = df_tp2['id'].astype(str)
    df_superager['id'] = df_superager['id'].astype(str).apply(lambda x: 'sub-' + x)

    # Drop columns that are not needed
    columns_to_keep = ['id', 'w1_age', 'w2_age',
                       'superager', 
                       'maintainer'
                       ]
     
    df_superager = df_superager[columns_to_keep]

    # Look only at the absolute values of the connectivity data
    df_tp1.iloc[:, 1:] = df_tp1.iloc[:, 1:].abs()  # Skip 'id' column
    df_tp2.iloc[:, 1:] = df_tp2.iloc[:, 1:].abs()

    print(df_tp1.head(5))

    # Merge TP1 and TP2 data
    df_change = pd.merge(df_tp1, df_tp2, on='id', suffixes=('_tp1', '_tp2'))

    # Extract age-related columns
    df_change = pd.merge(df_change, df_superager[['id', 'w1_age', 'w2_age']], on='id', how='inner')

    # Calculate the time difference
    time_diff = df_change['w2_age'] - df_change['w1_age']

    # Prepare a dictionary to store changes in connectivity
    change_data = {}

    print("Starting change calculation")

    # Calculate the change in connectivity, adjusted for time elapsed
    for column in df_tp1.columns:
        if column != 'id':
            change_column = column + '_change'
            change_data[change_column] = (df_change[column + '_tp2'] - df_change[column + '_tp1']) / time_diff

    # Convert the dictionary to a DataFrame and concatenate with df_change
    change_df = pd.DataFrame(change_data)
    df_change = pd.concat([df_change, change_df], axis=1)

    print("Change calculation complete")

    # Drop non-required columns and prepare the superager data
    # df_change = pd.merge(df_change, df_superager[['id', 'superager']], on='id', how='inner')
    df_change = pd.merge(df_change, df_superager[['id', 'superager', 'maintainer']], on='id', how='inner')

    # Concatenate change_df only with 'id' and 'superager', omit original data columns
    # df_change = pd.concat([df_change[['id', 'superager']], change_df], axis=1)
    df_change = pd.concat([df_change[['id', 'superager', 'maintainer']], change_df], axis=1)

    # Split the data into superagers and non-superagers, dropping metadata
    df_super = df_change[df_change['superager'] == 1].drop(columns=['id', 'superager'])
    df_non = df_change[df_change['superager'] == 0].drop(columns=['id', 'superager'])
    df_maint = df_change[df_change['maintainer'] == 1].drop(columns=['id', 'maintainer'])
    df_decline = df_change[df_change['maintainer'] == 0].drop(columns=['id', 'maintainer'])

    df_superager_maintainers = df_change[(df_change['superager'] == 1) & (df_change['maintainer'] == 1)].drop(columns=['id', 'superager', 'maintainer'])
    df_superager_decliners = df_change[(df_change['superager'] == 1) & (df_change['maintainer'] == 0)].drop(columns=['id', 'superager', 'maintainer'])
    df_non_superager_maintainers = df_change[(df_change['superager'] == 0) & (df_change['maintainer'] == 1)].drop(columns=['id', 'superager', 'maintainer'])
    df_non_superager_decliners = df_change[(df_change['superager'] == 0) & (df_change['maintainer'] == 0)].drop(columns=['id', 'superager', 'maintainer'])

    # Output the shape to verify column count
    # print(f"Superagers: {df_super.shape}, Non-superagers: {df_non.shape}")
    print(f"superagesr: {df_super.shape}, decliners: {df_decline.shape}")

    # Drop non-edge columns (like 'id', 'superager', 'w1_age', 'w2_age', 'time_diff')
    cols_to_drop = ['id', 'w1_age', 'w2_age', 'time_diff',
                    'superager', 
                    'maintainer'
                    ] + list(df_tp1.columns)
    
    df_super_edges = df_superager_maintainers.drop(columns=cols_to_drop, errors='ignore')
    df_non_edges = df_non_superager_decliners.drop(columns=cols_to_drop, errors='ignore')

    # Convert these to NumPy arrays for convenience
    arr_super = df_super_edges.to_numpy()  # shape = (num_super_subjects, num_edges)
    arr_non   = df_non_edges.to_numpy()    # shape = (num_non_subjects,   num_edges)

    # Remember which columns correspond to which edge
    edge_labels = df_super_edges.columns.tolist()
    num_edges   = len(edge_labels)

    # Run an independent t-test for each edge
    pvals = np.zeros(num_edges)
    for e in range(num_edges):
        # Compare superagers vs. non-superagers on column e
        tstat, pval = ttest_ind(arr_super[:, e], arr_non[:, e], nan_policy='omit')
        pvals[e] = pval

    print("Shape of arr_super:", arr_super.shape)
    print("Shape of arr_non:", arr_non.shape)
    print("First row of arr_super:", arr_super[0])
    print("First row of arr_non:", arr_non[0])

    print("Raw p-values:", pvals)
    significant_raw_pvals = (pvals < 0.05).sum()
    print(f"Number of raw significant p-values (<0.05): {significant_raw_pvals}")

    # Correct for multiple comparisons (FDR)
    rejected, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
    print("Rejected hypotheses:", rejected.sum())

    plt.hist(pvals_corrected, bins=50, edgecolor='k')
    plt.xlabel('p-value')
    plt.ylabel('Frequency')
    plt.title('Distribution of corrected p-values')

    plot_out_file = output_directory / "p_value_corrected_distribution.png"
    plt.savefig(plot_out_file) 

    plt.hist(pvals, bins=50, edgecolor='k')
    plt.xlabel('p-value')
    plt.ylabel('Frequency')
    plt.title('Distribution of p-values')

    plot_out_file = output_directory / "p_value_distribution.png"
    plt.savefig(plot_out_file) 

    # Pick a few columns to inspect
    example_columns = [
        "b'7Networks_LH_Default_Temp_1'-b'7Networks_LH_Default_Temp_2'_change", 
        "b'7Networks_RH_Default_pCunPCC_2'-b'7Networks_RH_Default_pCunPCC_3'_change"
    ]

    for col in example_columns:
        # Combine data in a single DataFrame for plotting
        df_plot = pd.concat([
            pd.DataFrame({
                'group': 'superagers',
                'values': df_super[col]
            }),
            pd.DataFrame({
                'group': 'decliners',
                'values': df_decline[col]
            })
        ], ignore_index=True)
    
    # Box plot
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='group', y='values', data=df_plot)
    plt.title(f'Box plot of {col}')
    plot_out_file = output_directory / "box_plot.png"
    plt.savefig(plot_out_file) 

    # Histogram (with overlayed KDE)
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df_plot, x='values', hue='group', kde=True, stat='density')
    plt.title(f'Histogram of {col}')
    plot_out_file = output_directory / "histogram.png"
    plt.savefig(plot_out_file) 

    # Save a CSV of significant edges
    # Optionally compute mean connectivity in each group for those edges
    mean_super = arr_super.mean(axis=0)
    mean_non   = arr_non.mean(axis=0)

    # Build a list of dictionaries to hold each “significant” edge’s info
    sig_edges = []
    for i, is_sig in enumerate(rejected):
        if is_sig:
            sig_edges.append({
                'edge_label': edge_labels[i],
                'p_uncorrected': pvals[i],
                'p_fdr': pvals_corrected[i],
                'mean_super': mean_super[i],
                'mean_non': mean_non[i],
            })

    print("Any significant edges:", any(rejected))

    df_sig = pd.DataFrame(sig_edges).sort_values('p_fdr')
    df_sig.to_csv(out_file, index=False)

    # Print summary to terminal
    print(f"Found {len(df_sig)} significant edges (FDR < 0.05). Saved to {out_file}")

if __name__ == "__main__":
    ses = "02"
    superager_file = "/home/rachel/Desktop/data/maintainer_superager_data.csv"
    # output_directory = Path(f"/home/rachel/Desktop/schaefer_analysis/connectivity_matrices/ses-{ses}/all_to_all_roi_matrices")
    # output_directory = Path(f"/home/rachel/Desktop/schaefer_analysis/connectivity_matrices/ses-{ses}/within_network_matrices")
    # connectivity_file = output_directory / "fisher_z_all_to_all_roi_matrix.csv"
    # connectivity_file = output_directory / "fisher_z_Default_within_network_matrix.csv"

    output_directory = Path(f"/home/rachel/Desktop/schaefer_analysis/connectivity_matrices/")
    ses = "01"
    connectivity_tp1 = Path(f"{output_directory}/ses-{ses}/all_to_all_roi_matrices/fisher_z_all_to_all_roi_matrix.csv")
    ses = "02"
    connectivity_tp2 = Path(f"{output_directory}/ses-{ses}/all_to_all_roi_matrices/fisher_z_all_to_all_roi_matrix.csv")

    # out_file = output_directory / "significant_edges_superagers_vs_non.csv"
    # out_file = output_directory / "significant_DMN_edges_superagers_vs_non.csv"
    out_file = output_directory / "significant_longitudinal.csv"

    main(
        connectivity_tp1=connectivity_tp1,
        connectivity_tp2=connectivity_tp2,
        superager_file=superager_file,
        out_file=out_file
    )