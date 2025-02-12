import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Union
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

# This script looks at differences in connectivity matrices (currently 
# for timepoint 1 vs 2 but can be easily changed to look at superagers 
# vs non-superagers 

def main(
        # connectivity_file: Union[str, Path], 
         connectivity_tp1: Union[str, Path], 
         connectivity_tp2: Union[str, Path], 
        #  superager_file: Union[str, Path], 
         out_file: Union[str, Path]
         ):
    """
    1) Reads a single CSV with all-subject connectivity ('all_to_all_roi_matrix.csv').
    2) Reads a CSV with superager status info (id, superager, maintainer).
    3) Merges them, filtering out any subjects missing from either file.
    4) Splits data into 'superagers' vs. 'non-superagers'.
    5) Runs an independent t-test on each ROI-to-ROI column.
    6) Corrects for multiple comparisons using FDR.
    7) Saves a CSV of only the edges that differ significantly (FDR < 0.05).
    """
    
    # # Load both CSVs into DataFrames
    # df_connectivity = pd.read_csv(connectivity_file)  # each row is a subject, columns = edges
    # df_superager = pd.read_csv(superager_file)        # columns: id, superager, maintainer

    # # Make 'id' columns match in both DataFrames (e.g., add "sub-" prefix in superager file)
    # df_connectivity['id'] = df_connectivity['id'].astype(str)
    # df_superager['id'] = df_superager['id'].astype(str).apply(lambda x: 'sub-' + x)

    # # Drop columns that are not needed
    # columns_to_keep = ['id', 'superager', 'w1_age', 'w2_age'] 
    # df_superager = df_superager[columns_to_keep]

    df_tp1 = pd.read_csv(connectivity_tp1)  # each row is a subject, columns = edges
    df_tp2 = pd.read_csv(connectivity_tp2) 

    # Merge on 'id' so each row has connectivity + superager info
    # df_connectivity = pd.merge(df_connectivity, df_superager, on='id', how='inner')

    # # Split into two groups: superagers == 1 vs. superagers == 0
    # df_super = df_merged[df_merged['superager'] == 1]
    # df_non   = df_merged[df_merged['superager'] == 0]

    # Extract 'id' columns
    super_ids = df_tp1['id'].astype(str)
    non_ids = df_tp2['id'].astype(str)

    # Find intersection of 'id' columns
    common_ids = set(super_ids).intersection(set(non_ids))
    print("Number of common ids:", len(common_ids))

    # Filter DataFrames to only include rows with 'id' in the intersection
    df_super_filtered = df_tp1[df_tp1['id'].isin(common_ids)]
    df_non_filtered = df_tp2[df_tp2['id'].isin(common_ids)]

    # Drop non-edge columns (like 'id', 'superager')
    cols_to_drop = ['id', 'superager']
    df_super_edges = df_super_filtered.drop(columns=cols_to_drop, errors='ignore')
    df_non_edges   = df_non_filtered.drop(columns=cols_to_drop, errors='ignore')

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

    plot_out_file = output_directory / "p_value_distribution.png"
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
    # ses = "01"
    # superager_file = "/home/rachel/Desktop/data/maintainer_superager_data.csv"
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
    out_file = output_directory / "significant_tp1_vs_tp2.csv"

    main(
        # connectivity_file=connectivity_file,
        connectivity_tp1=connectivity_tp1,
        connectivity_tp2=connectivity_tp2,
        # superager_file=superager_file,
        out_file=out_file
    )