import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from typing import Union
from scipy.stats import ttest_ind, ttest_rel, pearsonr
from statsmodels.stats.multitest import multipletests

# This thresholds the data by group (e.g. top 15% for superagers, then top 15% for non-superagers)
# and then merges the columns from both groups

def group_level_threshold(df, threshold, id_col='id', method='mean'):
    """
    1) Compute group-level average or median for each edge (column).
    2) Find the numeric cutoff for top `threshold` fraction.
    3) Remove all columns that do not meet (>=) the cutoff.
    4) Return the reduced DataFrame with just the retained columns (plus the id_col).

    Args:   
        df (pd.DataFrame): DataFrame where rows = participants, columns = edges + metadata.
        threshold (float): Fraction of edges to keep. For example, 0.1 = keep top 10%.
        id_col (str): Name of the ID column to exclude from thresholding.
        method (str): 'mean' or 'median' to compute the group-level measure.
    """
    df_out = df.copy(deep=True)
    numeric_cols = [c for c in df_out.columns if c != id_col]
    arr = df_out[numeric_cols].to_numpy(dtype=float)  # shape: (N, M)

    # 1) Compute aggregate measure (mean or median)
    if method == 'mean':
        agg_vals = arr.mean(axis=0)  # shape (M,)
    elif method == 'median':
        agg_vals = np.median(arr, axis=0)  # shape (M,)
    else:
        raise ValueError("method must be 'mean' or 'median'")

    # 2) Determine the cutoff
    sorted_agg = np.sort(agg_vals)
    cutoff_index = int(len(sorted_agg) * (1 - threshold))
    cutoff_index = max(0, min(cutoff_index, len(sorted_agg) - 1))
    cutoff_value = sorted_agg[cutoff_index]

    # 3) Create a mask for edges that meet or exceed that cutoff
    global_mask = (agg_vals >= cutoff_value)

    # "Retained" columns are those that pass the mask
    retained_cols = [col for col, keep in zip(numeric_cols, global_mask) if keep]

    # 4) Subset df_out to just the retained columns + the ID column
    df_out = df_out[[id_col] + retained_cols]

    return df_out

def check_memory_correlation(df_change, significant_edges, memory_var='w1_memory'):
    """
    For each edge in significant_edges, compute Pearson correlation with the given memory variable
    across all participants in df_change.

    Args:
      df_change (pd.DataFrame): The master dataframe with columns for edges and memory.
      significant_edges (list): List of edge column names (strings) to test.
      memory_var (str): Column name of the memory variable, e.g., 'w1_memory', 'w2_memory', or 'memory_slopes'.

    Returns:
      pd.DataFrame with columns [edge_label, r, p_uncorrected].
    """
    # Initialize lists to store results
    edge_labels = []
    r_vals = []
    p_vals = []

    # For each significant edge, compute correlation
    for edge in significant_edges:
        if edge not in df_change.columns:
            print(f"Warning: {edge} not found in df_change columns.")
            continue
        
        # Extract non-NaN rows
        valid_idx = df_change[[edge, memory_var]].dropna().index
        x = df_change.loc[valid_idx, edge].values
        y = df_change.loc[valid_idx, memory_var].values

        if len(x) < 3:
            # Not enough data points to correlate
            continue

        r, p = pearsonr(x, y)
        edge_labels.append(edge)
        r_vals.append(r)
        p_vals.append(p)

    results = pd.DataFrame({
        'edge_label': edge_labels,
        'r': r_vals,
        'p_uncorrected': p_vals
    })

    # Sort by p-value for easier interpretation
    results = results.sort_values('p_uncorrected')
    
    return results

def threshold_by_group_and_merge(df_grp1, df_grp2, threshold, id_col='id', method='mean'):
    """
    Applies group-level thresholding to each group separately, takes the union of retained columns,
    and returns df_grp1, df_grp2 each subset to that union of columns (plus the id_col).

    Args:
        df_grp1 (pd.DataFrame): DataFrame for group 1.
        df_grp2 (pd.DataFrame): DataFrame for group 2.
        threshold (float): Fraction of edges to keep. For example, 0.1 = keep top 10%.
        id_col (str): Name of the ID column to exclude from thresholding.
        method (str): 'mean' or 'median' to compute the group-level measure.
    """
    # Threshold each group separately
    df_grp1_thresh = group_level_threshold(df_grp1, threshold, id_col, method)
    df_grp2_thresh = group_level_threshold(df_grp2, threshold, id_col, method)

    # Columns from each thresholded dataframe (excluding id_col)
    grp1_cols = set(df_grp1_thresh.columns) - {id_col}
    grp2_cols = set(df_grp2_thresh.columns) - {id_col}

    # Union of columns
    merged_cols = [id_col] + sorted(list(grp1_cols.union(grp2_cols)))

    # Subset original data so we keep consistent indexing
    df_grp1_out = df_grp1.copy()
    df_grp2_out = df_grp2.copy()

    # Intersect with existing columns in case metadata is present
    df_grp1_out = df_grp1_out[[c for c in merged_cols if c in df_grp1_out.columns]]
    df_grp2_out = df_grp2_out[[c for c in merged_cols if c in df_grp2_out.columns]]

    return df_grp1_out, df_grp2_out

def main(
    connectivity_tp1: Union[str, Path],
    connectivity_tp2: Union[str, Path],
    superager_file: Union[str, Path],
    out_file: Union[str, Path],
    threshold: float = 0.15
       ):
    """
    1) Reads two CSVs with connectivity data for two time points (TP1 and TP2).
    2) Reads a CSV with superager status info (id, superager, maintainer, w1_age, w2_age).
    3) Creates "change" columns: (TP2 - TP1)/mem_time for longitudinal analysis.
    4) For each group comparison, runs:
       a) Threshold each group separately, merge columns from both groups.
       b) T-tests on "change" columns (longitudinal).
       c) Cross-sectional T-tests at TP1 only.
       d) Cross-sectional T-tests at TP2 only.
    5) Applies FDR correction for multiple comparisons, saves CSV of significant edges.
    6) Skips saving if no edges survive FDR.
    """

    # Load Data
    df_tp1 = pd.read_csv(connectivity_tp1).astype({'id': str})
    df_tp2 = pd.read_csv(connectivity_tp2).astype({'id': str})
    df_superager = pd.read_csv(superager_file).astype({'id': str})
    
    # Add 'sub-' prefix in superager file so IDs match, keep only needed columns
    df_superager['id'] = df_superager['id'].apply(lambda x: 'sub-' + x)
    columns_to_keep = ['id', 'w1_age', 'w2_age', 'superager', 'maintainer', 'mem_time', 'memory_slopes', 'w1_memory', 'w2_memory']
    df_superager = df_superager[columns_to_keep]

    # Merge TP1, TP2, and superager info
    df_change = pd.merge(df_tp1, df_tp2, on='id', suffixes=('_tp1', '_tp2'))
    df_change = pd.merge(df_change, df_superager, on='id', how='inner')

    # Create "change" columns: (TP2 - TP1)/mem_time
    # When running 'apply_proportional_threshold', each column is 0 or 1, so “change” will also be restricted in possible values.
    change_data = {}
    for column in df_tp1.columns:
        if column == 'id':
            continue
        col_tp1 = column + '_tp1'
        col_tp2 = column + '_tp2'
        if col_tp1 in df_change.columns and col_tp2 in df_change.columns:
            col_change = column + '_change'
            change_data[col_change] = (
                (df_change[col_tp2] - df_change[col_tp1])
                / df_change['mem_time'] # mem_time is age at TP2 - age at TP1
            )
    df_change = pd.concat([df_change, pd.DataFrame(change_data)], axis=1)

    # No need to remove negative values because they are removed in the thresholding function

    #  Apply group level threshold (e.g., top 10% strongest connections)
    columns_keep = [
        'w1_age', 'w2_age', 'mem_time', 'superager', 
        'maintainer', 'memory_slopes', 'w1_memory', 'w2_memory'
    ]

    # Build separate DataFrames for each group
    df_super       = df_change[df_change['superager'] == 1]
    df_non         = df_change[df_change['superager'] == 0]
    df_maint       = df_change[df_change['maintainer'] == 1]
    df_decl        = df_change[df_change['maintainer'] == 0]
    df_super_maint = df_change[(df_change['superager'] == 1) & (df_change['maintainer'] == 1)]
    df_super_decl  = df_change[(df_change['superager'] == 1) & (df_change['maintainer'] == 0)]
    df_non_maint   = df_change[(df_change['superager'] == 0) & (df_change['maintainer'] == 1)]
    df_non_decl    = df_change[(df_change['superager'] == 0) & (df_change['maintainer'] == 0)]
    
    # Each comparison is a tuple (group1_df, group2_df)
    comparisons = {
        "superagers_vs_nonsuperagers":     (df_super,       df_non),
        "maintainers_vs_decliners":        (df_maint,       df_decl),
        "superagers_vs_decliners":     (df_super,       df_decl),
        "superagers_vs_maintainers":     (df_super,       df_maint),
        "superagerMaint_vs_superagerDecl": (df_super_maint, df_super_decl),
        "nonSuperMaint_vs_nonSuperDecl":   (df_non_maint,   df_non_decl),
        "superagerMaint_vs_nonSuperDecl": (df_super_maint, df_non_decl),
        "nonSuperMaint_vs_superagerDecl":   (df_non_maint,   df_decl),
        # Run this to compare tp1 to tp2
        # "tp1_vs_tp2":                      (df_tp1,         df_tp2)
    }

    # Helper function to run T-tests & FDR on a given set of columns
    def run_ttest_and_fdr(df_group1, df_group2, label, out_stem, keep_suffix):
        """
        Runs t-tests on all columns in df_group1, df_group2 after dropping
        metadata. Then applies FDR correction, saves CSV of significant
        edges + hist plots of p-values. If no edges survive FDR, stops.
        
        Args:
        df_group1, df_group2: DataFrames containing columns to test
        label:      short text label for print statements
        out_stem:   base filename stem (e.g. 'longitudinal', 'tp1', etc.)
        keep_suffix: suffix to keep for column filtering
        """
        # 1) Subset to the relevant suffix + forced metadata columns
        df_g1 = df_group1[
            [c for c in df_group1.columns 
            if c.endswith(keep_suffix) or c in columns_keep or c == 'id']
        ]

        df_g2 = df_group2[
            [c for c in df_group2.columns 
            if c.endswith(keep_suffix) or c in columns_keep or c == 'id']
        ]

        # 2) Threshold by group
        df_g1, df_g2 = threshold_by_group_and_merge(
            df_g1, df_g2, threshold=0.1, id_col='id', method='mean'
        )

        # 3) Drop forced metadata, leaving only numeric columns
        columns_to_drop = ['id','w1_age','w2_age','mem_time','superager','maintainer', 'memory_slopes', 'w1_memory', 'w2_memory']  
        df_g1.drop(columns=columns_to_drop, inplace=True, errors="ignore")  
        df_g2.drop(columns=columns_to_drop, inplace=True, errors="ignore")  

        if df_g1.empty or df_g2.empty:
            print(f"[{label}][{out_stem}] One group is empty or no valid columns. Skipping.")
            return

        # Convert to numpy arrays for t-test
        arr1 = df_g1.to_numpy()
        arr2 = df_g2.to_numpy()
        edge_labels = df_g1.columns.tolist()

        # Initialize arrays to store p-values
        num_edges = len(edge_labels)
        pvals = np.zeros(num_edges)

        # T-test across each edge (column)
        for i in range(num_edges):
            _, pvals[i] = ttest_ind(arr1[:, i], arr2[:, i], nan_policy='omit')

        # Run this t-test if comparing tp1 to tp2 because the subjects are related
        # for i in range(num_edges):
        #     _, pvals[i] = ttest_rel(arr1[:, i], arr2[:, i], nan_policy='omit')

        # 1. PRINT: Uncorrected significant edges
        raw_signif = (pvals < 0.05).sum()
        print(f"[{label}][{out_stem}] {raw_signif} / {num_edges} ({(raw_signif / num_edges) * 100:.2f}%) edges p < 0.05 (uncorrected).")

        # 2. PRINT: FDR corrected significant edges
        # Apply False Discovery Rate (FDR) correction
        rejected, pvals_corr, _, _ = multipletests(pvals, method='fdr_bh')
        num_sig = rejected.sum()
        print(f"[{label}][{out_stem}] {num_sig} / {num_edges} ({(num_sig / num_edges) * 100:.2f}%) edges significant after FDR correction.")

        # Initialize a dictionary to store significant edge lists
        df_sig_uncorr = pd.DataFrame({
            'edge_label': edge_labels,
            'p_uncorrected': pvals
        })

        # Filter to get just significant edges based on uncorrected p-values
        significant_edge_labels = df_sig_uncorr[df_sig_uncorr['p_uncorrected'] < 0.05]['edge_label'].tolist()

        # 3. PRINT: Memory correlations
        # Check memory correlation for significant edges based on the analysis type
        if significant_edge_labels:
            # Select which memory variables to correlate based on out_stem
            if out_stem == "change":
                # For longitudinal analysis, only correlate with memory change
                results_change = check_memory_correlation(df_change, significant_edge_labels, memory_var='memory_slopes')
                sig_results_change = results_change[results_change['p_uncorrected'] < 0.05]
                if not sig_results_change.empty:
                    print(f"[{label}][{out_stem}] {len(sig_results_change)} significant memory correlations")
                else:
                    print(f"[{label}][{out_stem}] No significant memory correlations")
                    
            elif out_stem == "tp1":
                # For timepoint 1 analysis, only correlate with timepoint 1 memory
                results_w1 = check_memory_correlation(df_change, significant_edge_labels, memory_var='w1_memory')
                sig_results_w1 = results_w1[results_w1['p_uncorrected'] < 0.05]
                if not sig_results_w1.empty:
                    print(f"[{label}][{out_stem}] {len(sig_results_w1)} significant memory correlations")
                else:
                    print(f"[{label}][{out_stem}] No significant memory correlations")
                    
            elif out_stem == "tp2":
                # For timepoint 2 analysis, only correlate with timepoint 2 memory
                results_w2 = check_memory_correlation(df_change, significant_edge_labels, memory_var='w2_memory')
                sig_results_w2 = results_w2[results_w2['p_uncorrected'] < 0.05]
                if not sig_results_w2.empty:
                    print(f"[{label}][{out_stem}] {len(sig_results_w2)} significant memory correlations")
                else:
                    print(f"[{label}][{out_stem}] No significant memory correlations")

        # If no edges survive FDR, stop here
        if num_sig == 0:
            return

        # Save histogram plots for p-values & corrected p-values
        plt.figure()
        plt.hist(pvals, bins=50, edgecolor='k')
        plt.title(f"P-values ({label}, {out_stem})")
        plt.savefig(out_file.parent / f"pvals_{label}_{out_stem}.png")
        plt.close()

        plt.figure()
        plt.hist(pvals_corr, bins=50, edgecolor='k')
        plt.title(f"FDR-corrected P-values ({label}, {out_stem})")
        plt.savefig(out_file.parent / f"pvals_corr_{label}_{out_stem}.png")
        plt.close()

        # Compute group means
        mean_g1 = arr1.mean(axis=0)
        mean_g2 = arr2.mean(axis=0)

        sig_edges = []
        for i, is_sig in enumerate(rejected):
            if is_sig:
                sig_edges.append({
                    'edge_label': edge_labels[i],
                    'p_uncorrected': pvals[i],
                    'p_fdr': pvals_corr[i],
                    f'mean_{label}_group1': mean_g1[i],
                    f'mean_{label}_group2': mean_g2[i],
                })

        df_sig = pd.DataFrame(sig_edges).sort_values('p_fdr')
        csv_name = out_file.parent / f"{out_file.stem}_{label}_{out_stem}{out_file.suffix}"
        df_sig.to_csv(csv_name, index=False)
        print(f"[{label}][{out_stem}] Saved {len(df_sig)} significant edges to {csv_name}")
        print("------------------------------------------------")

    # MAIN LOOP: Run all comparisons
    for label, (df_grp1, df_grp2) in comparisons.items():
        print(f"\n=== Starting analysis: {label} ===")

        # 1) Longitudinal: use meta_cols_longitudinal
        run_ttest_and_fdr(df_grp1, df_grp2, label, out_stem="change", keep_suffix="_change")

        # 2) Cross-sectional at TP1
        run_ttest_and_fdr(df_grp1, df_grp2, label, out_stem="tp1", keep_suffix="_tp1")

        # 3) Cross-sectional at TP2
        run_ttest_and_fdr(df_grp1, df_grp2, label, out_stem="tp2", keep_suffix="_tp2")

        # 4) TP1 vs TP2 - NOTE that when running this analysis the change, tp1 and tp2 results are all the same 
        # run_ttest_and_fdr(df_grp1, df_grp2, label, out_stem="", meta_cols=meta_cols_tp1_tp2)

if __name__ == "__main__":

    output_directory = Path("/home/rachel/Desktop/schaefer_analysis/connectivity_matrices")
    superager_file    = "/home/rachel/Desktop/data/maintainer_superager_data.csv"

    # All ROIs
    connectivity_tp1 = output_directory / "ses-01/all_to_all_roi_matrices/fisher_z_all_to_all_roi_matrix.csv"
    connectivity_tp2 = output_directory / "ses-02/all_to_all_roi_matrices/fisher_z_all_to_all_roi_matrix.csv"

    # Network specific ROIs
    # connectivity_tp1 = output_directory / "ses-01/within_network_matrices/fisher_z_Default_within_network_matrix.csv"
    # connectivity_tp2 = output_directory / "ses-02/within_network_matrices/fisher_z_Default_within_network_matrix.csv"

    out_file = output_directory / "significant_longitudinal.csv"

    main(
        connectivity_tp1=connectivity_tp1,
        connectivity_tp2=connectivity_tp2,
        superager_file=superager_file,
        out_file=out_file
    )