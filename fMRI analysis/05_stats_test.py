import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from typing import Union
from scipy.stats import ttest_ind, ttest_rel
from statsmodels.stats.multitest import multipletests

# Unthresholded analysis 

def main(
        connectivity_tp1: Union[str, Path], 
        connectivity_tp2: Union[str, Path], 
        superager_file: Union[str, Path], 
        out_file: Union[str, Path]
       ):
    """
    1) Reads two CSVs with connectivity data for two time points (TP1 and TP2).
    2) Reads a CSV with superager status info (id, superager, maintainer, w1_age, w2_age).
    3) Creates "change" columns: (TP2 - TP1)/mem_time for longitudinal analysis.
    4) For each group comparison, runs:
       a) T-tests on "change" columns (longitudinal).
       b) Cross-sectional T-tests at TP1 only.
       c) Cross-sectional T-tests at TP2 only.
    5) Applies FDR correction for multiple comparisons, saves CSV of significant edges.
    6) Skips saving if no edges survive FDR.
    """

    # Load Data
    df_tp1 = pd.read_csv(connectivity_tp1).astype({'id': str})
    df_tp2 = pd.read_csv(connectivity_tp2).astype({'id': str})
    df_superager = pd.read_csv(superager_file).astype({'id': str})
    
    # Add 'sub-' prefix in superager file so IDs match, keep only needed columns
    df_superager['id'] = df_superager['id'].apply(lambda x: 'sub-' + x)
    columns_to_keep = ['id', 'w1_age', 'w2_age', 'superager', 'maintainer', 'mem_time']
    df_superager = df_superager[columns_to_keep]

    # Convert connectivity values to ABSOLUTE value
    # df_tp1.iloc[:, 1:] = df_tp1.iloc[:, 1:].abs()
    # df_tp2.iloc[:, 1:] = df_tp2.iloc[:, 1:].abs()

    # Remove NEGATIVE connectivity values 
    # df_tp1.iloc[:, 1:] = df_tp1.iloc[:, 1:].where(df_tp1.iloc[:, 1:] >= 0, np.nan)
    # df_tp2.iloc[:, 1:] = df_tp2.iloc[:, 1:].where(df_tp2.iloc[:, 1:] >= 0, np.nan)

    # Merge TP1, TP2, and superager info
    df_change = pd.merge(df_tp1, df_tp2, on='id', suffixes=('_tp1', '_tp2'))
    df_change = pd.merge(df_change, df_superager, on='id', how='inner')

    # Create "change" columns: (TP2 - TP1)/mem_time
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
        "tp1_vs_tp2":                      (df_tp1,         df_tp2)
    }

    # For the CHANGE analysis, we want to drop both raw tp1/tp2 columns
    meta_cols_longitudinal = (
        ['id','w1_age','w2_age','mem_time','superager','maintainer']
        + [col+'_tp1' for col in df_tp1.columns if col != 'id']
        + [col+'_tp2' for col in df_tp1.columns if col != 'id']
    )

    # For the CROSS-SECTIONAL analysis at TP1, we want to drop mem_time, 
    # plus the TP2 columns and “_change” columns, but KEEP columns ending in _tp1
    meta_cols_tp1 = (
        ['id','w1_age','w2_age','mem_time','superager','maintainer']
        + [col+'_tp2' for col in df_tp1.columns if col != 'id']
        + [col+'_change' for col in df_tp1.columns if col != 'id']
    )

    # For CROSS-SECTIONAL analysis at TP2:
    # keep everything that ends in _tp2, but drop _tp1 and _change columns.
    meta_cols_tp2 = (
        ['id','w1_age','w2_age','mem_time','superager','maintainer']
        + [col+'_tp1' for col in df_tp1.columns if col != 'id']
        + [col+'_change' for col in df_tp1.columns if col != 'id']
    )

    # For TP1 vs TP2 ananlysis:
    meta_cols_tp1_tp2 = (
        ['id','w1_age','w2_age','mem_time','superager','maintainer']
    )

    # Helper function to run T-tests & FDR on a given set of columns
    def run_ttest_and_fdr(df_group1, df_group2, label, out_stem, meta_cols):
        """
        Runs t-tests on all columns in df_group1, df_group2 after dropping
        metadata. Then applies FDR correction, saves CSV of significant
        edges + hist plots of p-values. If no edges survive FDR, stops.
        
        Args:
          df_group1, df_group2: DataFrames containing columns to test
          label:      short text label for print statements
          out_stem:   base filename stem (e.g. 'longitudinal', 'tp1', etc.)
          meta_cols:  list of metadata columns to drop before testing
        """
        # Drop metadata columns
        df_g1 = df_group1.drop(columns=meta_cols, errors='ignore')
        df_g2 = df_group2.drop(columns=meta_cols, errors='ignore')

        # FOR LIMITED ANALYSIS DROP ALL ROWS WITH ONLY 0 BECAUSE THEY CANT BE COMPARED
        df_g1 = df_g1.loc[(df_g1 != 0).any(axis=1)]
        df_g2 = df_g2.loc[(df_g2 != 0).any(axis=1)]

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

        # Set significance level to 0.05
        raw_signif = (pvals < 0.05).sum()
        print(f"[{label}][{out_stem}] {raw_signif} / {num_edges} ({(raw_signif / num_edges) * 100:.2f}%) edges p < 0.05 (uncorrected).")

        # FDR Correction
        rejected, pvals_corr, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
        num_sig = rejected.sum()
        print(f"[{label}][{out_stem}] {num_sig} edges survive FDR=0.05.")

        # If no significant edges, skip saving CSV & plots
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

    # Helper function to do cross-sectional tests at TP1 or TP2
    def run_cross_sectional(df_group1, df_group2, label, tp_suffix, meta_cols): 
        """
        For cross-sectional analysis (TP1 or TP2).
        Keeps only columns that end in _tp1 or _tp2 (except 'id').
        Then passes them to run_ttest_and_fdr.

        Args:
            df_group1, df_group2: DataFrames containing columns to test
            label:      short text label for print statements
            tp_suffix:  suffix to filter columns (e.g. '_tp1' or '_tp2')
            meta_cols:  list of metadata columns to drop before testing
        """
        # Copy data so can safely drop columns
        df_g1 = df_group1.copy()
        df_g2 = df_group2.copy()

        # Keep only columns that match the timepoint suffix (plus metadata)
        keep_cols = [c for c in df_g1.columns 
                     if (c.endswith(tp_suffix) or c in meta_cols)]
        
        df_g1 = df_g1[keep_cols]
        df_g2 = df_g2[keep_cols]

        # Then call run_ttest_and_fdr using out_stem = "tp1" or "tp2"
        run_ttest_and_fdr(df_g1, df_g2, label, out_stem=tp_suffix)

    # MAIN LOOP: Run all comparisons
    for label, (df_grp1, df_grp2) in comparisons.items():
        print(f"\n=== Starting analysis: {label} ===")

        # 1) Longitudinal: use meta_cols_longitudinal
        run_ttest_and_fdr(df_grp1, df_grp2, label, out_stem="change", meta_cols=meta_cols_longitudinal)

        # 2) Cross-sectional at TP1
        run_ttest_and_fdr(df_grp1, df_grp2, label, out_stem="tp1", meta_cols=meta_cols_tp1)

        # 3) Cross-sectional at TP2
        run_ttest_and_fdr(df_grp1, df_grp2, label, out_stem="tp2", meta_cols=meta_cols_tp2)

        # 4) TP1 vs TP2 - NOTE that when running this analysis the change, tp1 and tp2 results are all the same 
        # run_ttest_and_fdr(df_grp1, df_grp2, label, out_stem="", meta_cols=meta_cols_tp1_tp2)

if __name__ == "__main__":

    output_directory = Path("/home/rachel/Desktop/schaefer_analysis/connectivity_matrices")
    superager_file    = "/home/rachel/Desktop/data/maintainer_superager_data.csv"

    # All ROIs
    # connectivity_tp1 = output_directory / "ses-01/all_to_all_roi_matrices/fisher_z_all_to_all_roi_matrix.csv"
    # connectivity_tp2 = output_directory / "ses-02/all_to_all_roi_matrices/fisher_z_all_to_all_roi_matrix.csv"

    # Network specific ROIs
    connectivity_tp1 = output_directory / "ses-01/subcortical_matrices/fisher_z_Thalamus_bilateral_matrix.csv"
    connectivity_tp2 = output_directory / "ses-02/subcortical_matrices/fisher_z_Thalamus_bilateral_matrix.csv"

    out_file = output_directory / "significant_longitudinal.csv"

    main(
        connectivity_tp1=connectivity_tp1,
        connectivity_tp2=connectivity_tp2,
        superager_file=superager_file,
        out_file=out_file
    )