import numpy as np
import pandas as pd
from pathlib import Path
import os, re
import nibabel as nib
from collections import Counter
from nilearn.datasets import fetch_atlas_schaefer_2018
from functools import lru_cache

def get_subjects_to_process(output_folder, ses, id_csv_path, age_dir):
    """Generate a list of subjects to process ensuring each subject is 
    also in the id column of the CSV. Also, filters for any subs with
    <1.8 years follow-up time.

    Args:
        output_folder (Path): Path to the directory coupling results
        ses (str): Session ID (format: ses-01).
        id_csv_path (Path or str): Path to CSV file with 'id' column.
        age_dir (Path): Path to the directory containing the age data CSV.

    Returns:
        list: List of subject IDs to process.
    """
    # Load valid subject IDs from the CSV 
    ids = set(pd.read_csv(id_csv_path)['id'].astype(str))

    # Merge in the age data
    age_data = pd.read_csv(age_dir / "maintainer_superager_data.csv")
    age_data.columns = [re.sub(r"^w(\d)_(.*)", r"\2_\1", col) for col in age_data.columns] # Rename the columns
    age_data['id'] = 'sub-' + age_data['id'].astype(str) # Add 'sub-' to the id
    age_data_filt = age_data[['id', 'age_1', 'age_2']].copy() # Keep only the relevant columns

    # Create a new fu_time variable
    age_data_filt['fu_time'] = age_data_filt['age_2'] - age_data_filt['age_1'] 

    # Drop participant with fu_time NA
    age_data_filt = age_data_filt.dropna(subset=['fu_time'])

    # Create a set of valid ids that have fu_time >1.8 years
    valid_ids = set(age_data_filt[age_data_filt['fu_time'] > 1.8]['id'].astype(str))

    # Subset ids to only those in valid_ids
    ids = ids.intersection(valid_ids)

    subjects = []
    for fname in os.listdir(output_folder):
        if not fname.startswith("sub-") or not fname.endswith(f"{ses}_structure_function_coupling.csv"):
            continue

        # Extract subject ID 
        subject = fname.split("_")[0]

        sfc_path = output_folder / fname
        if sfc_path.exists() and subject in valid_ids:
            subjects.append(subject)

    return subjects


def flatten_connectivity_csv(matrix_csv, measure_col="pearson_rho"):
    """Reads an NxN connectivity CSV (214×214) and flattens it into a long DataFrame
    by taking the row-wise mean to match the format of SFC data which takes the rho
    of each ROI connection in the row, weighing them the same. Returns columns: 
    ROI_name, measure_col.

    Args:
        matrix_csv (str or Path): Path to the NxN CSV.
        measure_col (str): Name for the connectivity measure (e.g. "sc_value", "fc_value").

    Returns:
        pd.DataFrame with columns ["ROI_name", measure_col].
    """
    # Read the NxN matrix 
    df_matrix = pd.read_csv(matrix_csv, header=0, index_col=0)
    roi_list = df_matrix.index.tolist()  # ROI names

    # Compute the mean connectivity for each ROI (row-wise mean)
    avg_conn = df_matrix.mean(axis=1).values

    # Build the output DataFrame
    df_out = pd.DataFrame({
        "ROI_name": roi_list,
        measure_col: avg_conn
    })

    return df_out


_SCHAEFER_LABEL_TO_COUNT = None
def _get_schaefer_label_to_count():
    """Cached mapping of Schaefer 200 ROIs to their voxel counts
    to speed up weighted averaging.

    Returns:
        dict: Mapping of ROI label to voxel count.
    """
    global _SCHAEFER_LABEL_TO_COUNT
    if _SCHAEFER_LABEL_TO_COUNT is None:
        atl = fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7, resolution_mm=1)
        atlas = nib.load(atl.maps).get_fdata().astype(int)
        cortical_counts = Counter(atlas[atlas > 0].ravel())
        labels = [l.decode() if isinstance(l, bytes) else str(l) for l in atl.labels]
        _SCHAEFER_LABEL_TO_COUNT = {labels[i]: cortical_counts[i]
                                    for i in cortical_counts.keys()
                                    if 0 <= i < len(labels)}
    return _SCHAEFER_LABEL_TO_COUNT


@lru_cache(maxsize=None) 
def _get_subcort_counts(subject: str, ses: str):
    """Load and cache the voxel counts for subcortical ROIs from the 
    subject's aseg.mgz file.
    
    Args:
        subject (str): Subject ID (e.g. "sub-1234").
        ses (str): Session ID (e.g. "ses-01").
    
    Returns:
        Counter: Mapping of aseg label ID to voxel count.
    """
    cohort = "bbhi" if int(subject.split("-")[1]) > 5000 else "bbhi senior"
    if cohort == "bbhi":
        aseg_file = Path(f"/pool/guttmann/institut/BBHI/MRI/derivatives/freesurfer-reconall/{subject}_{ses}_run-01/mri/aseg.mgz")
    else:
        aseg_file = Path(f"/pool/guttmann/institut/UB/Superagers/MRI/derivatives/freesurfer-reconall/{subject}_{ses}/mri/aseg.mgz")
    aseg = nib.load(aseg_file).get_fdata().astype(int)
    return Counter(aseg[aseg > 0].ravel())


def save_grouped_roi_averages(csv_path, output_path, group_level, subject, ses):
    """Reads a subject's connectivity CSV (SFC/FC/SC),
    then groups & averages the coefficients either:
      - by ROI prefix (strip trailing _<digits>), or
      - by network (for cortical: 3rd "_" field; for subcortical: region name).

    Args:
        csv_path (str or Path): Path to the input CSV file.
        output_path (str or Path): Path to save the grouped averages CSV.
        group_level (str): Grouping level, either "ROI" or "network".
                           "ROI" groups by ROI prefix (eg PFCv) 
                           "network" groups by network name (eg DMN)
        subject (str): Subject ID 
        ses (str): Session ID
    """
    df = pd.read_csv(csv_path)
    df["ROI_name"] = df["ROI_name"].astype(str)

    # Cached Schaefer voxel counts per ROI label 
    label_to_count = _get_schaefer_label_to_count()
    df["weight"] = df["ROI_name"].map(label_to_count).fillna(1)

    # Then get them for the subcortical ROIs
    subcort_counts = _get_subcort_counts(subject, ses)

    subcort_map = {
        "Subcortical 201: Left Hippocampus": 17, "Subcortical 202: Left Amygdala": 18, "Subcortical 203: Left Pallidum": 13,
        "Subcortical 204: Left Putamen": 12, "Subcortical 205: Left Caudate": 11, "Subcortical 206: Left Accumbens": 26,
        "Subcortical 207: Left Thalamus": 10, "Subcortical 208: Right Hippocampus": 53, "Subcortical 209: Right Amygdala": 54,
        "Subcortical 210: Right Pallidum": 52, "Subcortical 211: Right Putamen": 51, "Subcortical 212: Right Caudate": 50,
        "Subcortical 213: Right Accumbens": 58, "Subcortical 214: Right Thalamus": 49,
    }

    # Map subcortical voxel counts and override weight where applicable
    df["aseg_id"] = df["ROI_name"].map(subcort_map)
    df["weight_subcort"] = df["aseg_id"].map(subcort_counts)

    # Prefer subcortical voxel count when ROI_name is subcortical
    df.loc[df["ROI_name"].str.startswith("Subcortical"), "weight"] = df["weight_subcort"]
    df["weight"] = df["weight"].fillna(1).astype(float)

    # Clean up helper columns
    df.drop(columns=["aseg_id", "weight_subcort"], inplace=True)

    # Group by ROI prefix (strip trailing _<digits>)
    if group_level.lower() == "roi":
        df["ROI_group"] = df["ROI_name"].str.replace(r'_\d+$', '', regex=True)
        df["w_rho"] = df["pearson_rho"] * df["weight"]
        agg = df.groupby("ROI_group", as_index=False).agg(
            wsum=("w_rho", "sum"),
            w=("weight", "sum"),
        )
        grouped = (
            agg.assign(pearson_rho=agg["wsum"] / agg["w"])[["ROI_group", "pearson_rho"]]
               .rename(columns={"ROI_group": "ROI_name"})
        )

    # Group by networks to have 7 cortical networks and 7 subcortical regions (classed as networks here)
    elif group_level.lower() == "network":
        def network_key(name):
            if name.startswith("Subcortical"):
                parts = name.split()
                # parts = ["Subcortical","208","Right","Hippocampus",...]
                region = " ".join(parts[3:]) # Combines the left and right hemispheres
                return region
            else:
                # Cortical ROIs: "7Networks_RH_Cont_pCun"
                parts = name.split("_")
                if len(parts) >= 3:
                    return parts[2]
                # If the name does not match the expected format, return it as is
                return name

        df["network"] = df["ROI_name"].apply(network_key)

        # Weighted average of pearson_rho by network using voxel-based weights
        tmp = df[["network", "pearson_rho", "weight"]].copy()
        tmp["w_rho"] = tmp["pearson_rho"] * tmp["weight"]
        agg = tmp.groupby("network", as_index=False).agg(
            wsum=("w_rho", "sum"),
            w=("weight", "sum"),
        )
        grouped = (
            agg.assign(pearson_rho=agg["wsum"] / agg["w"])[["network", "pearson_rho"]]
               .rename(columns={"network": "ROI_name"})
        )

    print(f"Conversion complete for {subject} {ses}!")

    grouped.to_csv(output_path, index=False)


def main():
    sessions = ["ses-01", "ses-02"] 
    group_level = "ROI"
    root_path = Path("/home/rachel/Desktop/schaefer_analysis/structure_function_coupling")
    fc_root_path = Path("/home/rachel/Desktop/schaefer_analysis/functional_connectivity/native_space")
    sc_root_path = Path("/home/rachel/Desktop/schaefer_analysis/structural_connectivity")
    csv_path = Path("/home/rachel/Desktop/data/clean_data_all.csv")
    age_dir = Path("/home/rachel/Desktop/data")

    # Get the list of subjects to process
    subjects_tp1 = get_subjects_to_process(root_path / "ses-01" / "individual_coupling_matrices", "ses-01", csv_path, age_dir)
    subjects_tp2 = get_subjects_to_process(root_path / "ses-02" / "individual_coupling_matrices", "ses-02", csv_path, age_dir)
    subjects = sorted(set(subjects_tp1) & set(subjects_tp2))
    print(f"Subjects: {len(subjects)}")

    # Make the flattened FC and SC CSVs for each subject
    for sub in subjects:
        for ses in sessions:
            ses_path_fc = fc_root_path / ses / "individual_connectivity_matrices"
            ses_path_sc = sc_root_path / ses / "individual_connectivity_matrices"

            # Functional connectivity 
            fc_csv = ses_path_fc / f"{sub}_{ses}_functional_connectivity_matrix_fisher_z.csv"
            if fc_csv.is_file():
                fc_flat = flatten_connectivity_csv(fc_csv, measure_col="pearson_rho")

                fc_output_dir = ses_path_fc / "grouped_rois"
                fc_output_dir.mkdir(parents=True, exist_ok=True)  # Ensure dir exists

                fc_output = fc_output_dir / f"{sub}_{ses}_functional_connectivity_flat.csv"
                fc_flat.to_csv(fc_output, index=False)  # Save the flattened version

                # Group the flattened CSV by ROI
                grouped_output = fc_output_dir / f"{sub}_{ses}_functional_connectivity_grouped_by_{group_level}.csv"
                # Adding group_level="network" allows looking at all of the DMN for example rather than individual ROIs
                save_grouped_roi_averages(fc_output, grouped_output, group_level = group_level, subject=sub, ses=ses) 

            # Structural connectivity 
            sc_csv = ses_path_sc / f"{sub}_{ses}_structural_connectivity_matrix_normalized.csv"
            if sc_csv.is_file():
                sc_flat = flatten_connectivity_csv(sc_csv, measure_col="pearson_rho")

                sc_output_dir = ses_path_sc / "grouped_rois"
                sc_output_dir.mkdir(parents=True, exist_ok=True)  # Ensure dir exists

                sc_output = sc_output_dir / f"{sub}_{ses}_structural_connectivity_flat.csv"
                sc_flat.to_csv(sc_output, index=False)  # Save the flattened version

                # Group the flattened CSV by ROI
                grouped_output = sc_output_dir / f"{sub}_{ses}_structural_connectivity_grouped_by_{group_level}.csv"
                save_grouped_roi_averages(sc_output, grouped_output, group_level = group_level, subject=sub, ses=ses)

            else:
                print(f"Missing SC CSV at path: {sc_csv}")

    # Make the grouped averages for each subject's SFC CSV   
    for sub in subjects:
        for ses in sessions:
            ses_path = root_path / ses / "individual_coupling_matrices"
            csv_path = ses_path / f"{sub}_{ses}_structure_function_coupling.csv"
            group_level = "ROI"
            output_path = ses_path / f"{sub}_{ses}_structure_function_coupling_grouped_by_{group_level}.csv"
            if csv_path.is_file():
                save_grouped_roi_averages(csv_path, output_path, group_level = group_level, subject=sub, ses=ses)


if __name__ == "__main__":
    main()