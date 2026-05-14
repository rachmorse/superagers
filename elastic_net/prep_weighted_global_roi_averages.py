from collections import Counter
from functools import lru_cache
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.datasets import fetch_atlas_schaefer_2018
from prep_data_for_en import get_subjects_to_process

DEBUG = False
_PRINTED_ROI_DOMAIN_EXAMPLES = False
_SUBCORT_DEBUG_PRINTS = 0
_SCHAEFER_LABEL_TO_COUNT = None  # Lazy-loaded cache of Schaefer ROI -> voxel count weights.
SFC_NETWORKS = ("SalVentAttn", "Cont", "Default")
SFC_NETWORK_COL_SUFFIX = {
    "SalVentAttn": "salience",
    "Cont": "control",
    "Default": "dmn",
}


def _dbg(msg: str):
    """Print debug message if DEBUG is True."""
    if DEBUG:
        print(f"[DEBUG] {msg}")


def get_weighted_roi_mean(df: pd.DataFrame, subject: str, ses: str) -> float:
    """Compute voxel-weighted global mean of `pearson_rho` across all ROIs.
    
    Args:
        df: DataFrame with columns "ROI_name" and "pearson_rho".
        subject: Subject ID.
        ses: Session ID.

    Returns:
        Weighted mean of `pearson_rho` across all ROIs.
    """
    valid = _prepare_weighted_df(df, subject, ses)
    if valid.empty or np.isclose(valid["weight"].sum(), 0.0):
        return np.nan
    return float((valid["pearson_rho"] * valid["weight"]).sum() / valid["weight"].sum())


def get_sfc_sensory_hmod_means(df: pd.DataFrame, subject: str, ses: str):
    """Compute weighted means split by sensory (Vis/SomMot) vs heteromodal.
    
    Args:
        df: DataFrame with columns "ROI_name" and "pearson_rho".
        subject: Subject ID.
        ses: Session ID.

    Returns:
        Tuple of (sensory_mean, hmod_mean), where each is the weighted mean of
        `pearson_rho` for the respective ROI rows.
    """
    sensory_networks = {"Vis", "SomMot"}

    valid = _prepare_weighted_df(df, subject, ses)
    if valid.empty:
        return np.nan, np.nan

    def _domain_from_roi(name: str) -> str:
        """Classify ROI as "sensory" vs "hmod" based on name. 
        Subcortical ROIs are classified as "hmod
        
        Args: name (str): ROI name.

        Returns:
            str: "sensory" or "hmod".
        """
        if name.startswith("Subcortical"):
            return "hmod"
        parts = name.split("_")
        network = parts[2] if len(parts) >= 3 else ""
        return "sensory" if network in sensory_networks else "hmod"

    global _PRINTED_ROI_DOMAIN_EXAMPLES
    valid["domain"] = valid["ROI_name"].astype(str).apply(_domain_from_roi)
    if DEBUG and not _PRINTED_ROI_DOMAIN_EXAMPLES:
        sensory_sample5 = valid.loc[valid["domain"] == "sensory", "ROI_name"].sample(5).tolist()
        hmod_sample10 = valid.loc[valid["domain"] == "hmod", "ROI_name"].sample(10).tolist()
        _dbg(f"Example sensory ROIs (sample 5): {sensory_sample5}")
        _dbg(f"Example hmod ROIs (sample 10): {hmod_sample10}")
        _PRINTED_ROI_DOMAIN_EXAMPLES = True

    out = {}
    for domain in ["sensory", "hmod"]:
        d = valid[valid["domain"] == domain]
        if d.empty or np.isclose(d["weight"].sum(), 0.0):
            out[domain] = np.nan
        else:
            out[domain] = float((d["pearson_rho"] * d["weight"]).sum() / d["weight"].sum())
    return out["sensory"], out["hmod"]


def get_sfc_network_means(
    df: pd.DataFrame, subject: str, ses: str, networks=SFC_NETWORKS
):
    """Compute weighted means for specific Schaefer networks.

    Args:
        df: DataFrame with columns "ROI_name" and "pearson_rho".
        subject: Subject ID.
        ses: Session ID.
        networks: Network names to compute means for (e.g. "Vis", "Default").

    Returns:
        Dict mapping each requested network name to its weighted mean `pearson_rho`.
    """
    valid = _prepare_weighted_df(df, subject, ses)
    if valid.empty:
        return {network: np.nan for network in networks}

    cortical = valid[~valid["ROI_name"].astype(str).str.startswith("Subcortical")].copy()
    if cortical.empty:
        return {network: np.nan for network in networks}

    def _network_from_roi(name: str) -> str:
        parts = str(name).split("_")
        return parts[2] if len(parts) >= 3 else ""

    cortical["network"] = cortical["ROI_name"].astype(str).apply(_network_from_roi)
    out = {}
    for network in networks:
        d = cortical[cortical["network"] == network]
        _dbg(f"Example ROIs in {network} network: {d['ROI_name'].sample(min(5, len(d))).tolist()}")
        if d.empty or np.isclose(d["weight"].sum(), 0.0):
            out[network] = np.nan
        else:
            out[network] = float((d["pearson_rho"] * d["weight"]).sum() / d["weight"].sum())
    return out


def _prepare_weighted_df(df: pd.DataFrame, subject: str, ses: str) -> pd.DataFrame:
    """Attach voxel-based weights and return valid rows for weighted averaging.
    
    Args:
        df: DataFrame with columns "ROI_name" and "pearson_rho".
        subject: Subject ID.
        ses: Session ID.

    Returns:
        DataFrame with additional weight column, filtered to rows with valid `pearson_rho`.
    """
    df = df.copy()
    df["ROI_name"] = df["ROI_name"].astype(str)
    n_before = len(df)
    df["pearson_rho"] = pd.to_numeric(df["pearson_rho"], errors="coerce")

    label_to_count = _get_schaefer_label_to_count()
    df["weight"] = df["ROI_name"].map(label_to_count)

    subcort_counts = _get_subcort_counts(subject, ses)
    subcort_map = {
        "Subcortical 201: Left Hippocampus": 17,
        "Subcortical 202: Left Amygdala": 18,
        "Subcortical 203: Left Pallidum": 13,
        "Subcortical 204: Left Putamen": 12,
        "Subcortical 205: Left Caudate": 11,
        "Subcortical 206: Left Accumbens": 26,
        "Subcortical 207: Left Thalamus": 10,
        "Subcortical 208: Right Hippocampus": 53,
        "Subcortical 209: Right Amygdala": 54,
        "Subcortical 210: Right Pallidum": 52,
        "Subcortical 211: Right Putamen": 51,
        "Subcortical 212: Right Caudate": 50,
        "Subcortical 213: Right Accumbens": 58,
        "Subcortical 214: Right Thalamus": 49,
    }
    df["aseg_id"] = df["ROI_name"].map(subcort_map)
    df["weight_subcort"] = df["aseg_id"].map(subcort_counts)
    df.loc[df["ROI_name"].str.startswith("Subcortical"), "weight"] = df["weight_subcort"]

    missing_weights = df.loc[df["weight"].isna(), "ROI_name"].unique().tolist()
    if missing_weights:
        preview = ", ".join(missing_weights[:10])
        raise ValueError(
            f"Missing voxel-based weights for {len(missing_weights)} ROI labels. "
            f"First labels: {preview}"
        )

    valid = df[df["pearson_rho"].notna()].copy()
    n_dropped = n_before - len(valid)
    if DEBUG and n_dropped > 0:
        _dbg(f"{subject} {ses}: dropped {n_dropped}/{n_before} rows with non-numeric pearson_rho")
    valid["weight"] = valid["weight"].astype(float)
    if DEBUG and np.isclose(valid["weight"].sum(), 0.0):
        _dbg(f"{subject} {ses}: weight sum is ~0 after filtering")
    return valid


def _get_schaefer_label_to_count():
    """Fetch Schaefer atlas info and return mapping of ROI label to voxel count, with caching."""
    global _SCHAEFER_LABEL_TO_COUNT
    if _SCHAEFER_LABEL_TO_COUNT is None:
        atl = fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7, resolution_mm=1)
        atlas = nib.load(atl.maps).get_fdata().astype(int)
        cortical_counts = Counter(atlas[atlas > 0].ravel())
        labels = [l.decode() if isinstance(l, bytes) else str(l) for l in atl.labels]
        max_roi_id = max(cortical_counts.keys())
        if len(labels) == max_roi_id + 1:
            roi_id_to_label = {
                i: labels[i] for i in cortical_counts.keys() if 0 <= i < len(labels)
            }
        elif len(labels) == max_roi_id:
            roi_id_to_label = {
                i: labels[i - 1] for i in cortical_counts.keys() if 1 <= i <= len(labels)
            }
        else:
            raise ValueError(
                f"Unexpected Schaefer label configuration: len(labels)={len(labels)}, "
                f"roi ids span 1..{max_roi_id}."
            )

        _SCHAEFER_LABEL_TO_COUNT = {
            roi_id_to_label[i]: cortical_counts[i]
            for i in cortical_counts.keys()
            if i in roi_id_to_label
        }
        _dbg(
            "Loaded Schaefer atlas voxel weights: "
            f"{len(_SCHAEFER_LABEL_TO_COUNT)} ROI labels"
        )
    return _SCHAEFER_LABEL_TO_COUNT


@lru_cache(maxsize=None)
def _get_subcort_counts(subject: str, ses: str):
    """Get voxel counts for subcortical regions from aseg.mgz for the given subject/session, with caching."""
    global _SUBCORT_DEBUG_PRINTS
    cohort = "bbhi" if int(subject.split("-")[1]) > 5000 else "bbhi senior"
    if cohort == "bbhi":
        aseg_file = Path(
            f"/pool/guttmann/institut/BBHI/MRI/derivatives/reconall_fs6/{subject}_{ses}_run-01/mri/aseg.mgz"
        )
    else:
        aseg_file = Path(
            f"/pool/guttmann/institut/UB/Superagers/MRI/derivatives/reconall_fs6/{subject}_{ses}/mri/aseg.mgz"
        )
    if DEBUG and _SUBCORT_DEBUG_PRINTS < 5:
        _dbg(f"Loading aseg for {subject} {ses} from: {aseg_file}")
        _SUBCORT_DEBUG_PRINTS += 1
    aseg = nib.load(aseg_file).get_fdata().astype(int)
    return Counter(aseg[aseg > 0].ravel())


def main():
    root_path = Path("/home/rachel/Desktop/schaefer_analysis/structure_function_coupling")
    fc_root_path = Path("/home/rachel/Desktop/schaefer_analysis/functional_connectivity/native_space")
    sc_root_path = Path("/home/rachel/Desktop/schaefer_analysis/structural_connectivity")
    age_dir = Path("/home/rachel/Desktop/data")

    subjects_tp1 = get_subjects_to_process(
        root_path / "ses-01" / "individual_coupling_matrices", "ses-01", age_dir
    )
    subjects_tp2 = get_subjects_to_process(
        root_path / "ses-02" / "individual_coupling_matrices", "ses-02", age_dir
    )
    subjects = sorted(set(subjects_tp1) | set(subjects_tp2))
    _dbg(
        f"Subjects found: tp1={len(subjects_tp1)}, tp2={len(subjects_tp2)} "
    )

    out_rows = []
    for i, sub in enumerate(subjects, start=1):
        row = {"subject_id": sub}
        _dbg(f"[{i}/{len(subjects)}] Processing {sub}")

        # SFC tp1 / tp2
        for ses, col in [("ses-01", "sfc_tp1_weighted_mean"), ("ses-02", "sfc_tp2_weighted_mean")]:
            sfc_csv = (
                root_path
                / ses
                / "individual_coupling_matrices"
                / f"{sub}_{ses}_structure_function_coupling.csv"
            )
            if sfc_csv.is_file():
                sfc_df = pd.read_csv(sfc_csv)
                row[col] = get_weighted_roi_mean(sfc_df, subject=sub, ses=ses)
                _dbg(f"  SFC {ses}: global weighted mean={row[col]:.6f}")
                sensory, hmod = get_sfc_sensory_hmod_means(sfc_df, subject=sub, ses=ses)
                if ses == "ses-01":
                    row["sfc_sensory_1"] = sensory
                    row["sfc_hmod_1"] = hmod
                else:
                    row["sfc_sensory_2"] = sensory
                    row["sfc_hmod_2"] = hmod
                _dbg(f"  SFC {ses}: sensory={sensory}, hmod={hmod}")
                net_means = get_sfc_network_means(sfc_df, subject=sub, ses=ses, networks=SFC_NETWORKS)
                tp_idx = "1" if ses == "ses-01" else "2"
                for network in SFC_NETWORKS:
                    suffix = SFC_NETWORK_COL_SUFFIX[network]
                    row[f"sfc_{suffix}_{tp_idx}"] = net_means[network]
                _dbg(
                    f"  SFC {ses}: dmn={net_means['Default']}, "
                    f"salience={net_means['SalVentAttn']}"
                )
            else:
                _dbg(f"  SFC {ses}: missing file {sfc_csv.name}")
                row[col] = np.nan
                if ses == "ses-01":
                    row["sfc_sensory_1"] = np.nan
                    row["sfc_hmod_1"] = np.nan
                else:
                    row["sfc_sensory_2"] = np.nan
                    row["sfc_hmod_2"] = np.nan
                tp_idx = "1" if ses == "ses-01" else "2"
                for network in SFC_NETWORKS:
                    suffix = SFC_NETWORK_COL_SUFFIX[network]
                    row[f"sfc_{suffix}_{tp_idx}"] = np.nan

        # FC tp1 / tp2
        for ses, col in [("ses-01", "fc_tp1_weighted_mean"), ("ses-02", "fc_tp2_weighted_mean")]:
            fc_csv = (
                fc_root_path
                / ses
                / "individual_connectivity_matrices"
                / "grouped_rois"
                / f"{sub}_{ses}_functional_connectivity_flat.csv"
            )
            if fc_csv.is_file():
                fc_flat = pd.read_csv(fc_csv)
                row[col] = get_weighted_roi_mean(fc_flat, subject=sub, ses=ses)
                _dbg(f"  FC  {ses}: weighted mean={row[col]:.6f}")
            else:
                _dbg(f"  FC  {ses}: missing file {fc_csv.name}")
                row[col] = np.nan

        # SC tp1 / tp2
        for ses, col in [("ses-01", "sc_tp1_weighted_mean"), ("ses-02", "sc_tp2_weighted_mean")]:
            sc_csv = (
                sc_root_path
                / ses
                / "individual_connectivity_matrices"
                / "grouped_rois"
                / f"{sub}_{ses}_structural_connectivity_flat.csv"
            )
            if sc_csv.is_file():
                sc_flat = pd.read_csv(sc_csv)
                row[col] = get_weighted_roi_mean(sc_flat, subject=sub, ses=ses)
                _dbg(f"  SC  {ses}: global weighted mean={row[col]:.6f}")
            else:
                _dbg(f"  SC  {ses}: missing file {sc_csv.name}")
                row[col] = np.nan

        out_rows.append(row)
        if i % 25 == 0 or i == len(subjects):
            _dbg(f"Progress: completed {i}/{len(subjects)} subjects")

    df_out = pd.DataFrame(out_rows)
    output_path = Path("/home/rachel/Desktop/data/weighted_global_roi_averages.csv")
    _dbg("NA counts by column:")
    for col, n_na in df_out.isna().sum().items():
        _dbg(f"  {col}: {n_na}")
    df_out.to_csv(output_path, index=False)
    print(f"Saved: {output_path} | n_subjects={len(df_out)}")


if __name__ == "__main__":
    main()
