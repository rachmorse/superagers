"""Compute voxel-weighted global, sensory, heteromodal and network SFC averages for analyses.R."""

import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Literal, Sequence, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.datasets import fetch_atlas_schaefer_2018

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    DATA_DIR,
    N_PARCELS_BY_ATLAS,
    SESSIONS,
    SFC_DATA_CSV,
    SFC_FILE,
    SFC_INDIVIDUAL_DIR,
    SFC_ROOTS,
    SUBCORTICAL_ASEG_CODES,
    SUBJECTS,
    WEIGHTED_AVERAGES_CSV,
    cohort_from_id,
    reconall_subject_dir,
)

ATLAS: Literal["schaefer200", "schaefer400"] = "schaefer200"

DEBUG = False
_PRINTED_ROI_DOMAIN_EXAMPLES = False
_SUBCORT_DEBUG_PRINTS = 0
NETWORK_FIELD = 2  # Index of the network name in Schaefer labels split on "_": 7Networks_LH_Vis_1 -> "Vis"
SENSORY_NETWORKS = {"Vis", "SomMot"}
SFC_NETWORKS = ("SalVentAttn", "Cont", "Default")
SFC_NETWORK_COL_SUFFIX = {
    "SalVentAttn": "salience",
    "Cont": "control",
    "Default": "dmn",
}


def _dbg(msg: str) -> None:
    """Print debug message if DEBUG is True."""
    if DEBUG:
        print(f"[DEBUG] {msg}")


def network_from_roi(name: str) -> str:
    """Get the Schaefer network name from an ROI label.

    Args:
        name: ROI label (e.g. 7Networks_LH_Vis_1).

    Returns:
        Network name (e.g. "Vis"), or "" if the label has no network part.
    """
    parts = str(name).split("_")
    return parts[NETWORK_FIELD] if len(parts) > NETWORK_FIELD else ""


def weighted_mean(df: pd.DataFrame) -> float:
    """Compute the weighted mean of `pearson_rho`.

    Args:
        df: DataFrame with columns "pearson_rho" and "weight".

    Returns:
        Weighted mean, or NaN if there are no rows or the weights sum to 0.
    """
    if df.empty or np.isclose(df["weight"].sum(), 0.0):
        return np.nan
    return float((df["pearson_rho"] * df["weight"]).sum() / df["weight"].sum())


def get_weighted_roi_mean(df: pd.DataFrame, subject: str, ses: str) -> float:
    """Compute voxel-weighted global mean of `pearson_rho` across all ROIs.

    Args:
        df: DataFrame with columns "ROI_name" and "pearson_rho".
        subject: Subject ID.
        ses: Session ID.

    Returns:
        Weighted mean of `pearson_rho` across all ROIs.
    """
    return weighted_mean(_prepare_weighted_df(df, subject, ses))


def get_sfc_sensory_hmod_means(df: pd.DataFrame, subject: str, ses: str) -> Tuple[float, float]:
    """Compute weighted means split by sensory (Vis/SomMot) vs heteromodal.

    Args:
        df: DataFrame with columns "ROI_name" and "pearson_rho".
        subject: Subject ID.
        ses: Session ID.

    Returns:
        Tuple of (sensory_mean, hmod_mean), where each is the weighted mean of
        `pearson_rho` for the respective ROI rows.
    """
    valid = _prepare_weighted_df(df, subject, ses)
    if valid.empty:
        return np.nan, np.nan

    def _domain_from_roi(name: str) -> str:
        """Classify ROI as "sensory" vs "hmod" based on name.

        Subcortical ROIs are classified as "hmod".

        Args:
            name: ROI name.

        Returns:
            "sensory" or "hmod".
        """
        if name.startswith("Subcortical"):
            return "hmod"
        return "sensory" if network_from_roi(name) in SENSORY_NETWORKS else "hmod"

    global _PRINTED_ROI_DOMAIN_EXAMPLES  # noqa: PLW0603
    valid["domain"] = valid["ROI_name"].astype(str).apply(_domain_from_roi)
    if DEBUG and not _PRINTED_ROI_DOMAIN_EXAMPLES:
        sensory_sample5 = valid.loc[valid["domain"] == "sensory", "ROI_name"].sample(5).tolist()
        hmod_sample10 = valid.loc[valid["domain"] == "hmod", "ROI_name"].sample(10).tolist()
        _dbg(f"Example sensory ROIs (sample 5): {sensory_sample5}")
        _dbg(f"Example hmod ROIs (sample 10): {hmod_sample10}")
        _PRINTED_ROI_DOMAIN_EXAMPLES = True

    return weighted_mean(valid[valid["domain"] == "sensory"]), weighted_mean(valid[valid["domain"] == "hmod"])


def get_sfc_network_means(
    df: pd.DataFrame, subject: str, ses: str, networks: Sequence[str] = SFC_NETWORKS
) -> Dict[str, float]:
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
        return dict.fromkeys(networks, np.nan)

    cortical = valid[~valid["ROI_name"].astype(str).str.startswith("Subcortical")].copy()
    if cortical.empty:
        return dict.fromkeys(networks, np.nan)

    cortical["network"] = cortical["ROI_name"].astype(str).apply(network_from_roi)
    out = {}
    for network in networks:
        d = cortical[cortical["network"] == network]
        _dbg(f"Example ROIs in {network} network: {d['ROI_name'].sample(min(5, len(d))).tolist()}")
        out[network] = weighted_mean(d)
    return out


def _prepare_weighted_df(df: pd.DataFrame, subject: str, ses: str) -> pd.DataFrame:
    """Attach voxel-based weights and return valid rows for weighted averaging.

    Args:
        df: DataFrame with columns "ROI_name" and "pearson_rho".
        subject: Subject ID.
        ses: Session ID.

    Returns:
        DataFrame with additional weight column, filtered to rows with valid `pearson_rho`.

    Raises:
        ValueError: If any ROI label has no voxel-based weight.
    """
    df = df.copy()
    df["ROI_name"] = df["ROI_name"].astype(str)
    n_before = len(df)
    df["pearson_rho"] = pd.to_numeric(df["pearson_rho"], errors="coerce")

    label_to_count = _get_schaefer_label_to_count()
    df["weight"] = df["ROI_name"].map(label_to_count)

    subcort_counts = _get_subcort_counts(subject, ses)
    subcort_aseg_ids = [(f"Left {name}", left) for name, (left, _) in SUBCORTICAL_ASEG_CODES.items()] + [
        (f"Right {name}", right) for name, (_, right) in SUBCORTICAL_ASEG_CODES.items()
    ]
    # Subcortical ROIs are numbered after the cortical parcels (201+ or 401+).
    subcort_map = {
        f"Subcortical {N_PARCELS_BY_ATLAS[ATLAS] + i}: {name}": aseg_id
        for i, (name, aseg_id) in enumerate(subcort_aseg_ids, start=1)
    }
    df["aseg_id"] = df["ROI_name"].map(subcort_map)
    df["weight_subcort"] = df["aseg_id"].map(subcort_counts)
    df.loc[df["ROI_name"].str.startswith("Subcortical"), "weight"] = df["weight_subcort"]

    missing_weights = df.loc[df["weight"].isna(), "ROI_name"].unique().tolist()
    if missing_weights:
        preview = ", ".join(missing_weights[:10])
        raise ValueError(f"Missing voxel-based weights for {len(missing_weights)} ROI labels. First labels: {preview}")

    valid = df[df["pearson_rho"].notna()].copy()
    n_dropped = n_before - len(valid)
    if DEBUG and n_dropped > 0:
        _dbg(f"{subject} {ses}: dropped {n_dropped}/{n_before} rows with non-numeric pearson_rho")
    valid["weight"] = valid["weight"].astype(float)
    if DEBUG and np.isclose(valid["weight"].sum(), 0.0):
        _dbg(f"{subject} {ses}: weight sum is ~0 after filtering")
    return valid


@lru_cache(maxsize=None)
def _get_schaefer_label_to_count() -> Dict[str, int]:
    """Fetch Schaefer atlas info and return mapping of ROI label to voxel count, with caching.

    Returns:
        Voxel count for each Schaefer ROI label.

    Raises:
        ValueError: If the atlas labels do not line up with the ROI IDs in the atlas image.
    """
    atl = fetch_atlas_schaefer_2018(n_rois=N_PARCELS_BY_ATLAS[ATLAS], yeo_networks=7, resolution_mm=1)
    atlas = nib.load(atl.maps).get_fdata().astype(int)
    cortical_counts = Counter(atlas[atlas > 0].ravel())
    labels = [label.decode() if isinstance(label, bytes) else str(label) for label in atl.labels]
    max_roi_id = max(cortical_counts)
    if len(labels) == max_roi_id + 1:
        roi_id_to_label = {i: labels[i] for i in cortical_counts if 0 <= i < len(labels)}
    elif len(labels) == max_roi_id:
        roi_id_to_label = {i: labels[i - 1] for i in cortical_counts if 1 <= i <= len(labels)}
    else:
        raise ValueError(
            f"Unexpected Schaefer label configuration: len(labels)={len(labels)}, roi ids span 1..{max_roi_id}."
        )

    label_to_count = {roi_id_to_label[i]: cortical_counts[i] for i in cortical_counts if i in roi_id_to_label}
    _dbg(f"Loaded Schaefer atlas voxel weights: {len(label_to_count)} ROI labels")
    return label_to_count


@lru_cache(maxsize=None)
def _get_subcort_counts(subject: str, ses: str) -> Counter:
    """Get voxel counts for subcortical regions from aseg.mgz for the given subject/session, with caching.

    Args:
        subject: Subject ID.
        ses: Session ID.

    Returns:
        Voxel count for each aseg label.
    """
    global _SUBCORT_DEBUG_PRINTS  # noqa: PLW0603
    aseg_file = reconall_subject_dir(subject, ses, cohort_from_id(subject)) / "mri" / "aseg.mgz"
    if DEBUG and _SUBCORT_DEBUG_PRINTS < 5:
        _dbg(f"Loading aseg for {subject} {ses} from: {aseg_file}")
        _SUBCORT_DEBUG_PRINTS += 1
    aseg = nib.load(aseg_file).get_fdata().astype(int)
    return Counter(aseg[aseg > 0].ravel())


def sfc_summary(sfc_csv: Path, subject: str, ses: str) -> Dict[str, float]:
    """Compute the weighted global, sensory, heteromodal and network SFC means for one session.

    Args:
        sfc_csv: Path to the subject's SFC CSV.
        subject: Subject ID.
        ses: Session ID.

    Returns:
        Output columns for the session and their values, all NaN if the SFC file is missing.
    """
    tp = "1" if ses == "ses-01" else "2"
    columns = [f"sfc_tp{tp}_weighted_mean", f"sfc_sensory_{tp}", f"sfc_hmod_{tp}"] + [
        f"sfc_{SFC_NETWORK_COL_SUFFIX[network]}_{tp}" for network in SFC_NETWORKS
    ]
    if not sfc_csv.is_file():
        _dbg(f"  SFC {ses}: missing file {sfc_csv.name}")
        return dict.fromkeys(columns, np.nan)

    sfc_df = pd.read_csv(sfc_csv)
    global_mean = get_weighted_roi_mean(sfc_df, subject=subject, ses=ses)
    _dbg(f"  SFC {ses}: global weighted mean={global_mean:.6f}")
    sensory, hmod = get_sfc_sensory_hmod_means(sfc_df, subject=subject, ses=ses)
    _dbg(f"  SFC {ses}: sensory={sensory}, hmod={hmod}")
    net_means = get_sfc_network_means(sfc_df, subject=subject, ses=ses, networks=SFC_NETWORKS)
    _dbg(f"  SFC {ses}: dmn={net_means['Default']}, salience={net_means['SalVentAttn']}")
    return dict(zip(columns, [global_mean, sensory, hmod, *(net_means[network] for network in SFC_NETWORKS)]))


def save_all_sfc(subjects: List[str], ses: str) -> None:
    """Save raw coupling as one subject x ROI table for analyses.R.

    Args:
        subjects: Subject IDs to include.
        ses: Session ID (format: ses-01).
    """
    individual_dir = SFC_ROOTS[ATLAS] / ses / SFC_INDIVIDUAL_DIR
    rows = {}
    for sub in subjects:
        sfc_csv = individual_dir / SFC_FILE.format(subject=sub, ses=ses)
        if sfc_csv.is_file():
            rows[sub] = pd.read_csv(sfc_csv).set_index("ROI_name")["pearson_rho"]
    output_path = DATA_DIR / SFC_DATA_CSV.format(ses=ses, atlas=ATLAS)
    pd.DataFrame(rows).T.to_csv(output_path)
    print(f"Saved: {output_path} | n_subjects={len(rows)}")


def main() -> None:
    """Compute the weighted SFC averages for every subject and save them with the raw SFC tables."""
    subjects = sorted(SUBJECTS)
    _dbg(f"Subjects found: {len(subjects)}")

    out_rows = []
    for i, sub in enumerate(subjects, start=1):
        row = {"subject_id": sub}
        _dbg(f"[{i}/{len(subjects)}] Processing {sub}")

        # SFC tp1 / tp2
        for ses in SESSIONS:
            sfc_csv = SFC_ROOTS[ATLAS] / ses / SFC_INDIVIDUAL_DIR / SFC_FILE.format(subject=sub, ses=ses)
            row.update(sfc_summary(sfc_csv, sub, ses))

        out_rows.append(row)
        if i % 25 == 0 or i == len(subjects):
            _dbg(f"Progress: completed {i}/{len(subjects)} subjects")

    df_out = pd.DataFrame(out_rows)
    output_path = DATA_DIR / WEIGHTED_AVERAGES_CSV.format(atlas=ATLAS)
    _dbg("NA counts by column:")
    for col, n_na in df_out.isna().sum().items():
        _dbg(f"  {col}: {n_na}")
    df_out.to_csv(output_path, index=False)
    print(f"Saved: {output_path} | n_subjects={len(df_out)}")

    for ses in SESSIONS:
        save_all_sfc(subjects, ses)


if __name__ == "__main__":
    main()
