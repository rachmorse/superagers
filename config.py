"""Settings and paths shared by the processing pipeline (masks, tractography, fMRI, structural connectivity, coupling).

Every data and software path the pipeline uses is defined here. Scripts only build per-subject file names from them.

ATLAS and TEST_SUBJECT are set at the top of each run_*_pipeline.py script which passes them to the scripts it runs.
The defaults here are only used when a script is run on its own (e.g. fmri_analysis/01_extract_subjects_timeseries.py
instead of run_fmri_pipeline.py).
"""

import os
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

# Atlas
ATLASES = ("schaefer200", "schaefer400")
ATLAS = os.environ.get("ATLAS", "schaefer200")
if ATLAS not in ATLASES:
    raise ValueError(f"Unknown ATLAS {ATLAS!r}, expected one of {ATLASES}")
N_PARCELS = int(ATLAS.removeprefix("schaefer"))

# Subcortical structures in mask value order (1 to 7) with their FreeSurfer aseg codes (left, right)
SUBCORTICAL_ASEG_CODES: Dict[str, Tuple[int, int]] = {
    "Hippocampus": (17, 53),
    "Amygdala": (18, 54),
    "Pallidum": (13, 52),
    "Putamen": (12, 51),
    "Caudate": (11, 50),
    "Accumbens": (26, 58),
    "Thalamus": (10, 49),
}

# Cohort
SUBJECTS_FILE = Path("/home/rachel/Desktop/data/final_cohort_ids.txt")  # Final longitudinal cohort
TEST_SUBJECT: Optional[str] = os.environ.get("TEST_SUBJECT") or None
SUBJECTS: List[str] = [TEST_SUBJECT] if TEST_SUBJECT else SUBJECTS_FILE.read_text().split()
SUPERAGER_CSV = Path("/home/rachel/Desktop/data/superager.csv")  # Superager classification per timepoint
COHORTS: List[Literal["bbhi", "bbhi senior"]] = ["bbhi", "bbhi senior"]
SESSIONS: List[Literal["ses-01", "ses-02"]] = ["ses-01", "ses-02"]
BBHI_MIN_ID = 6000  # BBHI IDs are above this, BBHI senior IDs below

# Software
FSL_DIR = Path("/vol/software/fsl_6_0_4")
FREESURFER_HOME = Path("/vol/software/freesurfer-6.0")
MRTRIX_BIN = Path("/home/rachel/miniconda3/bin")
SPM_PATH = Path("/home/rachel/spm12")
MATLAB_CMD = "/usr/local/bin/matlab -nodesktop -nosplash"

# Input data
BBHI_MRI = Path("/pool/guttmann/institut/BBHI/MRI")
SENIOR_MRI = Path("/pool/guttmann/institut/UB/Superagers/MRI")
RECONALL_DIRS: Dict[str, Path] = {
    "bbhi": BBHI_MRI / "derivatives" / "reconall_fs6",
    "bbhi senior": SENIOR_MRI / "derivatives" / "reconall_fs6",
}
DTI_DIRS: Dict[Tuple[str, str], Path] = {
    (cohort, ses): root / f"dtifit_{ses}_fsl-604"
    for cohort, root in [("bbhi", BBHI_MRI / "processed_data"), ("bbhi senior", SENIOR_MRI)]
    for ses in SESSIONS
}
RESTING_DIRS: Dict[Tuple[str, str], Path] = {
    ("bbhi", "ses-01"): BBHI_MRI / "processed_data" / "resting_preproc_fs6-recon",
    ("bbhi", "ses-02"): BBHI_MRI / "processed_data" / "resting_preproc_fs6-recon_tp2",
    ("bbhi senior", "ses-01"): SENIOR_MRI / "resting_preproc_fs6-recon",
    ("bbhi senior", "ses-02"): SENIOR_MRI / "resting_preproc_fs6-recon",
}
TRACTOGRAM_DIRS: Dict[str, Path] = {
    "bbhi": BBHI_MRI / "processed_data" / "tracto_SIFT2",
    "bbhi senior": SENIOR_MRI / "tracto_SIFT2",
}
TRACTOGRAM = "{subject}_{ses}_dwi_tractogram_10M.tck"
SIFT2_WEIGHTS = "{subject}_{ses}_dwi_tractogram_10M_SIFT2_weights.txt"
ANNOT_DIR = Path(__file__).resolve().parent / "fsaverage_masks"  # Schaefer annot files
SCHAEFER_ANNOT = f"{{hemi}}.Schaefer2018_{N_PARCELS}Parcels_7Networks_order.annot"
EDDY_CORRECTED = "eddy_corrected_data.nii.gz"  # Inside DTI_DIRS[(cohort, ses)] / f"{subject}_{ses}"
RESTING_BOLD = "{subject}_{ses}_run-01_rest_bold_ap_T1-space.nii.gz"  # Unscrubbed, inside resting_native_dir()
FRAMEWISE_DISPL = "framewise_displ.txt"

# Outputs: unsuffixed folders are the original run, each atlas writes to its own suffixed folders
# Subcortical masks and DWI registrations are atlas-independent, so all atlases reuse the ones in fsaverage
ROOT = Path("/home/rachel/Desktop/schaefer_analysis")
SHARED_MASK_ROOT = ROOT / "fsaverage"
MASK_ROOT = SHARED_MASK_ROOT if ATLAS == "schaefer200" else ROOT / f"fsaverage_{ATLAS}"
TRACTOGRAPHY_WD = ROOT / "tracto_wd"
TIMESERIES_ROOT = ROOT / f"timeseries_data_{ATLAS}" / "native_space"
TIMESERIES_FILE = f"{{subject}}_{{ses}}_subcortical_{ATLAS}_timeseries.csv"  # Inside TIMESERIES_ROOT / ses
LABELS_CSV = TIMESERIES_ROOT / "combined_labels.csv"  # ROI labels in mask order
FC_ROOT = ROOT / f"functional_connectivity_{ATLAS}" / "native_space"
SC_ROOT = ROOT / f"structural_connectivity_{ATLAS}"
SFC_ROOT = ROOT / f"structure_function_coupling_{ATLAS}"
ALL_TO_ALL_DIR = "all_to_all_roi_matrices"
ALL_TO_ALL_LABEL = "all_to_all_roi"
MATRIX_CSV = "{label}_matrix.csv"

# Per-subject mask files, relative to MASK_ROOT / ses / subject (SHARED_MASK_ROOT for the subcortical and transform files)
CORTICAL_T1_MASK = "t1_masks/{subject}_schaefer_volumetric_t1_{hemi}.nii.gz"  # hemi: lh or rh
SUBCORTICAL_T1_MASK = "subcortical_t1_masks/{subject}_{side}_subcortical14_t1.nii.gz"  # side: left or right
COMBINED_T1_MASK = f"subcortical_t1_masks/{{subject}}_{{ses}}_{ATLAS}_subcortical14_t1_space.nii.gz"
DWI_MASK = f"dwi_space_masks/{{subject}}_{{ses}}_{ATLAS}_subcortical14_dwi_space.nii.gz"
BOLD_MASK = f"bold_space_masks/{{subject}}_{{ses}}_{ATLAS}_subcortical14_bold_space.nii.gz"
T1W_TO_B0_MAT = "dwi_space_masks/T1w_to_b0.mat"


def cohort_from_id(subject: str) -> str:
    """Get a subject's cohort from their ID.

    Args:
        subject: Subject ID, with or without the "sub-" prefix (e.g. "sub-1014" or "1014").

    Returns:
        "bbhi" or "bbhi senior".
    """
    return "bbhi" if int(subject.removeprefix("sub-")) > BBHI_MIN_ID else "bbhi senior"


def reconall_subject_dir(subject: str, ses: str, cohort: str) -> Path:
    """Get a subject's recon-all folder.

    BBHI folders end in _run-01, BBHI senior folders do not.

    Args:
        subject: Subject ID (e.g. "sub-1014").
        ses: Session label (e.g. "ses-01").
        cohort: Cohort name ("bbhi" or "bbhi senior").

    Returns:
        Path to the subject's recon-all folder for that session.
    """
    name = f"{subject}_{ses}_run-01" if cohort == "bbhi" else f"{subject}_{ses}"
    return RECONALL_DIRS[cohort] / name


def resting_native_dir(subject: str, ses: str, cohort: str) -> Path:
    """Get the folder holding a subject's T1-space BOLD images and FD file.

    BBHI has a separate root directory per session, BBHI senior has session subfolders.

    Args:
        subject: Subject ID (e.g. "sub-1014").
        ses: Session label (e.g. "ses-01").
        cohort: Cohort name ("bbhi" or "bbhi senior").

    Returns:
        Path to the subject's native_T1 folder.
    """
    subject_dir = RESTING_DIRS[(cohort, ses)] / subject
    return subject_dir / "native_T1" if cohort == "bbhi" else subject_dir / ses / "native_T1"
