#!/usr/bin/env python3
"""Transform each subject's combined T1-space atlas into native DWI and native-T1 BOLD space."""

import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    BOLD_MASK,
    COHORTS,
    COMBINED_T1_MASK,
    DTI_DIRS,
    DWI_MASK,
    EDDY_CORRECTED,
    FSL_DIR,
    MASK_ROOT,
    RESTING_BOLD,
    SESSIONS,
    SHARED_MASK_ROOT,
    SUBJECTS,
    T1W_TO_B0_MAT,
    cohort_from_id,
    reconall_subject_dir,
    resting_native_dir,
)
from pipeline_utils import record_summary, run_step

N_JOBS = 6  # Subjects processed in parallel

# Intermediate files, relative to the subject's output folder (b0, T1 conversions and epi_reg images are removed afterwards)
B0_REF = "b0/{subject}_{ses}_b0.nii.gz"
T1_NIFTI = "t1_converted/{subject}_{ses}_conformed_T1.nii.gz"
BRAIN_NIFTI = "brain_converted/{subject}_{ses}_conformed_T1_brain.nii.gz"
BOLD_REFERENCE = "bold_space_masks/bold_reference.nii.gz"
EPI_REG_PREFIX = "dwi_space_masks/epi_reg"  # epi_reg writes epi_reg.mat, epi_reg_fast_wmedge.nii.gz, etc.

# epi_reg checks
EPI_REG_TIMEOUT_S = 1200  # 20 minutes
BBR_FAILURE = "ERROR::set_bbr_seg: could not find any boundary points"
MIN_WMEDGE_VOXELS = 2000  # Fewer white matter edge voxels than this is flagged as abnormally small


@dataclass
class SubjectPaths:
    """Files used to decide whether a subject needs DWI and/or BOLD processing.

    Attributes:
        subject: Subject ID.
        ses_label: Session label.
        schaefer_subcort_atlas: Combined T1-space atlas.
        bold_data: BOLD image.
        eddy_corrected: Eddy-corrected DWI series.
        dwi_mask_output: DWI-space atlas to create.
        bold_mask_output: BOLD-space atlas to create.
    """

    subject: str
    ses_label: str
    schaefer_subcort_atlas: Path
    bold_data: Optional[Path]
    eddy_corrected: Optional[Path]
    dwi_mask_output: Path
    bold_mask_output: Path


@dataclass
class DwiProcessingPaths:
    """Files used to create a subject's DWI-space atlas.

    Attributes:
        subject: Subject ID.
        ses_label: Session label.
        t1_mgz_path: recon-all T1.mgz.
        t1w_mask: Combined T1-space atlas.
        brain_mgz_path: recon-all brain.mgz.
        eddy_corrected: Eddy-corrected DWI series.
        b0_output: b0 reference image to create.
        out_b0_dir: Folder for the b0 reference.
        out_native_masks: Folder for the DWI-space atlas and registration files.
        output_path_dwi: DWI-space atlas to create.
        out_subject_dir: The subject's output folder for this session.
        shared_transform_mat: Existing T1w to b0 transform, reused if it exists.
    """

    subject: str
    ses_label: str
    t1_mgz_path: Path
    t1w_mask: Path
    brain_mgz_path: Path
    eddy_corrected: Path
    b0_output: Path
    out_b0_dir: Path
    out_native_masks: Path
    output_path_dwi: Path
    out_subject_dir: Path
    shared_transform_mat: Path


@dataclass
class BoldProcessingPaths:
    """Files used to create a subject's BOLD-space atlas.

    Attributes:
        subject: Subject ID.
        ses_label: Session label.
        t1_mgz_path: recon-all T1.mgz.
        t1w_mask: Combined T1-space atlas.
        bold_path: BOLD image.
        out_bold_masks: Folder for the BOLD-space atlas.
        output_path_bold: BOLD-space atlas to create.
        out_subject_dir: The subject's output folder for this session.
    """

    subject: str
    ses_label: str
    t1_mgz_path: Path
    t1w_mask: Path
    bold_path: Path
    out_bold_masks: Path
    output_path_bold: Path
    out_subject_dir: Path


def run_parallel_tasks(
    tasks: List[Tuple[str, object]], processor: Callable[[object], Optional[str]], job_name: str
) -> Tuple[List[str], List[str]]:
    """Run tasks sequentially or in parallel depending on N_JOBS.

    Args:
        tasks: (subject ID, paths) pairs to process.
        processor: Function that processes one subject's paths and returns the subject ID, or None on failure.
        job_name: Name of the job for printed messages (e.g. DWI or BOLD).

    Returns:
        Subject IDs that succeeded and subject IDs that failed.
    """
    successes = []
    failures = []
    if not tasks:
        return successes, failures

    cores = N_JOBS
    if cores <= 1:
        for subject, paths in tasks:
            result = processor(paths)
            if result:
                successes.append(subject)
            else:
                failures.append(subject)
        return successes, failures

    print(f"Running {job_name} for {len(tasks)} subjects with up to {cores} cores.")
    with ProcessPoolExecutor(max_workers=cores) as executor:
        future_to_subject = {executor.submit(processor, paths): subject for subject, paths in tasks}
        for future in as_completed(future_to_subject):
            subject = future_to_subject[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f"{job_name} processing crashed for {subject}: {exc}")
                failures.append(subject)
                continue

            if result:
                successes.append(subject)
            else:
                failures.append(subject)

    return successes, failures


def get_subjects_to_process(subject_infos: List[SubjectPaths]) -> Tuple[List[str], List[str], List[str]]:
    """Determine which subjects need DWI and/or BOLD processing.

    Args:
        subject_infos: Metadata for each subject including all relevant paths.

    Returns:
        A tuple containing three lists:
            - subjects_for_dwi: Subjects needing DWI mask processing.
            - subjects_for_bold: Subjects needing BOLD mask processing.
            - already_processed: Subjects already fully processed.
    """
    already_processed_dwi = []
    already_processed_bold = []
    already_processed = []
    missing_files = []
    subjects_for_dwi = []
    subjects_for_bold = []

    # Iterate over all possible subject directories
    for info in subject_infos:
        subject = info.subject
        # Check files separately to diagnose issues
        has_atlas = info.schaefer_subcort_atlas.exists()
        has_eddy = info.eddy_corrected.exists() if info.eddy_corrected else False
        has_bold = info.bold_data.exists() if info.bold_data else False
        has_dwi_mask = info.dwi_mask_output.exists()
        has_bold_mask = info.bold_mask_output.exists()

        # Check if the DWI and/or BOLD needs processing
        needs_dwi_processing = has_atlas and has_eddy and not has_dwi_mask
        needs_bold_processing = has_atlas and has_bold and not has_bold_mask

        # Categorize subjects
        if needs_dwi_processing:
            subjects_for_dwi.append(subject)
        if needs_bold_processing:
            subjects_for_bold.append(subject)
        if has_dwi_mask:
            already_processed_dwi.append(subject)
        if has_bold_mask:
            already_processed_bold.append(subject)
        if has_dwi_mask and has_bold_mask:
            already_processed.append(subject)

        if not has_atlas:
            missing_files.append(subject)
            print(f"Subject {subject} missing atlas file")
        elif not has_eddy and not has_bold:
            missing_files.append(subject)
            print(f"Subject {subject} missing both eddy and BOLD data")
            print(f"{info.bold_data} and {info.eddy_corrected} not found")
        elif not has_eddy:
            print(f"Subject {subject} missing eddy data")
            print(f"{info.eddy_corrected}")
        elif not has_bold:
            print(f"Subject {subject} missing BOLD data")
            print(f"{info.bold_data}")

    print(f"Total subjects evaluated: {len(subject_infos)}")
    print(f"Subjects already with DWI mask: {len(already_processed_dwi)}")
    print(f"Subjects already with BOLD mask: {len(already_processed_bold)}")
    print(f"Subjects already fully processed: {len(already_processed)}")
    print(f"Subjects missing required files: {len(missing_files)}")
    print(f"Subjects needing DWI processing: {len(subjects_for_dwi)}")
    print(f"Subjects needing BOLD processing: {len(subjects_for_bold)}")

    return subjects_for_dwi, subjects_for_bold, already_processed


def transform_t1w_to_dwi(
    t1w_mask: Path,
    t1_anat: Path,
    t1_brain: Path,
    eddy_corrected: Path,
    b0_ref: Path,
    *,
    output_path: Path,
    out_subject_dir: Path,
    subject: str,
    shared_transform_mat: Path,
) -> None:
    """Create a b0 reference and transform the T1w space mask to native DWI space using epi_reg and flirt.

    Args:
        t1w_mask: Path to the T1w space mask.
        t1_anat: Path to the T1 NIfTI image from recon-all.
        t1_brain: Path to the brain-extracted T1 image.
        eddy_corrected: Path to the eddy-corrected DWI series.
        b0_ref: Path to the b0 reference image.
        output_path: Path to save the DWI space mask file.
        out_subject_dir: The subject's output folder for this session.
        subject: Subject ID for logging.
        shared_transform_mat: Existing T1w to b0 transform. Used instead of rerunning epi_reg if it exists.

    Raises:
        FileNotFoundError: If the eddy-corrected data is missing, or the b0 reference or DWI mask is not created.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create b0 reference from the eddy-corrected data (first volume)
    b0_ref.parent.mkdir(parents=True, exist_ok=True)
    if not eddy_corrected.exists():
        print(f"ERROR: Eddy-corrected DWI file not found: {eddy_corrected}")
        raise FileNotFoundError(f"Eddy-corrected DWI not found: {eddy_corrected}")

    if not b0_ref.exists():
        cmd_b0 = [
            "fslroi",
            str(eddy_corrected),
            str(b0_ref),
            "0", "1",
        ]  # fmt: skip
        run_step(cmd_b0, "extracting b0 reference")

    if not b0_ref.exists():
        print(f"ERROR: Failed to create b0 reference: {b0_ref}")
        raise FileNotFoundError(f"b0 reference not created: {b0_ref}")

    # Reuse the existing transform so all atlases share the same registration
    if shared_transform_mat.exists():
        print(f"{subject}: reusing T1w to b0 transform {shared_transform_mat}")
        transform_mat = shared_transform_mat
    else:
        transform_mat = register_t1w_to_b0(t1_anat, t1_brain, b0_ref, out_subject_dir, subject)

    # Apply the transformation to the T1w mask to bring it into DWI space
    cmd_flirt = [
        "flirt",
        "-in", str(t1w_mask),
        "-ref", str(b0_ref),
        "-applyxfm",
        "-init", str(transform_mat),
        "-out", str(output_path),
        "-paddingsize", "0.0",
        "-interp", "nearestneighbour",
    ]  # fmt: skip
    run_step(cmd_flirt, "applying T1w to b0 transform")

    # flirt exits 0 even when it writes nothing (e.g., unreadable matrix)
    if not output_path.exists():
        print(f"ERROR: DWI mask not created: {output_path}")
        raise FileNotFoundError(f"DWI mask not created: {output_path}")

    print(f"Transformed T1w mask to DWI space: {output_path}")


def check_bbr_log(log_file: Path, subject: str) -> None:
    """Check the epi_reg log for the BBR boundary-points failure.

    Args:
        log_file: Path to the epi_reg log.
        subject: Subject ID for logging.

    Raises:
        RuntimeError: If the log reports the BBR failure.
    """
    if log_file.exists() and BBR_FAILURE in log_file.read_text():
        print(f"\nBBR FAILURE for {subject}: no boundary points found.\n")
        raise RuntimeError(f"BBR FAILURE for {subject}: no boundary points found.")


def register_t1w_to_b0(t1_anat: Path, t1_brain: Path, b0_ref: Path, out_subject_dir: Path, subject: str) -> Path:
    """Compute the T1w to b0 transform using epi_reg.

    Args:
        t1_anat: Path to the T1 NIfTI image from recon-all.
        t1_brain: Path to the brain-extracted T1 image.
        b0_ref: Path to the b0 reference image.
        out_subject_dir: The subject's output folder for this session.
        subject: Subject ID for logging.

    Returns:
        Path to the T1w to b0 transformation matrix.

    Raises:
        RuntimeError: If epi_reg times out, reports a BBR failure, or its white matter edge image looks wrong.
    """
    epi_reg_prefix = out_subject_dir / EPI_REG_PREFIX
    epi_reg_prefix.parent.mkdir(parents=True, exist_ok=True)

    # Run epi_reg to compute the transformation from b0 to T1w
    cmd = (
        f"source {FSL_DIR}/etc/fslconf/fsl.sh; "
        f"epi_reg "
        f"--epi={b0_ref} "
        f"--t1={t1_anat} "
        f"--t1brain={t1_brain} "
        f"--out={epi_reg_prefix}"
    )

    # Use a temporary file for logging to avoid buffer deadlocks
    log_file = epi_reg_prefix.parent / f"epi_reg_{subject}.log"

    try:
        with open(log_file, "w") as f:
            subprocess.run(
                ["bash", "-lc", cmd],
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=EPI_REG_TIMEOUT_S,
                check=True,
            )

    # Ive added a bunch of print statements here as this part sometimes throws errors
    # If a flag is thrown, the script will exit for that subject
    except subprocess.TimeoutExpired:
        print(f"\nERROR: epi_reg timed out for {subject} after 20 minutes.")
        raise RuntimeError(f"epi_reg timed out for {subject}") from None
    except subprocess.CalledProcessError:
        print(f"\nERROR: epi_reg failed for {subject}.")
        # Read the log file to check for specific errors
        check_bbr_log(log_file, subject)
        raise

    # Check log for BBR failure even on success (sometimes it doesn't return non-zero exit code)
    check_bbr_log(log_file, subject)

    # Clean up log file if successful
    log_file.unlink(missing_ok=True)

    # BBR errors seem to come from the WM segmentation step, so check that the output wmedge file is ok
    img_path = Path(f"{epi_reg_prefix}_fast_wmedge.nii.gz")
    if not img_path.exists():
        print(f"Stopping for {subject}: wmedge file does not exist.")
        raise RuntimeError(f"Stopping for {subject}: wmedge file does not exist.")
    data = nib.load(img_path).get_fdata()

    # NaN check
    if np.isnan(data).any():
        print(f"Stopping for {subject}: wmedge image contains NaNs.")
        raise RuntimeError(f"Stopping for {subject}: wmedge image contains NaNs.")

    # Non-zero voxel check (must NOT be empty)
    nonzero = np.count_nonzero(data)
    if nonzero == 0:
        print(f"Stopping for {subject}: wmedge image is EMPTY → FAST failed → BBR will fail.")
        raise RuntimeError(f"Stopping for {subject}: wmedge image is EMPTY → FAST failed → BBR will fail.")

    # Warn if extremely sparse
    if nonzero < MIN_WMEDGE_VOXELS:
        print(f"Warning for {subject}: wmedge image looks abnormally small ({nonzero} voxels).")

    print(f"{subject}: wmedge file looks valid ({nonzero} non-zero voxels). Continuing...")

    epi_to_t1_mat = Path(f"{epi_reg_prefix}.mat")
    transform_mat = out_subject_dir / T1W_TO_B0_MAT

    # Invert the transformation matrix to go from T1w to b0
    cmd_invert = [
        "convert_xfm",
        "-omat", str(transform_mat),
        "-inverse", str(epi_to_t1_mat),
    ]  # fmt: skip
    run_step(cmd_invert, "inverting the epi_reg transform")

    return transform_mat


def transform_t1w_to_bold(t1w_mask: Path, bold_path: Path, bold_ref_path: Path, output_path_bold: Path) -> None:
    """Transform the T1w space mask to BOLD native T1 space using mri_vol2vol.

    The BOLD image is already in the same anatomical space as T1 but not the same voxel grid.

    Args:
        t1w_mask: Path to the T1w space mask.
        bold_path: Path to the BOLD native T1 image.
        bold_ref_path: Path to save the first BOLD volume, used as the target grid.
        output_path_bold: Path to save the BOLD space mask.

    Raises:
        FileNotFoundError: If the BOLD image is missing, or the BOLD reference or BOLD mask is not created.
    """
    # Create output directory if it doesn't exist
    output_path_bold.parent.mkdir(parents=True, exist_ok=True)

    # Check if the BOLD file exists
    if not bold_path.exists():
        print(f"ERROR: BOLD file not found: {bold_path}")
        raise FileNotFoundError(f"BOLD file not found: {bold_path}")

    # Extract the first volume from the 4D BOLD images
    cmd_bold_ref = [
        "fslroi",
        str(bold_path),
        str(bold_ref_path),
        "0", "1",
    ]  # fmt: skip
    run_step(cmd_bold_ref, "extracting BOLD reference")

    # Verify the reference file was created
    if not bold_ref_path.exists():
        print(f"ERROR: Failed to create BOLD reference: {bold_ref_path}")
        raise FileNotFoundError(f"BOLD reference not created: {bold_ref_path}")

    # Resample mask onto BOLD grid using mri_vol2vol with regheader
    cmd_resample = [
        "mri_vol2vol",
        "--mov", str(t1w_mask),
        "--targ", str(bold_ref_path),
        "--o", str(output_path_bold),
        "--regheader",
        "--interp", "nearest",
    ]  # fmt: skip
    run_step(cmd_resample, "resampling mask onto BOLD grid")

    # Verify the output file was created
    if not output_path_bold.exists():
        print(f"ERROR: Failed to create BOLD space mask: {output_path_bold}")
        raise FileNotFoundError(f"BOLD space mask not created: {output_path_bold}")

    print(f"Transformed T1w mask to BOLD space: {output_path_bold}")


def ensure_t1_nifti(
    t1_mgz: Path, brain_mgz: Path, subject: str, ses_label: str, out_subject_dir: Path
) -> Tuple[Path, Path]:
    """Convert the subject's T1.mgz and brain.mgz to NIfTI.

    Args:
        t1_mgz: Path to the subject's recon-all T1.mgz
        brain_mgz: Path to the subject's recon-all brain.mgz
        subject: The subject ID
        ses_label: Session label
        out_subject_dir: Output directory for the subject

    Returns:
        Paths to the converted NIfTI T1 image and brain-extracted T1 image.

    Raises:
        FileNotFoundError: If T1.mgz or brain.mgz is missing.
    """
    if not t1_mgz.exists():
        raise FileNotFoundError(f"T1.mgz not found for {subject}: {t1_mgz}")
    if not brain_mgz.exists():
        raise FileNotFoundError(f"brain.mgz not found for {subject}: {brain_mgz}")

    t1_nifti = out_subject_dir / T1_NIFTI.format(subject=subject, ses=ses_label)
    brain_nifti = out_subject_dir / BRAIN_NIFTI.format(subject=subject, ses=ses_label)
    t1_nifti.parent.mkdir(parents=True, exist_ok=True)
    brain_nifti.parent.mkdir(parents=True, exist_ok=True)

    if not t1_nifti.exists():
        cmd_t1 = [
            "mri_convert",
            str(t1_mgz),
            str(t1_nifti),
        ]  # fmt: skip
        run_step(cmd_t1, "converting T1.mgz to NIfTI")
        print(f"Converted T1.mgz to NIfTI for {subject}")

    if not brain_nifti.exists():
        cmd_brain = [
            "mri_convert",
            str(brain_mgz),
            str(brain_nifti),
        ]  # fmt: skip
        run_step(cmd_brain, "converting brain.mgz to NIfTI")
        print(f"Converted brain.mgz to NIfTI for {subject}")

    return t1_nifti, brain_nifti


def process_subject_dwi(paths: DwiProcessingPaths) -> Optional[str]:
    """Process a single subject's DWI data using predefined file paths.

    Args:
        paths: Paths required for DWI processing.

    Returns:
        Subject ID if processing was successful, None otherwise.
    """
    subject = paths.subject
    print(f"\nProcessing DWI for {subject}...")

    try:
        # Step 0: Ensure T1 is in NIfTI format and brain-extracted image is available
        t1_anat, t1_brain = ensure_t1_nifti(
            paths.t1_mgz_path,
            paths.brain_mgz_path,
            subject,
            paths.ses_label,
            paths.out_subject_dir,
        )

        # Step 1: Transform T1w mask to native space using the transformation matrix
        transform_t1w_to_dwi(
            paths.t1w_mask,
            t1_anat,
            t1_brain,
            paths.eddy_corrected,
            paths.b0_output,
            output_path=paths.output_path_dwi,
            out_subject_dir=paths.out_subject_dir,
            subject=subject,
            shared_transform_mat=paths.shared_transform_mat,
        )

        print(f"Successfully created native space DWI mask for {subject}")

        # Step 2: Clean up individual subject's intermediate files
        for intermediate_dir in [paths.out_b0_dir, t1_anat.parent, t1_brain.parent]:
            if intermediate_dir.exists():
                shutil.rmtree(intermediate_dir)
        for f in paths.out_native_masks.glob("epi_reg*.nii.gz"):
            f.unlink()

        return subject
    except Exception as e:
        print(f"Error processing {subject}: {e}")
        return None


def process_subject_bold(paths: BoldProcessingPaths) -> Optional[str]:
    """Process a single subject's BOLD data using predefined file paths.

    Args:
        paths: Paths required for BOLD processing.

    Returns:
        Subject ID if processing was successful, None otherwise.
    """
    subject = paths.subject
    print(f"\nProcessing BOLD for {subject}...")

    try:
        transform_t1w_to_bold(
            paths.t1w_mask,
            paths.bold_path,
            paths.out_subject_dir / BOLD_REFERENCE,
            paths.output_path_bold,
        )

        print(f"Successfully created native space BOLD mask for {subject}")

        return subject
    except Exception as e:
        print(f"Error processing {subject}: {e}")
        return None


def build_subject_paths(
    subject: str, ses_label: str, cohort: str
) -> Tuple[SubjectPaths, DwiProcessingPaths, BoldProcessingPaths]:
    """Build all input and output paths for one subject and session.

    Args:
        subject: Subject ID.
        ses_label: Session label (e.g. 'ses-01').
        cohort: Cohort name (e.g., 'bbhi', 'bbhi senior').

    Returns:
        The subject's paths for deciding what to process, for DWI processing, and for BOLD processing.
    """
    out_subject_dir = MASK_ROOT / ses_label / subject
    recon_mri_dir = reconall_subject_dir(subject, ses_label, cohort) / "mri"
    t1w_mask = out_subject_dir / COMBINED_T1_MASK.format(subject=subject, ses=ses_label)
    dwi_mask_output = out_subject_dir / DWI_MASK.format(subject=subject, ses=ses_label)
    bold_mask_output = out_subject_dir / BOLD_MASK.format(subject=subject, ses=ses_label)
    eddy_corrected = DTI_DIRS[(cohort, ses_label)] / f"{subject}_{ses_label}" / EDDY_CORRECTED
    bold_path = resting_native_dir(subject, ses_label, cohort) / RESTING_BOLD.format(subject=subject, ses=ses_label)
    b0_output = out_subject_dir / B0_REF.format(subject=subject, ses=ses_label)

    subject_info = SubjectPaths(
        subject=subject,
        ses_label=ses_label,
        schaefer_subcort_atlas=t1w_mask,
        bold_data=bold_path,
        eddy_corrected=eddy_corrected,
        dwi_mask_output=dwi_mask_output,
        bold_mask_output=bold_mask_output,
    )
    dwi_paths = DwiProcessingPaths(
        subject=subject,
        ses_label=ses_label,
        t1_mgz_path=recon_mri_dir / "T1.mgz",
        t1w_mask=t1w_mask,
        brain_mgz_path=recon_mri_dir / "brain.mgz",
        eddy_corrected=eddy_corrected,
        b0_output=b0_output,
        out_b0_dir=b0_output.parent,
        out_native_masks=dwi_mask_output.parent,
        output_path_dwi=dwi_mask_output,
        out_subject_dir=out_subject_dir,
        shared_transform_mat=SHARED_MASK_ROOT / ses_label / subject / T1W_TO_B0_MAT,
    )
    bold_paths = BoldProcessingPaths(
        subject=subject,
        ses_label=ses_label,
        t1_mgz_path=recon_mri_dir / "T1.mgz",
        t1w_mask=t1w_mask,
        bold_path=bold_path,
        out_bold_masks=bold_mask_output.parent,
        output_path_bold=bold_mask_output,
        out_subject_dir=out_subject_dir,
    )
    return subject_info, dwi_paths, bold_paths


def main() -> None:
    """Main function to transform the Schaefer and subcortical (aseg) mask to native DWI and native T1 BOLD."""
    print("-----------------------Running 04_t1_to_dwi_bold.py-----------------------")

    for cohort in COHORTS:
        for ses_label in SESSIONS:
            print("-------------------------")
            print(f"Processing {cohort} {ses_label}")
            print("-------------------------")

            # Set up FSL so it runs correctly in this script
            # Automatically pulls 6.0.4 for consistency with fMRI analysis
            os.environ["FSLDIR"] = str(FSL_DIR)
            fsl_bin = f"{FSL_DIR}/bin"
            if fsl_bin not in os.environ["PATH"].split(":"):
                os.environ["PATH"] = f"{fsl_bin}:{os.environ['PATH']}"

            # Set FSL to output compressed NIFTI files
            os.environ["FSLOUTPUTTYPE"] = "NIFTI_GZ"

            subject_infos: List[SubjectPaths] = []
            dwi_inputs: Dict[str, DwiProcessingPaths] = {}
            bold_inputs: Dict[str, BoldProcessingPaths] = {}

            for subject in sorted(os.listdir(MASK_ROOT / ses_label)):
                if not subject.startswith("sub-") or subject not in SUBJECTS or cohort_from_id(subject) != cohort:
                    continue

                subject_info, dwi_inputs[subject], bold_inputs[subject] = build_subject_paths(
                    subject, ses_label, cohort
                )
                subject_infos.append(subject_info)

            subjects_for_dwi, subjects_for_bold, _ = get_subjects_to_process(subject_infos)

            # Check if there are subjects to process
            if not subjects_for_dwi and not subjects_for_bold:
                print("No subjects found that need processing.")
                continue

            dwi_tasks = [(subject, dwi_inputs[subject]) for subject in subjects_for_dwi]
            result_dwi, failed_dwi = run_parallel_tasks(dwi_tasks, process_subject_dwi, "DWI")

            bold_tasks = [(subject, bold_inputs[subject]) for subject in subjects_for_bold]
            result_bold, failed_bold = run_parallel_tasks(bold_tasks, process_subject_bold, "BOLD")

            print(f"Successfully processed DWI for {len(result_dwi)} subjects")
            print(f"Successfully processed BOLD for {len(result_bold)} subjects")
            print(f"Total subjects with successful processing: {len(set(result_dwi + result_bold))}")

            print(f"Failed to process DWI for {len(failed_dwi)} subjects")
            print(f"Failed to process BOLD for {len(failed_bold)} subjects")

            if failed_dwi:
                print(f"Failed DWI subjects: {failed_dwi}")
            if failed_bold:
                print(f"Failed BOLD subjects: {failed_bold}")

            record_summary(__file__, cohort, ses_label, len(result_dwi), failed_dwi, step="DWI")
            record_summary(__file__, cohort, ses_label, len(result_bold), failed_bold, step="BOLD")


if __name__ == "__main__":
    main()
