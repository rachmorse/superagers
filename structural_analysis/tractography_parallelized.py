"""Run SIFT2 tractography in parallel for the BBHI and BBHI senior cohorts.

Requires the MATLAB script 'spm_coregister_parcellation.m' in the same directory as this script.
"""

import json
import logging
import multiprocessing
import os
import socket
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, NamedTuple, Optional, Set, Tuple, Union

JOBS = 1  # Subjects run at the same time
THREADS = 20  # CPU threads per subject for MRtrix (total cores used = JOBS * THREADS)
TEST_SUBJECTS: Optional[Set[str]] = None  # e.g. {"sub-1014_ses-01"} to run only those
COHORTS: List[Literal["bbhi", "bbhi senior"]] = ["bbhi", "bbhi senior"]
SESSIONS: List[Literal["ses-01", "ses-02"]] = ["ses-01", "ses-02"]

DTI_DIRS: Dict[str, str] = {
    "bbhi": "/pool/guttmann/institut/BBHI/MRI/processed_data/dtifit_{ses}_fsl-604",
    "bbhi senior": "/pool/guttmann/institut/UB/Superagers/MRI/dtifit_{ses}_fsl-604",
}
TRACTOGRAM_DIRS: Dict[str, Path] = {
    "bbhi": Path("/pool/guttmann/institut/BBHI/MRI/processed_data/tracto_SIFT2"),
    "bbhi senior": Path("/pool/guttmann/institut/UB/Superagers/MRI/tracto_SIFT2"),
}
RECON_ALL_DIRS: Dict[str, Path] = {
    "bbhi": Path("/pool/guttmann/institut/BBHI/MRI/derivatives/reconall_fs6"),
    "bbhi senior": Path("/pool/guttmann/institut/UB/Superagers/MRI/derivatives/reconall_fs6"),
}

GIT_DIR = Path("/home/rachel/Desktop/superagers/structural_analysis")
WORKING_DIR_ROOT = Path("/home/rachel/Desktop/schaefer_analysis/tracto_wd")
SPM_PATH = Path("/home/rachel/spm12")
MATLAB_CMD = "/usr/local/bin/matlab -nodesktop -nosplash"
FSL_DIR = "/vol/software/fsl_6_0_4"
FREESURFER_HOME = "/vol/software/freesurfer-6.0"


class SubjectInfo(NamedTuple):
    """A subject-session to process.

    Attributes:
        subject_id: Subject-session ID, e.g. "sub-1014_ses-01".
        cohort: Cohort key, e.g. "bbhi".
        ses: Session, e.g. "ses-01".
        dti_dir: Directory containing the subject's DTI data.
    """

    subject_id: str
    cohort: str
    ses: str
    dti_dir: Path


def setup_logging(output_dir: Path) -> logging.Logger:
    """Sets up basic logging to file and console.

    Args:
        output_dir: Directory where the "logs" folder is created.

    Returns:
        Root logger.
    """
    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"tractography_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()


def setup_environment() -> None:
    """Sets up environment variables for FSL and FreeSurfer."""
    # Set up FSL
    os.environ["FSLDIR"] = FSL_DIR
    os.environ["PATH"] = f"{os.environ['FSLDIR']}/bin:" + os.environ["PATH"]
    # Set FSL to output uncompressed NIFTI files
    os.environ["FSLOUTPUTTYPE"] = "NIFTI"

    # Set up FreeSurfer
    os.environ["FREESURFER_HOME"] = FREESURFER_HOME
    os.environ["PATH"] = f"{os.environ['FREESURFER_HOME']}/bin:" + os.environ["PATH"]


def run_command(cmd: Union[str, List[str]], shell: bool = True) -> None:
    """Runs a shell command and checks for errors.

    Args:
        cmd: Command to run, as a string or list of arguments.
        shell: Whether to run in a shell.

    Raises:
        subprocess.CalledProcessError: If the command fails.
    """
    cmd_str = " ".join(str(c) for c in cmd) if isinstance(cmd, list) else cmd

    logging.info(f"Running command: {cmd_str}")

    try:
        subprocess.run(cmd_str, shell=shell, check=True, executable="/bin/bash", capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed: {e.cmd}")
        logging.error(f"STDOUT: {e.stdout}")
        logging.error(f"STDERR: {e.stderr}")
        raise


def find_recon_all_dir(subject_id: str, cohort: str) -> Optional[Path]:
    """Finds a subject's recon-all directory within its cohort.

    BBHI uses "sub-x_ses-0y_run-01", BBHI senior uses "sub-x_ses-0y".

    Args:
        subject_id: Subject-session ID, e.g. "sub-1014_ses-01".
        cohort: Cohort key in RECON_ALL_DIRS.

    Returns:
        The recon-all directory, or None if not found.
    """
    recon_root = RECON_ALL_DIRS[cohort]
    for candidate in (recon_root / f"{subject_id}_run-01", recon_root / subject_id):
        if candidate.exists():
            return candidate
    return None


def is_processed(subject: SubjectInfo) -> bool:
    """Checks whether SIFT2 weights already exist locally or in the shared directory.

    Args:
        subject: Subject to check.

    Returns:
        True if the subject's SIFT2 weights already exist.
    """
    weights_name = f"{subject.subject_id}_dwi_tractogram_10M_SIFT2_weights.txt"
    local_file = WORKING_DIR_ROOT / subject.subject_id / weights_name
    shared_file = TRACTOGRAM_DIRS[subject.cohort] / subject.subject_id / weights_name
    return local_file.exists() or shared_file.exists()


def get_subjects_to_process() -> List[SubjectInfo]:
    """Lists subjects with DTI and recon-all data that are not yet processed.

    Returns:
        Subjects to process, sorted by subject ID.
    """
    subjects = []
    for cohort in COHORTS:
        for ses in SESSIONS:
            dti_dir = Path(DTI_DIRS[cohort].format(ses=ses))
            if not dti_dir.exists():
                logging.warning(f"DTI directory not found: {dti_dir}")
                continue

            for sub_path in dti_dir.glob("sub-*"):
                subject = SubjectInfo(sub_path.name, cohort, ses, dti_dir)
                if find_recon_all_dir(subject.subject_id, cohort) is None:
                    logging.info(f"Skipping {subject.subject_id}, no recon-all found.")
                elif is_processed(subject):
                    logging.info(f"Skipping {subject.subject_id}, already processed.")
                else:
                    subjects.append(subject)

    return sorted(subjects, key=lambda s: s.subject_id)


def check_inputs(subject: str, input_corr_dwi: Path, fs_dir: Optional[Path]) -> bool:
    """Checks if the necessary input files and directories exist.

    Args:
        subject: Subject ID.
        input_corr_dwi: Path to the eddy corrected DWI data.
        fs_dir: Path to the subject's recon-all directory, or None if not found.

    Returns:
        True if inputs exist, False otherwise.
    """
    if not input_corr_dwi.exists():
        logging.error(f"{subject}: No eddy corrected data found at {input_corr_dwi}")
        return False
    if fs_dir is None:
        logging.error(f"{subject}: No recon-all directory found")
        return False
    return True


def run_coregistration(
    subject: str,
    wd: Path,
    *,
    input_corr_dwi: Path,
    fs_dir: Path,
    git_dir: Path,
    spm_path: Path,
    matlab_cmd: str,
) -> str:
    """Performs structural to diffusion coregistration.

    Copies and unzips the eddy corrected data, converts recon-all T1 and
    segmentation files to NIfTI format, extracts the b0 image, and estimates
    the coregistration using a MATLAB script. Always runs.

    Args:
        subject: Subject ID.
        wd: Working directory file prefix.
        input_corr_dwi: Path to the eddy corrected DWI data.
        fs_dir: Path to the subject's recon-all directory.
        git_dir: Directory containing the MATLAB script.
        spm_path: Path to the SPM installation.
        matlab_cmd: Command to execute MATLAB.

    Returns:
        Path to the unzipped eddy corrected DWI data.
    """
    logging.info(f"{subject}: Structural 2 diff. coregistration")

    # Copy and unzip eddy data
    run_command(f"cp -f {input_corr_dwi} {wd}_eddy_corrected_data.nii.gz")
    input_corr_dwi_temp = f"{wd}_eddy_corrected_data.nii"
    run_command(f"gunzip -f {wd}_eddy_corrected_data.nii.gz")

    # Convert recon-all T1 and aparc+aseg
    run_command(f"mrconvert -force {fs_dir}/mri/T1.mgz {wd}_fs2diff_coords_T1w.nii")
    run_command(f"mrconvert -force {fs_dir}/mri/aparc+aseg.mgz '{wd}_fs2diff_coords_aparc+aseg.nii'")

    # Extract b0
    run_command(f"fslroi {input_corr_dwi_temp} {wd}_b0.nii 0 1")

    # MATLAB Coregistration
    matlab_cmd_full = (
        f"{matlab_cmd} -r \"addpath(genpath('{spm_path}')); addpath('{git_dir}'); "
        f"spm_coregister_parcellation('{wd}_b0.nii', '{wd}_fs2diff_coords_T1w.nii', "
        f"'{wd}_fs2diff_coords_aparc+aseg.nii'); exit;\""
    )
    run_command(matlab_cmd_full)

    return input_corr_dwi_temp


def prepare_brain_mask(subject: str, wd: Path, input_corr_dwi_brain_mask: Path) -> None:
    """Prepares the brain mask by copying the T1-derived mask.

    Only uses the pre-computed T1w_brain_mask_dMRIres.nii.gz. If this mask is
    missing, processing for this subject fails.

    Args:
        subject: Subject ID.
        wd: Working directory file prefix.
        input_corr_dwi_brain_mask: Path to the pre-existing brain mask.

    Raises:
        FileNotFoundError: If the robust T1 mask is missing.
    """
    logging.info(f"{subject}: using robust T1 mask")
    if not Path(f"{wd}_mask.nii.gz").exists():
        if input_corr_dwi_brain_mask.exists():
            run_command(f"cp {input_corr_dwi_brain_mask} {wd}_mask.nii.gz")
        else:
            raise FileNotFoundError(f"Robust T1 mask not found for {subject} at {input_corr_dwi_brain_mask}")


def run_dwi2response(
    subject: str, wd: Path, *, dwi_file: str, input_corr_val: Path, input_corr_vec: Path, threads: int
) -> None:
    """Estimates the tissue response functions for spherical deconvolution.

    Args:
        subject: Subject ID.
        wd: Working directory file prefix.
        dwi_file: Path to the DWI data file.
        input_corr_val: Path to the bval file.
        input_corr_vec: Path to the bvec file.
        threads: Number of threads to use.
    """
    logging.info(f"{subject}: Response Function Estimation")
    if not Path(f"{wd}_desc-csf_response.txt").exists():
        # fmt: off
        run_command([
            "dwi2response", "dhollander", str(dwi_file),
            "-fslgrad", str(input_corr_vec), str(input_corr_val),
            f"{wd}_desc-wm_response.txt",
            f"{wd}_desc-gm_response.txt",
            f"{wd}_desc-csf_response.txt",
            "-mask", f"{wd}_mask.nii.gz",
            "-nthreads", str(threads),
        ])
        # fmt: on


def run_dwi2fod(
    subject: str, wd: Path, *, dwi_file: str, input_corr_val: Path, input_corr_vec: Path, threads: int
) -> None:
    """Estimates Fibre Orientation Distributions (FODs) using Multi-Shell Multi-Tissue CSD.

    Args:
        subject: Subject ID.
        wd: Working directory file prefix.
        dwi_file: Path to the DWI data file.
        input_corr_val: Path to the bval file.
        input_corr_vec: Path to the bvec file.
        threads: Number of threads to use.
    """
    logging.info(f"{subject}: FOD Estimation")
    if not Path(f"{wd}_desc-wm_fod.mif").exists():
        # fmt: off
        run_command([
            "dwi2fod", "msmt_csd", str(dwi_file),
            "-fslgrad", str(input_corr_vec), str(input_corr_val),
            f"{wd}_desc-wm_response.txt", f"{wd}_desc-wm_fod.mif",
            f"{wd}_desc-gm_response.txt", f"{wd}_desc-gm_fod.mif",
            f"{wd}_desc-csf_response.txt", f"{wd}_desc-csf_fod.mif",
            "-mask", f"{wd}_mask.nii.gz",
            "-nthreads", str(threads),
        ])
        # fmt: on


def run_5ttgen(subject: str, wd: Path) -> None:
    """Generates the 5-tissue-type (5TT) image for Anatomically-Constrained Tractography (ACT).

    Args:
        subject: Subject ID.
        wd: Working directory file prefix.
    """
    logging.info(f"{subject}: 5 Tissues")
    if not Path(f"{wd}_fs2diff_coords_aparc+aseg_5TT.nii").exists():
        # fmt: off
        run_command([
            "5ttgen", "freesurfer",
            f"{wd}_fs2diff_coords_aparc+aseg.nii",
            f"{wd}_fs2diff_coords_aparc+aseg_5TT.nii",
        ])
        # fmt: on


def run_tractography(subject: str, wd: Path, threads: int) -> None:
    """Performs tractography and SIFT filtering.

    Generates 10 million streamlines initially and estimates weights for them
    using SIFT2.

    Args:
        subject: Subject ID.
        wd: Working directory file prefix.
        threads: Number of threads to use.
    """
    logging.info(f"{subject}: Tractogram calculation")
    if not Path(f"{wd}_tractogram_10M.tck").exists():
        # fmt: off
        run_command([
            "tckgen", f"{wd}_desc-wm_fod.mif",
            f"{wd}_tractogram_10M.tck",
            "-act", f"{wd}_fs2diff_coords_aparc+aseg_5TT.nii",
            "-backtrack", "-crop_at_gmwmi",
            "-seed_dynamic", f"{wd}_desc-wm_fod.mif",
            "-maxlength", "250", "-select", "10M",
            "-mask", f"{wd}_mask.nii.gz",
            "-nthreads", str(threads),
        ])
        # fmt: on

    logging.info(f"{subject}: SIFT2 filtering")
    if not Path(f"{wd}_tractogram_10M_SIFT2_weights.txt").exists():
        # fmt: off
        run_command([
            "tcksift2", f"{wd}_tractogram_10M.tck",
            f"{wd}_desc-wm_fod.mif",
            f"{wd}_tractogram_10M_SIFT2_weights.txt",
            "-act", f"{wd}_fs2diff_coords_aparc+aseg_5TT.nii",
            "-nthreads", str(threads),
        ])
        # fmt: on


def cleanup_files(wd: Path) -> None:
    """Removes intermediate files to save disk space.

    Args:
        wd: Working directory file prefix.
    """
    files_to_remove = [
            f"{wd}_b0.nii",
            f"{wd}_b0.nii.gz",
            f"{wd}_eddy_corrected_data.nii",
            f"{wd}_eddy_corrected_data.nii.gz",
            f"{wd}_fs2diff_coords_aparc+aseg.nii",
            # f"{wd}_fs2diff_coords_aparc+aseg_5TT.nii",        # not deleting while switching to SIFT2
            f"{wd}_fs2diff_coords_aparc+aseg_nodes.nii.gz",
            f"{wd}_fs2diff_coords_T1w.nii",
            f"{wd}_mask.nii.gz",
            f"{wd}_desc-csf_response.txt",
            f"{wd}_desc-gm_response.txt",
            f"{wd}_desc-wm_response.txt",
            f"{wd}_desc-csf_fod.mif",
            f"{wd}_desc-gm_fod.mif",
            # f"{wd}_desc-wm_fod.mif",                           # not deleting while switching to SIFT2
    ]

    logging.info(f"Cleaning up intermediate files for {wd.name}")
    for f_path in files_to_remove:
        p = Path(f_path)
        if p.exists():
            p.unlink()


def process_subject(subject_info: SubjectInfo) -> Tuple[str, bool]:
    """Processes a single subject for tractography.

    Args:
        subject_info: Subject to process.

    Returns:
        Subject ID and whether processing succeeded.
    """
    subject = subject_info.subject_id
    id_dir = subject_info.dti_dir / subject

    # Create output directory for this subject
    subject_dir = WORKING_DIR_ROOT / subject
    subject_dir.mkdir(parents=True, exist_ok=True)

    # File prefix for outputs (e.g. /path/to/sub-01/sub-01_dwi_...)
    wd = subject_dir / f"{subject}_dwi"

    fs_dir = find_recon_all_dir(subject, subject_info.cohort)
    logging.info(f"{subject}: recon-all directory: {fs_dir}")

    # Input files
    input_corr_dwi = id_dir / "eddy_corrected_data.nii.gz"
    input_corr_val = id_dir / "BVAL_concat_APPA.bval"
    input_corr_vec = id_dir / "BVEC_concat_APPA.bvec"
    input_corr_dwi_brain_mask = id_dir / "T1w_brain_mask_dMRIres.nii.gz"

    if not check_inputs(subject, input_corr_dwi, fs_dir):
        return (subject, False)

    try:
        logging.info(f"Working on {subject}...")

        # 1. Coregistration
        print(f"Running coregistration for {subject}")
        input_corr_dwi_temp = run_coregistration(
            subject,
            wd,
            input_corr_dwi=input_corr_dwi,
            fs_dir=fs_dir,
            git_dir=GIT_DIR,
            spm_path=SPM_PATH,
            matlab_cmd=MATLAB_CMD,
        )

        # 2. Masking
        print(f"Running masking (copying T1 mask) for {subject}")
        prepare_brain_mask(subject, wd, input_corr_dwi_brain_mask)

        # 3. Response Function
        print(f"Running dwi2response for {subject}")
        run_dwi2response(
            subject,
            wd,
            dwi_file=input_corr_dwi_temp,
            input_corr_val=input_corr_val,
            input_corr_vec=input_corr_vec,
            threads=THREADS,
        )

        # 4. FOD Estimation
        print(f"Running dwi2fod for {subject}")
        run_dwi2fod(
            subject,
            wd,
            dwi_file=input_corr_dwi_temp,
            input_corr_val=input_corr_val,
            input_corr_vec=input_corr_vec,
            threads=THREADS,
        )

        # 5. 5TT Generation
        print(f"Running 5ttgen for {subject}")
        run_5ttgen(subject, wd)

        # 6. Tractography
        print(f"Running tractography for {subject}")
        run_tractography(subject, wd, THREADS)

        # 7. Cleanup
        cleanup_files(wd)

        logging.info(f"Finished {subject}")
        return (subject, True)

    except Exception as e:
        logging.error(f"Error processing {subject}: {e}")
        return (subject, False)


def create_dataset_description(output_path: str, spm_path: Path, final_subjects: str) -> None:
    """Creates a BIDS-compliant dataset_description.json file.

    Args:
        output_path: The path to the output directory where the JSON file will be saved.
        spm_path: The path to the SPM installation directory.
        final_subjects: A string listing the subjects that were processed.
    """
    # Get the versions of FSL, FreeSurfer, SPM and MRTrix
    try:
        with open(os.path.join(os.environ["FSLDIR"], "etc", "fslversion")) as f:
            fsl_version = f.read().strip()
    except (KeyError, OSError):
        fsl_version = "unknown"

    try:
        with open(os.path.join(os.environ["FREESURFER_HOME"], "build-stamp.txt")) as f:
            freesurfer_version = f.read().strip()
    except (KeyError, OSError):
        freesurfer_version = "unknown"

    try:
        spm_version = "unknown"
        with open(spm_path / "Contents.m") as f:
            for line in f:
                if "Version" in line and "SPM" in line:
                    spm_version = line.strip().lstrip('% ').strip()
                    break
    except OSError:
        spm_version = "unknown"

    try:
        mrtrix_version = subprocess.check_output(
            ["mrconvert", "-version"], text=True
        ).splitlines()[0]
    except Exception:
        mrtrix_version = "unknown"

    # Get the exact time
    current_date = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    # Get the hostname of the machine
    hostname = socket.gethostname()

    # Get the current user
    user = os.getlogin()

    description = {
        "Name": "Tractography Output " + current_date,
        "BIDSVersion": "1.10.1",
        "PipelineDescription": {
            "Name": "Tractography Pipeline",
            "Version": "1.0",
            "RunOnMachine": hostname,
            "RunByUser": user,
            "Software": [
                {
                    "Name": "FSL",
                    "Version": fsl_version
                },
                {
                    "Name": "SPM",
                    "Version": spm_version
                },
                {
                    "Name": "FreeSurfer",
                    "Version": freesurfer_version
                },
                {
                    "Name": "MRTrix",
                    "Version": mrtrix_version
                }
            ],
            "SubjectsProcessed": final_subjects
        },
    }

    output_file = os.path.join(output_path, f"dataset_description_{current_date}.json")

    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    try:
        with open(output_file, 'w') as f:
            json.dump(description, f, indent=2)
        print(f"Successfully created {output_file}")
    except Exception as e:
        print(f"Error creating dataset_description.json: {e}")


def log_subject_counts(subjects: List[SubjectInfo]) -> None:
    """Logs how many subjects will be processed per session and cohort.

    Args:
        subjects: Subjects to process.
    """
    session_counts = Counter(s.ses for s in subjects)
    cohort_counts = Counter(s.cohort for s in subjects)

    logging.info(f"Found {len(subjects)} subjects to process")
    for ses in SESSIONS:
        logging.info(f"  {ses}: {session_counts[ses]}")
    for cohort in COHORTS:
        logging.info(f"  {cohort}: {cohort_counts[cohort]}")
    logging.info(f"Parallel Jobs: {JOBS}, Threads per Job: {THREADS}")


def log_summary(results: List[Tuple[str, bool]]) -> List[str]:
    """Prints which subjects succeeded or failed.

    Args:
        results: Subject ID and success flag for each processed subject.

    Returns:
        IDs of the subjects that succeeded.
    """
    successful_subjects = [subject for subject, success in results if success]
    failed_subjects = [subject for subject, success in results if not success]

    print("\n------------------------------")
    print("Processing Summary")
    print("------------------------------")
    print(f"Successful Subjects ({len(successful_subjects)}):")
    for s in successful_subjects:
        print(f" - {s}")

    print(f"\nFailed Subjects ({len(failed_subjects)}):")
    for s in failed_subjects:
        print(f" - {s}")

    logging.info(f"Processing complete. Success: {len(successful_subjects)}, Failed: {len(failed_subjects)}")
    return successful_subjects


def main() -> None:
    """Runs tractography for all subjects that still need it."""
    # 1. Environment and logging
    setup_environment()
    WORKING_DIR_ROOT.mkdir(parents=True, exist_ok=True)
    setup_logging(Path.cwd())

    # 2. Find subjects (test subjects must also pass get_subjects_to_process)
    subjects = get_subjects_to_process()
    if TEST_SUBJECTS:
        logging.info("TEST MODE ENABLED: Filtering for TEST_SUBJECTS.")
        subjects = [s for s in subjects if s.subject_id in TEST_SUBJECTS]
    if not subjects:
        logging.info("No subjects found to process.")
        sys.exit(0)
    log_subject_counts(subjects)

    # 3. Run tractography in parallel
    with multiprocessing.Pool(processes=JOBS) as pool:
        results = pool.map(process_subject, subjects)

    # 4. Summary and dataset description
    successful_subjects = log_summary(results)
    create_dataset_description(str(WORKING_DIR_ROOT), SPM_PATH, ", ".join(successful_subjects))


if __name__ == "__main__":
    main()
