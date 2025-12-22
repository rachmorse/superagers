import os
import subprocess
import sys
import logging
from datetime import datetime
from pathlib import Path
import shutil
import glob
import multiprocessing
import json
import socket

# NOTE: The MATLAB script 'spm_coregister_parcellation.m' 
# should be in the same directory as this Python script.

def setup_logging(output_dir):
    """Setup basic logging to file and console
    
    Args:
        output_dir (Path): Path to the output directory where the log file will be saved.
    
    Returns:
        logging.Logger: Logger object for logging messages.
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


def setup_environment():
    """Set up environment variables for FSL and FreeSurfer."""
    # Set up FSL
    os.environ["FSLDIR"] = "/vol/software/fsl_6_0_4"
    os.environ["PATH"] = f"{os.environ['FSLDIR']}/bin:" + os.environ["PATH"]
    # Set FSL to output uncompressed NIFTI files
    os.environ["FSLOUTPUTTYPE"] = "NIFTI"

    # Set up FreeSurfer
    os.environ["FREESURFER_HOME"] = "/vol/software/freesurfer-6.0" 
    os.environ["PATH"] = f"{os.environ['FREESURFER_HOME']}/bin:" + os.environ["PATH"]


def run_command(cmd, shell=True):
    """Run a shell command and check for errors.
    
    Args:
        cmd (str or list): Command to run.
        shell (bool): Whether to run in a shell.
        
    Raises:
        subprocess.CalledProcessError: If the command fails.
    """
    if isinstance(cmd, list):
        cmd_str = ' '.join(str(c) for c in cmd)
    else:
        cmd_str = cmd
        
    logging.info(f"Running command: {cmd_str}")
    
    try:
        subprocess.run(cmd_str, shell=shell, check=True, executable="/bin/bash", capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed: {e.cmd}")
        logging.error(f"STDOUT: {e.stdout}")
        logging.error(f"STDERR: {e.stderr}")
        raise e


def get_subjects_to_process(dti_dirs, tracto_dirs, working_dir_root, recon_all_bbhi, recon_all_senior):
    """Generate a list of subjects to process from multiple directories.
    
    Args:
        dti_dirs (list of Path): List of directories containing subject DTI files.
        tracto_dirs (list of Path): List of directories containing tractography output files in shared directories.
        working_dir_root (Path): Root directory where processed subjects are stored.
        recon_all_bbhi (Path): Directory containing recon-all data for BBHI cohort.
        recon_all_senior (Path): Directory containing recon-all data for Senior cohort.

    Returns:
        list: List of tuples (subject_id, source_dir).
    """
    subjects_info = []
    
    # Scan each DTI directory for subjects
    for dti_dir in dti_dirs:
        if not dti_dir.exists():
            logging.warning(f"DTI directory not found: {dti_dir}")
            continue

        # Scan directory
        for sub_path in dti_dir.glob("sub-*"):
            subjects_info.append((sub_path.name, dti_dir))

    # Check recon-all directories and filter subjects
    valid_subjects_info = []

    for subject_id, source_dir in subjects_info:
        # BBHI: sub-x_ses-0y_run-01
        bbhi_recon = recon_all_bbhi / f"{subject_id}_run-01"

        # Senior: sub-x_ses-0y
        senior_recon = recon_all_senior / subject_id

        if bbhi_recon.exists() or senior_recon.exists():
            valid_subjects_info.append((subject_id, source_dir))
        else:
            logging.info(f"Skipping {subject_id}, no recon-all found.")

    # Filter out done subjects based on the output directory
    filtered_subjects = []

    for subject_id, source_dir in valid_subjects_info:
        wd_sub = working_dir_root / subject_id

        if wd_sub.exists() and any(wd_sub.glob("*_tractogram_10M_SIFT2_weights.txt")):
            logging.info(f"Skipping {subject_id}, already processed (in local directory).")
            continue

        found_in_shared = False
        for tracto_dir in tracto_dirs:
            tracto_file = tracto_dir / subject_id / f"{subject_id}_dwi_tractogram_10M_SIFT2_weights.txt"
            if tracto_file.exists():
                logging.info(f"Skipping {subject_id}, already processed (in shared directory).")
                found_in_shared = True
                break
        
        if found_in_shared:
            continue
            
        filtered_subjects.append((subject_id, source_dir))

    return sorted(filtered_subjects, key=lambda x: x[0])


def check_inputs(subject, input_corr_dwi, fs_dir):
    """Checks if the necessary input files and directories exist.

    Args:
        subject (str): Subject ID.
        input_corr_dwi (Path): Path to the eddy corrected DWI data.
        fs_dir (Path): Path to the subject's recon-all directory.

    Returns:
        bool: True if inputs exist, False otherwise.
    """
    if not input_corr_dwi.exists():
        logging.error(f"{subject}: No eddy corrected data found at {input_corr_dwi}")
        return False
    if not fs_dir.exists():
        logging.error(f"{subject}: No recon-all directory found at {fs_dir}")
        return False
    return True


def run_coregistration(subject, wd, input_corr_dwi, fs_dir, git_dir, spm_path, matlab_cmd):
    """Performs structural to diffusion coregistration.

    This function copies and unzips the eddy corrected data, converts recon-all
    T1 and segmentation files to NIfTI format, extracts the b0 image, and
    estimates the coregistration using a MATLAB script.

    Args:
        subject (str): Subject ID.
        wd (Path): Working directory.
        input_corr_dwi (Path): Path to the eddy corrected DWI data.
        fs_dir (Path): Path to the subject's recon-all directory.
        git_dir (Path): Directory containing the MATLAB script.
        matlab_cmd (str): Command to execute MATLAB.

    Returns:
        str: Path to the unzipped eddy corrected DWI data.
    """
    logging.info(f"{subject}: Structural 2 diff. coregistration")
    
    if not (Path(f"{wd}_fs2diff_coords_T1w_brainmask_diff.nii.gz").exists()):
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
            f"{matlab_cmd} -r \"addpath(genpath('{str(spm_path)}')); addpath('{git_dir}'); "
            f"spm_coregister_parcellation('{wd}_b0.nii', '{wd}_fs2diff_coords_T1w.nii', "
            f"'{wd}_fs2diff_coords_aparc+aseg.nii'); exit;\""
        )
        run_command(matlab_cmd_full)

        return input_corr_dwi_temp


def prepare_brain_mask(subject, wd, input_corr_dwi_brainMASK):
    """Prepares the brain mask by copying the T1-derived mask.

    This function exclusively uses the pre-computed T1w_brain_mask_dMRIres.nii.gz.
    If this mask is missing, processing for this subject will fail.

    Args:
        subject (str): Subject ID.
        wd (Path): Working directory.
        input_corr_dwi_brainMASK (Path): Path to the pre-existing brain mask.
    
    Raises:
        FileNotFoundError: If the robust T1 mask is missing.
    """
    logging.info(f"{subject}: using robust T1 mask")
    if not Path(f"{wd}_mask.nii.gz").exists():
        if input_corr_dwi_brainMASK.exists():
            run_command(f"cp {input_corr_dwi_brainMASK} {wd}_mask.nii.gz")
        else:
            raise FileNotFoundError(f"Robust T1 mask not found for {subject} at {input_corr_dwi_brainMASK}")


def run_dwi2response(subject, wd, dwi_file, input_corr_val, input_corr_vec, threads):
    """Estimates the tissue response functions for spherical deconvolution.

    Args:
        subject (str): Subject ID.
        wd (Path): Working directory.
        dwi_file (Path): Path to the DWI data file.
        input_corr_val (Path): Path to the bval file.
        input_corr_vec (Path): Path to the bvec file.
        threads (int): Number of threads to use.
    """
    logging.info(f"{subject}: Response Function Estimation")
    if not Path(f"{wd}_desc-csf_response.txt").exists():
        run_command([
            "dwi2response", "dhollander", str(dwi_file),
            "-fslgrad", str(input_corr_vec), str(input_corr_val),
            f"{wd}_desc-wm_response.txt",
            f"{wd}_desc-gm_response.txt",
            f"{wd}_desc-csf_response.txt",
            "-mask", f"{wd}_mask.nii.gz",
            "-nthreads", str(threads)
        ])


def run_dwi2fod(subject, wd, dwi_file, input_corr_val, input_corr_vec, threads):
    """Estimates Fibre Orientation Distributions (FODs) using Multi-Shell Multi-Tissue CSD.

    Args:
        subject (str): Subject ID.
        wd (Path): Working directory.
        dwi_file (Path): Path to the DWI data file.
        input_corr_val (Path): Path to the bval file.
        input_corr_vec (Path): Path to the bvec file.
        threads (int): Number of threads to use.
    """
    logging.info(f"{subject}: FOD Estimation")
    if not Path(f"{wd}_desc-wm_fod.mif").exists():
        run_command([
            "dwi2fod", "msmt_csd", str(dwi_file),
            "-fslgrad", str(input_corr_vec), str(input_corr_val),
            f"{wd}_desc-wm_response.txt", f"{wd}_desc-wm_fod.mif",
            f"{wd}_desc-gm_response.txt", f"{wd}_desc-gm_fod.mif",
            f"{wd}_desc-csf_response.txt", f"{wd}_desc-csf_fod.mif",
            "-mask", f"{wd}_mask.nii.gz",
            "-nthreads", str(threads)
        ])


def run_5ttgen(subject, wd):
    """Generates the 5-tissue-type (5TT) image for Anatomically-Constrained Tractography (ACT).

    Args:
        subject (str): Subject ID.
        wd (Path): Working directory.
    """
    logging.info(f"{subject}: 5 Tissues")
    if not Path(f"{wd}_fs2diff_coords_aparc+aseg_5TT.nii").exists():
        run_command([
            "5ttgen", "freesurfer", 
            f"{wd}_fs2diff_coords_aparc+aseg.nii",
            f"{wd}_fs2diff_coords_aparc+aseg_5TT.nii"
        ])


def run_tractography(subject, wd, threads):
    """Performs tractography and SIFT filtering.

    Generates 10 million streamlines initially and filters them down to 1 million
    using SIFT (Spherical-deconvolution Informed Filtering of Tractograms).

    Args:
        subject (str): Subject ID.
        wd (Path): Working directory.
        threads (int): Number of threads to use.
    """
    logging.info(f"{subject}: Tractogram calculation")
    if not Path(f"{wd}_tractogram_10M.tck").exists():
        run_command([
            "tckgen", f"{wd}_desc-wm_fod.mif",
            f"{wd}_tractogram_10M.tck",
            "-act", f"{wd}_fs2diff_coords_aparc+aseg_5TT.nii",
            "-backtrack", "-crop_at_gmwmi",
            "-seed_dynamic", f"{wd}_desc-wm_fod.mif",
            "-maxlength", "250", "-select", "10M",
            "-mask", f"{wd}_mask.nii.gz",
            "-nthreads", str(threads)
        ])

    logging.info(f"{subject}: SIFT2 filtering")
    if not Path(f"{wd}_tractogram_10M_SIFT2_weights.txt").exists():
        run_command([
            "tcksift2", f"{wd}_tractogram_10M.tck",
            f"{wd}_desc-wm_fod.mif",
            f"{wd}_tractogram_10M_SIFT2_weights.txt",
            "-act", f"{wd}_fs2diff_coords_aparc+aseg_5TT.nii",
            "-nthreads", str(threads)
        ])


def cleanup_files(wd):
    """Removes intermediate files to save disk space.

    Args:
        wd (Path): Working directory.
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
    
    # Uncomment to enable cleanup
    logging.info(f"Cleaning up intermediate files for {wd.name}")
    for f_path in files_to_remove:
        p = Path(f_path)
        if p.exists():
            p.unlink()


def process_subject(subject, dti_dir, recon_all_dir, working_dir_root, git_dir, spm_path, matlab_cmd, threads=10):
    """Process a single subject for tractography.
    
    Args:
        subject (str): Subject ID (e.g., 'sub-12345').
        dti_dir (Path): Directory containing DTI data (e.g. DWI_dtifit_tp1).
        recon_all_dir (Path): Directory containing FreeSurfer data.
        working_dir_root (Path): Root directory for working files.
        git_dir (Path): Directory containing dependencies (like colorlut).
        spm_path (Path): Path to SPM installation.
        matlab_cmd (str): Command to launch MATLAB.
        threads (int): Number of threads for MRTrix commands.
        
    Returns:
        tuple: (subject, success)
    """    
    # Directories
    id_dir = dti_dir / subject
    
    # Create output directory for this subject
    subject_dir = working_dir_root / subject
    subject_dir.mkdir(parents=True, exist_ok=True)
    
    # File prefix for outputs (e.g. /path/to/sub-01/sub-01_dwi_...)
    wd = subject_dir / f"{subject}_dwi"
    
    # Handle recon-all path:
    possible_fs_dirs = []
    
    if "_ses-" in subject:
        # e.g. 'sub-44010_ses-01' -> try '..._run-01' then base
        possible_fs_dirs.append(recon_all_dir / f"{subject}_run-01")
        possible_fs_dirs.append(recon_all_dir / f"{subject}")
    else:
        # Fallback
        possible_fs_dirs.append(recon_all_dir / f"{subject}_ses-01_run-01")
        possible_fs_dirs.append(recon_all_dir / f"{subject}_ses-01") # Just in case

    fs_dir = possible_fs_dirs[0] # Default
    for p in possible_fs_dirs:
        if p.exists():
            fs_dir = p
            break
            
    logging.info(f"DEBUG: Checking FS dirs: {[str(p) for p in possible_fs_dirs]} -> Selected: {fs_dir}") 
    
    # Input files
    input_corr_dwi = id_dir / "eddy_corrected_data.nii.gz"
    input_corr_val = id_dir / "BVAL_concat_APPA.bval"
    input_corr_vec = id_dir / "BVEC_concat_APPA.bvec"
    input_corr_dwi_brainMASK = id_dir / "T1w_brain_mask_dMRIres.nii.gz"
    
    if not check_inputs(subject, input_corr_dwi, fs_dir):
        return (subject, False)
        
    try:
        logging.info(f"Working on {subject}...")
        
        # 1. Coregistration
        print(f"Running coregistration for {subject}")
        input_corr_dwi_temp = run_coregistration(subject, wd, input_corr_dwi, fs_dir, git_dir, spm_path, matlab_cmd)

        # 2. Masking
        print(f"Running masking (copying T1 mask) for {subject}")
        prepare_brain_mask(subject, wd, input_corr_dwi_brainMASK)

        # 3. Response Function
        print(f"Running dwi2response for {subject}")
        run_dwi2response(subject, wd, input_corr_dwi_temp, input_corr_val, input_corr_vec, threads)

        # 4. FOD Estimation
        print(f"Running dwi2fod for {subject}")
        run_dwi2fod(subject, wd, input_corr_dwi_temp, input_corr_val, input_corr_vec, threads)

        # 5. 5TT Generation
        print(f"Running 5ttgen for {subject}")
        run_5ttgen(subject, wd)

        # 6. Tractography
        print(f"Running tractography for {subject}")
        run_tractography(subject, wd, threads)
            
        # 7. Cleanup
        cleanup_files(wd)

        logging.info(f"Finished {subject}")
        return (subject, True)

    except Exception as e:
        logging.error(f"Error processing {subject}: {e}")
        return (subject, False)


def create_dataset_description(output_path: str, spm_path: Path, final_subjects: str):
    """Create a BIDS-compliant dataset_description.json file.

    Args:
        output_path (str): The path to the output directory where the JSON file will be saved.
        spm_path (Path): The path to the SPM installation directory.
        final_subjects (str): A string listing the subjects that were processed.
    """
    # Get the versions of FSL, FreeSurfer, SPM and MRTrix
    try:
        with open(os.path.join(os.environ["FSLDIR"], "etc", "fslversion")) as f:
            fsl_version = f.read().strip()
    except:
        fsl_version = "unknown"

    try:
        with open(os.path.join(os.environ["FREESURFER_HOME"], "build-stamp.txt")) as f:
            freesurfer_version = f.read().strip()
    except:
        freesurfer_version = "unknown"

    try:
        spm_version = "unknown"
        with open(spm_path / "Contents.m") as f:
            for line in f:
                if "Version" in line and "SPM" in line:
                    spm_version = line.strip().lstrip('% ').strip()
                    break
    except:
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


def main():
    # Configuration
    jobs = 1          # Number of subjects to process in parallel
    threads = 20      # Number of threads per subject (for MRTrix)
    sessions = ["ses-01", "ses-02"]

    # Paths - using multiple cohorts
    dti_dirs = []
    tracto_dirs = []
    cohorts = ["bbhi"]

    for ses in sessions:
        # bbhi
        if "bbhi" in cohorts:
            dti_dirs.append(Path(f"/pool/guttmann/institut/BBHI/MRI/processed_data/dtifit_{ses}_fsl-604"))
            tracto_dirs.append(Path(f"/pool/guttmann/institut/BBHI/MRI/processed_data/tracto_SIFT2"))
        # bbhi senior
        if "bbhi senior" in cohorts:
            dti_dirs.append(Path(f"/pool/guttmann/institut/UB/Superagers/MRI/dtifit_{ses}_fsl-604"))
            tracto_dirs.append(Path(f"/pool/guttmann/institut/UB/Superagers/MRI/tracto_SIFT2"))
        
    recon_all_bbhi = Path("/pool/guttmann/institut/BBHI/MRI/derivatives/reconall_fs6")
    recon_all_senior = Path("/pool/guttmann/institut/UB/Superagers/MRI/derivatives/reconall_fs6")

    git_dir = Path("/home/rachel/Desktop/superagers/structural_analysis")
    working_dir_root = Path("/home/rachel/Desktop/schaefer_analysis/tracto_wd")
    
    spm_path = Path("/home/rachel/spm12")
    matlab_cmd = "/usr/local/bin/matlab -nodesktop -nosplash"
    
    # Setup
    setup_environment()
    
    if not working_dir_root.exists():
        working_dir_root.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(Path.cwd()) # Log to current dir

    # Get subjects
    subjects_info = get_subjects_to_process(dti_dirs, tracto_dirs, working_dir_root, recon_all_bbhi, recon_all_senior)
    
    # TEST MODE: Run on a specific list of subs. They must be in the list from get_subjects_to_process 
    # if subjects_info:
    #     logging.info(f"TEST MODE ENABLED: Filtering for specific test subjects.")
    #     # Filters the list of subjects to only include the target subject/s
    #     included_subs = {
    #         "sub-1283_ses-02",
    #         "sub-3079_ses-02",
    #         "sub-4141_ses-02",
    #         "sub-4145_ses-01",
    #         "sub-62333_ses-01",
    #         "sub-77813_ses-01",
    #         "sub-95133_ses-01",
    #         "sub-95272_ses-01",
    #         "sub-95653_ses-01",
    #         "sub-100204_ses-01",
    #         "sub-1014_ses-01",
    #     }
    #     subjects_info = [s for s in subjects_info if s[0] in included_subs]
    #     if not subjects_info:
    #          logging.info("Target test subject not found.")
    #          sys.exit(0)

    if not subjects_info:
        logging.info("No subjects found to process.")
        sys.exit(0)

    # Breakdown counts
    count_ses01 = 0
    count_ses02 = 0
    count_bbhi = 0
    count_senior = 0

    for _, d_path in subjects_info:
        s_path = str(d_path)
        if "ses-01" in s_path:
            count_ses01 += 1
        elif "ses-02" in s_path:
            count_ses02 += 1
        
        if "Superagers" in s_path:
            count_senior += 1
        else:
            count_bbhi += 1

    logging.info(f"Found {len(subjects_info)} subjects to process")
    logging.info(f"  Session 01: {count_ses01}")
    logging.info(f"  Session 02: {count_ses02}")
    logging.info(f"  BBHI: {count_bbhi}")
    logging.info(f"  BBHI Senior: {count_senior}")
    logging.info(f"Parallel Jobs: {jobs}, Threads per Job: {threads}")
    
    successful_subjects = []
    failed_subjects = []

    # Prepare arguments for starmap
    pool_args = []
    for subject, dti_dir in subjects_info:
        # Determine correct recon-all dir based on DTI path for each cohort
        if "BBHI" in str(dti_dir):
            current_recon_dir = recon_all_bbhi
        else:
            current_recon_dir = recon_all_senior

        pool_args.append((
            subject,
            dti_dir,
            current_recon_dir,
            working_dir_root,
            git_dir,
            spm_path,
            matlab_cmd,
            threads
        ))

    # Parallel execution
    with multiprocessing.Pool(processes=jobs) as pool:
        results = pool.starmap(process_subject, pool_args)

    for subject, success in results:
        if success:
            successful_subjects.append(subject)
            print(f"SUCCESS: {subject}")
        else:
            failed_subjects.append(subject)
            print(f"FAILED: {subject}")
            
    # Summary
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

    create_dataset_description(
        str(working_dir_root),
        spm_path,
        final_subjects=', '.join(successful_subjects)
    )


if __name__ == "__main__":
    main()
