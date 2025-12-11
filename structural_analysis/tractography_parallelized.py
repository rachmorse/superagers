import os
import subprocess
import sys
import logging
from datetime import datetime
from pathlib import Path
import shutil
import glob
import multiprocessing

# ------------------------------------------------------------------------------
# MATLAB Script Configuration
# ------------------------------------------------------------------------------
# The MATLAB script 'spm_coregister_parcellation.m' is expected to be in the
# same directory as this Python script.

def setup_logging(output_dir):
    """Setup basic logging to file and console
    
    Args:
        output_dir (Path): Path to the output directory where the log file will be saved.
    
    Returns:
        logging.Logger: Logger object for logging messages.
    """
    log_file = output_dir / f"tractography_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
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
    """Set up environment variables for FSL, FreeSurfer, and MATLAB."""
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
    logging.info(f"Running command: {cmd if isinstance(cmd, str) else ' '.join(cmd)}")
    subprocess.run(cmd, shell=shell, check=True, executable="/bin/bash")


def get_subjects_to_process(dti_dir, done_list_path=None):
    """Generate a list of subjects to process.
    
    Args:
        dti_dir (Path): Path to directory containing subject folders (e.g. DWI_dtifit_tp1).
        done_list_path (Path, optional): Path to a text file listing completed subjects.

    Returns:
        list: List of subject IDs (e.g., 'sub-1001').
    """
    if not dti_dir.exists():
        logging.warning(f"DTI directory not found: {dti_dir}")
        return []

    subjects = []
    
    # Scan directory
    for sub_path in dti_dir.glob("sub-*"):
        subjects.append(sub_path.name) # e.g., 'sub-12345'
        
    # Filter out done subjects if a list is provided
    if done_list_path and done_list_path.exists():
        with open(done_list_path, 'r') as f:
            done_subjects = set(line.strip() for line in f)
        subjects = [s for s in subjects if s not in done_subjects]
        
    return sorted(subjects)


def check_inputs(subject, input_corr_dwi, fs_dir):
    """Checks if the necessary input files and directories exist.

    Args:
        subject (str): Subject ID.
        input_corr_dwi (Path): Path to the eddy corrected DWI data.
        fs_dir (Path): Path to the subject's FreeSurfer directory.

    Returns:
        bool: True if inputs exist, False otherwise.
    """
    if not input_corr_dwi.exists():
        logging.error(f"{subject}: No eddy corrected data found at {input_corr_dwi}")
        return False
    if not fs_dir.exists():
        logging.error(f"{subject}: No FreeSurfer directory found at {fs_dir}")
        return False
    return True


def run_coregistration(subject, wd, input_corr_dwi, fs_dir, git_dir, matlab_cmd):
    """Performs structural to diffusion coregistration.

    This function copies and unzips the eddy corrected data, converts FreeSurfer
    T1 and segmentation files to NIfTI format, extracts the b0 image, and
    estimates the coregistration using a MATLAB script.

    Args:
        subject (str): Subject ID.
        wd (Path): Working directory.
        input_corr_dwi (Path): Path to the eddy corrected DWI data.
        fs_dir (Path): Path to the subject's FreeSurfer directory.
        git_dir (Path): Directory containing the scripts (for MATLAB path).
        matlab_cmd (str): Command to execute MATLAB.
    """
    logging.info(f"{subject}: Structural 2 diff. coregistration")
    
    if not (Path(f"{wd}_fs2diff_coords_T1w_brainmask_diff.nii.gz").exists()):
        # Copy and unzip eddy data
        run_command(f"cp -f {input_corr_dwi} {wd}_eddy_corrected_data.nii.gz")
        input_corr_dwi_temp = f"{wd}_eddy_corrected_data.nii" # No .gz
        run_command(f"gunzip -f {wd}_eddy_corrected_data.nii.gz")
        
        # Convert FreeSurfer T1 and aparc+aseg
        run_command(f"mrconvert -force {fs_dir}/mri/T1.mgz {wd}_fs2diff_coords_T1w.nii")
        run_command(f"mrconvert -force {fs_dir}/mri/aparc+aseg.mgz '{wd}_fs2diff_coords_aparc+aseg.nii'")
        
        # Extract b0
        run_command(f"fslroi {input_corr_dwi_temp} {wd}_b0.nii.gz 0 1")
        run_command(f"gzip -f -d {wd}_b0.nii.gz") # Becomes b0.nii
        
        # MATLAB Coregistration
        matlab_cmd_full = (
            f"{matlab_cmd} -r \"addpath('{git_dir}'); "
            f"spm_coregister_parcellation('{wd}_b0.nii', '{wd}_fs2diff_coords_T1w.nii', "
            f"'{wd}_fs2diff_coords_aparc+aseg.nii'); exit;\""
        )
        run_command(matlab_cmd_full)


def get_working_dwi_file(wd, input_corr_dwi):
    """Resolves the path to the DWI file to use for processing.

    It checks for an unzipped or zipped copy in the working directory first,
    falling back to the original input file.

    Args:
        wd (Path): Working directory.
        input_corr_dwi (Path): Path to the original input DWI file.

    Returns:
        Path: Path to the DWI file to use.
    """
    current_dwi = Path(f"{wd}_eddy_corrected_data.nii")
    if not current_dwi.exists():
        current_dwi = Path(f"{wd}_eddy_corrected_data.nii.gz")
    if not current_dwi.exists():
        current_dwi = input_corr_dwi
    return current_dwi


def run_dwi2mask(subject, wd, input_corr_dwi_brainMASK, dwi_file, input_corr_val, input_corr_vec, threads):
    """Generates a brain mask for the DWI data.

    If a pre-computed mask exists (input_corr_dwi_brainMASK), it is copied.
    Otherwise, dwi2mask is used to compute it from the DWI data.

    Args:
        subject (str): Subject ID.
        wd (Path): Working directory.
        input_corr_dwi_brainMASK (Path): Path to the pre-existing brain mask.
        dwi_file (Path): Path to the DWI data file.
        input_corr_val (Path): Path to the bval file.
        input_corr_vec (Path): Path to the bvec file.
        threads (int): Number of threads to use.
    """
    if not Path(f"{wd}_mask.nii.gz").exists():
        # Fallback logic to dwi2mask if not copied
        if input_corr_dwi_brainMASK.exists():
            run_command(f"cp {input_corr_dwi_brainMASK} {wd}_mask.nii.gz")
        else:
            run_command([
                "dwi2mask", str(dwi_file), f"{wd}_mask.nii.gz",
                "-fslgrad", str(input_corr_vec), str(input_corr_val),
                "-nthreads", str(threads)
            ])


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


def run_labelconvert(subject, wd, fs_colorlut, fs_default):
    """Converts the FreeSurfer parcellation to a format suitable for connectivity analysis.

    Args:
        subject (str): Subject ID.
        wd (Path): Working directory.
        fs_colorlut (Path): Path to the FreeSurfer ColorLUT file.
        fs_default (Path): Path to the default FreeSurfer label file.
    """
    logging.info(f"{subject}: Creating matrix nodes")
    if not Path(f"{wd}_fs2diff_coords_aparc+aseg_nodes.nii.gz").exists():
            run_command([
                "labelconvert",
                f"{wd}_fs2diff_coords_aparc+aseg.nii",
                str(fs_colorlut), str(fs_default),
                f"{wd}_fs2diff_coords_aparc+aseg_nodes.nii.gz",
                "-force"
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

    if not Path(f"{wd}_tractogram_1M_SIFT.tck").exists():
        run_command([
            "tcksift", f"{wd}_tractogram_10M.tck",
            f"{wd}_desc-wm_fod.mif",
            f"{wd}_tractogram_1M_SIFT.tck",
            "-act", f"{wd}_fs2diff_coords_aparc+aseg_5TT.nii",
            "-term_number", "1M",
            "-nthreads", str(threads)
        ])


def cleanup_files(subject, wd):
    """Removes intermediate files to save disk space.

    Args:
        subject (str): Subject ID.
        wd (Path): Working directory.
    """
    files_to_remove = [
            f"{wd}_b0.nii",
            f"{wd}_b0.nii.gz",
            f"{wd}_eddy_corrected_data.nii",
            f"{wd}_eddy_corrected_data.nii.gz",
            f"{wd}_fs2diff_coords_aparc+aseg.nii",
            f"{wd}_fs2diff_coords_aparc+aseg_5TT.nii",
            f"{wd}_fs2diff_coords_aparc+aseg_nodes.nii.gz",
            f"{wd}_fs2diff_coords_T1w.nii",
            f"{wd}_mask.nii.gz",
            f"{wd}_desc-csf_response.txt",
            f"{wd}_desc-gm_response.txt",
            f"{wd}_desc-wm_response.txt",
            f"{wd}_tractogram_10M.tck",
    ]
    
    # Uncomment to enable cleanup
    # for f_path in files_to_remove:
    #     p = Path(f_path)
    #     if p.exists():
    #         p.unlink()


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
    wd = working_dir_root / f"{subject}_dwi"
    fs_dir = recon_all_dir / f"{subject}_ses-01" 
    
    # Input files
    input_corr_dwi = id_dir / "eddy_corrected_data.nii.gz"
    input_corr_val = id_dir / "BVAL_concat_APPA.bval"
    input_corr_vec = id_dir / "BVEC_concat_APPA.bvec"
    input_corr_dwi_brainMASK = id_dir / "T1w_brain_mask_dMRIres.nii.gz"
    
    # Dependencies
    fs_default = git_dir / "dependences/fs_default.txt"
    fs_colorlut = git_dir / "dependences/FreeSurferColorLUT.txt"
    
    if not check_inputs(subject, input_corr_dwi, fs_dir):
        return (subject, False)
        
    try:
        logging.info(f"Working on {subject}...")
        
        # 1. Coregistration
        print(f"Running coregistration for {subject}")
        run_coregistration(subject, wd, input_corr_dwi, fs_dir, git_dir, matlab_cmd)

        # Resolve DWI file for subsequent steps
        dwi_file = get_working_dwi_file(wd, input_corr_dwi)

        # 2. Masking
        print(f"Running dwi2mask for {subject}")
        run_dwi2mask(subject, wd, input_corr_dwi_brainMASK, dwi_file, input_corr_val, input_corr_vec, threads)

        # 3. Response Function
        print(f"Running dwi2response for {subject}")
        run_dwi2response(subject, wd, dwi_file, input_corr_val, input_corr_vec, threads)

        # 4. FOD Estimation
        print(f"Running dwi2fod for {subject}")
        run_dwi2fod(subject, wd, dwi_file, input_corr_val, input_corr_vec, threads)

        # 5. 5TT Generation
        print(f"Running 5ttgen for {subject}")
        run_5ttgen(subject, wd)
            
        # 6. Label Conversion
        print(f"Running labelconvert for {subject}")
        run_labelconvert(subject, wd, fs_colorlut, fs_default)

        # 7. Tractography
        print(f"Running tractography for {subject}")
        run_tractography(subject, wd, threads)
            
        # 8. Cleanup
        cleanup_files(subject, wd)

        logging.info(f"Finished {subject}")
        return (subject, True)

    except Exception as e:
        logging.error(f"Error processing {subject}: {e}")
        return (subject, False)


def main():
    # --------------------------------------------------------------------------
    # Configuration
    # --------------------------------------------------------------------------
    # Resource Management
    jobs = 2          # Number of subjects to process in parallel
    threads = 10      # Number of threads per subject (for MRTrix)
    
    # Paths
    dti_dir = Path("/pool/guttmann/institut/BBHI/MRI/processed_data/DWI_dtifit_tp1")
    recon_all_dir = Path("/pool/guttmann/institut/BBHI/MRI/processed_data/freesurfer-reconall")
    git_dir = Path("/home/mariacabello/working_dir/multimodal/DTI/TRACTO_Saul")
    working_dir_root = Path("/home/mariacabello/working_dir/multimodal/DTI/tracto_wd")
    
    spm_path = Path("/home/rachel/spm12")
    matlab_cmd = "/usr/local/bin/matlab -nodesktop -nosplash"
    
    # Setup
    setup_environment()
    
    if not working_dir_root.exists():
        # working_dir_root.mkdir(parents=True, exist_ok=True)
        pass 

    logger = setup_logging(Path.cwd()) # Log to current dir
    
    # Use the directory where this script is running from as the location for the .m file
    script_dir = Path(__file__).resolve().parent

    # Get subjects
    subjects = get_subjects_to_process(dti_dir)
    
    if not subjects:
        print("No subjects found to process.")
        sys.exit(0)

    print(f"Found {len(subjects)} subjects to process.")
    print(f"Parallel Jobs: {jobs}, Threads per Job: {threads}")
    
    successful_subjects = []
    failed_subjects = []

    # Prepare arguments for starmap
    pool_args = []
    for subject in subjects:
        pool_args.append((
            subject,
            dti_dir,
            recon_all_dir,
            working_dir_root,
            git_dir,
            spm_path,
            matlab_cmd,
            threads
        ))

    # Parallel Execution with Multiprocessing Pool
    with multiprocessing.Pool(processes=jobs) as pool:
        # starmap blocks until all results are ready
        results = pool.starmap(process_subject, pool_args)

    for subject, success in results:
        if success:
            successful_subjects.append(subject)
            print(f"SUCCESS: {subject}")
        else:
            failed_subjects.append(subject)
            print(f"FAILED: {subject}")
            
    # --------------------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------------------
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


if __name__ == "__main__":
    main()
