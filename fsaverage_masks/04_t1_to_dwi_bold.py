#!/usr/bin/env python3
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class SubjectPaths:
    subject: str
    ses_label: str
    schaefer_subcort_atlas: Path
    bold_data: Optional[Path]
    eddy_corrected: Optional[Path]
    dwi_mask_output: Path
    bold_mask_output: Path


@dataclass
class DwiProcessingPaths:
    subject: str
    ses_label: str
    t1_mgz_path: Path
    t1w_mask: Path
    t1_brain_path: Path
    eddy_corrected: Path
    b0_output: Path
    out_b0_dir: Path
    out_native_masks: Path
    output_path_dwi: Path
    out_subject_dir: Path


@dataclass
class BoldProcessingPaths:
    subject: str
    ses_label: str
    t1_mgz_path: Path
    t1w_mask: Path
    bold_path: Path
    out_bold_masks: Path
    output_path_bold: Path
    out_subject_dir: Path


def run_parallel_tasks(tasks, processor, job_name):
    """Run tasks sequentially or in parallel depending on task count."""
    successes = []
    failures = []
    if not tasks:
        return successes, failures

    cores = 10
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
        future_to_subject = {
            executor.submit(processor, paths): subject for subject, paths in tasks
        }
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

    print(f"Subjects already with DWI mask: {len(already_processed_dwi)}")
    print(f"Subjects already with BOLD mask: {len(already_processed_bold)}")
    print(f"Subjects already fully processed: {len(already_processed)}")
    print(f"Subjects missing required files: {len(missing_files)}")     

    return subjects_for_dwi, subjects_for_bold, already_processed


def transform_t1w_to_dwi(t1w_mask, t1_anat, t1_brain, eddy_corrected, b0_ref, output_path, out_native_masks):
    """Create a b0 reference and transform the T1w space mask to native DWI space 
    using epi_reg and flirt.
    
    Args:
        t1w_mask (Path): Path to the T1w space mask.
        t1_anat (Path): Path to the T1 NIfTI image from recon-all.
        t1_brain (Path): Path to the brain-extracted T1 image.
        eddy_corrected (Path): Path to the eddy-corrected DWI series.
        b0_ref (Path): Path to the b0 reference image.
        output_path (Path): Path to save the DWI space mask file.
        out_native_masks (Path): Output directory to where the mask is saved.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create b0 reference from the eddy-corrected data (first volume)
    b0_ref.parent.mkdir(parents=True, exist_ok=True)
    if not eddy_corrected.exists():
        print(f"ERROR: Eddy-corrected DWI file not found: {eddy_corrected}")
        raise FileNotFoundError(f"Eddy-corrected DWI not found: {eddy_corrected}")

    if not b0_ref.exists():
        subprocess.run(
            f"fslroi {eddy_corrected} {b0_ref} 0 1",
            shell=True,
            check=True,
        )

    if not b0_ref.exists():
        print(f"ERROR: Failed to create b0 reference: {b0_ref}")
        raise FileNotFoundError(f"b0 reference not created: {b0_ref}")

    epi_reg_prefix = out_native_masks / "epi_reg"
    epi_reg_prefix.parent.mkdir(parents=True, exist_ok=True)

    # Run epi_reg to compute the transformation from b0 to T1w
    subprocess.run(
        [
            "epi_reg",
            f"--epi={b0_ref}",
            f"--t1={t1_anat}",
            f"--t1brain={t1_brain}",
            f"--out={epi_reg_prefix}",
        ],
        check=True,
    )

    epi_to_t1_mat = Path(f"{epi_reg_prefix}.mat")
    transform_mat = out_native_masks / "T1w_to_b0.mat"

    # Invert the transformation matrix to go from T1w to b0
    subprocess.run(
        [
            "convert_xfm",
            "-omat",
            str(transform_mat),
            "-inverse",
            str(epi_to_t1_mat),
        ],
        check=True,
    )

    # Apply the transformation to the T1w mask to bring it into DWI space
    subprocess.run(
        f"flirt -in {t1w_mask} -ref {b0_ref} -applyxfm -init {transform_mat} -out {output_path} -paddingsize 0.0 -interp nearestneighbour",
        shell=True,
        check=True,
    )

    print(f"Transformed T1w mask to DWI space: {output_path}")
    

def transform_t1w_to_bold(t1w_mask, bold_path, out_bold_masks, output_path_bold):
    """Transform the T1w space mask to BOLD native T1 space using mri_vol2vol.
    The BOLD image is already in the same anatomical space as T1 but not the same
    voxel grid.
    
    Args:
        t1w_mask (Path): Path to the T1w space mask.
        bold_path (Path): Path to the BOLD native T1 image.
        out_bold_masks (Path): Output directory to where the mask is saved.
        output_path_bold (Path): Path to save the BOLD space mask.
    """
    # Create output directory if it doesn't exist
    output_path_bold.parent.mkdir(parents=True, exist_ok=True)

    # Check if the BOLD file exists
    if not bold_path.exists():
        print(f"ERROR: BOLD file not found: {bold_path}")
        raise FileNotFoundError(f"BOLD file not found: {bold_path}")
    
    # Extract the first volume from the 4D BOLD images
    bold_ref_path = out_bold_masks / "bold_reference.nii.gz"
    subprocess.run(f"fslroi {bold_path} {bold_ref_path} 0 1", shell=True, check=True)

    # Verify the reference file was created
    if not bold_ref_path.exists():
        print(f"ERROR: Failed to create BOLD reference: {bold_ref_path}")
        raise FileNotFoundError(f"BOLD reference not created: {bold_ref_path}")
    
    # Resample mask onto BOLD grid using mri_vol2vol with regheader
    subprocess.run(
        [
            "mri_vol2vol",
            "--mov", str(t1w_mask),
            "--targ", str(bold_ref_path),
            "--o", str(output_path_bold),
            "--regheader",
            "--interp", "nearest",
        ],
        check=True,
    )
    
    # Verify the output file was created
    if not output_path_bold.exists():
        print(f"ERROR: Failed to create BOLD space mask: {output_path_bold}")
        raise FileNotFoundError(f"BOLD space mask not created: {output_path_bold}")

    print(f"Transformed T1w mask to BOLD space: {output_path_bold}")


def ensure_t1_nifti(t1_mgz, subject, ses_label, out_subject_dir):
    """Convert the subject's T1.mgz to NIfTI.
    
    Args:
        t1_mgz (Path): Path to the subject's recon-all T1.mgz
        subject (str): The subject ID
        ses_label (str): Session label 
        out_subject_dir (Path): Output directory for the subject
        
    Returns:
        Path: Path to the converted NIfTI T1 image
    """
    if not t1_mgz.exists():
        raise FileNotFoundError(f"T1.mgz not found for {subject}: {t1_mgz}")

    t1_dir = out_subject_dir / "t1_converted"
    t1_dir.mkdir(parents=True, exist_ok=True)
    t1_nifti = t1_dir / f"{subject}_{ses_label}_conformed_T1.nii.gz"

    if not t1_nifti.exists():
        subprocess.run(
            ["mri_convert", str(t1_mgz), str(t1_nifti)],
            check=True,
        )

    return t1_nifti


def ensure_t1_brain(t1_anat: Path, t1_brain: Path):
    """Generate a brain-extracted T1 volume required by epi_reg
    using FSL's BET tool.
    
    Args:
        t1_anat (Path): Path to the T1 NIfTI image from recon-all.
        t1_brain (Path): Path to save the brain-extracted T1 image.
    
    Returns:
        Path: Path to the brain-extracted T1 image.
    """
    t1_brain.parent.mkdir(parents=True, exist_ok=True)
    if not t1_brain.exists():
        subprocess.run(
            [
                "bet",
                str(t1_anat),
                str(t1_brain),
                "-R", # More robust 
                "-f", # Removes more non-brain tissue
                "0.2",
            ],
            check=True,
        )
    return t1_brain


def process_subject_dwi(paths: DwiProcessingPaths):
    """Process a single subject's DWI data using predefined file paths.
    
    Args:
        paths (DwiProcessingPaths): Paths required for DWI processing.
    
    Returns:
        str: Subject ID if processing was successful, None otherwise.
    """
    subject = paths.subject
    print(f"\nProcessing DWI for {subject}...")

    try:
        # Step 0: Ensure T1 is in NIfTI format and brain-extracted image is available
        t1_anat = ensure_t1_nifti(
            paths.t1_mgz_path,
            subject,
            paths.ses_label,
            paths.out_subject_dir,
        )
        t1_brain = ensure_t1_brain(t1_anat, paths.t1_brain_path)

        # Step 1: Transform T1w mask to native space using the transformation matrix
        transform_t1w_to_dwi(
            paths.t1w_mask,
            t1_anat,
            t1_brain,
            paths.eddy_corrected,
            paths.b0_output,
            paths.output_path_dwi,
            paths.out_native_masks,
        )

        print(f"Successfully created native space DWI mask for {subject}")

        # Step 4: Clean up individual subject's intermediate files
        if paths.out_b0_dir.exists():
            shutil.rmtree(paths.out_b0_dir)
        if (paths.out_subject_dir / "t1_converted").exists():
            shutil.rmtree(paths.out_subject_dir / "t1_converted")
        epi_reg_files = list((paths.out_subject_dir / "dwi_space_masks").glob("epi_reg*.nii.gz"))
        if epi_reg_files:
            for f in epi_reg_files:
                f.unlink()
        
        return subject
    except Exception as e:
        print(f"Error processing {subject}: {e}")
        return None


def process_subject_bold(paths: BoldProcessingPaths):
    """Process a single subject's BOLD data using predefined file paths.
    
    Args:
        paths (BoldProcessingPaths): Paths required for BOLD processing.
    
    Returns:
        str: Subject ID if processing was successful, None otherwise.
    """
    subject = paths.subject
    print(f"\nProcessing BOLD for {subject}...")

    try:
        transform_t1w_to_bold(
            paths.t1w_mask,
            paths.bold_path,
            paths.out_bold_masks,
            paths.output_path_bold,
        )

        print(f"Successfully created native space BOLD mask for {subject}")

        return subject
    except Exception as e:
        print(f"Error processing {subject}: {e}")
        return None


def main():
    """Main function to transform the Schaefer/Harvard-Oxford mask to native DWI and native T1 BOLD."""
    # Set up parameters
    cohorts = ["bbhi", "bbhi senior"]
    sessions = ["1", "2"]

    sys.stdout.flush() 
    print("-----------------------Running 04_t1_to_dwi_bold.py-----------------------")

    for cohort in cohorts:
        for ses in sessions: 
            print("-------------------------")
            print(f"Processing {cohort} ses-0{ses}")
            print("-------------------------")

            # Set up FSL so it runs correctly in this script
            os.environ["FSLDIR"] = "/home/rachel/fsl"
            os.environ["PATH"] = f"{os.environ['FSLDIR']}/bin:" + os.environ["PATH"]
            subprocess.run(["bash", "-c", "source /home/rachel/fsl/etc/fslconf/fsl.sh"], check=True)

            # Set FSL to output compressed NIFTI files
            os.environ["FSLOUTPUTTYPE"] = "NIFTI_GZ"

            out_dir = Path(f"/home/rachel/Desktop/schaefer_analysis/fsaverage/ses-0{ses}")
            dwi_bbhi_dir = Path(f"/pool/guttmann/institut/BBHI/MRI/processed_data/DWI_dtifit_tp{ses}")
            dwi_bbhi_senior_dir = Path(f"/pool/guttmann/institut/UB/Superagers/MRI/DTIFIT_TP{ses}")

            subject_infos: List[SubjectPaths] = []
            dwi_inputs: dict[str, DwiProcessingPaths] = {}
            bold_inputs: dict[str, BoldProcessingPaths] = {}

            for subject_dir in sorted(os.listdir(out_dir)):
                if not subject_dir.startswith("sub-"):
                    continue

                subject_num = int(subject_dir.split("-")[1])
                if cohort == "bbhi" and subject_num < 6000:
                    continue
                if cohort == "bbhi senior" and subject_num > 6000:
                    continue

                subject = subject_dir
                ses_label = f"ses-0{ses}"
                out_subject_dir = out_dir / subject
                t1w_mask = out_subject_dir / f"subcortical_t1_masks/{subject}_{ses_label}_schaefer200_subcortical14_t1_space.nii.gz"
                t1_brain = out_subject_dir / f"t1_converted/{subject}_{ses_label}_conformed_T1_brain.nii.gz"
                dwi_mask_output = out_subject_dir / f"dwi_space_masks/{subject}_{ses_label}_schaefer200_subcortical14_dwi_space.nii.gz"
                bold_mask_output = out_subject_dir / f"bold_space_masks/{subject}_{ses_label}_schaefer200_subcortical14_bold_space.nii.gz"

                if cohort == "bbhi":
                    t1_mgz_path = Path(f"/pool/guttmann/institut/BBHI/MRI/derivatives/freesurfer-reconall/{subject}_{ses_label}_run-01/mri/T1.mgz")
                    dwi_root_dir = dwi_bbhi_dir / subject
                    if ses == "2":
                        bold_path = Path(f"/pool/guttmann/institut/BBHI/MRI/processed_data/fMRI-preprocessed_tp{ses}/{subject}/native_T1/{subject}_{ses_label}_run-01_rest_bold_ap_T1-space.nii.gz")
                    else:
                        bold_path = Path(f"/pool/guttmann/institut/BBHI/MRI/processed_data/fMRI-preprocessed/{subject}/native_T1/{subject}_{ses_label}_run-01_rest_bold_ap_T1-space.nii.gz")
                else:
                    t1_mgz_path = Path(f"/pool/guttmann/institut/UB/Superagers/MRI/derivatives/freesurfer-reconall/{subject}_{ses_label}/mri/T1.mgz")
                    if ses == "1":
                        dwi_root_dir = dwi_bbhi_senior_dir / subject
                    else:
                        dwi_root_dir = dwi_bbhi_senior_dir / f"{subject}_ses-0{ses}"
                    bold_path = Path(f"/pool/guttmann/institut/UB/Superagers/MRI/resting_preprocessed/{subject}/{ses_label}/native_T1/{subject}_{ses_label}_run-01_rest_bold_ap_T1-space.nii.gz")

                eddy_corrected = dwi_root_dir / "eddy_corrected_data.nii.gz"
                out_b0_dir = out_subject_dir / "b0"
                out_native_masks = out_subject_dir / "dwi_space_masks"
                b0_output = out_b0_dir / f"{subject}_{ses_label}_b0.nii.gz"
                out_bold_masks = out_subject_dir / "bold_space_masks"

                subject_infos.append(
                    SubjectPaths(
                        subject=subject,
                        ses_label=ses_label,
                        schaefer_subcort_atlas=t1w_mask,
                        bold_data=bold_path,
                        eddy_corrected=eddy_corrected,
                        dwi_mask_output=dwi_mask_output,
                        bold_mask_output=bold_mask_output,
                    )
                )

                dwi_inputs[subject] = DwiProcessingPaths(
                    subject=subject,
                    ses_label=ses_label,
                    t1_mgz_path=t1_mgz_path,
                    t1w_mask=t1w_mask,
                    t1_brain_path=t1_brain,
                    eddy_corrected=eddy_corrected,
                    b0_output=b0_output,
                    out_b0_dir=out_b0_dir,
                    out_native_masks=out_native_masks,
                    output_path_dwi=dwi_mask_output,
                    out_subject_dir=out_subject_dir,
                )

                bold_inputs[subject] = BoldProcessingPaths(
                    subject=subject,
                    ses_label=ses_label,
                    t1_mgz_path=t1_mgz_path,
                    t1w_mask=t1w_mask,
                    bold_path=bold_path,
                    out_bold_masks=out_bold_masks,
                    output_path_bold=bold_mask_output,
                    out_subject_dir=out_subject_dir,
                )

            subjects_for_dwi, subjects_for_bold, already_processed = get_subjects_to_process(subject_infos)

            # Uncomment the following line to process specific subjects
            # subjects_for_dwi = ["sub-3020", "sub-159530", "sub-1171"] 
            # subjects_for_bold = ["sub-3020", "sub-159530", "sub-1171"] 
            
            # Process each subject
            result_dwi = []
            result_bold = []

            print(f"Subjects needing DWI processing: {len(subjects_for_dwi)}")
            print(f"Subjects needing BOLD processing: {len(subjects_for_bold)}")
            print(f"Already processed subjects: {len(already_processed)}")

            # Check if there are subjects to process
            if not subjects_for_dwi and not subjects_for_bold:
                print("No subjects found that need processing.")
                continue

            dwi_tasks = []
            missing_dwi = []
            for subject in subjects_for_dwi:
                subject_paths = dwi_inputs.get(subject)
                if not subject_paths:
                    print(f"Missing DWI path configuration for {subject}, skipping.")
                    missing_dwi.append(subject)
                    continue
                dwi_tasks.append((subject, subject_paths))

            result_dwi, failed_dwi = run_parallel_tasks(
                dwi_tasks, process_subject_dwi, "DWI"
            )
            failed_dwi.extend(missing_dwi)

            bold_tasks = []
            missing_bold = []
            for subject in subjects_for_bold:
                subject_paths = bold_inputs.get(subject)
                if not subject_paths:
                    print(f"Missing BOLD path configuration for {subject}, skipping.")
                    missing_bold.append(subject)
                    continue
                bold_tasks.append((subject, subject_paths))

            result_bold, failed_bold = run_parallel_tasks(
                bold_tasks, process_subject_bold, "BOLD"
            )
            failed_bold.extend(missing_bold)
                    
            print(f"Successfully processed DWI for {len(result_dwi)} subjects")
            print(f"Successfully processed BOLD for {len(result_bold)} subjects")
            print(f"Total subjects with successful processing: {len(set(result_dwi + result_bold))}")

            print(f"Failed to process DWI for {len(failed_dwi)} subjects")
            print(f"Failed to process BOLD for {len(failed_bold)} subjects")

            if failed_dwi:
                print(f"Failed DWI subjects: {failed_dwi}")
            if failed_bold:
                print(f"Failed BOLD subjects: {failed_bold}")

    # Return both result lists
    return result_dwi, result_bold

if __name__ == "__main__":
    main()
