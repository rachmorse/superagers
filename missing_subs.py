import pandas as pd
import os
from pathlib import Path
import re

def get_expected_dwi_files(subject_id, ses):
    """Returns a list of DWI file paths expected for the subject_id for a given session.
    Used to check for subjects who are missing structural data but do have functional data.

    Args:
        subject_id (str): The subject ID in the format 'sub-xxx'.
        ses (int): The session number (1 or 2).
    """
    # Get a numeric version of ID to be able to separate BBHI from BBHI senior file paths
    try:
        numeric_id = int(subject_id.split('-')[1])
    except (IndexError, ValueError):
        return []

    if numeric_id > 5000: # BBHI
        files = [
            f"/pool/guttmann/institut/BBHI/MRI/processed_data/DWI_dtifit_tp{ses}/{subject_id}/eddy_corrected_data.nii.gz",
            f"/pool/guttmann/institut/BBHI/MRI/derivatives/tracto_MSMTCSD/{subject_id}/ses-0{ses}/dwi/{subject_id}_ses-0{ses}_model-MSMTCSD_tractogram.tck"
        ] 
    else:
        if ses == 1:
            files = [
                f"/pool/guttmann/institut/UB/Superagers/MRI/DTIFIT_TP{ses}/{subject_id}/eddy_corrected_data.nii.gz",
                f"/pool/guttmann/institut/UB/Superagers/MRI/derivatives/tracto_MSMTCSD/{subject_id}/ses-0{ses}/dwi/{subject_id}_ses-0{ses}_model-MSMTCSD_tractogram.tck"
            ]
        else:
            files = [
                f"/pool/guttmann/institut/UB/Superagers/MRI/DTIFIT_TP{ses}/{subject_id}_ses-0{ses}/eddy_corrected_data.nii.gz",
                f"/pool/guttmann/institut/UB/Superagers/MRI/derivatives/tracto_MSMTCSD/{subject_id}/ses-0{ses}/dwi/{subject_id}_ses-0{ses}_model-MSMTCSD_tractogram.tck"
            ]

    return files


def gather_dwi_checks(sub_ids, ses):
    """For each subject, check the expected DWI files for a given session.
    Returns a dict: subject -> { 'eddy': bool, 'tract': bool } indicating file existence.

    Args:
        sub_ids (list): List of subject IDs in the format 'sub-xxx'.
        ses (int): The session number (1 or 2).
    """
    results = {}
    for subject_id in sub_ids:
        dwi_files = get_expected_dwi_files(subject_id, ses)
        if len(dwi_files) < 2:
            continue

        eddy_found = os.path.exists(dwi_files[0])
        tract_found = os.path.exists(dwi_files[1])
        results[subject_id] = {'eddy': eddy_found, 'tract': tract_found}
    return results


def check_poor_registration(subject):
    """Check if a subject has poor registration based on structural analysis logs.
    
    Args:
        subject (str): Subject ID
        
    Returns:
        bool: True if poor registration was detected, False otherwise
    """  
    log_file = Path(f"/home/rachel/Desktop/superagers/structural_analysis/nohup_struct.out")
    
    # Check if log file exists
    if not log_file.exists():
        print(f"Warning: Log file {log_file} not found")
        return False
    
    # Search for poor registration warning in the log file
    try:
        with open(log_file, 'r') as f:
            log_content = f.read()
            if f"WARNING: Poor registration detected for {subject}" in log_content:
                return True
    except Exception as e:
        print(f"Error reading log file {log_file}: {e}")
    
    return False


def summarize_dwi_results(ses_number, all_ids):
    """Summarizes the results of DWI checks for a given session.

    Args:
        ses_number (int): The session number (1 or 2).
        all_ids (list): List of all subject IDs.
    """
    dwi_status = gather_dwi_checks(all_ids, ses_number)
    found_tract_ids = [sub for sub, res in dwi_status.items() if res['tract']]
    missing_eddy = [sub for sub, res in dwi_status.items() if not res['eddy']]
    missing_tract = [sub for sub, res in dwi_status.items() if res['eddy'] and not res['tract']]

    # Check subs with poor registration
    poor_reg = []
    for subject in found_tract_ids:
        if check_poor_registration(subject):
            poor_reg.append(subject)

    print(f"\n=== DWI Check Summary for Timepoint {ses_number} (subs with functional but not structural data) ===")
    if missing_eddy:
        print(f"----------------------------")
        print(f"Subjects dropped because no eddy_corrected_data.nii.gz: {len(missing_eddy)} subjects")
        print(', '.join(missing_eddy))
        print("-----------------------------")
    else:
        print("No subjects missing eddy_corrected_data.nii.gz")
        print("-----------------------------")

    if missing_tract:
        print(f"Subjects dropped because no dwi_tractogram_1M_SIFT.tck (but with eddy_corrected_data.nii.gz): {len(missing_tract)} subjects")
        print(', '.join(missing_tract))
        print("-----------------------------")
    else:
        print("No subjects missing <sub-xxx>_dwi_tractogram_1M_SIFT.tck")
        print("-----------------------------")

    if poor_reg:
        print(f"Subjects dropped because poor structural registration: {len(poor_reg)} subjects")
        print(', '.join(poor_reg))
        print("-----------------------------")
    else:
        print("No subjects dropped due to poor structural registration.")
        print("-----------------------------")

    return missing_tract, poor_reg


def get_expected_fmri_files(subject_id, ses):
    """Returns a list of fMRI file paths expected for subject_id for a given session.
    Used to check the files for subjects who are missing func data but do have structural data.

    Args:
        subject_id (str): The subject ID in the format 'sub-xxx'.
        ses (int): The session number (1 or 2).
    """
    # Get a numeric version of ID to be able to separate BBHI from BBHI senior file paths
    try:
        numeric_id = int(subject_id.split('-')[1])
    except (IndexError, ValueError):
        return []

    if numeric_id > 5000: # BBHI
        if ses == 1:
            files = [
                f"/pool/guttmann/institut/BBHI/MRI/processed_data/fMRI-preprocessed/{subject_id}/native_T1/{subject_id}_ses-01_run-01_rest_bold_ap_T1-space.nii.gz",
                f"/home/rachel/Desktop/schaefer_analysis/fsaverage/ses-01/{subject_id}/bold_space_masks/{subject_id}_ses-01_schaefer200_subcortical14_bold_space.nii.gz"
            ]
        else:
            files = [
                f"/pool/guttmann/institut/BBHI/MRI/processed_data/fMRI-preprocessed_tp2/{subject_id}/native_T1/{subject_id}_ses-02_run-01_rest_bold_ap_T1-space.nii.gz",
                f"/home/rachel/Desktop/schaefer_analysis/fsaverage/ses-02/{subject_id}/bold_space_masks/{subject_id}_ses-02_schaefer200_subcortical14_bold_space.nii.gz"
            ]
    else:
        if ses == 1:
            files = [
                f"/pool/guttmann/institut/UB/Superagers/MRI/resting_preprocessed/{subject_id}/ses-01/native_T1/{subject_id}_ses-01_run-01_rest_bold_ap_T1-space.nii.gz",
                f"/home/rachel/Desktop/schaefer_analysis/fsaverage/ses-01/{subject_id}/bold_space_masks/{subject_id}_ses-01_schaefer200_subcortical14_bold_space.nii.gz"
            ]
        else:
            files = [
                f"/pool/guttmann/institut/UB/Superagers/MRI/resting_preprocessed/{subject_id}/ses-02/native_T1/{subject_id}_ses-02_run-01_rest_bold_ap_T1-space.nii.gz",
                f"/home/rachel/Desktop/schaefer_analysis/fsaverage/ses-02/{subject_id}/bold_space_masks/{subject_id}_ses-02_schaefer200_subcortical14_bold_space.nii.gz"
            ]
    return files


def gather_fmri_checks(sub_ids, ses):
    """For each subject in sub_ids, check the expected fMRI files for a given session.
    Return a dict: subject -> {'fmri': bool} indicating file existence.

    Args:
        sub_ids (list): List of subject IDs in the format 'sub-xxx'.
        ses (int): The session number (1 or 2).
    """
    results = {}
    for subject_id in sub_ids:
        fmri_files = get_expected_fmri_files(subject_id, ses)
        if len(fmri_files) < 2:
            continue

        bold_found = os.path.exists(fmri_files[0])
        mask_found = os.path.exists(fmri_files[1])

        results[subject_id] = {'bold': bold_found, 'mask': mask_found}
    return results


def check_scrubbing(subject):
    """Check if a subject was excluded due to scrubbing based 
    on nohup scrubbing logs.
    
    Args:
        subject (str): Subject ID
        
    Returns:
        bool: True if excluded due to scrubbing, False otherwise
    """   
    log_file = Path(f"/home/rachel/Desktop/superagers/fmri_analysis/nohup_timeseries.out")
    
    # Check if log file exists
    if not log_file.exists():
        print(f"Warning: Log file {log_file} not found")
        return False
    
    # Search for excessive motion warning in the log file
    try:
        with open(log_file, 'r') as f:
            log_content = f.read()
            if f"Excluding {subject} due to excessive motion" in log_content:
                return True
    except Exception as e:
        print(f"Error reading log file {log_file}: {e}")
    
    return False


def check_truncated_file(subject):
    """Check if a subject has a truncated BOLD file that caused issues with the atlas.
    
    Args:
        subject (str): Subject ID
        
    Returns:
        bool: True if truncation was detected, False otherwise
    """  
    log_file = Path(f"/home/rachel/Desktop/superagers/fsaverage_masks/nohup_fsaverage.out")
    
    # Check if log file exists
    if not log_file.exists():
        print(f"Warning: Log file {log_file} not found")
        return False
    
    # Search for truncated warning in the log file
    try:
        with open(log_file, 'r') as f:
            log_content = f.read()
            pattern = re.compile(rf"\+\+\s*WARNING:\s*nifti_read_buffer\([^)]*{subject}")
            match = pattern.search(log_content)
            if match:
                return True
    except Exception as e:
        print(f"Error reading log file {log_file}: {e}")
    
    return False   


def summarize_fmri_results(ses_number, all_ids):
    """Summarizes the results of fMRI checks for a given session.

    Args:
        ses_number (int): The session number (1 or 2).
        all_ids (list): List of all subject IDs.
    """
    fmri_status = gather_fmri_checks(all_ids, ses_number)
    found_ids = [sub for sub, res in fmri_status.items() if res['bold']]
    missing_bold = [sub for sub, res in fmri_status.items() if not res['bold']]
    missing_mask_prelim = [sub for sub, res in fmri_status.items() if not res['mask']]

    # Check if the subs missing the mask are not missing T1 (e.g., they should have the mask)
    missing_mask = [sub for sub in missing_mask_prelim if sub not in missing_bold]

    # Check subs with poor registration
    scrub_exc = []
    for subject in found_ids:
        if check_scrubbing(subject):
            scrub_exc.append(subject)
    
    trunc_bold = []
    for subject in missing_mask:
        if check_truncated_file(subject):
            trunc_bold.append(subject)

    print(f"\n=== fMRI Check Summary for Timepoint {ses_number} (subs with structural but not functional data) ===")

    if missing_bold:
        print("-----------------------------")
        print(f"Subjects dropped because no rest_bold_ap_T1-space.nii.gz: {len(missing_bold)} subjects")
        print(', '.join(missing_bold))
        if "sub-4036" in missing_bold:
            print("NOTE: sub-4036 has invalid fMRI data and was dropped.")
        print("-----------------------------")
    else:
        print("No subjects missing BOLD file(s).")
        print("-----------------------------")
    
    if missing_mask:
        print(f"Subjects dropped because no schaefer200_subcortical14_bold_space.nii.gz: {len(missing_mask)} subjects")
        print(', '.join(missing_mask))
        print(f"NOTE: These subjects had truncated BOLD files: {', '.join(trunc_bold)}")
        print("-----------------------------")

    else:
        print("No subjects missing Schaefer mask file.")
        print("-----------------------------")

    if scrub_exc:
        print(f"Subjects dropped due to excessive movement (scrubbing): {len(scrub_exc)} subjects")
        print(', '.join(scrub_exc))
        print("-----------------------------")
    else:
        print("No subjects dropped due to scrubbing.")
        print("-----------------------------")

    return scrub_exc, trunc_bold


def main():
    csv_path = "/home/rachel/Desktop/data/superager.csv"
    df = pd.read_csv(csv_path)
    # Ensure IDs are in the format 'sub-xxx'
    all_ids = []
    for i in df['id'].dropna().tolist():
        if isinstance(i, int) or (isinstance(i, str) and i.isdigit()):
             all_ids.append(f"sub-{i}")

    # Reconall dir
    bbhi_reconall_dir = Path('/pool/guttmann/institut/BBHI/MRI/derivatives/freesurfer-reconall')
    bbhi_senior_reconall_dir = Path('/pool/guttmann/institut/UB/Superagers/MRI/derivatives/freesurfer-reconall')

    # --- Session 1 ---
    print("\nProcessing Session 1...")

    # Filter all_ids to only include subjects with recon-all done
    all_ids_tp1 = []

    for sub in all_ids:
        num = int(sub.replace("sub-", ""))
        if num > 6000: 
            if (bbhi_reconall_dir / f"{sub}_ses-01_run-01").exists():
                all_ids_tp1.append(sub)
        else: 
            if (bbhi_senior_reconall_dir / f"{sub}_ses-01").exists():
                all_ids_tp1.append(sub)

    print(f"Number of subjects with recon-all done tp1: {len(all_ids_tp1)}")

    dwi_status_ses1 = gather_dwi_checks(all_ids_tp1, 1)
    fmri_status_ses1 = gather_fmri_checks(all_ids_tp1, 1)

    # Identify who has what
    has_dwi_ses1 = [sub for sub, res in dwi_status_ses1.items() if res['eddy'] and res['tract']]
    has_fmri_ses1 = [sub for sub, res in fmri_status_ses1.items() if res['bold'] and res['mask']]
    has_fmri_dwi_ses1 = [
        sub for sub in fmri_status_ses1
        if fmri_status_ses1[sub]['bold']
        and sub in dwi_status_ses1
        and dwi_status_ses1[sub]['tract']
    ]

    # Summarize
    # For func_not_struct: explain why structural is missing
    missing_tract_tp1, poor_reg_tp1 = summarize_dwi_results(1, all_ids)

    # For struct_not_func: explain why functional is missing
    scrub_exc_tp1, trunc_bold_tp1 = summarize_fmri_results(1, all_ids)

    # --- Session 2 ---
    print("\nProcessing Session 2...")

    # Filter all_ids to only include subjects with recon-all done
    all_ids_tp2 = []

    for sub in all_ids:
        num = int(sub.replace("sub-", ""))
        if num > 6000: 
            if (bbhi_reconall_dir / f"{sub}_ses-02_run-01").exists():
                all_ids_tp2.append(sub)
        else: 
            if (bbhi_senior_reconall_dir / f"{sub}_ses-02").exists():
                all_ids_tp2.append(sub)

    print(f"Number of subjects with recon-all done tp2: {len(all_ids_tp2)}")
    
    dwi_status_ses2 = gather_dwi_checks(all_ids_tp2, 2)
    fmri_status_ses2 = gather_fmri_checks(all_ids_tp2, 2)

    # Identify who has what
    has_dwi_ses2 = [sub for sub, res in dwi_status_ses2.items() if res['eddy'] and res['tract']]
    has_fmri_ses2 = [sub for sub, res in fmri_status_ses2.items() if res['bold'] and res['mask']]
    has_fmri_dwi_ses2 = [
        sub for sub in fmri_status_ses2
        if fmri_status_ses2[sub]['bold']
        and sub in dwi_status_ses2
        and dwi_status_ses2[sub]['tract']
    ]

    # Summarize
    missing_tract_tp2, poor_reg_tp2 = summarize_dwi_results(2, all_ids)
    scrub_exc_tp2, trunc_bold_tp2 = summarize_fmri_results(2, all_ids)

    # Print summary of subjects with both DWI and fMRI
    has_fmri_dwi_long = [
        sub for sub in fmri_status_ses1
        if fmri_status_ses1[sub]['bold']
        and sub in dwi_status_ses1
        and dwi_status_ses1[sub]['tract']
        and sub in dwi_status_ses2
        and dwi_status_ses2[sub]['tract']
        and sub in fmri_status_ses2
        and fmri_status_ses2[sub]['bold']
    ]

    # Calculate the intersection of scrub_exc_tp1 and scrub_exc_tp2
    scrub_exc_tp1_tp2 = list(set(scrub_exc_tp1) & set(scrub_exc_tp2))
    missing_tract_tp1_tp2 = list(set(missing_tract_tp1) & set(missing_tract_tp2))
    trunc_bold_tp1_tp2 = list(set(trunc_bold_tp1) & set(trunc_bold_tp2))
    poor_reg_tp1_tp2 = list(set(poor_reg_tp1) & set(poor_reg_tp2))

    print(f"Subjects with both DWI and fMRI at timepoint 1: {len(has_fmri_dwi_ses1)-len(scrub_exc_tp1)-len(trunc_bold_tp1)-len(poor_reg_tp1)}")
    print(f"Subjects with DWI at timepoint 1: {len(has_dwi_ses1)}")
    print(f"Subjects with fMRI at timepoint 1: {len(has_fmri_ses1)}")
    print(f"---------------")
    print(f"Subjects with both DWI and fMRI at timepoint 2: {len(has_fmri_dwi_ses2)-len(scrub_exc_tp2)-len(trunc_bold_tp2)-len(poor_reg_tp2)}")
    print(f"Subjects with DWI at timepoint 2: {len(has_dwi_ses2)}")
    print(f"Subjects with fMRI at timepoint 2: {len(has_fmri_ses2)}")
    print(f"---------------")
    print(f"Subjects with both DWI and fMRI at both timepoints: {len(has_fmri_dwi_long)-len(scrub_exc_tp1_tp2)-len(trunc_bold_tp1_tp2)-len(poor_reg_tp1_tp2)}")
    print("---------------")
    print(f"Subjects missing tracts at timepoint 1: {len(missing_tract_tp1)}")
    print(f"Subjects missing tracts at timepoint 2: {len(missing_tract_tp2)}")
    print(f"Subjects missing tracts at both timepoints: {len(missing_tract_tp1_tp2)} - {', '.join(missing_tract_tp1_tp2)}")

if __name__ == "__main__":
    main()


    