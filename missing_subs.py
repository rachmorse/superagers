import pandas as pd
import os
from pathlib import Path
import re

def get_expected_dwi_files(subject_id, ses):
    """
    Returns a list of DWI file paths expected for the subject_id for a given session.
    Used to check these for subjects who are missing structural data but do have functional data.

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
                f"/pool/guttmann/institut/BBHI/MRI/processed_data/DWI_dtifit_tp1/{subject_id}/eddy_corrected_data.nii.gz",
                f"/pool/guttmann/institut/BBHI/MRI/processed_data/tracto_MSMTCSD_TP1/{subject_id}_dwi_tractogram_1M_SIFT.tck"
            ] 
        else:
            files = [
                f"/pool/guttmann/institut/BBHI/MRI/processed_data/DWI_dtifit_tp2/{subject_id}/eddy_corrected_data.nii.gz",
                f"/pool/guttmann/institut/BBHI/MRI/processed_data/tracto_MSMTCSD_TP2/{subject_id}_dwi_tractogram_1M_SIFT.tck"
            ]
    else:
        if ses == 1:
            files = [
                f"/pool/guttmann/institut/UB/Superagers/MRI/DTIFIT_TP1/{subject_id}/eddy_corrected_data.nii.gz",
                f"/pool/guttmann/institut/UB/Superagers/MRI/tracto_MSMTCSD_TP1/{subject_id}_dwi_tractogram_1M_SIFT.tck"
            ]
        else:
            files = [
                f"/pool/guttmann/institut/UB/Superagers/MRI/DTIFIT_TP2/{subject_id}_ses-02/eddy_corrected_data.nii.gz",
                f"/pool/guttmann/institut/UB/Superagers/MRI/tracto_MSMTCSD_TP2/{subject_id}_dwi_tractogram_1M_SIFT.tck"
            ]
    return files

def gather_dwi_checks(sub_ids, ses):
    """
    For each subject, check the expected DWI files for a given session.
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

def summarize_dwi_results(results_dict, ses_number):
    """
    Summarizes the results of DWI checks for a given session.

    Args:
        results_dict (dict): Dictionary with subject IDs as keys and file existence as values.
        ses_number (int): The session number (1 or 2).
    """
    found_tract_ids = [sub for sub, r in results_dict.items() if r['tract']]
    missing_eddy = [sub for sub, r in results_dict.items() if not r['eddy']]
    missing_tract = [sub for sub, r in results_dict.items() if not r['tract']]

    # Check subs with poor registration
    poor_reg = []
    for subject in found_tract_ids:
        if check_poor_registration(subject):
            poor_reg.append(subject)

    # Get list of subs who dont fit any category
    odd_cases = [sub for sub in found_tract_ids if sub not in poor_reg]

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
        print(f"Subjects dropped because no dwi_tractogram_1M_SIFT.tck: {len(missing_tract)} subjects")
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

    if odd_cases:
        print(f"Subjects who were not dropped due to any of these: {len(odd_cases)} subjects")
        print(', '.join(odd_cases))
        print("-----------------------------")

def get_expected_fmri_files(subject_id, ses):
    """
    Returns a list of fMRI file paths expected for subject_id for a given session.
    Used to check the files for subjects who are missing func data but do have structural data.

    Args:
        subject_id (str): The subject ID in the format 'sub-xxx'.
        ses (int): The session number (1 or 2).
    """
    try:
        numeric_id = int(subject_id.split('-')[1])
    except (IndexError, ValueError):
        return []

    if numeric_id > 5000:
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
    """
    For each subject in sub_ids, check the expected fMRI files for a given session.
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

def check_scrubbing_bbhi_senior(subject):
    """Check if a BBHI senior subject was excluded due to scrubbing 
    based on nohup scrubbing logs.
    
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
    
    # Search for poor registration warning in the log file
    try:
        with open(log_file, 'r') as f:
            log_content = f.read()
            if f"Excluding {subject} due to excessive motion" in log_content:
                return True
    except Exception as e:
        print(f"Error reading log file {log_file}: {e}")
    
    return False

def check_scrubbing_bbhi(subject, ses):
    """Check if a BBHI subject was excluded due to scrubbing 
    based on all_fwd file.

    Args:
        subject (str): Subject ID in the format "sub-XXXX".
        ses (str): Session (e.g., "01" or "02").

    """
    if ses == 1:
        fwd_csv = Path("/pool/guttmann/institut/BBHI/MRI/processed_data/fMRI-preprocessed/all_fwd.csv")
    else:
        fwd_csv = Path("/pool/guttmann/institut/BBHI/MRI/processed_data/fMRI-preprocessed_tp2/all_fwd.csv")
    
    # Make sure the file exists
    if not fwd_csv.exists():
        print(f"Warning: FWD CSV file not found at {fwd_csv}")
        return False
    
    try:
        # Load the CSV
        df = pd.read_csv(fwd_csv)
        subject_col = df.columns[0]
        
        # Filter rows matching the subject ID at hand
        subject_rows = df[df[subject_col] == subject]
        if subject_rows.empty:
            return False
        
        subject_row = subject_rows.iloc[0]
        fwd_values = subject_row.iloc[1:].astype(float)  # Exclude the first column (the subject ID) from the FWD values
        fraction_exceed_05 = (fwd_values > 0.5).mean()  # Check what fraction of columns exceed 0.5
        
        # Check who is excluded based on 30% threshold
        if fraction_exceed_05 > 0.3:
            return True
        
    except Exception as e:
        print(f"Error processing FWD CSV for {subject}: {e}")
    
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
            pattern = re.compile(rf"Image Exception : #22.*{subject}")
            match = pattern.search(log_content)
            if match:
                return True
    except Exception as e:
        print(f"Error reading log file {log_file}: {e}")
    
    return False   

def summarize_fmri_results(results_dict, ses_number):
    """
    Summarizes the results of fMRI checks for a given session.

    Args:
        results_dict (dict): Dictionary with subject IDs as keys and file existence as values.
        ses_number (int): The session number (1 or 2).
    """
    found_ids = [sub for sub, r in results_dict.items() if r['bold']]
    missing_bold = [sub for sub, r in results_dict.items() if not r['bold']]
    missing_mask_prelim = [sub for sub, r in results_dict.items() if not r['mask']]

    # Check if the subs missing the mask are not missing T1 (e.g., they should have the mask)
    missing_mask = [sub for sub in missing_mask_prelim if sub not in missing_bold]

    # Check subs with poor registration
    scrub_exc = []
    for subject in found_ids:
        if check_scrubbing_bbhi_senior(subject):
            scrub_exc.append(subject)
    
    scrub_exc_bbhi = []
    for subject in found_ids:
        if check_scrubbing_bbhi(subject, ses_number):
            scrub_exc_bbhi.append(subject)
    
    trunc_bold = []
    for subject in missing_mask:
        if check_truncated_file(subject):
            trunc_bold.append(subject)

    # Get a list of all subjects dropped due to scrubbing
    scrub_exc_all = set(scrub_exc) | set(scrub_exc_bbhi)  

    # Get list of subs who dont fit any category
    odd_cases = [
        sub
        for sub in found_ids
        if sub not in scrub_exc_all
        and sub not in missing_bold
        and sub not in missing_mask
    ]

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

    if scrub_exc_all:
        print(f"Subjects dropped due to excessive movement (scrubbing): {len(scrub_exc_all)} subjects")
        print(', '.join(scrub_exc_all))
        print("-----------------------------")
    else:
        print("No subjects dropped due to scrubbing.")
        print("-----------------------------")

    if odd_cases:
        print(f"Subjects who were not dropped due to any of these: {len(odd_cases)} subjects")
        print(', '.join(odd_cases))
        print("-----------------------------")

def main():
    csv_path = "/home/rachel/Desktop/data/clean_data_all.csv"
    df = pd.read_csv(csv_path)

    # Get initial sub lists
    struct_not_func_tp1 = df.loc[(~df['struct_all_1'].isna()) & (df['func_all_1'].isna()), 'id'].tolist()
    struct_not_func_tp2 = df.loc[(~df['struct_all_2'].isna()) & (df['func_all_2'].isna()), 'id'].tolist()
    func_not_struct_tp1 = df.loc[(~df['func_all_1'].isna()) & (df['struct_all_1'].isna()), 'id'].tolist()
    func_not_struct_tp2 = df.loc[(~df['func_all_2'].isna()) & (df['struct_all_2'].isna()), 'id'].tolist()

    # Run functions
    dwi_results_ses1 = gather_dwi_checks(func_not_struct_tp1, 1)
    summarize_dwi_results(dwi_results_ses1, 1)

    dwi_results_ses2 = gather_dwi_checks(func_not_struct_tp2, 2)
    summarize_dwi_results(dwi_results_ses2, 2)

    fmri_results_ses1_struct_missing = gather_fmri_checks(struct_not_func_tp1, 1)
    summarize_fmri_results(fmri_results_ses1_struct_missing, 1)

    fmri_results_ses2_struct_missing = gather_fmri_checks(struct_not_func_tp2, 2)
    summarize_fmri_results(fmri_results_ses2_struct_missing, 2)

    n = df['sfc_all_slopes'].notna().sum() 
    print(f"Number of subjects with all data: {n}")

if __name__ == "__main__":
    main()


    