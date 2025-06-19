import pandas as pd
import os

def main():
    """
    This script checks two main scenarios, each at TP1 (time point 1) and TP2 (time point 2):

    1) func_not_struct_tpX (subjects who have functional data but are missing structural data):
       - We verify:
         (A) DWI existence (eddy_corrected_data.nii.gz + tractogram).
         (B) fMRI-preprocessed files (just to confirm their actual presence on disk, if desired).

    2) struct_not_func_tpX (subjects who have structural data but are missing functional data):
       - We verify the fMRI-preprocessed files.

    The script prints summaries for each group at each time point.
    """

    # ---------------------------------------------------------------------------
    # LOAD CSV
    # ---------------------------------------------------------------------------
    csv_path = "/home/rachel/Desktop/data/clean_data_all.csv"
    df = pd.read_csv(csv_path)

    # ---------------------------------------------------------------------------
    # IDENTIFY SUBJECT GROUPS
    # ---------------------------------------------------------------------------
    # "struct_not_func" means struct present, func missing
    struct_not_func_tp1 = df.loc[(~df['struct_all_1'].isna()) & (df['func_all_1'].isna()), 'id'].tolist()
    struct_not_func_tp2 = df.loc[(~df['struct_all_2'].isna()) & (df['func_all_2'].isna()), 'id'].tolist()

    # "func_not_struct" means func present, struct missing
    func_not_struct_tp1 = df.loc[(~df['func_all_1'].isna()) & (df['struct_all_1'].isna()), 'id'].tolist()
    func_not_struct_tp2 = df.loc[(~df['func_all_2'].isna()) & (df['struct_all_2'].isna()), 'id'].tolist()

    print("IDs with struct_all_1 but not func_all_1:", struct_not_func_tp1)
    print("IDs with func_all_1 but not struct_all_1:", func_not_struct_tp1)
    print("IDs with struct_all_2 but not func_all_2:", struct_not_func_tp2)
    print("IDs with func_all_2 but not struct_all_2:", func_not_struct_tp2)

    # ---------------------------------------------------------------------------
    # PART A: CHECK DWI FOR SUBJECTS WHO ARE func_not_struct (MISSING STRUCT)
    # ---------------------------------------------------------------------------
    
    def get_expected_dwi_files(subject_id, ses):
        """
        Returns a list of DWI file paths expected for the subject_id at session ses.
        We check these for subjects who are missing structural data but do have functional data.
        """
        try:
            numeric_id = int(subject_id.split('-')[1])
        except (IndexError, ValueError):
            return []

        if numeric_id > 5000:
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
            # numeric_id <= 5000
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
        For each subject in sub_ids, check the expected DWI files for session ses.
        Return a dict: subject -> { 'eddy': bool, 'tract': bool } indicating file existence.
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

    def summarize_dwi_results(results_dict, ses_number):
        found_eddy_ids = [sub for sub, r in results_dict.items() if r['eddy']]
        found_tract_ids = [sub for sub, r in results_dict.items() if r['tract']]
        missing_both_ids = [sub for sub, r in results_dict.items() if (not r['eddy']) and (not r['tract'])]
        missing_eddy = [sub for sub, r in results_dict.items() if not r['eddy']]
        missing_tract = [sub for sub, r in results_dict.items() if not r['tract']]

        print(f"\n=== DWI Check Summary for Session {ses_number} (func_not_struct) ===")
        if found_eddy_ids:
            print(f"File eddy_corrected_data.nii.gz found for: {', '.join(found_eddy_ids)}")
        else:
            print("File eddy_corrected_data.nii.gz found for: None")

        if found_tract_ids:
            print(f"File <sub-xxx>_dwi_tractogram_1M_SIFT.tck found for: {', '.join(found_tract_ids)}")
        else:
            print("File <sub-xxx>_dwi_tractogram_1M_SIFT.tck found for: None")

        if missing_both_ids:
            print(f"Subjects missing both DWI files: {', '.join(missing_both_ids)}")
        else:
            print("No subject is missing both DWI files.")

        if missing_eddy:
            print(f"Subjects missing eddy_corrected_data.nii.gz: {', '.join(missing_eddy)}")
        else:
            print("No subjects missing eddy_corrected_data.nii.gz")

        if missing_tract:
            print(f"Subjects missing <sub-xxx>_dwi_tractogram_1M_SIFT.tck: {', '.join(missing_tract)}")
        else:
            print("No subjects missing <sub-xxx>_dwi_tractogram_1M_SIFT.tck")

    # Run DWI checks for func_not_struct
    dwi_results_ses1 = gather_dwi_checks(func_not_struct_tp1, 1)
    summarize_dwi_results(dwi_results_ses1, 1)

    dwi_results_ses2 = gather_dwi_checks(func_not_struct_tp2, 2)
    summarize_dwi_results(dwi_results_ses2, 2)

    # ---------------------------------------------------------------------------
    # PART B: CHECK fMRI-PREPROCESSED FILES FOR struct_not_func (MISSING FUNC)
    # ---------------------------------------------------------------------------

    def get_expected_fmri_files(subject_id, ses):
        """
        Returns a list of fMRI file paths expected for subject_id at session ses.
        We check these for subjects who are missing func data but do have structural data.
        """
        try:
            numeric_id = int(subject_id.split('-')[1])
        except (IndexError, ValueError):
            return []

        if numeric_id > 5000:
            if ses == 1:
                files = [f"/pool/guttmann/institut/BBHI/MRI/processed_data/fMRI-preprocessed/{subject_id}/native_T1/{subject_id}_ses-01_run-01_rest_bold_ap_T1-space.nii.gz"]
            else:
                files = [f"/pool/guttmann/institut/BBHI/MRI/processed_data/fMRI-preprocessed_tp2/{subject_id}/native_T1/{subject_id}_ses-02_run-01_rest_bold_ap_T1-space.nii.gz"]
        else:
            if ses == 1:
                files = [f"/pool/guttmann/institut/UB/Superagers/MRI/resting_preprocessed/{subject_id}/ses-01/native_T1/{subject_id}_ses-01_run-01_rest_bold_ap_T1-space.nii.gz"]
            else:
                files = [f"/pool/guttmann/institut/UB/Superagers/MRI/resting_preprocessed/{subject_id}/ses-02/native_T1/{subject_id}_ses-02_run-01_rest_bold_ap_T1-space.nii.gz"]
        return files

    def gather_fmri_checks(sub_ids, ses):
        """
        For each subject in sub_ids, check the expected fMRI files for session = ses.
        Return a dict: subject -> {'fmri': bool} indicating file existence.
        """
        results = {}
        for subject_id in sub_ids:
            fmri_files = get_expected_fmri_files(subject_id, ses)
            if not fmri_files:
                continue

            # We'll consider the subject as 'fmri': True only if ALL expected files exist
            all_found = all(os.path.exists(fpath) for fpath in fmri_files)
            results[subject_id] = {'fmri': all_found}
        return results

    def summarize_fmri_results(results_dict, ses_number, label="struct_not_func"):
        found_ids = [sub for sub, r in results_dict.items() if r['fmri']]
        missing_ids = [sub for sub, r in results_dict.items() if not r['fmri']]

        print(f"\n=== fMRI Check Summary for Session {ses_number} ({label}) ===")
        if found_ids:
            print(f"Subjects with fMRI file found: {', '.join(found_ids)}")
        else:
            print("No subject with fMRI file found.")

        if missing_ids:
            print(f"Subjects missing fMRI file(s): {', '.join(missing_ids)}")
        else:
            print("No subject missing fMRI file(s).")

    fmri_results_ses1_struct_missing = gather_fmri_checks(struct_not_func_tp1, 1)
    summarize_fmri_results(fmri_results_ses1_struct_missing, 1, "struct_not_func")

    fmri_results_ses2_struct_missing = gather_fmri_checks(struct_not_func_tp2, 2)
    summarize_fmri_results(fmri_results_ses2_struct_missing, 2, "struct_not_func")


if __name__ == "__main__":
    main()


    