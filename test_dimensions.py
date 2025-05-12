import nibabel as nib
from pathlib import Path

def check_image_compatibility(func_path, label_path):
    func_img = nib.load(str(func_path))
    label_img = nib.load(str(label_path))
    
    print(f"Functional image shape: {func_img.shape}")
    print(f"Functional image affine:\n{func_img.affine}")
    print(f"Label image shape: {label_img.shape}")
    print(f"Label image affine:\n{label_img.affine}")
    print("="*40)

ses = "02"
timepoint = "2"
threshold = "0.5"
cohort = "bbhi senior"   
root = Path("/home/rachel/Desktop/schaefer_analysis") 

subjects = ["sub-1144", "sub-4005", "sub-4078", "sub-3085", "sub-4101", "sub-4013", "sub-4062"]

if cohort == "bbhi":
    if timepoint == "1":
        base_dir = Path(f"/pool/guttmann/institut/BBHI/MRI/processed_data/fMRI-preprocessed")
    else:
        base_dir = Path(f"/pool/guttmann/institut/BBHI/MRI/processed_data/fMRI-preprocessed_tp2")
else:
    base_dir = Path("/pool/guttmann/institut/UB/Superagers/MRI/resting_preprocessed")

for subject in subjects:
    label_file = Path(f"{root}/fsaverage/ses-{ses}/{subject}/bold_space_masks/{subject}_ses-{ses}_schaefer200_subcortical14_bold_space.nii.gz")
    if cohort == "bbhi":
        func_file = Path(f"{base_dir}/{subject}/native_T1/{subject}_ses-{ses}_run-01_rest_bold_ap_T1-space_scrubbed-interp-05.nii.gz")
        # This is because subjects who did not need any scrubbing do not have a separate scrubbed file
        if not func_file.exists():
            func_file = Path(f"{base_dir}/{subject}/native_T1/{subject}_ses-{ses}_run-01_rest_bold_ap_T1-space.nii.gz")
    else:
        if ses == "01":
            func_file = Path(f"{base_dir}/{subject}/ses-{ses}/native_T1/{subject}_ses-{ses}_run-01_rest_bold_ap_T1-space_scrubbed-interp-05.nii.gz")
            if not func_file.exists():
                func_file = Path(f"{base_dir}/{subject}/ses-{ses}/native_T1/{subject}_ses-{ses}_run-01_rest_bold_ap_T1-space.nii.gz")
        else:
            func_file = Path(f"{base_dir}/{subject}/ses-{ses}/native_T1/{subject}_ses-{ses}_run-01_rest_bold_ap_T1-space_scrubbed_0.5.nii.gz")
            if not func_file.exists():
                func_file = Path(f"{base_dir}/{subject}/ses-{ses}/native_T1/{subject}_ses-{ses}_run-01_rest_bold_ap_T1-space.nii.gz")

    print(f"Subject: {subject}")
    check_image_compatibility(func_file, label_file)