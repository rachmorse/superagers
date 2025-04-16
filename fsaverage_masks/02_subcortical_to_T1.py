import os
import subprocess

# Print environment variables
print(os.environ.get('ASEG_FILE', ''))
print(os.environ.get('REFERENCE_FILE', ''))
print(os.environ.get('OUTPUT_FOLDER', ''))

# Get environment variables
ASEG_FILE = os.environ.get('ASEG_FILE')
REFERENCE_FILE = os.environ.get('REFERENCE_FILE')
OUTPUT_FOLDER = os.environ.get('OUTPUT_FOLDER')

# Commented if condition
#if os.path.isfile(REFERENCE_FILE):
#LEFT

#17 L_Hippocampus -> 1
#18 L_Amygdala -> 2
#13 L_Pallidum -> 3
#12 L_Putamen -> 4
#11 L_Caudate -> 5
#26 L_Accumbens -> 6
#10 L_Thalamus -> 7

left_subcortical_labels = "17 18 13 12 11 26 10"

i = 1
for lab in left_subcortical_labels.split():
    lab = int(lab)
    # Execute the commands using subprocess
    subprocess.run(['mri_binarize', '--i', ASEG_FILE, '--match', str(lab), '--o', f'{OUTPUT_FOLDER}/tmp.mgz'])
    print("1")
    #subprocess.run(['mri_vol2vol', '--mov', f'{OUTPUT_FOLDER}/tmp.mgz', '--regheader', '--targ', REFERENCE_FILE, '--o', f'{OUTPUT_FOLDER}/tmp_sbref.mgz', '--interp', 'nearest'])
    print("2")
    subprocess.run(['mri_convert', '--in_type', 'mgz', '--out_type', 'nii', f'{OUTPUT_FOLDER}/tmp.mgz', f'{OUTPUT_FOLDER}/tmp.nii.gz'])
    print("3")
    #subprocess.run(['fslmaths', f'{OUTPUT_FOLDER}/tmp.nii.gz', '-bin', f'{OUTPUT_FOLDER}/tmp_binmask.nii.gz'])
    print("4")
    # Equivalent to ${left_subcortical_labels%% *} - first element of the space-separated string
    first_lab = int(left_subcortical_labels.split()[0])
    if lab == first_lab:
        subprocess.run(['fslmaths', f'{OUTPUT_FOLDER}/tmp.nii.gz', '-mul', str(i), f'{OUTPUT_FOLDER}/left_subcortical14_T1.nii.gz'])
        print("5")
    else:
        subprocess.run(['fslmaths', f'{OUTPUT_FOLDER}/tmp.nii.gz', '-mul', str(i), f'{OUTPUT_FOLDER}/tmp.nii.gz'])
        print("5")
        subprocess.run(['fslmaths', f'{OUTPUT_FOLDER}/left_subcortical14_T1.nii.gz', '-add', f'{OUTPUT_FOLDER}/tmp.nii.gz', f'{OUTPUT_FOLDER}/left_subcortical14_T1.nii.gz'])
        print("6")
    i += 1

#RIGTH

#53 R_Hippocampus -> 1
#54 R_Amygdala -> 2
#52 R_Pallidum -> 3
#51 R_Putamen -> 4
#50 R_Caudate -> 5
#58 R_Accumbens -> 6
#49 R_Thalamus -> 7

right_subcortical_labels = "53 54 52 51 50 58 49"

i = 1
for lab in right_subcortical_labels.split():
    lab = int(lab)
    subprocess.run(['mri_binarize', '--i', ASEG_FILE, '--match', str(lab), '--o', f'{OUTPUT_FOLDER}/tmp.mgz'])
    print("1")
    #subprocess.run(['mri_vol2vol', '--mov', f'{OUTPUT_FOLDER}/tmp.mgz', '--regheader', '--targ', REFERENCE_FILE, '--o', f'{OUTPUT_FOLDER}/tmp_sbref.mgz', '--interp', 'nearest'])
    print("2")
    subprocess.run(['mri_convert', '--in_type', 'mgz', '--out_type', 'nii', f'{OUTPUT_FOLDER}/tmp.mgz', f'{OUTPUT_FOLDER}/tmp.nii.gz'])
    print("3")
    #subprocess.run(['fslmaths', f'{OUTPUT_FOLDER}/tmp.nii.gz', '-bin', f'{OUTPUT_FOLDER}/tmp_binmask.nii.gz'])
    print("4")
    # Equivalent to ${right_subcortical_labels%% *} - first element of the space-separated string
    first_lab = int(right_subcortical_labels.split()[0])
    if lab == first_lab:
        subprocess.run(['fslmaths', f'{OUTPUT_FOLDER}/tmp.nii.gz', '-mul', str(i), f'{OUTPUT_FOLDER}/right_subcortical14_T1.nii.gz'])
        print("5")
    else:
        subprocess.run(['fslmaths', f'{OUTPUT_FOLDER}/tmp.nii.gz', '-mul', str(i), f'{OUTPUT_FOLDER}/tmp.nii.gz'])
        print("5")
        subprocess.run(['fslmaths', f'{OUTPUT_FOLDER}/right_subcortical14_T1.nii.gz', '-add', f'{OUTPUT_FOLDER}/tmp.nii.gz', f'{OUTPUT_FOLDER}/right_subcortical14_T1.nii.gz'])
        print("6")
    i += 1

# Cleanup files
#subprocess.run(['rm', f'{OUTPUT_FOLDER}/tmp_binmask.nii.gz'])
subprocess.run(['rm', f'{OUTPUT_FOLDER}/tmp.nii.gz'])
subprocess.run(['rm', f'{OUTPUT_FOLDER}/tmp.mgz'])
#fi