# Root directory containing all subject data for the Glasser374 atlas processing
ROOT="/home/mariacabello/working_dir/multimodal/atlases/glasser374/all_subjects_senior"
import os
import nibabel as nib
import numpy as np
# Get list of all subject directories
subjs=os.listdir(ROOT)
# Filter for subjects that don't have the final glasser374_T1.nii.gz file yet
todo=[s for s in subjs if not os.path.exists(ROOT+"/"+s+"/ses-01/glasser374_T1.nii.gz")]
#T1
#s="sub-100197"
for s in todo:
    # Check if the right subcortical data exists
    if os.path.exists(ROOT+"/"+s+"/ses-01/right_subcortical14_T1.nii.gz"):# and not os.path.exists(ROOT+"/"+s+"/ses-01/glasser374_T1.nii.gz"):
        print(s)
        id_user=s
        # Load the individual brain atlas components
        lc=nib.load(ROOT+"/"+s+"/ses-01/glasser_volumetric_T1_lh.nii.gz").get_fdata()
        rc=nib.load(ROOT+"/"+s+"/ses-01/glasser_volumetric_T1_rh.nii.gz").get_fdata()
        ls=nib.load(ROOT+"/"+s+"/ses-01/left_subcortical14_T1.nii.gz").get_fdata()
        rs=nib.load(ROOT+"/"+s+"/ses-01/right_subcortical14_T1.nii.gz").get_fdata()
        
        # Adjust values to create non-overlapping ranges for each region
        lc_adjusted = np.where(lc != 0, lc , 0)       # lc: values from 1 to 180 (keeping 0 as 0)
        rc_adjusted = np.where(rc != 0, rc + 180, 0)     # rc: values from 181 to 360 (keeping 0 as 0)
        ls_adjusted = np.where(ls != 0, ls + 360, 0)     # ls: values from 361 to 367 (keeping 0 as 0)
        rs_adjusted = np.where(rs != 0, rs + 367, 0)     # rs: values from 368 to 374 (keeping 0 as 0)
        
        
        # Get the affine transformation matrix from the original image
        affine=nib.load(ROOT+"/"+s+"/ses-01/glasser_volumetric_T1_lh.nii.gz").affine
        
        # Create a matrix to store the final result
        resultado = np.zeros_like(lc)
        
        # Create a stacked matrix for comparison
        stacked = np.stack([lc_adjusted, rc_adjusted, ls_adjusted, rs_adjusted], axis=-1)
        
        # Identify positions with overlap (more than one non-zero value)
        overlap_mask = np.sum(stacked != 0, axis=-1) > 1
        
        # Extract overlap values
        overlap_values = stacked[overlap_mask]
        overlap_values = np.unique(overlap_values,axis=0,return_counts=True)
        
        output_file = "overlap_values_percentages.txt"
        
        # Open file in append mode
        with open(output_file, "a") as file:
           
            # Iterate through each overlap instance
            for idx, voxel_values in enumerate(overlap_values[0], start=1):
                lc_value = voxel_values[0]
                rc_value = voxel_values[1]
                ls_value = voxel_values[2]
                rs_value = voxel_values[3]
                n_values_overlaped = overlap_values[1][idx-1]
                lc_perc = 0.
                rc_perc = 0.
                ls_perc = 0. 
                rs_perc = 0.
                # Calculate percentage of overlap for each region
                if lc_value != 0: lc_perc = n_values_overlaped*100/sum(sum(sum(lc_adjusted==lc_value)))
                if rc_value != 0: rc_perc = n_values_overlaped*100/sum(sum(sum(rc_adjusted==rc_value)))
                if ls_value != 0: ls_perc = n_values_overlaped*100/sum(sum(sum(ls_adjusted==ls_value)))
                if rs_value != 0: rs_perc = n_values_overlaped*100/sum(sum(sum(rs_adjusted==rs_value)))

                # Write overlap information to file
                new_line = f"{id_user}, {idx}, {n_values_overlaped}, {lc_value}, {rc_value}, {ls_value}, {rs_value}, {lc_perc}, {rc_perc}, {ls_perc}, {rs_perc}\n"
                file.write(new_line)

        # For voxels with overlap, keep the value from the area with the highest percentage of affected voxels
        for idx, voxel_values in enumerate(overlap_values[0], start=1):
            lc_value = voxel_values[0]
            rc_value = voxel_values[1]
            ls_value = voxel_values[2]
            rs_value = voxel_values[3]
            n_values_overlaped = overlap_values[1][idx-1]
            lc_perc = 0.
            rc_perc = 0.
            ls_perc = 0. 
            rs_perc = 0.
            # Calculate percentage of overlap for each region
            if lc_value != 0: lc_perc = n_values_overlaped*100/sum(sum(sum(lc_adjusted==lc_value)))
            if rc_value != 0: rc_perc = n_values_overlaped*100/sum(sum(sum(rc_adjusted==rc_value)))
            if ls_value != 0: ls_perc = n_values_overlaped*100/sum(sum(sum(ls_adjusted==ls_value)))
            if rs_value != 0: rs_perc = n_values_overlaped*100/sum(sum(sum(rs_adjusted==rs_value)))
            # Find the region with the highest overlap percentage
            max_perc_idx=np.argmax(np.array([lc_perc,rc_perc,ls_perc,rs_perc]))
            # Assign values based on the region with highest percentage
            resultado[(overlap_mask) &
                      (stacked[:,:,:,0]==lc_value) &
                      (stacked[:,:,:,1]==rc_value) &
                      (stacked[:,:,:,2]==ls_value) &
                      (stacked[:,:,:,3]==rs_value)] = np.array([lc_value,rc_value,ls_value,rs_value])[max_perc_idx]

        # For non-overlapping areas, simply sum the values (only one will be non-zero)
        resultado[~overlap_mask]=np.sum(stacked[~overlap_mask], axis=-1)
        
        
        # Check that all 374 areas are preserved
        unique_values = np.unique(resultado)
        if unique_values.shape[0] != 375:
            print(unique_values.shape[0])
        # Log incomplete Glasser atlas information
        with open("glasser_incompleto.txt","a") as f: 
            f.write(f"{s} {unique_values.shape[0]}")
        
        # Save the combined atlas
        nib.save(nib.Nifti1Image(resultado,affine),
                 ROOT+"/"+s+"/ses-01/glasser374_T1.nii.gz")