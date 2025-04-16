import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Path to your MGZ file
file_path = '/home/rachel/Desktop/schaefer_analysis/fsaverage/ses-02/sub-3087/sub-3087_ses-02_schaefer200_aparc+aseg.mgz'

# Load the MGZ file
img = nib.load(file_path)
print(f"File loaded successfully: {file_path}")

# Get the image data
data = img.get_fdata()
print(f"Data shape: {data.shape}")

# Find all unique labels in the volume
unique_labels = np.unique(data)
print(f"Total unique labels found: {len(unique_labels)}")

# Convert to integers for easier handling
unique_labels = [int(label) for label in unique_labels]

# Create ranges for expected Schaefer labels
expected_lh = list(range(1001, 1101))  # LH: 1001-1100
expected_rh = list(range(2001, 2101))  # RH: 2001-2100
expected_schaefer = expected_lh + expected_rh

# Check which expected labels are present
present_schaefer = [label for label in unique_labels if label in expected_schaefer]
missing_schaefer = [label for label in expected_schaefer if label not in unique_labels]

# Check for left and right hemisphere coverage
present_lh = [label for label in unique_labels if 1001 <= label <= 1100]
present_rh = [label for label in unique_labels if 2001 <= label <= 2100]

print("\n=== SCHAEFER ATLAS COVERAGE ===")
print(f"Expected Schaefer ROIs: 200")
print(f"Present Schaefer ROIs: {len(present_schaefer)}")
print(f"Missing Schaefer ROIs: {len(missing_schaefer)}")
print(f"Left hemisphere ROIs present: {len(present_lh)}/100")
print(f"Right hemisphere ROIs present: {len(present_rh)}/100")

# Identify subcortical structures that are present
subcortical_range = list(range(1, 100))  # Most subcortical structures are below 100
present_subcortical = [label for label in unique_labels if label in subcortical_range]

print("\n=== SUBCORTICAL STRUCTURE COVERAGE ===")
print(f"Present subcortical structures: {len(present_subcortical)}")
print(f"Subcortical labels: {sorted(present_subcortical)}")

# Check if any missing ROIs form a pattern
if missing_schaefer:
    print("\n=== MISSING ROI ANALYSIS ===")
    missing_lh = [label for label in missing_schaefer if 1001 <= label <= 1100]
    missing_rh = [label for label in missing_schaefer if 2001 <= label <= 2100]
    
    print(f"Missing left hemisphere ROIs: {len(missing_lh)}")
    if missing_lh:
        print(f"First few missing LH ROIs: {missing_lh[:10]}...")
    
    print(f"Missing right hemisphere ROIs: {len(missing_rh)}")
    if missing_rh:
        print(f"First few missing RH ROIs: {missing_rh[:10]}...")

# Calculate voxel counts for each label to see if some have very few voxels
voxel_counts = Counter(data.flatten().astype(int))
label_counts = pd.DataFrame([
    {"Label": label, "Voxels": count} 
    for label, count in voxel_counts.items()
    if label in expected_schaefer
])

# Low voxel count might indicate ROIs that are barely represented
if not label_counts.empty:
    low_voxel_rois = label_counts[label_counts['Voxels'] < 10].sort_values('Voxels')
    
    print("\n=== ROIs WITH LOW REPRESENTATION ===")
    print(f"ROIs with fewer than 10 voxels: {len(low_voxel_rois)}")
    if not low_voxel_rois.empty:
        print(low_voxel_rois)

# Save the detailed analysis to files
detailed_results = pd.DataFrame({
    'Label': unique_labels,
    'Voxel_Count': [voxel_counts[label] for label in unique_labels]
})
detailed_results.to_csv('schaefer_roi_analysis.csv', index=False)


# If the result is severe (many missing ROIs), suggest solutions
if len(missing_schaefer) > 50:
    print("\n=== POTENTIAL ISSUES AND SOLUTIONS ===")
    print("1. Check if the annotation files used in your aparc2aseg command were correct")
    print("   - Ensure the Schaefer200 annotations were used for both hemispheres")
    print("2. Verify the FreeSurfer subject directory has all necessary files")
    print("3. The volume might be in a different space than expected")
    print("4. Try running mri_aparc2aseg directly with debugging flags:")
    print("   - mri_aparc2aseg --s <subject> --o output.mgz --annot Schaefer2018_200Parcels_7Networks_order --debug")

    import nibabel as nib
import numpy as np
from nibabel import freesurfer as fs
import os
import pandas as pd
import matplotlib.pyplot as plt

# Path to your annotation file
annot_file = '/home/rachel/Desktop/schaefer_analysis/fsaverage/ses-02/sub-3087/lh.sub-3087_ses-02_Schaefer2018_200Parcels_7Networks_order.annot'

# Function to read and analyze annotation files
def analyze_annotation_file(annot_file):
    print(f"Analyzing annotation file: {annot_file}")
    
    # Check if file exists
    if not os.path.exists(annot_file):
        print(f"ERROR: Annotation file not found at {annot_file}")
        return None
    
    try:
        # Read the annotation file
        labels, ctab, names = fs.io.read_annot(annot_file)
        
        # Convert binary names to strings
        names = [name.decode('utf-8') if isinstance(name, bytes) else name for name in names]
        
        print(f"Successfully loaded annotation file with {len(names)} labels")
        
        # Check if there are expected number of regions (should be 100 per hemisphere plus unknown/medial wall)
        expected_count = 101  # 100 regions + 1 for unknown/medial wall
        if len(names) != expected_count:
            print(f"WARNING: Expected {expected_count} labels but found {len(names)}")
        
        # Create DataFrame of region information
        regions_df = pd.DataFrame({
            'Index': range(len(names)),
            'Name': names,
            'R': [ctab[i, 0] for i in range(len(names))],
            'G': [ctab[i, 1] for i in range(len(names))],
            'B': [ctab[i, 2] for i in range(len(names))],
            'Label': [ctab[i, 4] for i in range(len(names))],
            'Count': [np.sum(labels == i) for i in range(len(names))]
        })
        
        # Calculate percentage of surface covered by each region
        total_vertices = len(labels)
        regions_df['Percentage'] = regions_df['Count'] / total_vertices * 100
        
        return regions_df, labels, names
    
    except Exception as e:
        print(f"ERROR reading annotation file: {e}")
        return None

# Analyze left hemisphere annotation
lh_info = analyze_annotation_file(annot_file)

if lh_info is not None:
    lh_df, lh_labels, lh_names = lh_info
    
    # Display summary of the annotation file
    print("\n=== ANNOTATION FILE SUMMARY ===")
    print(f"Total labeled regions: {len(lh_names)}")
    print(f"Total vertices in surface: {len(lh_labels)}")
    
    # Look for Schaefer atlas naming patterns
    schaefer_regions = [name for name in lh_names if '7Networks' in name]
    print(f"Schaefer network regions found: {len(schaefer_regions)}")
    
    # Check for expected label range
    expected_label_range = range(1001, 1101)
    labels_in_range = [label for label in lh_df['Label'] if label in expected_label_range]
    print(f"Labels in expected range (1001-1100): {len(labels_in_range)}")
    
    # Find the first few regions to confirm naming pattern
    if schaefer_regions:
        print("\nSample region names:")
        for name in schaefer_regions[:5]:
            print(f"  - {name}")
    
    # Display statistics about vertex counts
    print("\n=== REGION COVERAGE STATISTICS ===")
    print(f"Median vertices per region: {lh_df['Count'].median()}")
    print(f"Minimum vertices per region: {lh_df['Count'].min()}")
    small_regions = lh_df[lh_df['Count'] < 10].sort_values('Count')
    if not small_regions.empty:
        print(f"\nRegions with fewer than 10 vertices ({len(small_regions)}):")
        print(small_regions[['Name', 'Count', 'Percentage']])
    
    # Check if any expected labels are missing
    if len(labels_in_range) < 100:
        missing_labels = set(expected_label_range) - set(labels_in_range)
        print(f"\nMissing labels from expected range: {len(missing_labels)}")
        if missing_labels:
            print(f"First few missing labels: {sorted(list(missing_labels))[:10]}")
    
    # Look for the unknown/medial wall region
    medial_wall = lh_df[lh_df['Name'].str.contains('Background|Medial_Wall|unknown', case=False, regex=True)]
    if not medial_wall.empty:
        print(f"\nMedial wall/background region found: {medial_wall.iloc[0]['Name']}")
        print(f"Medial wall covers {medial_wall.iloc[0]['Percentage']:.2f}% of the surface")
    else:
        print("\nWARNING: No medial wall/background region found")
    
    # Save the detailed results
    lh_df.to_csv('lh_annotation_analysis.csv', index=False)
    print("\nDetailed results saved to 'lh_annotation_analysis.csv'")
    
    # Create histogram of region sizes
    plt.figure(figsize=(12, 6))
    plt.hist(lh_df['Count'][lh_df['Count'] > 0], bins=30)
    plt.title('Distribution of Region Sizes in Left Hemisphere Annotation')
    plt.xlabel('Number of Vertices')
    plt.ylabel('Number of Regions')
    plt.savefig('lh_annotation_region_sizes.png')
    print("Region size distribution plot saved to 'lh_annotation_region_sizes.png'")
    
    # Check right hemisphere for completeness
    rh_annot_file = annot_file.replace('lh.', 'rh.')
    if os.path.exists(rh_annot_file):
        print(f"\nRight hemisphere annotation file exists: {rh_annot_file}")
        print("Run this script again with the right hemisphere file to check it too.")
    else:
        print(f"\nWARNING: Right hemisphere annotation file not found at: {rh_annot_file}")
    
    # Provide diagnosis and recommendations
    print("\n=== DIAGNOSIS AND RECOMMENDATIONS ===")
    
    # Check if we have fewer than expected regions
    if len(schaefer_regions) < 100:
        print("ISSUE: The annotation file does not contain all 100 expected Schaefer regions.")
        print("SOLUTION: Verify the annotation was created correctly with all 200 parcels.")
        print("  - Check that the correct template was used for annotation creation")
        print("  - Consider re-running the annotation generation process")
    elif len(labels_in_range) < 100:
        print("ISSUE: The annotation file has the correct number of regions,")
        print("       but they don't have the expected label values (1001-1100).")
        print("SOLUTION: Check how the annotation file was created - the label")
        print("       values assigned to regions aren't matching what aparc2aseg expects.")
    elif small_regions.shape[0] > 10:
        print("ISSUE: Many regions have very few vertices, which may not project")
        print("       properly into the volume with aparc2aseg.")
        print("SOLUTION: This might be due to the resolution of your surface or")
        print("       problems with the annotation creation. Consider checking")
        print("       the surface quality or using a different approach.")
    else:
        print("The annotation file appears to contain the expected regions and")
        print("label values. The issue is likely with the aparc2aseg process.")
        print("SOLUTION: Check your Nipype/FreeSurfer command parameters:")
        print("  - Make sure you're using the correct --annot flag")
        print("  - Try running mri_aparc2aseg directly as a test")
        print("  - Add --debug flag to see more detailed output")
        print("  - Check that white and pial surfaces are properly aligned")
else:
    print("Could not analyze the annotation file. Please check the file path and format.")