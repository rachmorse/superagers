import pandas as pd
import numpy as np
from pathlib import Path

# This script takes the output of fMRI Schaefer 200 atlas processing and turns it into an excel file to use in making a Gephi network visualization.

# Load the CSV data
ses = "02"
# data = pd.read_csv(f"/home/rachel/Desktop/schaefer_analysis/connectivity_matrices/ses-{ses}/all_to_all_roi_matrices/superagers_average.csv")
# data = pd.read_csv(f"/home/rachel/Desktop/schaefer_analysis/connectivity_matrices/ses-{ses}/all_to_all_roi_matrices/non_superagers_average.csv")
# data = pd.read_csv(f"/home/rachel/Desktop/schaefer_analysis/connectivity_matrices/ses-{ses}/all_to_all_roi_matrices/maintainers_average.csv")
data = pd.read_csv(f"/home/rachel/Desktop/schaefer_analysis/connectivity_matrices/ses-{ses}/all_to_all_roi_matrices/non_superager_decliners_average.csv")
data = pd.read_csv(f"/home/rachel/Desktop/schaefer_analysis/connectivity_matrices/ses-{ses}/all_to_all_roi_matrices/superager_maintainers_average.csv")

# Extract the subject data
# subject_id = "superager"
# subject_id = "non-superager"
# subject_id = "maintainer"
# subject_id = "non_superager_decliners"
subject_id = "superager_maintainers"

subject_data = data.loc[data['id'] == subject_id].iloc[0, 1:].values

# Calculate the number of elements for a full 200 x 200 matrix's lower triangle
expected_num_elements = (200 * 199) // 2

# Check if the data length matches expected lower triangle elements length
if len(subject_data) != expected_num_elements:
    raise ValueError(f"Unexpected number of elements. Expected {expected_num_elements} but got {len(subject_data)}.")

# Build the 200x200 matrix
connectivity_matrix = np.zeros((200, 200))
tril_indices = np.tril_indices(200, -1)
connectivity_matrix[tril_indices] = subject_data
connectivity_matrix += connectivity_matrix.T

# Print the matrix or use it as needed
print(connectivity_matrix)
print(connectivity_matrix.shape)

# Get the lower triangle of the matrix, excluding the diagonal.
lower_triangle = np.tril(connectivity_matrix, k=-1)

numrois = 200

# Prepare index vector
index = np.arange(1, numrois + 1)

# Initialize variables
i = 0
final_output = []

# Iterate over each column
while i < numrois - 1:
    output = []

    # Extract values from lower triangle, ensuring indices are valid
    values = lower_triangle[i+1:numrois, i]

    # Prepare 'Source' and 'Target' columns
    sources = index[i+1:numrois]
    targets = np.full(sources.shape, i + 1)

    # Combine in the required format
    for s, t, v in zip(sources, targets, values):
        output.append([s, t, 0, 0, 0, 0, v])  # Fill with zeroes for unused columns

    final_output.extend(output)

    i += 1

# Convert to df
df_output = pd.DataFrame(final_output)

# Add column headings
column_headers = ["Source", "Target", "Type", "ID", "Label", "Interval", "Weight"]
df_output.columns = column_headers

# Fill 'Type' column with 'Undirected'
df_output['Type'] = 'Undirected'

# Fill 'ID' column with sequential numbers starting from 1
df_output['ID'] = range(1, len(df_output) + 1)

# Set 'Label' and 'Interval' columns to empty strings
df_output['Label'] = ''
df_output['Interval'] = ''

# Save to a CSV file
output_file_path = f"/home/rachel/Desktop/schaefer_analysis/gephi/{subject_id}_ses-{ses}_reshaped_for_Gephi.csv"
df_output.to_csv(output_file_path, index=False)

print(f"Completed processing {subject_id}.")