import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nilearn.input_data import NiftiLabelsMasker


def extract_timeseries(atlas_file: str, fmri_file: str, error_log_path: Path) -> Optional[np.ndarray]:
    """Extracts timeseries data from a BOLD image using an atlas mask.

    Args:
        atlas_file (str): Path to the atlas file (mask).
        fmri_file (str): Path to the fMRI preprocessed BOLD image file.
        error_log_path (Path): Path to the error log file.

    Returns:
        np.ndarray: Extracted timeseries data, or None if an error occurred.

    Raises:
        FileNotFoundError: If the fMRI or atlas file is not found.
        ValueError: If the mask type is not recognized.
    """
    try:
        if not os.path.exists(fmri_file):
            raise FileNotFoundError(f"BOLD file {fmri_file} not found.")

        if not os.path.exists(atlas_file):
            raise FileNotFoundError(f"Schaefer atlas file {atlas_file} not found.")

        # Load the atlas file
        atlas_img = nib.load(atlas_file)

        # Use NiftiLabelsMasker to extract the timeseries
        masker = NiftiLabelsMasker(labels_img=atlas_img, standardize=False)
        print("Extracting timeseries...")
        timeseries = masker.fit_transform(fmri_file)

        return timeseries

    except Exception as e:
        with open(error_log_path, "a") as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Error processing atlas {atlas_file} and fMRI {fmri_file}:\n")
            f.write(f"{e!s}\n\n")
        return None


def visualize_timeseries(
    subject_id: str,
    timeseries: np.ndarray,
    roi_indices: List[int],
):
    """Visualize the timeseries for specified ROIs.

    Args:
        subject_id (str): Subject ID.
        timeseries (np.ndarray): The timeseries data to be visualized.
        roi_indices (List[int]): List of ROI indices to visualize.
    """
    # Visualize Timeseries for specified ROIs
    for idx in roi_indices:
        plt.figure(figsize=(10, 4))
        plt.plot(timeseries[:, idx])
        plt.title(f"Timeseries for ROI {idx} - Subject {subject_id}")
        plt.xlabel("Time points")
        plt.ylabel("BOLD signal")
        plt.show()
