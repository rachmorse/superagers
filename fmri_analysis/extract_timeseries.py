"""Functions to extract and plot ROI timeseries from a BOLD image using an atlas mask."""

from datetime import datetime
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nilearn.maskers import NiftiLabelsMasker


def extract_timeseries(atlas_file: Path, fmri_file: Path, error_log_path: Path) -> Optional[np.ndarray]:
    """Extracts timeseries data from a BOLD image using an atlas mask.

    Errors are written to the error log rather than raised, so one failed subject does not stop the run.

    Args:
        atlas_file: Path to the atlas file (mask).
        fmri_file: Path to the fMRI preprocessed BOLD image file.
        error_log_path: Path to the error log file.

    Returns:
        Extracted timeseries data, or None if an error occurred (e.g. a missing file or
        atlas and BOLD images with different spatial dimensions).
    """
    try:
        if not Path(fmri_file).exists():
            raise FileNotFoundError(f"BOLD file {fmri_file} not found.")

        if not Path(atlas_file).exists():
            raise FileNotFoundError(f"Schaefer atlas file {atlas_file} not found.")

        # Load the atlas and BOLD file
        atlas_img = nib.load(atlas_file)
        fmri_img = nib.load(fmri_file)

        # Check whether the atlas and BOLD files have the same shape
        if atlas_img.shape[:3] != fmri_img.shape[:3]:
            raise ValueError(
                f"Atlas and fMRI images must have the same spatial dimensions. "
                f"Atlas shape: {atlas_img.shape}, fMRI shape: {fmri_img.shape}"
            )

        # Use NiftiLabelsMasker to extract the timeseries
        masker = NiftiLabelsMasker(labels_img=atlas_img, standardize=False)

        print("Extracting timeseries...")
        timeseries = masker.fit_transform(fmri_file)

        print("Timeseries extraction complete.")
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
) -> None:
    """Visualize the timeseries for specified ROIs.

    Args:
        subject_id: Subject ID.
        timeseries: The timeseries data to be visualized.
        roi_indices: List of ROI indices to visualize.
    """
    # Visualize Timeseries for specified ROIs
    for idx in roi_indices:
        plt.figure(figsize=(10, 4))
        plt.plot(timeseries[:, idx])
        plt.title(f"Timeseries for ROI {idx} - Subject {subject_id}")
        plt.xlabel("Time points")
        plt.ylabel("BOLD signal")
        plt.show()
