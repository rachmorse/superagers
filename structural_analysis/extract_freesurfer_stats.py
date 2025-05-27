#!/usr/bin/env python3

import os
import re
import subprocess
import pandas as pd
from pathlib import Path

# NOTE about FreeSurfer naming:
# “aparc” refers to “automatic parcellation,” used to label cortical regions in ?h.aparc.stats files.
# “aseg” stands for “automatic segmentation,” which labels subcortical structures (including the hippocampus) in aseg.stats.

"""
This script extracts hippocampal volumes (labels #17 for left hippocampus,
#53 for right hippocampus) and white matter hypointensities (label #77)
using asegstats2table. 

Comment/uncomment the relevant blocks for Superager vs. BBHI data.
"""

##############################################################################
# Create the wrapper script
##############################################################################
def create_wrapper_script():
    """
    Create a wrapper script to run asegstats2table and aparcstats2table with
    Python 2. Note this is needed because Freesurfer 6 (used for recon-all)
    expects Python 2, not Python 3.
    """
    wrapper_script = """#!/bin/bash
PYTHON2=/usr/bin/python2
ASEGSTATS2TABLE=/home/rachel/freesurfer/freesurfer/bin/asegstats2table
APARCSTATS2TABLE=/home/rachel/freesurfer/freesurfer/bin/aparcstats2table
if [[ "$1" == "--aparc" ]]; then
    shift
    $PYTHON2 $APARCSTATS2TABLE "$@"
else
    $PYTHON2 $ASEGSTATS2TABLE "$@"
fi
"""
    with open("run_asegstats2table.sh", "w") as f:
        f.write(wrapper_script)
    os.chmod("run_asegstats2table.sh", 0o755)

##############################################################################
# Check for required segments in aseg.stats
##############################################################################
def has_required_segments(stats_file, required_segments):
    """
    Checks if aseg.stats has all required segmentation labels.

    Args:   
        stats_file (Path): Path to the aseg.stats file
        required_segments (list of str): List of segmentation labels to check for
    """
    if not stats_file.is_file():
        return False

    content = stats_file.read_text()
    # Look for lines containing " <segment> " to confirm presence of each
    return all(any(f" {seg} " in line for line in content.splitlines()) 
               for seg in required_segments)

##############################################################################
# Get unprocessed subjects
##############################################################################
def get_unprocessed_subjects(subject_list, output_file):
    """
    Given a list of subjects (e.g., ['sub-001_ses-01', ...]) and an output CSV file,
    return only those subjects not present in the output CSV.

    Args:
        subject_list (list of str): Potential subjects for a session
        output_file (Path): Path to the CSV file with existing data

    Returns:
        list of str: Subjects not yet processed in that output file
    """
    if not output_file.exists():
        print(f"Output CSV file {output_file} not found. Assuming no subjects processed yet.")
        return subject_list

    try:
        df = pd.read_csv(output_file)
    except Exception as e:
        print(f"Warning: could not read file {output_file}, error: {e}")
        print("Returning the full list of subjects.")
        return subject_list

    if "Measure" not in df.columns:
        print(f"Warning: 'Measure' column not found in {output_file}, cannot filter processed subjects. Returning the full list.")
        return subject_list

    # Attempt to parse which subjects have been processed by extracting "sub-XXX_ses-YY" from the 'Measure' column
    df["Measure"] = df["Measure"].astype(str)
    processed_regex = df["Measure"].str.extract(r'(sub-\d+_ses-\d+)')

    # Get the first column (index 0) to convert from DataFrame to Series, then use unique()
    if processed_regex.shape[1] > 0:  # Check if any columns were extracted
        processed_set = set(processed_regex[0].dropna().unique())
    else:
        processed_set = set()  # No matching patterns found

    unprocessed = [sub for sub in subject_list if sub not in processed_set]
    return unprocessed

##############################################################################
# Append CSV files
##############################################################################
def append_csv_files(final_csv: Path, new_csv: Path):
    """
    Merge contents of new_csv into final_csv by matching on "Measure."
    This ensures thickness columns and volume columns appear in a single row per subject.
    """

    # If final_csv doesn't exist or is empty, create a DataFrame with id only
    if final_csv.exists() and final_csv.stat().st_size > 0:
        final_df = pd.read_csv(final_csv)
    else:
        final_df = pd.DataFrame(columns=["id"])
    print(f"Initial final_df head: {final_df.head(3)}")

    # Read CSV with thickness or volume data
    new_df = pd.read_csv(new_csv)
    
    # If the new_df is the volume df, subset to only needed columns
    if "Left-Hippocampus" in new_df.columns:
        # Rename columns
        column_mapping = {
            "Measure:volume": "id",
            "Left-Hippocampus": "Left_Hippocampus_volume",
            "Right-Hippocampus": "Right_Hippocampus_volume",
            "WM-hypointensities": "WM_hypointensities_volume"
        }
        for old_col, new_col in column_mapping.items():
            if old_col in new_df.columns:
                new_df.rename(columns={old_col: new_col}, inplace=True)
        
        # Subset to only keep relevant columns     
        columns_to_keep = [
            "id",
            "Left_Hippocampus_volume",
            "Right_Hippocampus_volume",
            "WM_hypointensities_volume",
        ]
        new_df = new_df[columns_to_keep]

    # If the new_df is the thickness df, rename columns and subset to only needed columns
    elif "rh_bankssts_thickness" in new_df.columns:
        # Rename the id column
        new_df = new_df.rename(columns={"rh.aparc.thickness": "id"})

        # Subset to only keep relevant columns
        columns_to_keep = [
            "id",
            "rh_MeanThickness_thickness"
        ]
        
        new_df = new_df[columns_to_keep]
        print(f"before merge: {new_df.head(3)}")

    # If the new_df is the left hemisphere thickness df, rename columns and subset to only needed columns
    elif "lh_bankssts_thickness" in new_df.columns:
        # Rename the id column
        new_df = new_df.rename(columns={"lh.aparc.thickness": "id"})

        # Subset to only keep relevant columns
        columns_to_keep = [
            "id",
            "lh_MeanThickness_thickness"
        ]
        
        new_df = new_df[columns_to_keep]
        print(f"before merge: {new_df.head(3)}")

    # Merge each of the df for each participant id added the _new suffix
    merged_df = pd.merge(final_df, new_df, on="id", how="outer", suffixes=("", "_new"))

    # Consolidate any columns that appear as duplicates with "_new" suffix.
    for col in merged_df.columns:
        if col.endswith("_new"):
            base_col = col.replace("_new", "")
            merged_df[base_col] = merged_df[base_col].fillna(merged_df[col])
            merged_df.drop(columns=[col], inplace=True)

    print(f"merge df head: {merged_df.head(3)}")

    # Write updated CSV
    merged_df.to_csv(final_csv, index=False)
    print(f"Wrote updated data to {final_csv}")

##############################################################################
# Main
##############################################################################
def main():
    # Create the wrapper script
    create_wrapper_script()

    if cohort == "bbhi":
        subject_csv = "/home/rachel/Desktop/data/bbhi_ids_tp1.csv"
        root_path = Path("/pool/guttmann/institut/BBHI/MRI/processed_data/freesurfer-reconall")
    else:
        subject_csv = None
        root_path = Path("/pool/guttmann/institut/UB/Superagers/MRI/freesurfer-reconall")

    # Define session / output directory
    sessions = ["ses-01", "ses-02"]
    output_dir = Path("/home/rachel/Desktop/data/")
    os.environ["SUBJECTS_DIR"] = str(root_path)  # Required by Freesurfer

    # Required segmentation labels
    required_segments = ["17", "53", "77"]  # Left HC, Right HC, WM hypointensities

    # Collect a list of failed subjects for final printing
    failed_subjects = []

    # Create a list of subjects to process
    if subject_csv:
        df = pd.read_csv(subject_csv)
        if "id" not in df.columns:
            print(f"Error: 'id' column not found in {subject_csv}.")
            return
        base_subj_list = sorted({f"sub-{str(i).strip()}" for i in df["id"]})
        print(f"Found {len(base_subj_list)} unique base subjects from CSV.")
    else:
        all_dirs = [
            d for d in os.listdir(root_path)
            if (root_path / d).is_dir() and d.startswith("sub-")
        ]
        pattern = re.compile(r"^(sub-\d+)(?:_ses-\d+)?$")
        base_ids = set()
        for d in all_dirs:
            match = pattern.match(d)
            if match:
                base_ids.add(match.group(1))
        base_subj_list = sorted(base_ids)
        print(f"Found {len(base_subj_list)} unique base subjects by directory listing.")

    # For each timepoint, process each subject individually
    for ses in sessions:
        session_output = output_dir / f"structural_volume_thickness_{ses}.csv"

        unprocessed_subjects = get_unprocessed_subjects([f"{base_subj}_{ses}" for base_subj in base_subj_list], session_output)
        
        for subj in unprocessed_subjects:
            stats_dir = root_path / subj / "stats"
            if not stats_dir.is_dir():
                print(f"Warning: stats directory not found for {subj}, skipping.")
                continue

            stats_file = stats_dir / "aseg.stats"
            if not has_required_segments(stats_file, required_segments):
                print(f"Warning: {subj} missing required segments: {required_segments}, skipping.")
                failed_subjects.append(subj)
                continue

            # Build a temporary CSV for just this subject
            temp_csv = output_dir / f"temp_{subj}.csv"

            command = [
                "./run_asegstats2table.sh",
                "--subjects", subj,
                "--meas", "volume",
                "--segno", "17", "53", "77",
                "--delimiter", "comma",
                "--tablefile", str(temp_csv)
            ]

            try:
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as exc:
                print(f"Error: asegstats2table failed for {subj}, skipping.\n{exc}")
                failed_subjects.append(subj)
                if temp_csv.exists():
                    temp_csv.unlink()
                continue

            if not temp_csv.exists():
                print(f"Error: output CSV for {subj} not created, skipping.")
                failed_subjects.append(subj)
                continue

            # Append single-subject CSV to session CSV
            append_csv_files(session_output, temp_csv)

            # Delete the temporary CSV after appending
            if temp_csv.exists():
                temp_csv.unlink()

            # Cortical thickness extraction
            for hemi in ["lh", "rh"]:
                temp_thick_csv = output_dir / f"temp_{subj}_{hemi}_thickness.csv"
                aparc_command = [
                    "./run_asegstats2table.sh",
                    "--aparc",                # This tells the wrapper script to run aparcstats2table because both commands are in the same script
                    "--subjects", subj,
                    "--hemi", hemi,
                    "--meas", "thickness",
                    "--parc", "aparc",
                    "--delimiter", "comma",
                    "--tablefile", str(temp_thick_csv)
                ]
                try:
                    subprocess.run(aparc_command, check=True)
                except subprocess.CalledProcessError as exc:
                    print(f"Error: aparcstats2table failed for {subj} {hemi}, skipping.\n{exc}")
                    failed_subjects.append(f"{subj}_{hemi}_aparc")
                    if temp_thick_csv.exists():
                        temp_thick_csv.unlink()
                    continue

                if not temp_thick_csv.exists():
                    print(f"Error: output CSV for {subj} {hemi} thickness not created, skipping.")
                    failed_subjects.append(f"{subj}_{hemi}_aparc_output")
                    continue

                # Append thickness CSV to output 
                append_csv_files(session_output, temp_thick_csv)

                if temp_thick_csv.exists():
                    temp_thick_csv.unlink()

        print(f"\nFinished session {ses}. Results in {session_output}\n")

    # Print a list of subjects that failed
    if failed_subjects:
        print("Subjects that failed to process or were skipped due to missing segments:")
        for subj in sorted(set(failed_subjects)):
            print("  ", subj)
    else:
        print("All subjects processed successfully.")

if __name__ == "__main__":
    # Set the cohort variable (already processed tp1 and 2)
    cohort = "bbhi"  

    main()

    