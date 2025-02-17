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

- Reads "sub-XXX" from a CSV if given, or from directory otherwise.
- Appends "_ses-01" (or "_ses-02") to each subject for that time point.
- Runs one single asegstats2table call per session, creating a separate CSV
  for each.

Comment/uncomment the relevant blocks for Superager vs. BBHI data.

NOTE:
The only changes below ensure that results are appended
to existing CSV files instead of overwriting them.
"""

def create_wrapper_script():
    """
    Create a wrapper script to run asegstats2table with Python 2.
    Note this is needed because Freesurfer 6 which we used to run
    recon-all requires Python 2 not Python 3.
    """
    wrapper_script = """#!/bin/bash
PYTHON2=/usr/bin/python2
ASEGSTATS2TABLE=/home/rachel/freesurfer/freesurfer/bin/asegstats2table
$PYTHON2 $ASEGSTATS2TABLE "$@"
"""
    with open("run_asegstats2table.sh", "w") as f:
        f.write(wrapper_script)
    os.chmod("run_asegstats2table.sh", 0o755)

def get_unprocessed_subjects(subject_list, output_file):
    """
    Given a list of subjects (e.g., ['sub-001_ses-01', ...]) and an output CSV file,
    return only those subjects that are not present in the output CSV.

    Args:
        subject_list (list): Full list of potential subjects for a session.
        output_file (Path): Path to the CSV file with existing data.

    Returns:
        list: Subjects not yet processed.
    """
    if not output_file.exists():
        print(f"Output CSV file {output_file} not found. Assuming no subjects processed yet.")
        return subject_list

    # Attempt to parse which subjects have been processed by extracting "sub-XXX_ses-YY" from the "Measure" column
    df = pd.read_csv(output_file)
    if "Measure" not in df.columns:
        print(f"Warning: 'Measure' column not found in {output_file}, cannot filter processed subjects. Returning the full list.")
        return subject_list

    processed_regex = df["Measure"].str.extract(r'(sub-\d+_ses-\d+)')
    processed_set = set(processed_regex.dropna().unique())

    unprocessed = [sub for sub in subject_list if sub not in processed_set]
    return unprocessed

def append_csv_files(final_csv: Path, new_csv: Path):
    """
    Append the contents of the new CSV file to the final CSV file
    without overwriting existing rows. Skips the header line if
    the final CSV already exists (to avoid duplication).
    """
    # If final CSV doesn't exist, just rename the new one
    if not final_csv.exists():
        new_csv.rename(final_csv)
        return

    # Otherwise, read lines from new_csv and append to final_csv
    with final_csv.open("a") as main_f, new_csv.open("r") as temp_f:
        first_line = True
        for line in temp_f:
            # If this is the first line of the new CSV and final CSV isn't empty, skip header
            if first_line and main_f.tell() != 0 and line.strip().startswith("Measure"):
                first_line = False
                continue
            main_f.write(line)
            first_line = False

def main():
    # Create the wrapper script
    create_wrapper_script()

    # ---------------------------------------------------------------------
    # Switch between Superagers and BBHI by commenting/uncommenting:

    # Superagers:
    # subject_csv = None
    # root_path = Path("/pool/guttmann/institut/UB/Superagers/MRI/freesurfer-reconall")

    # BBHI:
    subject_csv = "/home/rachel/Desktop/data/clean_bbhi.csv"
    root_path = Path("/pool/guttmann/institut/BBHI/MRI/processed_data/freesurfer-reconall")
    # ---------------------------------------------------------------------

    # Define session / output directory
    sessions = ["ses-01", "ses-02"]
    output_dir = Path("/home/rachel/Desktop/data/")
    os.environ["SUBJECTS_DIR"] = str(root_path)  # Required by Freesurfer

    # Gather all possible base subjects
    if subject_csv:
        df = pd.read_csv(subject_csv)
        if "id" not in df.columns:
            print(f"Error: 'id' column not found in {subject_csv}.")
            return
        # Convert each ID to sub-XXX format if not already
        base_subj_list = sorted({f"sub-{str(i).strip()}" for i in df["id"]})
        print(f"Found {len(base_subj_list)} unique base subjects from CSV.")
    else:
        # If no CSV, gather from directory names (sub-XXX or sub-XXX_ses-YY)
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

    # For each session, build the full subject name "sub-XXX_ses-YY",
    # skip if it doesn't exist, then remove those already processed, then append results.
    for ses in sessions:
        all_this_session = []
        for base_subj in base_subj_list:
            directory_name = f"{base_subj}_{ses}"  # e.g., "sub-001_ses-01"
            stats_dir = root_path / directory_name / "stats"
            if stats_dir.is_dir():
                all_this_session.append(directory_name)
            else:
                print(f"Warning: directory does not exist for {directory_name}, skipping.")

        if not all_this_session:
            print(f"No subjects found for {ses}.")
            continue

        session_output = output_dir / f"hippocampus_and_wm_hypointensities_{ses}.csv"
        unprocessed = get_unprocessed_subjects(all_this_session, session_output)

        if not unprocessed:
            print(f"All subjects already processed for {ses}. Skipping.")
            continue

        # Write to a temporary CSV, then append to session_output
        temp_output = output_dir / f"temp_{session_output.name}"
        command = [
            "./run_asegstats2table.sh",
            "--subjects",
            *unprocessed,
            "--meas", "volume",
            "--segno", "17", "53", "77",  # left HC, right HC, WM hypointensities
            "--delimiter", "comma",
            "--tablefile", str(temp_output)
        ]

        print(f"\n[Session: {ses}] Processing {len(unprocessed)} unprocessed subjects.")
        print("Command:\n", " ".join(command), "\n")

        try:
            subprocess.run(command, check=True)
            print(f"Session {ses} completed successfully. Results in {temp_output}")
            # Append new results to session_output
            append_csv_files(session_output, temp_output)
        except subprocess.CalledProcessError as exc:
            print(f"Error running asegstats2table for session {ses}: {exc}")
        finally:
            if temp_output.exists():
                temp_output.unlink()

if __name__ == "__main__":
    main()