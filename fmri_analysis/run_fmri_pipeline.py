#!/usr/bin/env python3
"""Run the fMRI pipeline: extract ROI timeseries, then compute functional connectivity."""

import datetime
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Literal, Optional, Tuple

ATLAS: Literal["schaefer200", "schaefer400"] = "schaefer200"
TEST_SUBJECT: Optional[str] = None  # e.g. "sub-1014" to run only that subject

SCRIPTS: List[str] = [
    "01_extract_subjects_timeseries.py",
    "02_compute_subject_functional_connectivity.py",
]


def run_script(script_path: Path, log_path: Path) -> Tuple[int, str]:
    """Run a script and append stdout/stderr to the shared log.

    Args:
        script_path: Path to the script to run.
        log_path: Path to the log file to append output.

    Returns:
        The script's exit code (1 if the script was not found) and its name or error message.
    """
    if not script_path.exists():
        msg = f"Script not found: {script_path}"
        print(msg)
        return 1, msg

    print(f"Running {script_path.name}...")
    with log_path.open("a") as log_file:
        log_file.write(f"=== START {script_path.name} ===\n")
        result = subprocess.run(
            [sys.executable, "-u", str(script_path)],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log_file.write(f"=== END {script_path.name} (exit {result.returncode}) ===\n\n")

    if result.returncode == 0:
        print(f"Successfully completed {script_path.name}")
    else:
        print(f"{script_path.name} failed with exit code {result.returncode}")

    print("-" * 30)
    return result.returncode, script_path.name


def main() -> None:
    """Run each fMRI script in order, stopping at the first failure."""
    # Pass ATLAS and TEST_SUBJECT to config.py in the scripts started below
    os.environ["ATLAS"] = ATLAS
    if TEST_SUBJECT:
        os.environ["TEST_SUBJECT"] = TEST_SUBJECT
    else:
        os.environ.pop("TEST_SUBJECT", None)

    base_dir = Path(__file__).resolve().parent
    log_path = base_dir / f"nohup_fmri_{ATLAS}_{datetime.date.today():%Y%m%d}.out"

    start_time = datetime.datetime.now()
    print(f"Starting {ATLAS} fMRI pipeline at {start_time}" + (f" for {TEST_SUBJECT} only" if TEST_SUBJECT else ""))

    failures: List[str] = []
    for script in SCRIPTS:
        exit_code, name = run_script(base_dir / script, log_path)
        if exit_code != 0:
            failures.append(f"{name} (exit {exit_code})")
            # Connectivity depends on the timeseries so stop here
            break

    if failures:
        print("Completed with failures:")
        for item in failures:
            print(f"  - {item}")
    else:
        print("All scripts completed successfully.")

    end_time = datetime.datetime.now()
    elapsed = end_time - start_time
    hours = elapsed.total_seconds() / 3600

    print(f"Completed at {end_time}")
    print(f"Total time: {hours:.2f} hours")


if __name__ == "__main__":
    main()
