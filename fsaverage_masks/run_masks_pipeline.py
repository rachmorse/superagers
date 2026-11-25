#!/usr/bin/env python3
"""Run the masks pipeline: transform the Schaefer and subcortical atlases into native T1, DWI and BOLD space."""

import datetime
import os
import sys
from pathlib import Path
from typing import List, Literal, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_utils import format_summary, run_script, summary_path

ATLAS: Literal["schaefer200", "schaefer400"] = "schaefer200"
TEST_SUBJECT: Optional[str] = None  # e.g. "sub-1014" to run only that subject

SCRIPTS: List[str] = [
    "01_fsaverage_to_t1.py",
    "02_subcortical_to_t1.py",
    "03_combine_t1_atlases.py",
    "04_t1_to_dwi_bold.py",
]
if ATLAS != "schaefer200":
    SCRIPTS.remove("02_subcortical_to_t1.py")


def main() -> None:
    """Run each masks script in order, then print the summary of all scripts."""
    # Make the scripts below use the ATLAS and TEST_SUBJECT set at the top
    os.environ["ATLAS"] = ATLAS
    if TEST_SUBJECT:
        os.environ["TEST_SUBJECT"] = TEST_SUBJECT
    else:
        os.environ.pop("TEST_SUBJECT", None)

    base_dir = Path(__file__).resolve().parent
    log_path = base_dir / f"nohup_fsaverage_{ATLAS}_{datetime.date.today():%Y%m%d}.out"

    # Clear results from any previous run so the summary only covers this one
    summary_path(__file__, ATLAS).unlink(missing_ok=True)

    start_time = datetime.datetime.now()
    print(f"Starting {ATLAS} pipeline at {start_time}" + (f" for {TEST_SUBJECT} only" if TEST_SUBJECT else ""))

    failures: List[str] = []
    for script in SCRIPTS:
        exit_code, name = run_script(base_dir / script, log_path)
        if exit_code != 0:
            failures.append(f"{name} (exit {exit_code})")

    if failures:
        print("Completed with failures:")
        for item in failures:
            print(f"  - {item}")
    else:
        print("All scripts completed successfully.")

    # Print the summary and add it to the end of the log so it isn't buried in the per-subject output
    summary = format_summary(__file__, ATLAS)
    print(summary)
    with log_path.open("a") as log_file:
        log_file.write(summary + "\n")

    end_time = datetime.datetime.now()
    elapsed = end_time - start_time
    hours = elapsed.total_seconds() / 3600

    print(f"Completed at {end_time}")
    print(f"Total time: {hours:.2f} hours")


if __name__ == "__main__":
    main()
