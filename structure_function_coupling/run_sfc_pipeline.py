#!/usr/bin/env python3
"""Run the structure-function coupling pipeline: split the group matrices per subject, then compute coupling."""

import datetime
import os
import sys
from pathlib import Path
from typing import List, Literal, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_utils import run_script

ATLAS: Literal["schaefer200", "schaefer400"] = "schaefer200"
TEST_SUBJECT: Optional[str] = None  # e.g. "sub-1014" to run only that subject

SCRIPTS: List[str] = [
    "01_convert_to_individual_matrix.py",
    "02_structural_functional_coupling.py",
]


def main() -> None:
    """Run each coupling script in order, stopping at the first failure."""
    # Make the scripts below use the ATLAS and TEST_SUBJECT set at the top
    os.environ["ATLAS"] = ATLAS
    if TEST_SUBJECT:
        os.environ["TEST_SUBJECT"] = TEST_SUBJECT
    else:
        os.environ.pop("TEST_SUBJECT", None)

    base_dir = Path(__file__).resolve().parent
    log_path = base_dir / f"nohup_sfc_{ATLAS}_{datetime.date.today():%Y%m%d}.out"

    start_time = datetime.datetime.now()
    print(
        f"Starting {ATLAS} structure-function coupling pipeline at {start_time}"
        + (f" for {TEST_SUBJECT} only" if TEST_SUBJECT else "")
    )

    failures: List[str] = []
    for script in SCRIPTS:
        exit_code, name = run_script(base_dir / script, log_path)
        if exit_code != 0:
            failures.append(f"{name} (exit {exit_code})")
            # Coupling needs the individual matrices so stop here
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
