#!/usr/bin/env python3
"""Run the masks pipeline: transform the Schaefer and subcortical atlases into native T1, DWI and BOLD space."""

import datetime
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Literal, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_utils import format_summary, summary_path

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
    """Run each masks script in order, then print the summary of all scripts."""
    # Pass ATLAS and TEST_SUBJECT to config.py in the scripts started below
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
