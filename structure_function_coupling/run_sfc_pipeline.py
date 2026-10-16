#!/usr/bin/env python3

import subprocess
import sys
from pathlib import Path
from typing import List, Literal, Optional, Tuple
import datetime


ATLAS: Literal["schaefer200", "schaefer400"] = "schaefer200"
TEST_SUBJECT: Optional[str] = None  # e.g. "sub-1014" to run only that subject
SUBJECTS_FILE = Path("/home/rachel/Desktop/data/final_cohort_ids.txt")  # final longitudinal cohort 
SUBJECTS: List[str] = [TEST_SUBJECT] if TEST_SUBJECT else SUBJECTS_FILE.read_text().split()
SESSIONS: List[Literal["ses-01", "ses-02"]] = ["ses-01", "ses-02"]
N_PARCELS = int(ATLAS.removeprefix("schaefer"))
N_ROIS = N_PARCELS + 14  # plus 14 subcortical
IS_FISHER_Z = True
SAVE_RAW = True
SAVE_FISHER_Z = True

ROOT = Path("/home/rachel/Desktop/schaefer_analysis")
SUFFIX = f"_{ATLAS}"

# Unsuffixed folders are the original run, each atlas reads and writes its own suffixed folders
FC_ROOT = ROOT / f"functional_connectivity{SUFFIX}" / "native_space"
STRUCT_ROOT = ROOT / f"structural_connectivity{SUFFIX}"
SFC_ROOT = ROOT / f"structure_function_coupling{SUFFIX}"

SCRIPTS: List[str] = [
    "01_convert_to_individual_matrix.py",
    "02_structural_functional_coupling.py",
]


def run_script(script_path: Path, log_path: Path) -> Tuple[int, str]:
    """Run a script and append stdout/stderr to the shared log.

    Args:
        script_path: Path to the script to run.
        log_path: Path to the log file to append output.
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
    base_dir = Path(__file__).resolve().parent
    log_path = base_dir / f"nohup_sfc_{ATLAS}_{datetime.date.today():%Y%m%d}.out"

    start_time = datetime.datetime.now()
    print(f"Starting {ATLAS} structure-function coupling pipeline at {start_time}")

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
