#!/usr/bin/env python3

import subprocess
import sys
from pathlib import Path
from typing import List, Tuple
import datetime


SCRIPTS: List[str] = [
    "01_fsaverage_to_t1.py",
    "02_subcortical_to_t1.py",
    "03_combine_t1_atlases.py",
    "04_t1_to_dwi_bold.py",
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
    log_path = base_dir / "nohup_fsaverage.out"

    start_time = datetime.datetime.now()
    print(f"Starting pipeline at {start_time}")

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

    end_time = datetime.datetime.now()
    elapsed = end_time - start_time
    hours = elapsed.total_seconds() / 3600

    print(f"Completed at {end_time}")
    print(f"Total time: {hours:.2f} hours")


if __name__ == "__main__":
    main()
