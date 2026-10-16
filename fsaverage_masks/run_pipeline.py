#!/usr/bin/env python3

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple
import datetime


ATLAS: Literal["schaefer200", "schaefer400"] = "schaefer400"
TEST_SUBJECT: Optional[str] = None  # e.g. "sub-1014" to run only that subject
SUBJECTS_FILE = Path("/home/rachel/Desktop/data/final_cohort_ids.txt")  # final longitudinal cohort 
SUBJECTS: List[str] = [TEST_SUBJECT] if TEST_SUBJECT else SUBJECTS_FILE.read_text().split()
COHORTS: List[Literal["bbhi", "bbhi senior"]] = ["bbhi", "bbhi senior"]
SESSIONS: List[Literal["ses-01", "ses-02"]] = ["ses-01", "ses-02"]
N_JOBS = 6
N_PARCELS = int(ATLAS.removeprefix("schaefer"))

RECONALL_DIRS: Dict[str, Path] = {
    "bbhi": Path("/pool/guttmann/institut/BBHI/MRI/derivatives/reconall_fs6"),
    "bbhi senior": Path("/pool/guttmann/institut/UB/Superagers/MRI/derivatives/reconall_fs6"),
}
FSL_DIR = Path("/vol/software/fsl_6_0_4")

# Subcortical masks and DWI registrations are atlas-independent so all atlases reuse the ones in fsaverage
SHARED_ROOT = Path("/home/rachel/Desktop/schaefer_analysis/fsaverage")
OUTPUT_ROOT = SHARED_ROOT if ATLAS == "schaefer200" else Path(f"/home/rachel/Desktop/schaefer_analysis/fsaverage_{ATLAS}")

SCRIPTS: List[str] = [
    "01_fsaverage_to_t1.py",
    "02_subcortical_to_t1.py",
    "03_combine_t1_atlases.py",
    "04_t1_to_dwi_bold.py",
]
if ATLAS != "schaefer200":
    SCRIPTS.remove("02_subcortical_to_t1.py")

# Each script appends its per cohort/session counts here so the pipeline can print one summary at the end
SUMMARY_PATH = Path(__file__).resolve().parent / "pipeline_summaries" / f"pipeline_summary_{ATLAS}.jsonl"


def record_summary(script: str, cohort: str, session: str, n_successful: int, failed: List[str], step: str = "") -> None:
    """Record the success/failure counts for one cohort and session of a script.

    Args:
        script: Name of the script reporting the results.
        cohort: Cohort processed.
        session: Session processed.
        n_successful: Number of subjects processed successfully.
        failed: IDs of subjects that failed.
        step: Optional sub-step label (e.g. DWI or BOLD).
    """
    entry = {"script": script, "cohort": cohort, "session": session, "step": step,
             "n_successful": n_successful, "failed": list(failed)}
    SUMMARY_PATH.parent.mkdir(exist_ok=True)
    with SUMMARY_PATH.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def format_summary() -> str:
    """Build the end-of-pipeline summary from the recorded script results."""
    lines = ["", "=" * 60, f"PIPELINE SUMMARY ({ATLAS})", "=" * 60]
    if not SUMMARY_PATH.exists():
        lines.append("No results recorded (no subjects needed processing).")
        return "\n".join(lines)

    entries = [json.loads(line) for line in SUMMARY_PATH.read_text().splitlines() if line.strip()]
    current_script = None
    total_successful = 0
    total_failed = 0
    for e in entries:
        if e["script"] != current_script:
            current_script = e["script"]
            lines.append(f"\n{current_script}")
        label = f"{e['cohort']} {e['session']}" + (f" {e['step']}" if e["step"] else "")
        lines.append(f"  {label}: {e['n_successful']} successful, {len(e['failed'])} failed")
        if e["failed"]:
            lines.append(f"    Failed: {', '.join(e['failed'])}")
        total_successful += e["n_successful"]
        total_failed += len(e["failed"])

    lines.append(f"\nTotal across all scripts: {total_successful} successful, {total_failed} failed")
    lines.append("=" * 60)
    return "\n".join(lines)


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
    log_path = base_dir / f"nohup_fsaverage_{ATLAS}.out"

    # Clear results from any previous run so the summary only covers this one
    SUMMARY_PATH.unlink(missing_ok=True)

    start_time = datetime.datetime.now()
    print(f"Starting {ATLAS} pipeline at {start_time}")

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
    summary = format_summary()
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
