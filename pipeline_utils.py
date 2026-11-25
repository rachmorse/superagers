"""Helpers shared by the pipeline scripts.

The run_*_pipeline.py scripts run each step script with run_script. Each step script records its success/failure
counts per cohort and session, and the run_*_pipeline.py script prints them as one summary at the end of the run.
"""

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

from config import ATLAS

logger = logging.getLogger(__name__)


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


def run_step(cmd: List[str], description: str) -> None:
    """Run a command, logging its error output if it fails.

    Args:
        cmd: Command to run.
        description: What the command does, for the error message (e.g. "in mri_binarize").

    Raises:
        subprocess.CalledProcessError: If the command fails.
    """
    try:
        subprocess.run(cmd, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error {description}: {e.stderr}")
        raise


def summary_path(script: str, atlas: str) -> Path:
    """Get the run summary file for the pipeline folder a script belongs to.

    Args:
        script: Path to any script in the pipeline folder (usually __file__).
        atlas: Atlas the run uses.

    Returns:
        Path to the summary file, one per pipeline folder and atlas.
    """
    return Path(script).resolve().parent / "pipeline_summaries" / f"pipeline_summary_{atlas}.jsonl"


def record_summary(
    script: str, cohort: str, session: str, n_successful: int, failed: List[str], *, step: str = ""
) -> None:
    """Record the success/failure counts for one cohort and session of a script.

    Args:
        script: Path to the script reporting the results (usually __file__).
        cohort: Cohort processed.
        session: Session processed.
        n_successful: Number of subjects processed successfully.
        failed: IDs of subjects that failed.
        step: Optional sub-step label (e.g. DWI or BOLD).
    """
    entry = {
        "script": Path(script).name,
        "cohort": cohort,
        "session": session,
        "step": step,
        "n_successful": n_successful,
        "failed": list(failed),
    }
    path = summary_path(script, ATLAS)
    path.parent.mkdir(exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def format_summary(script: str, atlas: str) -> str:
    """Build the end-of-pipeline summary from the recorded script results.

    Args:
        script: Path to the run_*_pipeline.py script (usually __file__).
        atlas: Atlas the run used.

    Returns:
        The summary text.
    """
    path = summary_path(script, atlas)
    lines = ["", "=" * 60, f"PIPELINE SUMMARY ({atlas})", "=" * 60]
    if not path.exists():
        lines.append("No results recorded (no subjects needed processing).")
        return "\n".join(lines)

    entries = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
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
