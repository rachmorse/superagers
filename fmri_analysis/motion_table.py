"""Supplementary table of head motion at baseline and follow-up by superager group."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analyses"))
from demographics_table import DATA_DIR, P_FLOOR, SESSIONS, load_data, summary_row

BBHI_MRI = Path("/pool/guttmann/institut/BBHI/MRI")
SENIOR_MRI = Path("/pool/guttmann/institut/UB/Superagers/MRI")
BBHI_MIN_ID = 6000  # To separate BBHI IDs from BBHI senior

# Rows per timepoint: (label, column stem)
MOTION_ROWS = [
    ("Mean framewise displacement (mm)", "mean_fd"),
    ("Volumes scrubbed (%)", "pct_scrubbed"),
]
GROUP_LABELS = {"Whole cohort": "Whole cohort", "Non-superager": "Non-superagers", "Superager": "Superagers"}


def native_t1_dir(sub: str, ses: str) -> Path:
    """Return the resting-state native T1 folder for a subject/session.

    Args:
        sub: Cleaned subject ID.
        ses: Session label (e.g. 'ses-01').

    Returns:
        Folder holding framewise_displ.txt and the scrubbing JSON.
    """
    if int(sub) > BBHI_MIN_ID:
        preproc = "resting_preproc_fs6-recon" if ses == "ses-01" else "resting_preproc_fs6-recon_tp2"
        return BBHI_MRI / "processed_data" / preproc / f"sub-{sub}" / "native_T1"
    return SENIOR_MRI / "resting_preproc_fs6-recon" / f"sub-{sub}" / ses / "native_T1"


def motion(sub: str, ses: str) -> tuple[float, float]:
    """Get mean FD and percentage of scrubbed volumes (FD > 0.5 mm).

    Args:
        sub: Cleaned subject ID.
        ses: Session label (e.g. 'ses-01').

    Returns:
        Mean FD (mm) and percentage of volumes scrubbed.
    """
    d = native_t1_dir(sub, ses)
    fd = pd.read_csv(d / "framewise_displ.txt")["FramewiseDisplacement"]
    scrub = json.loads(next(d.glob("*scrubbed*.json")).read_text())
    return fd.mean(), scrub["NumFramesScrubbed"] / scrub["TotalFrames"] * 100


def load_motion_data() -> pd.DataFrame:
    """Add head motion measures to the final analysis sample.

    Returns:
        Per-subject table from load_data with mean_fd and pct_scrubbed _1/_2 columns.
    """
    df = load_data()
    for tp, ses in SESSIONS.items():
        df[f"mean_fd_{tp}"], df[f"pct_scrubbed_{tp}"] = zip(*[motion(s, ses) for s in df.index])
    return df


def build_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build the head motion table.

    Args:
        df: Per-subject table from load_motion_data.

    Returns:
        Table with whole cohort and group columns (n in headers), and p-values.
    """
    groups = {
        "Whole cohort": df,
        "Non-superager": df[df["group"] == "Non-superager"],
        "Superager": df[df["group"] == "Superager"],
    }
    rows = []
    for tp, section in [("1", "Baseline"), ("2", "Follow-up")]:
        rows.append({"Variable": section})
        rows += [summary_row(groups, label, f"{stem}_{tp}", "sd", True) for label, stem in MOTION_ROWS]

    table = pd.DataFrame(rows)[["Variable", *groups, "p_value"]]
    table["p_value"] = table["p_value"].map(lambda p: "" if pd.isna(p) else "<0.0001" if p < P_FLOOR else f"{p:.4f}")
    headers = {g: f"{GROUP_LABELS[g]} (n={len(d)})" for g, d in groups.items()}
    return table.fillna("").rename(columns={**headers, "p_value": "p-value"})


if __name__ == "__main__":
    table = build_table(load_motion_data())
    print(table.to_string(index=False))
    out = DATA_DIR / "motion_table.csv"
    table.to_csv(out, index=False)
    print(f"\nSaved to {out}")
