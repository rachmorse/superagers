"""Sample characteristics table at baseline and follow-up by superager group."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ATLAS = "schaefer200"  # Same sample for Schaefer 200 and 400
DATA_DIR = Path("/home/rachel/Desktop/data")
RAW_DIR = DATA_DIR / "raw"
BBHI_MRI = Path("/pool/guttmann/institut/BBHI/MRI")
SENIOR_MRI = Path("/pool/guttmann/institut/UB/Superagers/MRI")
SESSIONS = {"1": "ses-01", "2": "ses-02"}
BBHI_MIN_ID = 6000  # To separate BBHI IDs from BBHI senior
P_FLOOR = 1e-4  # Smaller p-values are shown as <0.0001

# Rows per timepoint, grouped by subsection: (label, column stem, format, run t-test)
# Stems get _1/_2 appended per timepoint
TIMEPOINT_SUBSECTIONS = {
    "Cognition": [
        ("MMSE", "mmse", "range", True),
        ("RAVLT delayed recall†", "delayed_recall_raw", "sd", True),
        ("Trail Making Test B (s)†", "tmt_b_raw", "sd", True),
        ("Semantic fluency†", "sem_fluency_raw", "sd", True),
        ("Backward digit span†", "inverse_digits_raw", "sd", True),
    ],
    "fMRI": [
        ("Mean framewise displacement (mm)", "mean_fd", "sd", True),
        ("Volumes scrubbed (%)", "pct_scrubbed", "sd", True),
    ],
}


def clean_id(x: object) -> str:
    """Strip a 'sub-' prefix and return the bare ID string.

    Args:
        x: Subject identifier, with or without a 'sub-' prefix.

    Returns:
        The numeric ID as a string.
    """
    return str(x).replace("sub-", "")


def analysis_ids(sup: pd.DataFrame) -> set[str]:
    """Reproduce analysis_ids from analyses.R: the final analysis sample.

    Subjects must be present in the cleaned table, both SFC files and the weighted ROI file,
    with complete weighted SFC at both timepoints.

    Args:
        sup: The cleaned superager table (must have an 'id' column).

    Returns:
        Cleaned subject IDs in the final analysis sample.
    """
    s1 = pd.read_csv(DATA_DIR / f"all_sfc_data_ses-01_{ATLAS}.csv")
    s2 = pd.read_csv(DATA_DIR / f"all_sfc_data_ses-02_{ATLAS}.csv")
    w = pd.read_csv(DATA_DIR / f"weighted_global_roi_averages_{ATLAS}.csv")
    # Rename to <variable>_<timepoint>, as in analyses.R
    w.columns = w.columns.str.replace(r"^w([12])_(.*)$", r"\2_\1", regex=True).str.replace(
        r"^(.*)_tp([12])_(.*)$", r"\1_\3_\2", regex=True
    )
    w_id = next(c for c in w.columns if c in ("subject_id", "id"))
    w["idc"] = w[w_id].map(clean_id)

    in_all = (
        set(sup["id"].map(clean_id))
        & set(s1.iloc[:, 0].map(clean_id))
        & set(s2.iloc[:, 0].map(clean_id))
        & set(w["idc"])
    )
    w = w[w["idc"].isin(in_all)]
    complete = set(w.loc[w[["sfc_weighted_mean_1", "sfc_weighted_mean_2"]].notna().all(axis=1), "idc"])
    return in_all & complete


def load_mmse(path: str, id_col: str, mmse_col: str) -> pd.Series:
    """Read MMSE scores from a raw data file.

    Args:
        path: File name in the raw data folder (.csv or .xlsx).
        id_col: Subject ID column.
        mmse_col: MMSE column.

    Returns:
        MMSE scores indexed by cleaned subject ID.
    """
    f = RAW_DIR / path
    df = pd.read_excel(f) if f.suffix == ".xlsx" else pd.read_csv(f, low_memory=False)
    s = pd.Series(pd.to_numeric(df[mmse_col], errors="coerce").values, index=df[id_col].map(clean_id))
    return s.dropna().groupby(level=0).first()


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


def load_data() -> pd.DataFrame:
    """Build the per-subject table for the final analysis sample.

    Returns:
        One row per subject with _1/_2 columns for each timepoint.
    """
    sup = pd.read_csv(DATA_DIR / "superager.csv")
    sup["id"] = sup["id"].map(clean_id)
    df = sup[sup["id"].isin(analysis_ids(sup))].set_index("id")
    # Rename w1_x/w2_x to x_1/x_2
    df.columns = df.columns.str.replace(r"^w([12])_(.*)$", r"\2_\1", regex=True)

    mmse_1 = pd.concat([
        load_mmse("bbhi senior data.xlsx", "ID", "MMSE"),
        load_mmse("BBHI Data Timept1 NPS.csv", "id", "w1_mmse"),
    ])
    mmse_2 = pd.concat([
        load_mmse("BBHI Data Timept2 NPS.csv", "id", "mmse"),
        load_mmse("Ministerio2024Wave2_DATA_2026-01-20_1245.csv", "record_id_np_w2", "mmse_w2"),
    ])
    df["mmse_1"] = mmse_1.groupby(level=0).first().reindex(df.index)
    df["mmse_2"] = mmse_2.groupby(level=0).first().reindex(df.index)

    for tp, ses in SESSIONS.items():
        df[f"mean_fd_{tp}"], df[f"pct_scrubbed_{tp}"] = zip(*[motion(s, ses) for s in df.index])
    df["followup_duration"] = df["age_2"] - df["age_1"]
    df["group"] = np.where(df["superager_long"] == 1, "Superager", "Non-superager")
    return df


def fmt(x: pd.Series, style: str) -> str:
    """Format a column summary as mean (SD) or mean (range).

    Args:
        x: Values to summarise.
        style: 'sd' or 'range'.

    Returns:
        Formatted summary.
    """
    x = x.dropna()
    if style == "range":
        lo, hi = (f"{v:.0f}" if float(v).is_integer() else f"{v:.1f}" for v in (x.min(), x.max()))
        return f"{x.mean():.2f} ({lo}–{hi})"
    return f"{x.mean():.2f} ({x.std():.2f})"


def summary_row(groups: dict[str, pd.DataFrame], label: str, col: str, style: str, test: bool) -> dict:
    """Summarise one variable for each group with a Welch t-test.

    Args:
        groups: Data for each table column, keyed by column name.
        label: Row label.
        col: Variable to summarise.
        style: 'sd' or 'range'.
        test: Whether to compare superagers and non-superagers.

    Returns:
        Table row.
    """
    p = np.nan
    if test:
        p = stats.ttest_ind(
            groups["Superager"][col].dropna(), groups["Non-superager"][col].dropna(), equal_var=False
        ).pvalue
    return {"Variable": label, **{g: fmt(d[col], style) for g, d in groups.items()}, "p_value": p}


def build_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build the sample characteristics table.

    Args:
        df: Per-subject table from load_data.

    Returns:
        Table with whole cohort and group columns, and p-values.
    """
    groups = {
        "Whole cohort": df,
        "Non-superager": df[df["group"] == "Non-superager"],
        "Superager": df[df["group"] == "Superager"],
    }
    sex = df["sex"].str.lower()
    rows = [
        {"Variable": "N", **{g: str(len(d)) for g, d in groups.items()}},
        {
            "Variable": "Female, n (%)",
            **{
                g: f"{(d['sex'].str.lower() == 'female').sum()} ({(d['sex'].str.lower() == 'female').mean() * 100:.1f}%)"
                for g, d in groups.items()
            },
            "p_value": stats.chi2_contingency(pd.crosstab(df["group"], sex))[1],
        },
        summary_row(groups, "Education (years)", "YoE", "sd", True),
    ]

    for tp, section in [("1", "Baseline"), ("2", "Follow-up")]:
        section_rows = [{"Variable": section}, summary_row(groups, "Age", f"age_{tp}", "range", True)]
        if tp == "2":
            section_rows.append(summary_row(groups, "Follow-up duration (years)", "followup_duration", "sd", True))
        for subsection, variables in TIMEPOINT_SUBSECTIONS.items():
            section_rows.append({"Variable": subsection})
            for label, stem, style, test in variables:
                section_rows.append(summary_row(groups, label, f"{stem}_{tp}", style, test))
        # Section column lets results.Rmd look up cells, e.g. baseline vs follow-up age
        rows += [{"Section": section, **row} for row in section_rows]

    table = pd.DataFrame(rows)[["Section", "Variable", *groups, "p_value"]]
    table["p_value"] = table["p_value"].map(lambda p: "" if pd.isna(p) else "<0.0001" if p < P_FLOOR else f"{p:.4f}")
    return table.fillna("")


if __name__ == "__main__":
    table = build_table(load_data())
    print(table.to_string(index=False))
    out = DATA_DIR / "demographics_table.csv"
    table.to_csv(out, index=False)
    print(f"\nSaved to {out}")
