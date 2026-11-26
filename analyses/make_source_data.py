"""Write the Source Data excel file for the main figures with one sheet per figure.

Values come from the same result files the figure scripts use.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO), str(REPO / "en_svm_models"), str(REPO / "structure_function_coupling")]

import plot_group_connectivity_figure as pgc  # noqa: E402
import plot_sfc_difference as psd  # noqa: E402
import plot_sfc_em_cortex as pse  # noqa: E402
from plot_feature_importance import (  # noqa: E402
    FEATURE_TYPE_STYLE,
    MODEL_TITLES,
    NETWORK_LABELS,
    TOP_N,
    load_and_prepare,
)

from config import ML_ATLAS as EN_ATLAS  # noqa: E402
from config import ML_RESULTS_DIR as RESULTS_DIR  # noqa: E402

OUTPUT_PATH = Path(__file__).parent / "figures" / "source_data.xlsx"
EN_MODELS = {"A": "t1", "B": "t1_slope"}  # Figure 2 panel -> feature set
PARCELLATION_LABELS = {"schaefer200": "Schaefer 200", "schaefer400": "Schaefer 400"}
NETWORK_NAMES = {  # Figure 3 label -> full name
    "Global": "Global",
    "Heteromodal": "Heteromodal",
    "DMN": "Default mode network",
    "ECN": "Executive control network",
    "SN": "Salience network",
    "Sensory": "Sensory",
}

Block = tuple[str, pd.DataFrame, bool]  # (title, table, write index)


def figure_1() -> list[Block]:
    """Group-mean structure-function coupling, structural and functional connectivity at baseline.

    Returns:
        One block per panel.
    """
    sc, fc, sfc = pgc.load_or_compute_group_means(Path(pgc.__file__).parent / f"group_means_{pgc.ATLAS}.npz")
    np.fill_diagonal(sc, 0)  # As plotted
    first_file = next(
        pgc.sfc_path(sub, ses) for ses in pgc.SESSIONS for sub in pgc.SUBJECTS if pgc.sfc_path(sub, ses).exists()
    )
    regions = pd.Index(pd.read_csv(first_file)["ROI_name"], name="Region")

    def matrix(values: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(values, index=regions, columns=regions)

    return [
        (
            "Structure-function coupling (mean Pearson r per region)",
            pd.DataFrame({"Region": regions, "Structure-function coupling (r)": sfc}),
            False,
        ),
        ("Structural connectivity (mean normalized white matter tracts, region × region)", matrix(sc), True),
        ("Functional connectivity (mean Fisher-z correlation, region × region)", matrix(fc), True),
    ]


def figure_2() -> list[Block]:
    """Top elastic net structure-function coupling features by permutation importance.

    Returns:
        One block per model panel.
    """
    blocks = []
    for letter, features in EN_MODELS.items():
        is_slope = features == "t1_slope"
        df = load_and_prepare(
            RESULTS_DIR / f"{EN_ATLAS}_SFC_ROI_{features}_long_include_feature_importance.csv", is_slope
        )
        # Same selection as plot_roi_panel
        top = df[df["perm_importance_mean"] > 0].sort_values("perm_importance_mean").tail(TOP_N).iloc[::-1]
        region = top["feature"].str.replace(r"_(1|slope)$", "", regex=True) if is_slope else top["feature"]
        table = pd.DataFrame({
            "Rank": range(1, len(top) + 1),
            "Region (figure label)": top["label"],
            "Schaefer atlas region": region,
            "Network": top["network"].map(NETWORK_LABELS),
            "Feature type": top["feature_type"].map(lambda t: FEATURE_TYPE_STYLE[t]["label"]),
            "Permutation importance": top["perm_importance_mean"],
        })
        blocks.append((f"Panel {letter}: {MODEL_TITLES[features]}", table, False))
    return blocks


def figure_3() -> list[Block]:
    """Superager minus typical older adult coupling map and linear mixed-effects model results.

    Returns:
        One block per panel.
    """
    diff = pd.read_csv(psd.average_dir(pse.MAIN_ATLAS) / f"{psd.AVERAGE_NAME}.csv")
    blocks = [
        (
            f"Panel A: Structure-function coupling difference, superagers minus typical older adults "
            f"({PARCELLATION_LABELS[pse.MAIN_ATLAS]}, mean of both timepoints)",
            pd.DataFrame({
                "Region": diff["ROI_name"],
                "Structure-function coupling difference (Fisher z)": diff[psd.AVERAGE_NAME],
            }),
            False,
        )
    ]

    lme = pd.read_csv(pse.LME_CSV)
    for letter, (model, (title, *_)) in zip("BC", pse.OUTCOMES.items(), strict=True):
        df = lme[(lme["model"] == model) & lme["network"].isin(pse.NETWORK_MAP) & lme["atlas"].isin(pse.ATLAS_STYLE)]
        df = df.assign(network=df["network"].map(pse.NETWORK_MAP))
        df = df.assign(order=df["network"].map(pse.NETWORKS.index)).sort_values(["order", "atlas"])  # Figure order
        table = pd.DataFrame({
            "Network": df["network"].map(NETWORK_NAMES),
            "Parcellation": df["atlas"].map(PARCELLATION_LABELS),
            "Standardized β": df["coef"],
            "95% CI lower": df["ci_lo"],
            "95% CI upper": df["ci_hi"],
            "p (uncorrected)": df["p_value"],
            "p (FDR-corrected)": df["p_fdr"],
        })
        blocks.append((f"Panel {letter}: {title.split(')', 1)[1].strip()}", table, False))
    return blocks


SHEETS = {  # Sheet name -> (figure title, block builder)
    "Figure 1": (
        "Figure 1. Structural connectivity, functional connectivity, and structure-function coupling methods",
        figure_1,
    ),
    "Figure 2": ("Figure 2. Structure-function coupling feature importance in elastic net models", figure_2),
    "Figure 3": ("Figure 3. Structure-function coupling, episodic memory, and superager status analyses", figure_3),
}


def write_sheet(writer: pd.ExcelWriter, name: str, heading: str, blocks: list[Block]) -> None:
    """Write a figure's blocks down one sheet, each under a bold title.

    Args:
        writer: Open Excel writer.
        name: Sheet name.
        heading: Figure title for the first row.
        blocks: (title, table, write index) per panel.
    """
    row = 2  # Below the heading
    for title, table, index in blocks:
        table.to_excel(writer, sheet_name=name, startrow=row + 1, index=index)
        writer.sheets[name].cell(row=row + 1, column=1, value=title).font = Font(bold=True)
        row += len(table) + 4  # Title, header, and a blank row

    ws = writer.sheets[name]
    ws.cell(row=1, column=1, value=heading).font = Font(bold=True, size=13)
    ws.column_dimensions["A"].width = 40
    for col in range(2, ws.max_column + 1):
        ws.column_dimensions[get_column_letter(col)].width = 22


def main() -> None:
    """Build every figure sheet and save the workbook."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
        for name, (heading, build) in SHEETS.items():
            write_sheet(writer, name, heading, build())
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
