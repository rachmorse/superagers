"""Flow diagram of superager classification at baseline and follow-up."""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

from demographics_table import DATA_DIR, analysis_ids, clean_id

CRITERIA_TEXT = (
    "Superager criteria applied at baseline and follow-up:\n"
    "Memory: RAVLT delayed recall ≥ 12 (young-adult norms for ages 16–29)\n"
    "Non-memory: age- and education-adjusted (no lower than 1SD below the mean)\n"
    "on TMT-B, semantic fluency, and backward digit span"
)


def compute_ns() -> dict:
    """Compute the counts for each box in the classification flow diagram.

    Returns:
        Counts keyed by box.
    """
    sup = pd.read_csv(DATA_DIR / "superager.csv")
    sup = sup[sup["id"].map(clean_id).isin(analysis_ids(sup))]
    bl_sa, bl_no = sup[sup["superager_tp1"] == 1], sup[sup["superager_tp1"] == 0]
    return {
        "total": len(sup),
        "bl_sa": len(bl_sa),
        "bl_no": len(bl_no),
        "sa_sa": int((bl_sa["superager_tp2"] == 1).sum()),
        "sa_no": int((bl_sa["superager_tp2"] == 0).sum()),
        "no_sa": int((bl_no["superager_tp2"] == 1).sum()),
        "no_no": int((bl_no["superager_tp2"] == 0).sum()),
        "final_sa": int((sup["superager_long"] == 1).sum()),
        "final_no": int((sup["superager_long"] == 0).sum()),
    }


def draw_flow(ns: dict) -> None:
    """Draw and save the classification flow diagram.

    Args:
        ns: Counts as returned by compute_ns.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.7))
    ax.set_xlim(0, 8)
    ax.set_ylim(1.2, 7.95)
    ax.axis("off")

    box_style = dict(linewidth=0.8, edgecolor="black", facecolor="white")
    line_kw = dict(color="black", lw=0.8)
    arrow_kw = dict(arrowstyle="->", shrinkA=0, **line_kw)
    fontsize = 8

    def box(cx: float, cy: float, w: float, h: float, text: str) -> tuple[float, float, float]:
        """Draw a centred text box.

        Args:
            cx: Box centre x.
            cy: Box centre y.
            w: Box width.
            h: Box height.
            text: Box label.

        Returns:
            Centre x, centre y and height, for connecting arrows.
        """
        ax.add_patch(mpatches.FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="square,pad=0", **box_style))
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize, multialignment="center")
        return cx, cy, h

    def elbow(src: tuple[float, float, float], dst: tuple[float, float, float]) -> None:
        """Connect two boxes: down from src, across, then an arrow down into dst.

        Args:
            src: Source box as returned by box.
            dst: Destination box as returned by box.
        """
        y0, y1 = src[1] - src[2] / 2, dst[1] + dst[2] / 2
        mid = (y0 + y1) / 2
        ax.plot([src[0], src[0], dst[0]], [y0, mid, mid], **line_kw)
        ax.annotate("", xy=(dst[0], y1), xytext=(dst[0], mid), arrowprops=arrow_kw)

    h = 0.75
    xs = [1.0, 3.0, 5.0, 7.0]
    w = 1.9
    met, not_met = "met\nsuperager criteria", "did not meet\nsuperager criteria"

    top = box(4.0, 7.5, 3.0, h, f"Final analysis sample\n$n$ = {ns['total']}")
    bl_sa = box(2.0, 6.0, 2.6, h, f"Baseline: {met}\n$n$ = {ns['bl_sa']}")
    bl_no = box(6.0, 6.0, 2.6, h, f"Baseline: {not_met}\n$n$ = {ns['bl_no']}")
    fu = [
        box(xs[0], 4.5, w, h, f"Follow-up: {met}\n$n$ = {ns['sa_sa']}"),
        box(xs[1], 4.5, w, h, f"Follow-up: {not_met}\n$n$ = {ns['sa_no']}"),
        box(xs[2], 4.5, w, h, f"Follow-up: {met}\n$n$ = {ns['no_sa']}"),
        box(xs[3], 4.5, w, h, f"Follow-up: {not_met}\n$n$ = {ns['no_no']}"),
    ]
    final_sa = box(xs[0], 3.0, w, h, f"Superagers\n$n$ = {ns['final_sa']}")
    final_no = box(5.0, 3.0, 3.0, h, f"Non-superagers\n$n$ = {ns['final_no']}")

    for parent, children in [(top, [bl_sa, bl_no]), (bl_sa, fu[:2]), (bl_no, fu[2:])]:
        for child in children:
            elbow(parent, child)
    elbow(fu[0], final_sa)

    # Merge the three non-superager branches into one arrow
    y0, y1 = fu[1][1] - h / 2, final_no[1] + h / 2
    mid = (y0 + y1) / 2
    for child in fu[1:]:
        ax.plot([child[0], child[0]], [y0, mid], **line_kw)
    ax.plot([fu[1][0], fu[3][0]], [mid, mid], **line_kw)
    ax.annotate("", xy=(final_no[0], y1), xytext=(final_no[0], mid), arrowprops=arrow_kw)

    # Criteria box spanning the figure width
    ax.add_patch(mpatches.FancyBboxPatch((0.05, 1.25), 7.9, 1.1, boxstyle="square,pad=0", linestyle="--", **box_style))
    ax.text(0.2, 1.8, CRITERIA_TEXT, ha="left", va="center", fontsize=fontsize)

    out = Path(__file__).resolve().parent / "figures" / "classification_flow_figure.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved to {out}")


if __name__ == "__main__":
    print("Computing classification counts...")
    ns = compute_ns()
    for k, v in ns.items():
        print(f"  {k}: {v}")
    print("\nDrawing classification flow diagram...")
    draw_flow(ns)
