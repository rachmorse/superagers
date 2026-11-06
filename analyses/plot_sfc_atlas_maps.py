#!/usr/bin/env python3
"""Supplementary figure: SFC difference maps for Schaefer 200 and 400 with one row per atlas."""

from pathlib import Path

import matplotlib.pyplot as plt
import plot_sfc_difference as psd

OUTPUT_PATH = Path(__file__).parent / "figures" / "sfc_diff_atlas_maps.png"
ROW_LABELS = {"schaefer200": "Schaefer 200", "schaefer400": "Schaefer 400"}


def main() -> None:
    """Stack both atlases' brain rows above a shared colorbar and save the figure."""
    # Layout, in inches
    fig_w = 15.0  # Same width as main text fig
    label_w = 1.8  # Atlas labels left of each row
    row_gap = 0.3  # Between the two brain rows
    cbar_space = 0.9  # Colorbar plus its label under the rows

    views = {atlas: psd.brain_views(atlas) for atlas in ROW_LABELS}
    row_w = fig_w - label_w
    sizes = {atlas: psd.brain_row_sizes(images, row_w) for atlas, images in views.items()}
    row_hs = [max(h for _, h in s) for s in sizes.values()]
    fig_h = sum(row_hs) + row_gap * (len(row_hs) - 1) + cbar_space
    fig = plt.figure(figsize=(fig_w, fig_h))

    # Rows top to bottom, each labeled on the left
    top = fig_h
    for (atlas, label), row_h in zip(ROW_LABELS.items(), row_hs, strict=True):
        bottom = top - row_h
        psd.draw_brain_row(fig, views[atlas], sizes[atlas], label_w, bottom)
        fig.text((label_w - 0.2) / fig_w, (bottom + row_h / 2) / fig_h, label, ha="right",
                 va="center", fontsize=16, color=psd.INK)
        top = bottom - row_gap

    psd.add_colorbar(fig, label_w + row_w / 2, cbar_space - 0.35)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved: {OUTPUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
