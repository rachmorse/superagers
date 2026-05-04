import shutil
import sys
from pathlib import Path
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.image import imread
import yabplot as yab

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analyses"))
from plot_sfc_difference import crop_colorbar, trim_whitespace

FEATURE_IMPORTANCE_T1_CSV = Path(
    "/home/rachel/Desktop/superagers/elastic_net/results/SFC_ROI_t1_long_include_feature_importance.csv"
)
FEATURE_IMPORTANCE_SLOPE_CSV = Path(
    "/home/rachel/Desktop/superagers/elastic_net/results/SFC_ROI_t1_slope_long_include_feature_importance.csv"
)
OUTPUT_PATH = Path(
    "/home/rachel/Desktop/superagers/elastic_net/figures_tables/feature_importance_brain.png"
)
TEMP_PATH = Path(
    "/home/rachel/Desktop/superagers/elastic_net/figures_tables/_brain_temp"
)

# Sub-region abbreviation to full name from Schaefer 200 atlas)
ROI_ABBREV = {
    "AntTemp":    "Anterior temporal",
    "Aud":        "Auditory",
    "Cent":       "Central",
    "Cinga":      "Cingulate anterior",
    "Cingm":      "Mid-cingulate",
    "Cingp":      "Cingulate posterior",
    "Cing":       "Cingulate",
    "ExStrInf":   "Extrastriate inferior",
    "ExStrSup":   "Extrastriate superior",
    "ExStr":      "Extrastriate",
    "FEF":        "Frontal eye fields",
    "FPole":      "Frontal pole",
    "FrMed":      "Frontal medial",
    "FrOperIns":  "Frontal operculum insula",
    "FrOper":     "Frontal operculum",
    "IFG":        "Inferior frontal gyrus",
    "Ins":        "Insula",
    "IPL":        "Inferior parietal lobule",
    "IPS":        "Intraparietal sulcus",
    "Med":        "Medial",
    "OFC":        "Orbital frontal cortex",
    "ParMed":     "Parietal medial",
    "ParOcc":     "Parietal occipital",
    "ParOper":    "Parietal operculum",
    "Par":        "Parietal",
    "pCunPCC":    "Precuneus/PCC",
    "pCun":       "Precuneus",
    "PFCdPFCm":   "Dorsal medial PFC",
    "PFCd":       "Dorsal PFC",
    "PFCld":      "Lateral dorsal PFC",
    "PFClv":      "Lateral ventral PFC",
    "PFCl":       "Lateral PFC",
    "PFCmp":      "Medial posterior PFC",
    "PFCm":       "Medial PFC",
    "PFCv":       "Ventral PFC",
    "PFC":        "Prefrontal cortex",
    "PHC":        "Parahippocampal cortex",
    "Post":       "Posterior",
    "PostC":      "Post-central",
    "PrCd":       "Precentral dorsal",
    "PrCv":       "Precentral ventral",
    "PrC":        "Precentral",
    "RSC":        "Retrosplenial cortex",
    "Rsp":        "Retrosplenial",
    "S2":         "S2",
    "SPL":        "Superior parietal lobule",
    "ST":         "Superior temporal",
    "Striate":    "Striate",
    "StriCal":    "Striate calcarine",
    "TempOccPar": "Temporal occipital parietal",
    "TempOcc":    "Temporal occipital",
    "TempPar":    "Temporal parietal",
    "TempPole":   "Temporal pole",
    "Temp":       "Temporal",
}

# Hatch style and legend label per feature type
FEATURE_TYPE_STYLE = {
    "baseline": {"hatch": "",    "label": "Baseline"},
    "slope":    {"hatch": "///", "label": "Annual change"},
}

# Network display names and colours
NETWORK_COLORS = {
    "Default":     "#7B4F9E",   # purple
    "Cont":        "#E8A030",   # orange
    "DorsAttn":    "#2D8B57",   # green
    "SalVentAttn": "#EE6677",   # pink/red
    "SomMot":      "#4477AA",   # blue
    "Limbic":      "#CCBB44",   # yellow-olive
    "Vis":         "#AA3377",   # magenta
    "Subcortical": "#66CCEE",   # light blue
}

NETWORK_LABELS = {
    "Default":     "Default mode",
    "Cont":        "Executive control",
    "DorsAttn":    "Dorsal attention",
    "SalVentAttn": "Salience",
    "SomMot":      "Somatomotor",
    "Limbic":      "Limbic",
    "Vis":         "Visual",
    "Subcortical": "Subcortical",
}

NETWORK_ORDER  = list(NETWORK_COLORS.keys())
VMIN           = 0.5
VMAX           = len(NETWORK_ORDER) + 0.5
CORTICAL_ATLAS = "schaefer200"
STYLE          = "glossy"

SUBCORTICAL_NAME_MAP = {
    "left hippocampus":  "Left-Hippocampus",
    "left amygdala":     "Left-Amygdala",
    "left pallidum":     "Left-Pallidum",
    "left putamen":      "Left-Putamen",
    "left caudate":      "Left-Caudate",
    "left accumbens":    "Left-Accumbens-area",
    "left thalamus":     "Left-Thalamus",
    "right hippocampus": "Right-Hippocampus",
    "right amygdala":    "Right-Amygdala",
    "right pallidum":    "Right-Pallidum",
    "right putamen":     "Right-Putamen",
    "right caudate":     "Right-Caudate",
    "right accumbens":   "Right-Accumbens-area",
    "right thalamus":    "Right-Thalamus",
}


def extract_network(feature: str) -> str:
    """Extract the network name from a feature string.

    Args:
        feature: Raw feature name from the CSV.

    Returns:
        Network key matching NETWORK_COLORS, or "Other".
    """
    if feature.startswith("7Networks_"):
        # e.g. 7Networks_LH_DorsAttn_FEF_1 to DorsAttn
        parts = feature.split("_")
        return parts[2] if len(parts) > 2 else "Other"
    if feature.startswith("Subcortical"):
        return "Subcortical"
    return "Other"


def extract_feature_type(feature: str) -> str:
    """Determine whether a feature is a baseline or slope measurement.

    Args:
        feature: Raw feature name from the CSV.

    Returns:
        "slope" if the feature ends with _slope, otherwise "baseline".
    """
    return "slope" if feature.endswith("_slope") else "baseline"


def make_short_label(feature: str, is_slope_model: bool = False) -> str:
    """Shorten a ROI feature name for use as an axis tick label.

    Includes the trailing parcel index (e.g. ``_7``) in parentheses
    so every label is unambiguous.

    Args:
        feature: Raw feature name from the CSV.
        is_slope_model: If True, strip _slope / _1 suffixes first.

    Returns:
        A shortened, readable label string with parcel index in parentheses.
    """
    if is_slope_model:
        # Strip the _1 (baseline) or _slope suffix before shortening
        if feature.endswith("_slope"):
            feature = feature[: -len("_slope")]
        elif feature.endswith("_1"):
            feature = feature[: -len("_1")]

    if feature.startswith("7Networks_"):
        # Drop the "7Networks" prefix
        parts = feature.split("_", 3)  # ['7Networks', 'LH', 'Network', 'rest']
        hemi = parts[1]  # LH / RH
        network = parts[2]
        rest = parts[3] if len(parts) > 3 else parts[2]

        # Extract parcel index before abbreviation lookup
        parcel_idx = None
        rest_parts = rest.rsplit("_", 1)
        if len(rest_parts) == 2 and rest_parts[1].isdigit():
            parcel_idx = rest_parts[1]
            rest = rest_parts[0]
        elif rest.isdigit():
            # e.g. 7Networks_LH_SomMot_1
            parcel_idx = rest

        raw = ROI_ABBREV.get(rest, rest) if not rest.isdigit() else NETWORK_LABELS.get(network, network)
        full = raw[0].lower() + raw[1:]  # lowercase first letter only, preserve acronyms
        # For vague sub-region names, prepend the network for context
        if rest in {"Med", "Post"}:
            net_label = NETWORK_LABELS.get(network, network).lower()
            label = f"{hemi} {net_label} {full}"
        else:
            label = f"{hemi} {full}"
        if parcel_idx is not None:
            label = f"{label} ({parcel_idx})"
        return label
    if feature.startswith("Subcortical"):
        # "Subcortical 214: Right Thalamus" to "R Thalamus"
        colon = feature.find(":")
        label = feature[colon + 2:] if colon != -1 else feature
        label = label.replace("Right ", "R ").replace("Left ", "L ")
        return label[0] + label[1:].lower()
    return feature


def load_and_prepare(csv_path: Path, is_slope_model: bool = False) -> pd.DataFrame:
    """Load a feature importance CSV and add derived columns.

    Args:
        csv_path: Path to the feature importance CSV.
        is_slope_model: If True, parse _1 / _slope suffixes to populate
            feature_type and strip them from tick labels.

    Returns:
        DataFrame with added columns: network, feature_type, label.
    """
    df = pd.read_csv(csv_path)
    df = df[~df["feature"].str.startswith("cov_")].copy()
    df["network"] = df["feature"].apply(extract_network)
    df["feature_type"] = df["feature"].apply(extract_feature_type) if is_slope_model else "baseline"
    df["label"] = df["feature"].apply(
        lambda f: make_short_label(f, is_slope_model=is_slope_model)
    )
    return df


def _network_legend_handles(present_networks):
    """Build Patch handles for the network colour legend.

    Args:
        present_networks: Iterable of network keys present in the data.

    Returns:
        List of mpatches.Patch objects.
    """
    return [
        mpatches.Patch(color=NETWORK_COLORS[n], label=NETWORK_LABELS.get(n, n))
        for n in NETWORK_COLORS if n in present_networks
    ]


def plot_roi_panel(ax: plt.Axes, df: pd.DataFrame, panel_label: str,
                   is_slope_model: bool = False) -> None:
    """Plot the top 20 ROIs by permutation importance.

    Bars are coloured by network. For the slope model, slope features are
    additionally hatched.

    Args:
        ax: Target axes.
        df: Prepared DataFrame from `load_and_prepare`.
        panel_label: Panel label string (e.g. "A" or "B").
        is_slope_model: If True, apply hatching to slope features.
    """
    top20 = (
        df[df["perm_importance_mean"] > 0]
        .sort_values("perm_importance_mean", ascending=True)
        .tail(20)
        .reset_index(drop=True)
    )

    spacing = 0.65  # <1 compresses vertical gap between bars
    bar_h = 0.45

    for i, row in top20.iterrows():
        style = FEATURE_TYPE_STYLE.get(row["feature_type"], FEATURE_TYPE_STYLE["baseline"])
        hatch = style["hatch"] if is_slope_model else ""
        ax.barh(
            i * spacing, row["perm_importance_mean"],
            height=bar_h,
            color=NETWORK_COLORS.get(row["network"], "#AAAAAA"),
            hatch=hatch,
            edgecolor="#444444" if hatch else "none",
            linewidth=0.5,
        )

    ax.set_ylim(-spacing / 2, (len(top20) - 1) * spacing + spacing / 2)
    ax.set_yticks([i * spacing for i in range(len(top20))])
    ax.set_yticklabels(top20["label"], fontsize=9)
    ax.set_xlabel("Permutation importance", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=9)

    handles = _network_legend_handles(top20["network"].unique())
    if is_slope_model:
        type_handles = [
            mpatches.Patch(facecolor="#AAAAAA", hatch=style["hatch"],
                           edgecolor="#444444" if style["hatch"] else "none",
                           label=style["label"])
            for style in FEATURE_TYPE_STYLE.values()
        ]
        handles += [mpatches.Patch(visible=False, label="")] + type_handles

    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.9,
              title="Network", title_fontsize=9, handlelength=1, handleheight=0.8)
    ax.text(-0.55, 1.05, panel_label, transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="top")


def render_brain_images(temp_path: Path) -> list:
    """Render cortical and subcortical network surface images via yabplot.

    Renders three cortical views (left lateral, superior, right lateral) and
    one superior subcortical view. Each region is coloured by its Yeo network
    using NETWORK_COLORS and NETWORK_ORDER. Intermediate PNGs are written to
    temp_path and returned as trimmed numpy arrays.

    Args:
        temp_path: Directory for intermediate PNG files (created if absent).

    Returns:
        List of four numpy image arrays in order: left lateral cortical,
        superior cortical, right lateral cortical, superior subcortical.
    """
    temp_path.mkdir(parents=True, exist_ok=True)
    cmap = ListedColormap([NETWORK_COLORS[n] for n in NETWORK_ORDER])

    yab_regions   = yab.get_atlas_regions(atlas=CORTICAL_ATLAS, category="cortical")
    cortical_data = {}
    for region in yab_regions:
        for part in region.split("_"):
            if part in NETWORK_COLORS:
                cortical_data[region] = float(NETWORK_ORDER.index(part) + 1)
                break

    cort_kw = dict(
        data=cortical_data, atlas=CORTICAL_ATLAS,
        vminmax=[VMIN, VMAX], cmap=cmap,
        bmesh="midthickness", figsize=(600, 600),
        style=STYLE, display_type="object",
    )

    arrs = []
    for view in ["left_lateral", "superior", "right_lateral"]:
        png = temp_path / f"cort_{view}.png"
        print(f"  Rendering cortical ({view})…")
        yab.plot_cortical(views=[view], export_path=str(png), **cort_kw)
        arrs.append(trim_whitespace(crop_colorbar(imread(str(png)))))

    subcort_val      = float(NETWORK_ORDER.index("Subcortical") + 1)
    valid_yab_names  = set(yab.get_atlas_regions(atlas="aseg", category="subcortical"))
    subcortical_data = {
        n: subcort_val for n in SUBCORTICAL_NAME_MAP.values()
        if n in valid_yab_names
    }
    png = temp_path / "subcort_superior.png"
    print("  Rendering subcortical (superior)…")
    yab.plot_subcortical(
        views=["superior"], export_path=str(png),
        data=subcortical_data, atlas="aseg",
        vminmax=[VMIN, VMAX], cmap=cmap,
        figsize=(600, 600), style=STYLE,
        bmesh_color="lightgray", bmesh_alpha=0.15,
        display_type="object",
    )
    arrs.append(trim_whitespace(crop_colorbar(imread(str(png)))))

    return arrs


def main():
    """Generate and save the combined feature importance and brain surface figure.

    Panel A (baseline model) and B (baseline + annual change model) are stacked
    vertically. Panel C shows cortical and anterior subcortical brain views
    arranged horizontally below, centred under A and B.
    """
    df_t1    = load_and_prepare(FEATURE_IMPORTANCE_T1_CSV,    is_slope_model=False)
    df_slope = load_and_prepare(FEATURE_IMPORTANCE_SLOPE_CSV, is_slope_model=True)

    brain_arrs = render_brain_images(TEMP_PATH)

    fig_w     = 7.0
    brain_h   = 2
    row_gap   = 0
    brain_gap = 0.2
    fig_h     = 9.0 

    # Scale all brain images to brain_h; shrink proportionally if too wide
    max_arr_h     = max(a.shape[0] for a in brain_arrs)
    h_scale       = 0.6 * (brain_h / max_arr_h)
    b_widths      = [a.shape[1] * h_scale for a in brain_arrs]
    b_heights     = [a.shape[0] * h_scale for a in brain_arrs]
    total_brain_w = sum(b_widths) + brain_gap * (len(brain_arrs) - 1)
    if total_brain_w > fig_w:
        s             = fig_w / total_brain_w
        b_widths      = [w * s for w in b_widths]
        b_heights     = [h * s for h in b_heights]
        total_brain_w = fig_w

    # Panels A (top) and B (middle)
    brain_frac = (brain_h + row_gap) / fig_h + 0.02
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(fig_w, fig_h))
    plot_roi_panel(ax_a, df_t1,    "A", is_slope_model=False)
    plot_roi_panel(ax_b, df_slope, "B", is_slope_model=True)
    ax_a.set_title("Baseline model",                 fontsize=11, fontweight="bold", pad=8)
    ax_b.set_title("Baseline + annual change model", fontsize=11, fontweight="bold", pad=8)
    plt.tight_layout(pad=0.5, rect=[0, brain_frac, 1, 1])
    plt.subplots_adjust(left=0.42, bottom=brain_frac)

    # Panel C — brain views centred horizontally at the bottom
    fig.text(0.11, brain_frac - 0.04, "C", fontsize=14, fontweight="bold", va="top")

    start_x  = (fig_w - total_brain_w) / 2 + 0.3
    x_cursor = start_x
    for arr, w_in, h_in in zip(brain_arrs, b_widths, b_heights):
        y_off = (brain_h - h_in) / 2
        ax = fig.add_axes([
            x_cursor / fig_w,
            y_off / fig_h,
            w_in / fig_w,
            h_in / fig_h,
        ])
        ax.imshow(arr)
        ax.axis("off")
        x_cursor += w_in + brain_gap

    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved to {OUTPUT_PATH}")
    shutil.rmtree(TEMP_PATH)
    plt.show()


if __name__ == "__main__":
    main()
