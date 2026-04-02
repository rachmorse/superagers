from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

FEATURE_IMPORTANCE_T1_CSV = Path(
    "/home/rachel/Desktop/superagers/elastic_net/SFC_ROI_t1_long_include_feature_importance.csv"
)
FEATURE_IMPORTANCE_SLOPE_CSV = Path(
    "/home/rachel/Desktop/superagers/elastic_net/SFC_ROI_t1_slope_long_include_feature_importance.csv"
)
OUTPUT_PATH = Path(
    "/home/rachel/Desktop/superagers/elastic_net/feature_importance_by_network.png"
)

# Sub-region abbreviation → full name (Schaefer atlas)
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
    "slope":    {"hatch": "///", "label": "Slope"},
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


def extract_network(feature: str) -> str:
    """Extract the network name from a feature string.

    Args:
        feature: Raw feature name from the CSV. 

    Returns:
        Network key matching.
    """
    if feature.startswith("7Networks_"):
        # e.g. 7Networks_LH_DorsAttn_FEF_1  ->  DorsAttn
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
        ``"slope"`` if the feature ends with ``_slope``, otherwise ``"baseline"``.
    """
    return "slope" if feature.endswith("_slope") else "baseline"


def make_short_label(feature: str, drop_trailing_index: bool = False,
                     is_slope_model: bool = False) -> str:
    """Shorten a ROI feature name for use as an axis tick label.

    Args:
        feature: Raw feature name from the CSV 
        drop_trailing_index: If True, strip the trailing _N parcel index
            from cortical ROI names (e.g. "LH FEF_1" to "LH FEF").

    Returns:
        A shortened more readable label string.
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
        if drop_trailing_index:
            # Strip trailing "_<digits>" parcel index
            rest_parts = rest.rsplit("_", 1)
            if len(rest_parts) == 2 and rest_parts[1].isdigit():
                rest = rest_parts[0]
        raw = ROI_ABBREV.get(rest, rest) if not rest.isdigit() else NETWORK_LABELS.get(network, network)
        full = raw[0].lower() + raw[1:]  # lowercase first letter only, preserve acronyms
        # For vague sub-region names, prepend the network for context
        if rest in {"Med", "Post"}:
            net_label = NETWORK_LABELS.get(network, network).lower()
            return f"{hemi} {net_label} {full}"
        return f"{hemi} {full}"
    if feature.startswith("Subcortical"):
        # "Subcortical 214: Right Thalamus"  ->  "R Thalamus"
        colon = feature.find(":")
        label = feature[colon + 2:] if colon != -1 else feature
        label = label.replace("Right ", "R ").replace("Left ", "L ")
        return label[0] + label[1:].lower()
    return feature


def load_and_prepare(csv_path: Path, is_slope_model: bool = False) -> pd.DataFrame:
    """Load a feature importance CSV and add derived columns.

    Args:
        csv_path: Path to the feature importance CSV.
        is_slope_model: If True, parse ``_1`` / ``_slope`` suffixes to populate
            ``feature_type`` and strip them from tick labels.

    Returns:
        DataFrame with added columns: ``network``, ``feature_type``, ``label``.
    """
    df = pd.read_csv(csv_path)
    df = df[~df["feature"].str.startswith("cov_")].copy()
    df["network"] = df["feature"].apply(extract_network)
    df["feature_type"] = df["feature"].apply(extract_feature_type) if is_slope_model else "baseline"
    df["label"] = df["feature"].apply(
        lambda f: make_short_label(f, drop_trailing_index=True, is_slope_model=is_slope_model)
    )
    return df


def _style_ax(ax: plt.Axes, panel_label: str, x_offset: float = -0.15) -> None:
    """Remove top/right spines and stamp a bold panel label.

    Args:
        ax: Target axes.
        panel_label: Single letter (e.g. ``"A"``).
        x_offset: Horizontal position of the label in axes coordinates.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=7)
    ax.text(x_offset, 1.05, panel_label, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top")


def _network_legend_handles(present_networks):
    """Build Patch handles for the network colour legend.

    Args:
        present_networks: Iterable of network keys present in the data.

    Returns:
        List of ``mpatches.Patch`` objects.
    """
    return [
        mpatches.Patch(color=NETWORK_COLORS[n], label=NETWORK_LABELS.get(n, n))
        for n in NETWORK_COLORS if n in present_networks
    ]


def plot_network_panel(ax: plt.Axes, df: pd.DataFrame, panel_label: str) -> None:
    """Plot mean permutation importance per network (single bar per network).

    Args:
        ax: Target axes.
        df: Prepared DataFrame from ``load_and_prepare``.
        panel_label: Panel label string (e.g. ``"A"``).
    """
    net_agg = (
        df.groupby("network")["perm_importance_mean"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    net_agg["color"] = net_agg["network"].map(NETWORK_COLORS)
    net_agg["display"] = net_agg["network"].map(NETWORK_LABELS)

    ax.bar(range(len(net_agg)), net_agg["perm_importance_mean"],
           color=net_agg["color"], width=0.7, edgecolor="none")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlim(-0.5, len(net_agg) - 0.5)
    ax.set_xticks(range(len(net_agg)))
    ax.set_xticklabels(net_agg["display"], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Mean permutation importance", fontsize=8)
    ax.set_xlabel("Network", fontsize=8)
    _style_ax(ax, panel_label)


def plot_network_panel_grouped(ax: plt.Axes, df: pd.DataFrame, panel_label: str) -> None:
    """Plot mean permutation importance per network, grouped by feature type.

    Each network shows two bars side by side: baseline (solid) and slope
    (hatched). Networks are ordered by their combined mean importance.

    Args:
        ax: Target axes.
        df: Prepared DataFrame from `load_and_prepare` with `is_slope_model=True`.
        panel_label: Panel label string (e.g. "C").
    """
    network_order = (
        df.groupby("network")["perm_importance_mean"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    net_type_agg = (
        df.groupby(["network", "feature_type"])["perm_importance_mean"]
        .mean()
        .reset_index()
    )

    bar_width = 0.38
    offsets = {"baseline": -bar_width / 2, "slope": bar_width / 2}

    for i, net in enumerate(network_order):
        for ftype, style in FEATURE_TYPE_STYLE.items():
            row = net_type_agg[
                (net_type_agg["network"] == net) & (net_type_agg["feature_type"] == ftype)
            ]
            if row.empty:
                continue
            ax.bar(
                i + offsets[ftype], row["perm_importance_mean"].values[0],
                width=bar_width,
                color=NETWORK_COLORS.get(net, "#AAAAAA"),
                hatch=style["hatch"],
                edgecolor="#444444" if style["hatch"] else "none",
                linewidth=0.5,
            )

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlim(-0.5, len(network_order) - 0.5)
    ax.set_xticks(range(len(network_order)))
    ax.set_xticklabels(
        [NETWORK_LABELS.get(n, n) for n in network_order],
        rotation=45, ha="right", fontsize=7,
    )
    ax.set_ylabel("Mean permutation importance", fontsize=8)
    ax.set_xlabel("Network", fontsize=8)

    type_handles = [
        mpatches.Patch(facecolor="#AAAAAA", hatch=style["hatch"],
                       edgecolor="#444444" if style["hatch"] else "none",
                       label=style["label"])
        for style in FEATURE_TYPE_STYLE.values()
    ]
    ax.legend(handles=type_handles, fontsize=6.5, loc="upper right",
              framealpha=0.9, title="Feature type", title_fontsize=7)
    _style_ax(ax, panel_label)


def plot_roi_panel(ax: plt.Axes, df: pd.DataFrame, panel_label: str,
                   is_slope_model: bool = False) -> None:
    """Plot the top 20 ROIs by permutation importance.

    Only positive-importance ROIs are shown. Bars are coloured by network.
    For the slope model, slope features are additionally hatched.

    Args:
        ax: Target axes.
        df: Prepared DataFrame from ``load_and_prepare``.
        panel_label: Panel label string (e.g. ``"B"`` or ``"D"``).
        is_slope_model: If True, apply hatching to slope features.
    """
    # Sort ascending so most important ROI sits at the top of the horizontal chart
    top20 = (
        df[df["perm_importance_mean"] > 0]
        .sort_values("perm_importance_mean", ascending=True)
        .tail(20)
        .reset_index(drop=True)
    )

    # Where two parcels share the same label, append the parcel index to distinguish them
    duplicated = top20["label"].duplicated(keep=False)
    if duplicated.any():
        def _parcel_index(feature: str, slope_model: bool) -> str:
            """Extract trailing parcel index from a feature name."""
            if slope_model:
                for suffix in ("_slope", "_1"):
                    if feature.endswith(suffix):
                        feature = feature[: -len(suffix)]
                        break
            return feature.rsplit("_", 1)[-1]

        top20.loc[duplicated, "label"] = top20.loc[duplicated].apply(
            lambda r: f"{r['label']} ({_parcel_index(r['feature'], is_slope_model)})", axis=1
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
    ax.set_yticklabels(top20["label"], fontsize=7)
    ax.set_xlabel("Permutation importance", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=7)

    handles = _network_legend_handles(top20["network"].unique())
    if is_slope_model:
        type_handles = [
            mpatches.Patch(facecolor="#AAAAAA", hatch=style["hatch"],
                           edgecolor="#444444" if style["hatch"] else "none",
                           label=style["label"])
            for style in FEATURE_TYPE_STYLE.values()
        ]
        handles += [mpatches.Patch(visible=False, label="")] + type_handles

    ax.legend(handles=handles, loc="lower right", fontsize=6.5, framealpha=0.9,
              title="Network", title_fontsize=7, handlelength=1, handleheight=0.8)
    ax.text(-0.35, 1.05, panel_label, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top")


def main():
    """Generate and save the two-panel feature importance figure.

    Panel A shows mean permutation importance aggregated by functional network.
    Panel B shows the top 20 individual ROIs by permutation importance, coloured
    by their network membership. The figure is saved as a 300 dpi PNG to
    ``OUTPUT_PATH``.
    """
    df_t1 = load_and_prepare(FEATURE_IMPORTANCE_T1_CSV, is_slope_model=False)
    df_slope = load_and_prepare(FEATURE_IMPORTANCE_SLOPE_CSV, is_slope_model=True)

    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, 4),
    )

    plot_roi_panel(axes[0], df_t1, "A", is_slope_model=False)
    plot_roi_panel(axes[1], df_slope, "B", is_slope_model=True)
    axes[0].set_title("Baseline model", fontsize=9, fontweight="bold", pad=8)
    axes[1].set_title("Baseline + annual change model", fontsize=9, fontweight="bold", pad=8)

    plt.tight_layout(pad=1.5)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Saved to {OUTPUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
