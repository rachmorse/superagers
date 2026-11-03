"""Plot elastic net vs linear SVM permutation importance for every SFC feature (supplementary figure)."""

from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd
from plot_feature_importance import NETWORK_COLORS, NETWORK_LABELS, RESULTS_DIR, load_and_prepare
from prep_data_for_en import ATLAS, DATA_DIR

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "analyses" / "figures" / f"en_svm_importance_{ATLAS}.png"
AGREEMENT_CSV = DATA_DIR / f"en_svm_feature_agreement_{ATLAS}.csv"  # For results.Rmd
N_LABELS = 5  # Label the elastic net's top features
TOP_N = 50  # Same number of top SFC features as the supplementary table
PANELS = [("t1", "Baseline model"), ("t1_slope", "Baseline + annual change model")]


def load_pair(features: str) -> pd.DataFrame:
    """Merge elastic net and SVM importance for one feature set.

    Args:
        features: "t1" or "t1_slope"

    Returns:
        pd.DataFrame: One row per feature with elastic net and SVM importance.
    """
    is_slope = features == "t1_slope"
    en = load_and_prepare(RESULTS_DIR / f"{ATLAS}_SFC_ROI_{features}_long_include_feature_importance.csv", is_slope)
    svm = load_and_prepare(RESULTS_DIR / f"{ATLAS}_SVM_SFC_ROI_{features}_long_include_feature_importance.csv", is_slope)
    return en.merge(svm[["feature", "perm_importance_mean"]], on="feature", suffixes=("_en", "_svm"))


def plot_panel(ax: plt.Axes, df: pd.DataFrame, title: str, letter: str) -> None:
    """Scatter one feature set, colored by network.

    Args:
        ax: Axes to draw on
        df: Output of load_pair
        title: Panel title
        letter: Panel letter
    """
    x, y = df["perm_importance_mean_en"], df["perm_importance_mean_svm"]
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    pad = (hi - lo) * 0.05
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#999999", lw=1, ls="--", zorder=1)
    ax.axhline(0, color="#DDDDDD", lw=0.8, zorder=0)
    ax.axvline(0, color="#DDDDDD", lw=0.8, zorder=0)

    # Filled = baseline, open = annual change
    for ftype, filled in [("baseline", True), ("slope", False)]:
        sub = df[df["feature_type"] == ftype]
        colors = sub["network"].map(NETWORK_COLORS).fillna("#AAAAAA")
        ax.scatter(sub["perm_importance_mean_en"], sub["perm_importance_mean_svm"], s=28,
                   facecolors=colors if filled else "none", edgecolors="white" if filled else colors,
                   linewidths=0.5 if filled else 1.2, zorder=2)

    # Elastic net top features stacked in the empty upper left in point height order so lines don't cross
    top = df.nlargest(N_LABELS, "perm_importance_mean_en").sort_values("perm_importance_mean_svm", ascending=False)
    for i, (_, r) in enumerate(top.iterrows()):
        label = r["label"] + (", change" if r["feature_type"] == "slope" else "")
        ax.annotate(label, (r["perm_importance_mean_en"], r["perm_importance_mean_svm"]),
                    xytext=(0.03, 0.96 - i * 0.1), textcoords="axes fraction", ha="left", va="center",
                    fontsize=8, color="#333333",
                    arrowprops={"arrowstyle": "-", "color": "#AAAAAA", "lw": 0.6, "shrinkB": 4,
                                "relpos": (1, 0.5)})  # Line starts at the end of the label

    rho = x.corr(y, method="spearman")
    ax.text(0.97, 0.04, f"Spearman ρ = {rho:.2f}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("Elastic net feature importance (ΔAUC)", fontsize=10)
    ax.set_ylabel("Linear SVM feature importance (ΔAUC)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.text(-0.15, 1.05, letter, transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def feature_agreement(df: pd.DataFrame, features: str) -> dict:
    """Summarize how closely the two models agree on feature importance.

    Args:
        df: Output of load_pair
        features: "t1" or "t1_slope" 

    Returns:
        dict: Spearman rho, feature count, and how many top TOP_N features both models share.
    """
    top_en = set(df.nlargest(TOP_N, "perm_importance_mean_en")["feature"])
    top_svm = set(df.nlargest(TOP_N, "perm_importance_mean_svm")["feature"])
    return {
        "features": features,
        "spearman_rho": df["perm_importance_mean_en"].corr(df["perm_importance_mean_svm"], method="spearman"),
        "n_features": len(df),
        "top_n": TOP_N,
        "top_n_shared": len(top_en & top_svm),
    }


def main() -> None:
    """Draw both panels with a shared network legend then save the figure and agreement stats."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2))
    agreement = []
    for ax, (features, title), letter in zip(axes, PANELS, "AB"):
        df = load_pair(features)
        plot_panel(ax, df, title, letter)
        agreement.append(feature_agreement(df, features))

    handles = [mlines.Line2D([], [], marker="o", ls="", color=c, markersize=7, label=NETWORK_LABELS[n])
               for n, c in NETWORK_COLORS.items()]
    handles += [mlines.Line2D([], [], marker="o", ls="", color="#666666", markersize=7, label="Baseline"),
                mlines.Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor="#666666",
                              markersize=7, label="Annual change")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved figure: {OUTPUT_PATH}")

    pd.DataFrame(agreement).to_csv(AGREEMENT_CSV, index=False)
    print(pd.DataFrame(agreement).to_string(index=False))
    print(f"Saved agreement stats: {AGREEMENT_CSV}")


if __name__ == "__main__":
    main()
