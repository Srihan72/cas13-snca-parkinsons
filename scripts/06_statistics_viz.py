"""
06_statistics_viz.py
--------------------
Statistical analysis and publication-quality figures for the SNCA Cas13
gRNA design pipeline.

ANOVA
-----
One-way ANOVA comparing predicted knockdown efficiency across three
transcript regions of NM_000345.4 (SNCA):
  • 5'UTR   : positions   1 –  69  (n ≈   12)
  • CDS     : positions  70 – 492  (n ≈  467)
  • 3'UTR   : positions 493 – 3177 (n ≈ 2021)

Figures (300 DPI PNG)
---------------------
  Fig 1 – Bar chart : top 10 composite scores ± model uncertainty
  Fig 2 – Scatter   : target-site accessibility vs predicted efficiency
  Fig 3 – H-bar     : XGBoost feature importances (top 15, colour-coded)
  Fig 4 – Histogram : BLAST hit profile for the 100 screened candidates
"""

from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import xgboost as xgb
from scipy.stats import pearsonr

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE         = Path(__file__).parent.parent
RANKED_CSV   = BASE / "output" / "final_ranked_candidates.csv"
FULL_CSV     = BASE / "output" / "grna_candidates_predicted.csv"
MODEL_PATH   = BASE / "models" / "xgboost_efficiency.json"
OUT_DIR      = BASE / "output"

# ── Feature name list (must match 03_xgboost_model.py) ────────────────────────
_NTS  = list("ACGU")
_DINS = ["".join(p) for p in product(_NTS, repeat=2)]
ALL_FEAT_COLS = ["gc"] + _NTS + _DINS + ["norm_pos", "mfe", "accessibility"]

# ── SNCA NM_000345.4 region boundaries ────────────────────────────────────────
REGIONS = {"5′UTR": (1, 69), "CDS": (70, 492), "3′UTR": (493, 3177)}

# Model uncertainty from 5-fold CV (MAE from script 03)
CV_MAE = 0.0287

# ── Plot style ────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "axes.titlesize"   : 12,
    "axes.labelsize"   : 11,
    "xtick.labelsize"  : 9,
    "ytick.labelsize"  : 9,
    "legend.fontsize"  : 9,
    "figure.dpi"       : 100,
    "savefig.dpi"      : 300,
})

PALETTE_10  = sns.color_palette("Blues_d", 10)[::-1]
ACCENT      = "#E84855"
DARK        = "#2D3142"


# ── ANOVA ─────────────────────────────────────────────────────────────────────
def run_anova(full_df: pd.DataFrame) -> dict:
    groups = {}
    for region, (lo, hi) in REGIONS.items():
        mask = (full_df["start"] >= lo) & (full_df["start"] <= hi)
        groups[region] = full_df.loc[mask, "predicted_efficiency"].values

    f_stat, p_val = stats.f_oneway(*groups.values())

    print("=" * 60)
    print("ONE-WAY ANOVA: predicted efficiency by transcript region")
    print("=" * 60)
    print(f"\nRegion        N      Mean ± SD")
    for region, vals in groups.items():
        print(f"  {region:<8}  {len(vals):>4}   "
              f"{vals.mean():.4f} ± {vals.std():.4f}")

    print(f"\nF-statistic : {f_stat:.4f}")
    print(f"p-value     : {p_val:.2e}")
    alpha = 0.05
    conclusion = "SIGNIFICANT" if p_val < alpha else "NOT significant"
    print(f"Result      : {conclusion} at α = {alpha}")

    # Post-hoc Tukey HSD (manual pairwise t-tests with Bonferroni)
    pairs = [("5′UTR", "CDS"), ("5′UTR", "3′UTR"), ("CDS", "3′UTR")]
    n_pairs = len(pairs)
    print(f"\nPairwise t-tests (Bonferroni-corrected, α = {alpha/n_pairs:.4f}):")
    for r1, r2 in pairs:
        t, p = stats.ttest_ind(groups[r1], groups[r2], equal_var=False)
        sig = "*" if p < alpha / n_pairs else "ns"
        print(f"  {r1} vs {r2:<8}  t = {t:+.3f}   p = {p:.2e}  {sig}")

    print()
    return {"f_stat": f_stat, "p_val": p_val, "groups": groups}


# ── Figure 1: Top 10 composite scores ─────────────────────────────────────────
def fig_composite_bars(ranked: pd.DataFrame):
    top10 = ranked.head(10).copy()
    labels = [f"#{r}\n{s[:10]}…" if len(s) > 10 else f"#{r}\n{s}"
              for r, s in zip(top10["rank"], top10["spacer"])]

    # Error bar: uncertainty propagated through composite formula
    # composite = 0.5*eff + 0.4*spec - 0.1*penalty
    # all top10 have spec=1, penalty=0 → Δcomposite = 0.5 × CV_MAE
    yerr = 0.5 * CV_MAE

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        range(10), top10["composite_score"],
        yerr=yerr, capsize=4, error_kw={"elinewidth": 1.2, "ecolor": "dimgrey"},
        color=PALETTE_10, edgecolor="white", linewidth=0.6,
        zorder=3,
    )

    # Value labels on bars
    for bar, val in zip(bars, top10["composite_score"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + yerr + 0.0005,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=8, color=DARK,
        )

    ax.set_xticks(range(10))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("Candidate (rank & spacer prefix)", labelpad=6)
    ax.set_ylabel("Composite Score")
    ax.set_title("Top 10 SNCA Cas13 gRNA Candidates — Composite Score", fontweight="bold")
    ax.set_ylim(
        top10["composite_score"].min() - 0.005,
        top10["composite_score"].max() + 0.010,
    )
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

    # Legend for error bar
    ax.errorbar([], [], yerr=1, fmt="none", color="dimgrey",
                capsize=4, label=f"±{CV_MAE/2:.4f} (model uncertainty, ½×CV MAE)")
    ax.legend(loc="lower right", framealpha=0.9)

    fig.tight_layout()
    path = OUT_DIR / "fig1_composite_scores.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")


# ── Figure 2: Accessibility vs predicted efficiency ───────────────────────────
def fig_accessibility_scatter(ranked: pd.DataFrame):
    df = ranked.copy()
    top10_mask = df["rank"] <= 10

    r_val, p_val = pearsonr(df["accessibility"], df["predicted_efficiency"])

    fig, ax = plt.subplots(figsize=(7, 5))

    # All 100 candidates
    sc = ax.scatter(
        df.loc[~top10_mask, "accessibility"],
        df.loc[~top10_mask, "predicted_efficiency"],
        c=df.loc[~top10_mask, "mfe"],
        cmap="coolwarm_r",
        s=50, alpha=0.7, edgecolors="none", zorder=2, label="Candidates 11–100",
    )
    # Top 10 highlighted
    ax.scatter(
        df.loc[top10_mask, "accessibility"],
        df.loc[top10_mask, "predicted_efficiency"],
        c=df.loc[top10_mask, "mfe"],
        cmap="coolwarm_r",
        s=130, marker="*", edgecolors=DARK, linewidths=0.5, zorder=4,
        label="Top 10 candidates",
    )

    # Regression line
    x = df["accessibility"].values
    m, b = np.polyfit(x, df["predicted_efficiency"].values, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, m * x_line + b, color=ACCENT, linewidth=1.5,
            linestyle="--", zorder=3, alpha=0.85)

    cbar = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label("MFE (kcal/mol)", fontsize=9)

    ax.set_xlabel("Target-site Accessibility (fraction unpaired bases)")
    ax.set_ylabel("Predicted Knockdown Efficiency")
    ax.set_title("RNA Accessibility vs Predicted Efficiency\n(100 BLAST-screened candidates)",
                 fontweight="bold")
    ax.legend(loc="lower right", bbox_to_anchor=(0.99, 0.20), framealpha=0.9)

    ax.text(0.97, 0.03,
            f"Pearson r = {r_val:.3f},  p = {p_val:.2e}",
            transform=ax.transAxes, va="bottom", ha="right",
            fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    fig.tight_layout()
    path = OUT_DIR / "fig2_accessibility_scatter.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")


# ── Figure 3: XGBoost feature importances ─────────────────────────────────────
def fig_feature_importance():
    model = xgb.XGBRegressor()
    model.load_model(str(MODEL_PATH))

    importances = model.feature_importances_        # shape (24,)
    feat_imp = pd.Series(importances, index=ALL_FEAT_COLS).sort_values(ascending=True)
    feat_imp = feat_imp.tail(15)                     # top 15

    # Colour by category
    def category(name):
        if name in ("mfe",):             return "Structural (MFE)"
        if name in ("accessibility",):   return "Structural (Access.)"
        if name in ("norm_pos",):        return "Transcript position"
        if name == "gc":                 return "GC content"
        if name in list("ACGU"):         return "Mono-nucleotide freq."
        return "Di-nucleotide freq."

    cat_colors = {
        "Structural (MFE)"       : "#E84855",
        "Structural (Access.)"   : "#F4845F",
        "Transcript position"    : "#9B59B6",
        "GC content"             : "#27AE60",
        "Mono-nucleotide freq."  : "#2ECC71",
        "Di-nucleotide freq."    : "#2E86AB",
    }
    colors = [cat_colors[category(f)] for f in feat_imp.index]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(feat_imp.index, feat_imp.values,
                   color=colors, edgecolor="white", linewidth=0.4, zorder=3)

    # Value labels
    for bar, val in zip(bars, feat_imp.values):
        ax.text(val + 0.0005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)

    # Legend
    legend_patches = [
        mpatches.Patch(color=col, label=cat)
        for cat, col in cat_colors.items()
        if cat in [category(f) for f in feat_imp.index]
    ]
    ax.legend(handles=legend_patches, loc="lower right",
              framealpha=0.9, fontsize=8, title="Feature category")

    ax.set_xlabel("Feature Importance (gain)")
    ax.set_title("XGBoost Model — Top 15 Feature Importances", fontweight="bold")
    ax.xaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    path = OUT_DIR / "fig3_feature_importance.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")


# ── Figure 4: BLAST hit profile ───────────────────────────────────────────────
def fig_blast_hits(ranked: pd.DataFrame):
    """Stacked bar chart: total BLAST hits broken into SNCA self-hits,
    off-target hits (≤3 mm), and 'clean' (no hits) per candidate."""

    df = ranked.head(30).copy()    # top 30 for readability
    df["total_hits_reported"] = df["snca_self_hits"] + df["offtarget_hits"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Left panel: stacked bar for top 30 ----------------------------------
    ax = axes[0]
    x   = np.arange(len(df))
    b1  = ax.bar(x, df["snca_self_hits"],  color="#4ECDC4", label="SNCA self-hits",
                 edgecolor="white", linewidth=0.4, zorder=3)
    b2  = ax.bar(x, df["offtarget_hits"],
                 bottom=df["snca_self_hits"],
                 color=ACCENT, label="Off-target hits (≤3 mm)",
                 edgecolor="white", linewidth=0.4, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([f"#{r}" for r in df["rank"]], fontsize=7.5, rotation=45, ha="right")
    ax.set_xlabel("Candidate rank")
    ax.set_ylabel("Number of BLAST hits")
    ax.set_title("BLAST Hit Profile — Top 30 Candidates", fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", framealpha=0.9)

    # Annotate "clean" above each bar
    for xi, (sh, ot) in enumerate(zip(df["snca_self_hits"], df["offtarget_hits"])):
        if sh == 0 and ot == 0:
            ax.text(xi, 0.05, "✓", ha="center", va="bottom",
                    fontsize=8, color="#27AE60", fontweight="bold")

    # --- Right panel: predicted efficiency vs off-target hits (all 100) ------
    ax2 = axes[1]
    jitter = np.random.default_rng(0).uniform(-0.1, 0.1, len(ranked))
    colors_pt = [
        "#27AE60" if h == 0 else ACCENT
        for h in ranked["offtarget_hits"]
    ]
    ax2.scatter(
        ranked["offtarget_hits"] + jitter,
        ranked["predicted_efficiency"],
        c=colors_pt, s=55, alpha=0.75, edgecolors="none", zorder=3,
    )
    ax2.set_xlabel("Off-target Hit Count (≤3 mismatches)")
    ax2.set_ylabel("Predicted Knockdown Efficiency")
    ax2.set_title("Off-target Hits vs Predicted Efficiency\n(all 100 candidates)",
                  fontweight="bold")
    ax2.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax2.xaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax2.set_axisbelow(True)

    clean_patch = mpatches.Patch(color="#27AE60", label="0 off-targets")
    offt_patch  = mpatches.Patch(color=ACCENT,    label="≥1 off-target")
    ax2.legend(handles=[clean_patch, offt_patch], loc="lower right", framealpha=0.9)

    pct_clean = (ranked["offtarget_hits"] == 0).mean() * 100
    ax2.text(0.03, 0.03, f"{pct_clean:.0f}% candidates: zero off-targets",
             transform=ax2.transAxes, va="bottom", ha="left",
             fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    fig.tight_layout(w_pad=3)
    path = OUT_DIR / "fig4_offtarget_distribution.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ranked = pd.read_csv(RANKED_CSV)
    full   = pd.read_csv(FULL_CSV)

    print(f"Loaded {len(ranked)} ranked candidates")
    print(f"Loaded {len(full)} full candidates for ANOVA\n")

    # ── ANOVA ─────────────────────────────────────────────────────────────────
    anova_result = run_anova(full)

    # ── Figures ───────────────────────────────────────────────────────────────
    print("Generating figures…\n")
    fig_composite_bars(ranked)
    fig_accessibility_scatter(ranked)
    fig_feature_importance()
    fig_blast_hits(ranked)

    print("\n" + "=" * 60)
    print("All 4 figures saved to output/ at 300 DPI")
    print("=" * 60)


if __name__ == "__main__":
    main()
