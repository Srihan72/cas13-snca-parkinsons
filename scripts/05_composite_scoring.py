"""
05_composite_scoring.py
-----------------------
Compute a composite score for each of the 100 BLAST-analysed SNCA gRNA
candidates and produce a final ranked shortlist.

Formula
-------
  specificity_score  = 1 / (1 + offtarget_hits)
  off_target_penalty = min(offtarget_hits / 10, 1.0)
  composite          = 0.5 × predicted_efficiency
                     + 0.4 × specificity_score
                     - 0.1 × off_target_penalty

Weight rationale:
  • predicted_efficiency carries the most weight (50 %) — captures
    sequence-level Cas13 activity from the Wessels model.
  • specificity_score (40 %) — rewards transcriptome uniqueness;
    diminishing penalty as off-targets accumulate (harmonic form).
  • off_target_penalty (−10 %) — linear demerit scaled to severity,
    capped at 1 to prevent extreme values dominating.

Output: output/final_ranked_candidates.csv
"""

from pathlib import Path
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE     = Path(__file__).parent.parent
IN_CSV   = BASE / "output" / "blast_offtarget_results.csv"
OUT_CSV  = BASE / "output" / "final_ranked_candidates.csv"

W_EFF    = 0.5
W_SPEC   = 0.4
W_PENALT = 0.1


def composite_score(pred_eff: float, n_offtargets: int) -> tuple[float, float, float]:
    spec    = 1.0 / (1.0 + n_offtargets)
    penalty = min(n_offtargets / 10.0, 1.0)
    score   = W_EFF * pred_eff + W_SPEC * spec - W_PENALT * penalty
    return score, spec, penalty


def main():
    df = pd.read_csv(IN_CSV)
    print(f"Loaded {len(df)} BLAST-analysed candidates from {IN_CSV.name}")
    print(f"Columns: {df.columns.tolist()}\n")

    # ── Score each candidate ─────────────────────────────────────────────────
    results = [
        composite_score(row.predicted_efficiency, int(row.offtarget_hits))
        for row in df.itertuples()
    ]
    df["specificity_score"]  = [r[1] for r in results]
    df["off_target_penalty"] = [r[2] for r in results]
    df["composite_score"]    = [r[0] for r in results]

    # ── Rank ─────────────────────────────────────────────────────────────────
    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)

    # ── Save ─────────────────────────────────────────────────────────────────
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(df)} ranked candidates → {OUT_CSV}\n")

    # ── Summary stats ─────────────────────────────────────────────────────────
    print("Composite score statistics:")
    print(f"  Range  : {df['composite_score'].min():.6f} – {df['composite_score'].max():.6f}")
    print(f"  Mean   : {df['composite_score'].mean():.6f}")
    print(f"  Std    : {df['composite_score'].std():.6f}")
    print(f"  Unique : {df['composite_score'].nunique()} distinct values\n")

    # Component ranges
    print("Component breakdown:")
    print(f"  predicted_efficiency : {df['predicted_efficiency'].min():.4f} – {df['predicted_efficiency'].max():.4f}")
    print(f"  specificity_score    : {df['specificity_score'].min():.4f} – {df['specificity_score'].max():.4f}")
    print(f"  off_target_penalty   : {df['off_target_penalty'].min():.4f} – {df['off_target_penalty'].max():.4f}")
    print(f"  composite_score      : {df['composite_score'].min():.4f} – {df['composite_score'].max():.4f}")

    # ── Top 20 display ────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("TOP 20 FINAL RANKED SNCA gRNA CANDIDATES")
    print("=" * 80)

    disp = df.head(20)[[
        "rank", "spacer", "strand", "start", "gc",
        "mfe", "accessibility", "predicted_efficiency",
        "offtarget_hits", "specificity_score", "composite_score",
    ]].copy()

    disp["gc"]                   = (disp["gc"] * 100).round(1).astype(str) + "%"
    disp["mfe"]                  = disp["mfe"].map("{:.1f}".format)
    disp["accessibility"]        = disp["accessibility"].map("{:.3f}".format)
    disp["predicted_efficiency"] = disp["predicted_efficiency"].map("{:.4f}".format)
    disp["specificity_score"]    = disp["specificity_score"].map("{:.4f}".format)
    disp["composite_score"]      = disp["composite_score"].map("{:.6f}".format)
    disp = disp.set_index("rank")

    print(disp.to_string())


if __name__ == "__main__":
    main()
