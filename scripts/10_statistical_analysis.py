"""
10_statistical_analysis.py
--------------------------
Comprehensive statistical analysis of the SNCA Cas13 gRNA design pipeline.
Writes a formatted report to output/statistical_analysis_report.txt covering:

  1. Bootstrap CIs (10,000 resamples) on predicted knockdown efficiency for
     the top 10 ranked candidates (output/final_ranked_candidates.csv).
     Each candidate's prediction is a single XGBoost point estimate rather
     than a set of repeat measurements, so its uncertainty is modelled by
     resampling residual noise ~ N(0, sigma), with sigma derived from the
     regressor's 5-fold CV MAE (CV_MAE = 0.0287, see scripts 03/06; for
     ~Normal residuals, sigma = MAE * sqrt(pi/2)).
  2. Mann-Whitney U test + rank-biserial correlation: predicted efficiency,
     CDS vs 3'UTR candidates (output/grna_candidates_predicted.csv).
  3. Cohen's d effect size for the same CDS vs 3'UTR comparison.
  4. Pearson and Spearman correlations between accessibility and predicted
     efficiency across all 2,500 candidates, with 95% CIs (Fisher z for
     Pearson, percentile bootstrap for Spearman).
  5. One-sample t-tests: each transcript region vs the pooled overall mean.
  6. Pairwise Bonferroni-corrected Welch t-tests between transcript regions.

Transcript-region boundaries follow NM_000345.4 (SNCA), matching script 06:
  5'UTR : 1-69, CDS : 70-492, 3'UTR : 493-3177  (by gRNA `start` position)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE        = Path(__file__).parent.parent
RANKED_CSV  = BASE / "output" / "final_ranked_candidates.csv"
FULL_CSV    = BASE / "output" / "grna_candidates_predicted.csv"
OUT_DIR     = BASE / "output"
REPORT_PATH = OUT_DIR / "statistical_analysis_report.txt"

# ── Constants ─────────────────────────────────────────────────────────────────
REGIONS  = {"5'UTR": (1, 69), "CDS": (70, 492), "3'UTR": (493, 3177)}
N_BOOT   = 10_000
ALPHA    = 0.05
CV_MAE   = 0.0287                          # 5-fold CV MAE, XGBoost regressor (scripts 03/06)
NOISE_SD = CV_MAE * np.sqrt(np.pi / 2)      # MAE -> SD assuming ~Normal residuals
RNG      = np.random.default_rng(42)


# ── Helpers ───────────────────────────────────────────────────────────────────
def region_groups(df: pd.DataFrame) -> dict:
    groups = {}
    for region, (lo, hi) in REGIONS.items():
        mask = (df["start"] >= lo) & (df["start"] <= hi)
        groups[region] = df.loc[mask, "predicted_efficiency"].to_numpy()
    return groups


def bootstrap_point_ci(point_estimate, n_boot=N_BOOT, rng=RNG):
    """Percentile bootstrap CI for a single point prediction, resampling
    residual noise N(0, NOISE_SD) derived from the model's CV MAE."""
    samples = point_estimate + rng.normal(0.0, NOISE_SD, n_boot)
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return samples.mean(), lo, hi


def rank_biserial_u(x, y):
    """Mann-Whitney U statistic plus the rank-biserial correlation effect size."""
    n1, n2 = len(x), len(y)
    u, p = stats.mannwhitneyu(x, y, alternative="two-sided")
    r = 1.0 - (2.0 * u) / (n1 * n2)
    return u, p, r


def cohens_d(x, y):
    n1, n2 = len(x), len(y)
    pooled_sd = np.sqrt(((n1 - 1) * x.var(ddof=1) + (n2 - 1) * y.var(ddof=1)) / (n1 + n2 - 2))
    return (x.mean() - y.mean()) / pooled_sd


def mean_ci(values, confidence=1 - ALPHA):
    n = len(values)
    m, sem = values.mean(), stats.sem(values)
    h = sem * stats.t.ppf((1 + confidence) / 2, n - 1)
    return m, m - h, m + h


def diff_ci_welch(x, y, confidence=1 - ALPHA):
    """Welch confidence interval on the difference of means (x - y)."""
    n1, n2 = len(x), len(y)
    m1, m2 = x.mean(), y.mean()
    v1, v2 = x.var(ddof=1), y.var(ddof=1)
    se = np.sqrt(v1 / n1 + v2 / n2)
    df = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    h = se * stats.t.ppf((1 + confidence) / 2, df)
    diff = m1 - m2
    return diff, diff - h, diff + h


# ── Report sections ───────────────────────────────────────────────────────────
def section_bootstrap_top10(ranked: pd.DataFrame, lines: list):
    lines.append("=" * 78)
    lines.append("1. BOOTSTRAP CONFIDENCE INTERVALS — TOP 10 CANDIDATES")
    lines.append("=" * 78)
    lines.append(f"Method: percentile bootstrap, {N_BOOT:,} resamples per candidate, residual")
    lines.append(f"noise ~ N(0, {NOISE_SD:.4f}) derived from the XGBoost regressor's 5-fold")
    lines.append(f"CV MAE ({CV_MAE:.4f}); see scripts 03/06.\n")

    for i, pe in enumerate(ranked.head(10)["predicted_efficiency"], start=1):
        mean, lo, hi = bootstrap_point_ci(pe)
        lines.append(
            f"Top candidate #{i}: predicted KD = {mean * 100:.1f}% "
            f"(95% CI: {lo * 100:.1f}% - {hi * 100:.1f}%, n={N_BOOT:,} bootstrap iterations)"
        )
    lines.append("")


def section_mannwhitney(groups: dict, lines: list):
    lines.append("=" * 78)
    lines.append("2. MANN-WHITNEY U TEST — PREDICTED EFFICIENCY, CDS vs 3'UTR")
    lines.append("=" * 78)
    cds, utr3 = groups["CDS"], groups["3'UTR"]
    u, p, r = rank_biserial_u(cds, utr3)
    sig = "significant" if p < ALPHA else "not significant"
    lines.append(f"CDS    : n = {len(cds):>4}   median = {np.median(cds):.4f}")
    lines.append(f"3'UTR  : n = {len(utr3):>4}   median = {np.median(utr3):.4f}")
    lines.append(f"\nU statistic               : {u:.1f}")
    lines.append(f"p-value                   : {p:.3e}  ({sig} at α = {ALPHA})")
    lines.append(f"Rank-biserial correlation : {r:+.4f}")
    lines.append("")
    return cds, utr3


def section_cohens_d(cds, utr3, lines: list):
    lines.append("=" * 78)
    lines.append("3. COHEN'S d — PREDICTED EFFICIENCY, CDS vs 3'UTR")
    lines.append("=" * 78)
    d = cohens_d(cds, utr3)
    magnitude = (
        "negligible" if abs(d) < 0.2 else
        "small"      if abs(d) < 0.5 else
        "medium"     if abs(d) < 0.8 else
        "large"
    )
    lines.append(f"Cohen's d               : {d:+.4f}  ({magnitude} effect)")
    lines.append(f"Mean(CDS) - Mean(3'UTR) : {cds.mean() - utr3.mean():+.4f}")
    lines.append("")


def section_correlations(full: pd.DataFrame, lines: list):
    lines.append("=" * 78)
    lines.append("4. ACCESSIBILITY vs PREDICTED EFFICIENCY — ALL 2,500 CANDIDATES")
    lines.append("=" * 78)
    x = full["accessibility"].to_numpy()
    y = full["predicted_efficiency"].to_numpy()

    pr = stats.pearsonr(x, y)
    pr_lo, pr_hi = pr.confidence_interval(confidence_level=1 - ALPHA)
    lines.append(f"Pearson  r   = {pr.statistic:+.4f}   p = {pr.pvalue:.3e}")
    lines.append(f"             95% CI (Fisher z-transform): [{pr_lo:+.4f}, {pr_hi:+.4f}]")

    sr = stats.spearmanr(x, y)
    boot = stats.bootstrap(
        (x, y),
        statistic=lambda a, b: stats.spearmanr(a, b).statistic,
        paired=True, n_resamples=N_BOOT, method="percentile", rng=RNG,
    )
    lines.append(f"\nSpearman rho = {sr.statistic:+.4f}   p = {sr.pvalue:.3e}")
    lines.append(
        f"             95% CI (percentile bootstrap, n={N_BOOT:,}): "
        f"[{boot.confidence_interval.low:+.4f}, {boot.confidence_interval.high:+.4f}]"
    )
    lines.append("")


def section_one_sample_ttests(groups: dict, overall_mean: float, lines: list):
    lines.append("=" * 78)
    lines.append("5. ONE-SAMPLE t-TESTS — EACH REGION vs OVERALL MEAN")
    lines.append("=" * 78)
    lines.append(f"Overall mean predicted efficiency (n = 2,500): {overall_mean:.4f}\n")
    for region, vals in groups.items():
        t, p = stats.ttest_1samp(vals, overall_mean)
        m, lo, hi = mean_ci(vals)
        sig = "significant" if p < ALPHA else "not significant"
        lines.append(
            f"{region:<6}  n = {len(vals):>4}   mean = {m:.4f}   "
            f"95% CI on mean: [{lo:.4f}, {hi:.4f}]"
        )
        lines.append(f"        t = {t:+.3f}   p = {p:.3e}  ({sig} at α = {ALPHA})")
    lines.append("")


def section_pairwise_ttests(groups: dict, lines: list):
    lines.append("=" * 78)
    lines.append("6. PAIRWISE BONFERRONI-CORRECTED t-TESTS BETWEEN REGIONS")
    lines.append("=" * 78)
    pairs = [("5'UTR", "CDS"), ("5'UTR", "3'UTR"), ("CDS", "3'UTR")]
    n_pairs = len(pairs)
    lines.append(
        f"Welch's t-test (unequal variances), Bonferroni-corrected α = {ALPHA / n_pairs:.4f}\n"
    )
    for r1, r2 in pairs:
        x, y = groups[r1], groups[r2]
        t, p = stats.ttest_ind(x, y, equal_var=False)
        p_corr = min(p * n_pairs, 1.0)
        diff, lo, hi = diff_ci_welch(x, y)
        sig = "*" if p_corr < ALPHA else "ns"
        lines.append(f"{r1} vs {r2}")
        lines.append(f"  t = {t:+.3f}   p = {p:.3e}   p_corrected = {p_corr:.3e}  [{sig}]")
        lines.append(f"  mean diff = {diff:+.4f}   95% CI on difference: [{lo:+.4f}, {hi:+.4f}]")
    lines.append("")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ranked = pd.read_csv(RANKED_CSV)
    full   = pd.read_csv(FULL_CSV)
    groups = region_groups(full)
    overall_mean = full["predicted_efficiency"].mean()

    lines = [
        "=" * 78,
        "SNCA CAS13 gRNA PIPELINE — STATISTICAL ANALYSIS REPORT",
        "=" * 78,
        f"Ranked candidates  : {RANKED_CSV.name}  (n = {len(ranked)})",
        f"Full candidate set : {FULL_CSV.name}  (n = {len(full)})",
        "",
    ]

    section_bootstrap_top10(ranked, lines)
    cds, utr3 = section_mannwhitney(groups, lines)
    section_cohens_d(cds, utr3, lines)
    section_correlations(full, lines)
    section_one_sample_ttests(groups, overall_mean, lines)
    section_pairwise_ttests(groups, lines)

    report = "\n".join(lines)
    print(report)

    REPORT_PATH.write_text(report + "\n")
    print(f"\nSaved → {REPORT_PATH}")


if __name__ == "__main__":
    main()
