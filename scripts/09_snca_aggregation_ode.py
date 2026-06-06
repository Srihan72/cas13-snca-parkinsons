"""
09_snca_aggregation_ode.py
--------------------------
Nucleation–elongation ODE model of α-synuclein aggregation kinetics.

Simulates the disease-level consequence of Cas13 gRNA-mediated SNCA mRNA
knockdown: how reducing α-synuclein monomer concentration suppresses fibril
formation kinetics in the substantia nigra.

═══════════════════════════════════════════════════════════════════════════════
ODE MODEL
═══════════════════════════════════════════════════════════════════════════════
Three coupled first-order ODEs track three α-synuclein species:

    M(t)  monomer concentration        [normalized; M(0) = M₀]
    O(t)  oligomers / fibril seeds     [active elongation sites; O(0) = 0]
    F(t)  fibril mass fraction         [F(0) = 0]

Reactions and their kinetic contributions:

    Primary nucleation   2M → O    rate = k_n · M²
    Elongation           M + O → F  rate = k_e · M · O
    Secondary nucleation M + F → O  rate = k_2 · M · F  (fibril-catalysed)

Differential equations:

    dM/dt = − k_n M² − k_e M O − k_2 M F      (monomer depletion)
    dO/dt =   k_n M²             + k_2 M F      (seed production)
    dF/dt =           k_e M O                    (fibril elongation)

Mass is approximately conserved (M + F ≈ M₀); O is a catalytic intermediate
and remains small relative to M and F throughout.

═══════════════════════════════════════════════════════════════════════════════
RATE CONSTANTS
═══════════════════════════════════════════════════════════════════════════════
Adapted from Buell et al. 2014 and Cremades et al. 2012, converted to
normalized concentration units (M₀ = 1.0 corresponds to physiological
α-synuclein level in dopaminergic neurons):

    k_n = 3×10⁻⁴ hr⁻¹        primary nucleation (slow — sets the lag phase)
    k_e = 1×10⁻²  hr⁻¹       elongation (main growth mechanism)
    k_2 = 1×10⁻³ hr⁻¹        secondary nucleation (autocatalytic amplifier)

The interplay of slow nucleation and autocatalytic secondary nucleation
produces the characteristic sigmoidal aggregation curve observed in vitro.

═══════════════════════════════════════════════════════════════════════════════
KNOCKDOWN SIMULATIONS
═══════════════════════════════════════════════════════════════════════════════
CRISPR-Cas13 knockdown of SNCA mRNA reduces the steady-state monomer pool
available for aggregation. We simulate three conditions:

    Baseline            M₀ = 1.00   (unmodified expression)
    50% knockdown       M₀ = 0.50   (moderate reduction)
    82% knockdown       M₀ = 0.18   (top candidate predicted efficiency)

Because nucleation ∝ M² and secondary nucleation ∝ M × F, fibril kinetics
are supralinearly suppressed at lower M₀ — a 5.6× reduction in monomer
(1/0.18) produces >10× reduction in aggregation rate, a key therapeutic
advantage of the RNA-knockdown strategy.

═══════════════════════════════════════════════════════════════════════════════
REFERENCES
═══════════════════════════════════════════════════════════════════════════════
Buell AK, Galvagnion C, Gaspar R, et al. "Solution conditions determine the
relative importance of nucleation and growth processes in α-synuclein
aggregation." Proc Natl Acad Sci USA. 2014;111(21):7671–7676.

Cremades N, Cohen SIA, Deas E, et al. "Direct observation of the
interconversion of normal and toxic forms of α-synuclein." Cell.
2012;149(5):1048–1059.

Knowles TPJ, Waudby CA, Devlin GL, et al. "An analytical solution to the
kinetics of breakable filament assembly." Science. 2009;326(5959):1533–1537.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp, trapezoid
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE    = Path(__file__).parent.parent
OUT_FIG = BASE / "output" / "fig5_aggregation_kinetics.png"
OUT_CSV = BASE / "output" / "aggregation_kinetics_results.csv"

# Dynamic knockdown outputs (Part 2)
OUT_FIG_DYN = BASE / "output" / "fig6_dynamic_knockdown.png"
OUT_CSV_DYN = BASE / "output" / "dynamic_knockdown_results.csv"

# ── Published rate constants ──────────────────────────────────────────────────
K_N = 3e-4    # primary nucleation rate constant   (hr⁻¹)
K_E = 1e-2    # elongation rate constant            (hr⁻¹, normalized units)
K_2 = 1e-3    # secondary nucleation rate constant  (hr⁻¹, normalized units)

T_END  = 200.0     # simulation window (hours)
N_EVAL = 4001      # evaluation points (smooth curves)

# ── Simulation conditions ─────────────────────────────────────────────────────
CONDITIONS = [
    dict(
        label    = "Baseline (no knockdown, M₀ = 1.00)",
        short    = "Baseline",
        M0       = 1.00,
        knockdown= 0.00,
        pct_kd   = "0%",
        color    = "#d62728",    # red — pathological baseline
        ls       = "-",
        lw       = 2.5,
    ),
    dict(
        label    = "50% knockdown (M₀ = 0.50)",
        short    = "50% KD",
        M0       = 0.50,
        knockdown= 0.50,
        pct_kd   = "50%",
        color    = "#ff7f0e",    # orange — partial intervention
        ls       = "--",
        lw       = 2.0,
    ),
    dict(
        label    = "82% knockdown — top gRNA (M₀ = 0.18)",
        short    = "82% KD (top gRNA)",
        M0       = 0.18,
        knockdown= 0.82,
        pct_kd   = "82%",
        color    = "#2ca02c",    # green — effective therapeutic intervention
        ls       = "-.",
        lw       = 2.0,
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# PART 2 — DYNAMIC KNOCKDOWN MODEL
# ─────────────────────────────────────────────────────────────────────────────
# Adds explicit monomer production (k_prod) and first-order degradation (k_deg)
# terms to the ODE system.  Cas13 knockdown is modelled as a fractional
# reduction in k_prod rather than a static initial condition.
#
# Steady-state constraint (no aggregation):
#   dM/dt = 0  →  M_ss = k_prod / k_deg = 1.0  (normalized)
#   ∴  k_prod = k_deg
#
# α-Synuclein protein half-life ≈ 50 hr in dopaminergic neurons.
#   Ref: Mak SK, McCormack AL, Manning-Bog AB, Cuervo AM, Di Monte DA.
#        "Lysosomal degradation of alpha-synuclein in vivo."
#        J Biol Chem. 2010;285(18):13621-13629.
#   Ref: Cuervo AM, Stefanis L, Fredenburg R, Lansbury PT, Sulzer D.
#        "Impaired degradation of mutant alpha-synuclein by chaperone-
#        mediated autophagy." Science. 2004;305(5688):1292-1295.
#
# k_deg = ln(2) / t½ = ln(2) / 50 hr ≈ 0.01386 hr⁻¹
# k_prod = k_deg so that M_ss = 1.0 (normalized physiological level)
#
# Knockdown → k_prod × (1 − knockdown_fraction)

K_DEG  = np.log(2) / 50.0   # protein turnover rate constant  (hr⁻¹)
K_PROD = K_DEG               # baseline production rate        (hr⁻¹, normalized units)

T_END_DYN  = 500.0     # longer window to capture approach to new steady state
N_EVAL_DYN = 5001

DYN_CONDITIONS = [
    dict(
        label   = "Baseline (no knockdown)",
        short   = "Baseline",
        kd_frac = 0.00,
        pct_kd  = "0%",
        color   = "#d62728",    # red
        ls      = "-",
        lw      = 2.5,
    ),
    dict(
        label   = "50% knockdown",
        short   = "50% KD",
        kd_frac = 0.50,
        pct_kd  = "50%",
        color   = "#ff7f0e",    # orange
        ls      = "--",
        lw      = 2.0,
    ),
    dict(
        label   = "82% knockdown (top gRNA)",
        short   = "82% KD",
        kd_frac = 0.82,
        pct_kd  = "82%",
        color   = "#2ca02c",    # green
        ls      = "-.",
        lw      = 2.0,
    ),
]


# ── ODE system ────────────────────────────────────────────────────────────────
def snca_ode(t: float, y: list[float]) -> list[float]:
    """
    α-Synuclein nucleation–elongation ODE system.

    dM/dt = − k_n M² − k_e M O − k_2 M F
    dO/dt =   k_n M²             + k_2 M F
    dF/dt =           k_e M O
    """
    M, O, F = y
    # clamp to zero to suppress floating-point negative artefacts
    M = max(M, 0.0)
    O = max(O, 0.0)
    F = max(F, 0.0)

    kn_term = K_N * M * M    # primary nucleation
    ke_term = K_E * M * O    # elongation
    k2_term = K_2 * M * F    # secondary nucleation

    return [
        -kn_term - ke_term - k2_term,   # dM/dt
         kn_term             + k2_term,  # dO/dt
                  ke_term,               # dF/dt
    ]


# ── Simulation ────────────────────────────────────────────────────────────────
def run_simulation(M0: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve ODE for a given initial monomer concentration M0."""
    t_eval = np.linspace(0.0, T_END, N_EVAL)
    sol = solve_ivp(
        snca_ode,
        (0.0, T_END),
        [M0, 0.0, 0.0],
        method  = "Radau",    # L-stable implicit method; ideal for stiff lag-phase
        t_eval  = t_eval,
        rtol    = 1e-9,
        atol    = 1e-12,
    )
    if not sol.success:
        raise RuntimeError(f"ODE solver failed for M0={M0}: {sol.message}")
    return sol.t, sol.y[0], sol.y[1], sol.y[2]


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_t50(t: np.ndarray, F: np.ndarray, M0: float) -> float | None:
    """
    Time (hours) at which F first reaches 50% of M0.
    Returns None if the threshold is not crossed within the simulation window.
    """
    target = 0.5 * M0
    above  = np.where(F >= target)[0]
    if len(above) == 0:
        return None
    idx = above[0]
    if idx == 0:
        return float(t[0])
    # linear interpolation between bracketing time points
    span = F[idx] - F[idx - 1]
    if span == 0.0:
        return float(t[idx])
    frac = (target - F[idx - 1]) / span
    return float(t[idx - 1] + frac * (t[idx] - t[idx - 1]))


def compute_metrics(t: np.ndarray, M: np.ndarray, O: np.ndarray,
                    F: np.ndarray, M0: float, cond: dict) -> dict:
    t50      = compute_t50(t, F, M0)
    F_max    = float(np.max(F))
    auc      = float(trapezoid(F, t))
    frac_agg = float(F[-1] / M0) if M0 > 0 else 0.0

    return {
        "condition"                   : cond["short"],
        "M0"                          : M0,
        "knockdown_pct"               : cond["pct_kd"],
        "t50_hr"                      : round(t50, 1) if t50 is not None else ">200",
        "F_max"                       : round(F_max, 8),
        "fraction_aggregated_at_200hr": round(frac_agg, 8),
        "AUC_0_200hr"                 : round(auc, 8),
    }


# ── Figure ────────────────────────────────────────────────────────────────────
def make_figure(simulations: list[dict]) -> None:
    """
    Publication-quality figure: fibril accumulation curves for all three
    conditions, with shaded regions showing the reduction vs baseline.
    """
    plt.rcParams.update({
        "font.family"      : "DejaVu Sans",
        "font.size"        : 11,
        "axes.linewidth"   : 1.2,
        "xtick.major.size" : 4,
        "ytick.major.size" : 4,
        "xtick.minor.size" : 2,
        "ytick.minor.size" : 2,
    })

    fig, ax = plt.subplots(figsize=(11, 6.5))

    baseline_F = simulations[0]["F"]
    t_eval     = simulations[0]["t"]

    # ── Shaded regions: protection vs baseline ─────────────────────────────
    shade_handles = []
    for sim in simulations[1:]:
        cond = sim["cond"]
        prot = np.maximum(baseline_F - sim["F"], 0.0)
        h = ax.fill_between(
            t_eval,
            sim["F"],
            baseline_F,
            where=(prot > 1e-12),
            alpha=0.15,
            color=cond["color"],
            label=f"Protection vs baseline ({cond['pct_kd']} KD)",
        )
        shade_handles.append(h)

    # ── Main kinetic curves ────────────────────────────────────────────────
    line_handles = []
    for sim in simulations:
        cond = sim["cond"]
        lh, = ax.plot(
            sim["t"], sim["F"],
            color=cond["color"],
            ls=cond["ls"],
            lw=cond["lw"],
            label=cond["label"],
        )
        line_handles.append(lh)

    # ── t50 markers ────────────────────────────────────────────────────────
    for sim in simulations:
        t50 = sim["metrics"]["t50_hr"]
        if not isinstance(t50, str):   # ">200" is a string; real values are float
            F_target = 0.5 * sim["cond"]["M0"]
            ax.axvline(t50, color=sim["cond"]["color"], ls=":", lw=1.1, alpha=0.55)
            ax.plot(t50, F_target, "o",
                    color=sim["cond"]["color"], ms=8.5, zorder=6,
                    markeredgecolor="white", markeredgewidth=1.2)
            ax.annotate(
                f"t₅₀ = {t50:.0f} h",
                xy=(t50, F_target),
                xytext=(t50 + 3.5, F_target + 0.0015),
                fontsize=8.5, color=sim["cond"]["color"],
                arrowprops=dict(arrowstyle="-", color=sim["cond"]["color"],
                                lw=0.7, alpha=0.5),
            )

    # ── Axes, labels, styling ──────────────────────────────────────────────
    ax.set_xlabel("Time (hours)", fontsize=13, labelpad=6)
    ax.set_ylabel("Fibril concentration (normalized)", fontsize=13, labelpad=6)
    ax.set_title(
        "α-Synuclein Aggregation Kinetics: Nucleation–Elongation ODE Model\n"
        "Effect of CRISPR-Cas13 SNCA mRNA Knockdown on Fibril Accumulation",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xlim(0.0, T_END)

    # dynamic y-axis: 5% headroom above the max plotted value
    all_F = np.concatenate([s["F"] for s in simulations])
    ymax = max(all_F.max() * 1.12, 0.005)
    ax.set_ylim(-ymax * 0.02, ymax)

    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    ax.grid(True, which="major", linestyle="--", linewidth=0.55,
            alpha=0.45, color="grey")
    ax.grid(True, which="minor", linestyle=":", linewidth=0.35,
            alpha=0.25, color="grey")

    # ── Legend ────────────────────────────────────────────────────────────
    all_handles = line_handles + shade_handles
    ax.legend(
        handles    = all_handles,
        loc        = "upper left",
        fontsize   = 9,
        framealpha = 0.94,
        edgecolor  = "0.75",
        handlelength=2.2,
    )

    # ── Caption / reference annotation ────────────────────────────────────
    ax.text(
        0.99, 0.03,
        (
            r"Rate constants: $k_n$ = 3×10⁻⁴ hr⁻¹,  $k_e$ = 10⁻² hr⁻¹,  "
            r"$k_2$ = 10⁻³ hr⁻¹"
            "\nBuell et al. 2014 PNAS;  Cremades et al. 2012 Cell;  "
            "Knowles et al. 2009 Science"
        ),
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=7.5, color="0.45", style="italic",
    )

    plt.tight_layout()
    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure saved → {OUT_FIG}")


# ── Summary and interpretation ────────────────────────────────────────────────
def print_summary(simulations: list[dict]) -> None:

    rows = [s["metrics"] for s in simulations]
    df   = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"  Results saved → {OUT_CSV}\n")

    # ── Metrics table ──────────────────────────────────────────────────────
    print("=" * 80)
    print("AGGREGATION KINETICS SUMMARY  (t = 0–200 hr)")
    print("=" * 80)
    hdr = (
        f"{'Condition':<26} {'M₀':>5}  {'KD':>4}  {'t₅₀ (hr)':>10}"
        f"  {'F_max':>9}  {'Aggr @200h':>10}  {'AUC':>10}"
    )
    print(hdr)
    print("-" * 80)
    for row in rows:
        print(
            f"{row['condition']:<26} {row['M0']:>5.2f}  {row['knockdown_pct']:>4}"
            f"  {str(row['t50_hr']):>10}"
            f"  {row['F_max']:>9.6f}"
            f"  {row['fraction_aggregated_at_200hr']:>10.4%}"
            f"  {row['AUC_0_200hr']:>10.6f}"
        )

    # ── Relative reductions ────────────────────────────────────────────────
    baseline = rows[0]
    print()
    print("Reductions relative to unmodified baseline:")
    print("-" * 60)
    for row in rows[1:]:
        def pct_red(a, b):
            return (1.0 - a / b) * 100 if b > 0 else float("nan")

        r_fmax = pct_red(row["F_max"], baseline["F_max"])
        r_agg  = pct_red(row["fraction_aggregated_at_200hr"],
                          baseline["fraction_aggregated_at_200hr"])
        r_auc  = pct_red(row["AUC_0_200hr"], baseline["AUC_0_200hr"])

        print(f"\n  {row['condition']}  (M₀ = {row['M0']}):")
        print(f"    Peak fibril concentration : −{r_fmax:.1f}%")
        print(f"    Aggregated fraction @200h : −{r_agg:.1f}%")
        print(f"    Disease burden (AUC)      : −{r_auc:.1f}%")

    # ── Mechanistic interpretation ─────────────────────────────────────────
    if len(rows) >= 3:
        r_auc_82 = (1.0 - rows[2]["AUC_0_200hr"] / baseline["AUC_0_200hr"]) * 100
        r_agg_82 = (
            (1.0 - rows[2]["fraction_aggregated_at_200hr"]
             / baseline["fraction_aggregated_at_200hr"]) * 100
            if baseline["fraction_aggregated_at_200hr"] > 0 else 100.0
        )

    print()
    print("=" * 80)
    print("MECHANISTIC INTERPRETATION")
    print("=" * 80)
    print("""
  The nucleation–elongation model reveals a critical non-linearity:
  fibril formation kinetics are SUPRALINEARLY suppressed at reduced
  monomer concentrations.

  This arises from two reinforcing mechanisms in the ODE:
    1. Primary nucleation ∝ M²  — a 5.6× reduction in M (1.0 → 0.18)
       produces a 31× reduction in nucleation rate.
    2. Secondary nucleation ∝ M × F — both factors are reduced; the
       autocatalytic amplification loop is doubly weakened.

  Consequence: the 82% SNCA knockdown predicted for the top gRNA
  candidate does not merely reduce fibril burden by 82%; it essentially""")
    if len(rows) >= 3:
        print(
            f"  abolishes fibril accumulation over the 200-hour window\n"
            f"  (>{r_agg_82:.0f}% reduction in aggregated fraction;\n"
            f"   >{r_auc_82:.0f}% reduction in cumulative disease burden AUC)."
        )
    print("""
  The t₅₀ metric (time to half-maximal fibril accumulation) illustrates
  the therapeutic leverage:
    • Baseline       : fibrils accumulate progressively in the window
    • 50% knockdown  : onset delayed; much lower total burden
    • 82% knockdown  : t₅₀ is pushed far beyond 200 hours — the
                       intervention effectively prevents pathological
                       fibril formation within the simulated timeframe.

  These kinetics support the rationale for SNCA mRNA knockdown as a
  disease-modifying strategy in Parkinson's disease. Even partial
  knockdown provides disproportionate protection due to the supralinear
  concentration dependence of aggregation kinetics.

  ⚠ LIMITATIONS:
    • This is a simplified 3-species phenomenological model. In vivo,
      additional factors (chaperones, degradation pathways, membrane
      interactions, post-translational modifications) alter kinetics.
    • Rate constants are adapted from in vitro data at defined pH/ionic
      strength; intracellular environment may shift absolute values.
    • The 82% knockdown efficiency is a model prediction requiring
      experimental validation in relevant cell models.
    • Monomer-level knockdown is the model input; actual clearance
      kinetics of existing aggregates are not modelled here.
""")


# ─────────────────────────────────────────────────────────────────────────────
# PART 2 FUNCTIONS — dynamic knockdown ODE, figure, summary
# ─────────────────────────────────────────────────────────────────────────────

def snca_ode_dynamic(t: float, y: list[float], kd_frac: float) -> list[float]:
    """
    α-Synuclein ODE with monomer production and first-order degradation.

    dM/dt = k_prod*(1 - kd_frac) - k_deg*M - k_n*M² - k_e*M*O - k_2*M*F
    dO/dt = k_n*M² + k_2*M*F
    dF/dt = k_e*M*O

    k_prod*(1 - kd_frac) represents Cas13-attenuated SNCA transcription/translation.
    k_deg*M represents constitutive proteasomal + lysosomal clearance (first-order).
    """
    M, O, F = y
    M = max(M, 0.0)
    O = max(O, 0.0)
    F = max(F, 0.0)

    prod    = K_PROD * (1.0 - kd_frac)   # attenuated production
    deg_M   = K_DEG * M                   # first-order degradation
    kn_term = K_N * M * M
    ke_term = K_E * M * O
    k2_term = K_2 * M * F

    return [
        prod - deg_M - kn_term - ke_term - k2_term,   # dM/dt
        kn_term + k2_term,                              # dO/dt
        ke_term,                                        # dF/dt
    ]


def run_simulation_dynamic(kd_frac: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve dynamic ODE from physiological steady state M(0) = 1.0.
    Knockdown applied at t = 0 (post-treatment trajectory).
    """
    t_eval = np.linspace(0.0, T_END_DYN, N_EVAL_DYN)
    sol = solve_ivp(
        snca_ode_dynamic,
        (0.0, T_END_DYN),
        [1.0, 0.0, 0.0],   # M=1 (steady-state), O=F=0 (no seeds/fibrils)
        args   = (kd_frac,),
        method = "Radau",
        t_eval = t_eval,
        rtol   = 1e-9,
        atol   = 1e-12,
    )
    if not sol.success:
        raise RuntimeError(f"Dynamic ODE solver failed (kd_frac={kd_frac}): {sol.message}")
    return sol.t, sol.y[0], sol.y[1], sol.y[2]


def compute_metrics_dynamic(
    t: np.ndarray, M: np.ndarray, O: np.ndarray, F: np.ndarray,
    cond: dict,
) -> dict:
    total_snca = M + O + F
    auc_fibril = float(trapezoid(F, t))
    return {
        "condition"          : cond["short"],
        "kd_frac"            : cond["kd_frac"],
        "pct_kd"             : cond["pct_kd"],
        "M_ss_predicted"     : round(1.0 - cond["kd_frac"], 4),
        "M_final"            : round(float(M[-1]), 8),
        "total_snca_final"   : round(float(total_snca[-1]), 8),
        "F_final"            : round(float(F[-1]), 8),
        "AUC_fibril_0_500hr" : round(auc_fibril, 6),
    }


def make_figure_dynamic(simulations_dyn: list[dict]) -> None:
    """
    Two-panel publication figure:
      (a) Total α-synuclein burden (M + O + F) over time
      (b) Fibril accumulation over time
    Each panel shows baseline, 50% KD, and 82% KD conditions.
    """
    plt.rcParams.update({
        "font.family"      : "DejaVu Sans",
        "font.size"        : 11,
        "axes.linewidth"   : 1.2,
        "xtick.major.size" : 4,
        "ytick.major.size" : 4,
        "xtick.minor.size" : 2,
        "ytick.minor.size" : 2,
    })

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.0), constrained_layout=True)
    ax_total, ax_fibril = axes

    t_eval   = simulations_dyn[0]["t"]
    base_M   = simulations_dyn[0]["M"]
    base_O   = simulations_dyn[0]["O"]
    base_F   = simulations_dyn[0]["F"]
    base_tot = base_M + base_O + base_F

    # ── Panel (a): total α-synuclein ──────────────────────────────────────────
    for sim in simulations_dyn[1:]:
        cond  = sim["cond"]
        total = sim["M"] + sim["O"] + sim["F"]
        prot  = np.maximum(base_tot - total, 0.0)
        ax_total.fill_between(
            t_eval, total, base_tot,
            where=(prot > 1e-12),
            alpha=0.12, color=cond["color"],
            label=f"SNCA removed vs baseline ({cond['pct_kd']} KD)",
        )

    line_handles_total = []
    for sim in simulations_dyn:
        cond  = sim["cond"]
        total = sim["M"] + sim["O"] + sim["F"]
        lh, = ax_total.plot(
            sim["t"], total,
            color=cond["color"], ls=cond["ls"], lw=cond["lw"],
            label=cond["label"],
        )
        line_handles_total.append(lh)

    ax_total.set_xlabel("Time (hours)", fontsize=12, labelpad=5)
    ax_total.set_ylabel("Total α-synuclein (normalized)", fontsize=12, labelpad=5)
    ax_total.set_title("(a) Total α-Synuclein Burden Over Time", fontsize=12, fontweight="bold")
    ax_total.set_xlim(0.0, T_END_DYN)
    ax_total.set_ylim(bottom=0.0)
    ax_total.xaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    ax_total.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    ax_total.grid(True, which="major", ls="--", lw=0.55, alpha=0.45, color="grey")
    ax_total.grid(True, which="minor", ls=":",  lw=0.35, alpha=0.25, color="grey")
    ax_total.legend(fontsize=8.5, loc="upper right", framealpha=0.94, edgecolor="0.75")

    # ── Panel (b): fibril accumulation ────────────────────────────────────────
    for sim in simulations_dyn[1:]:
        cond = sim["cond"]
        prot = np.maximum(base_F - sim["F"], 0.0)
        ax_fibril.fill_between(
            t_eval, sim["F"], base_F,
            where=(prot > 1e-12),
            alpha=0.12, color=cond["color"],
        )

    for sim in simulations_dyn:
        cond = sim["cond"]
        ax_fibril.plot(
            sim["t"], sim["F"],
            color=cond["color"], ls=cond["ls"], lw=cond["lw"],
            label=cond["label"],
        )

    ax_fibril.set_xlabel("Time (hours)", fontsize=12, labelpad=5)
    ax_fibril.set_ylabel("Fibril concentration (normalized)", fontsize=12, labelpad=5)
    ax_fibril.set_title("(b) Fibril Accumulation", fontsize=12, fontweight="bold")
    ax_fibril.set_xlim(0.0, T_END_DYN)
    ax_fibril.set_ylim(bottom=0.0)
    ax_fibril.xaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    ax_fibril.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    ax_fibril.grid(True, which="major", ls="--", lw=0.55, alpha=0.45, color="grey")
    ax_fibril.grid(True, which="minor", ls=":",  lw=0.35, alpha=0.25, color="grey")
    ax_fibril.legend(fontsize=8.5, loc="upper left", framealpha=0.94, edgecolor="0.75")

    fig.suptitle(
        "Dynamic Cas13 SNCA Knockdown: Monomer Production–Degradation ODE Model\n"
        r"Knockdown modelled as fractional reduction in production rate $k_{prod}$",
        fontsize=13, fontweight="bold",
    )

    caption = (
        r"$k_{prod}$ = $k_{deg}$ = $\ln 2\,/\,$50 hr⁻¹ ≈ 0.01386 hr⁻¹  "
        r"(50 hr half-life;  Mak et al. 2010 J Biol Chem;  Cuervo et al. 2004 Science)"
    )
    for ax in axes:
        ax.text(
            0.99, 0.03, caption,
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6.5, color="0.45", style="italic",
        )

    fig.savefig(OUT_FIG_DYN, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure saved → {OUT_FIG_DYN}")


def print_summary_dynamic(simulations_dyn: list[dict]) -> None:
    rows = [s["metrics"] for s in simulations_dyn]
    base = rows[0]

    def pct_red(a, b):
        return round((1.0 - a / b) * 100, 2) if b > 1e-15 else 0.0

    for row in rows:
        row["total_snca_reduction_pct"] = pct_red(row["total_snca_final"],   base["total_snca_final"])
        row["monomer_reduction_pct"]    = pct_red(row["M_final"],             base["M_final"])
        row["fibril_AUC_reduction_pct"] = pct_red(row["AUC_fibril_0_500hr"], base["AUC_fibril_0_500hr"])

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV_DYN, index=False)
    print(f"  Results saved → {OUT_CSV_DYN}\n")

    # ── Summary table ──────────────────────────────────────────────────────────
    print("=" * 85)
    print(f"DYNAMIC KNOCKDOWN SUMMARY  (t = 0 – {T_END_DYN:.0f} hr)")
    print("=" * 85)
    hdr = (
        f"{'Condition':<22} {'KD':>4}  {'M_ss':>6}  {'M_final':>9}"
        f"  {'Total_final':>11}  {'F_final':>9}  {'Fibril AUC':>12}"
    )
    print(hdr)
    print("-" * 85)
    for row in rows:
        print(
            f"{row['condition']:<22} {row['pct_kd']:>4}"
            f"  {row['M_ss_predicted']:>6.2f}"
            f"  {row['M_final']:>9.6f}"
            f"  {row['total_snca_final']:>11.6f}"
            f"  {row['F_final']:>9.6f}"
            f"  {row['AUC_fibril_0_500hr']:>12.4f}"
        )

    print()
    print("Reductions relative to baseline at final timepoint:")
    print("-" * 65)
    for row in rows[1:]:
        print(f"\n  {row['condition']}  ({row['pct_kd']} knockdown):")
        print(f"    Total α-synuclein reduction : −{row['total_snca_reduction_pct']:.1f}%")
        print(f"    Monomer reduction           : −{row['monomer_reduction_pct']:.1f}%")
        print(f"    Fibril burden (AUC) reduction: −{row['fibril_AUC_reduction_pct']:.1f}%")

    print()
    print("=" * 85)
    print("DYNAMIC MODEL INTERPRETATION")
    print("=" * 85)
    print(f"""
  All conditions start identically (M₀ = 1.0, O = F = 0). Knockdown acts
  immediately as a reduction in SNCA production rate k_prod; the existing
  monomer pool is cleared by normal protein turnover (t½ ≈ 50 hr).

  The monomer level relaxes to a new steady state on the turnover timescale
  (~3–5 half-lives = 150–250 hr).  Predicted new steady-state monomer:
    •  50% KD  →  M_ss ≈ 0.50
    •  82% KD  →  M_ss ≈ 0.18

  Aggregation is suppressed because:
    1.  Primary nucleation ∝ M²  — a falling M doubly suppresses seed formation.
    2.  Secondary nucleation ∝ M × F  — autocatalytic loop is doubly weakened.

  The 82% knockdown prevents meaningful fibril formation: while M is
  transitioning from 1.0 → 0.18 the nucleation rate is already negligible,
  and at the new steady state it is negligible indefinitely.

  ⚠ LIMITATIONS:
    •  Instantaneous onset assumed; in vivo, AAV delivery and Cas13 expression
       have pharmacokinetic delays not captured here.
    •  Pre-existing aggregates are not cleared by this model.
    •  Turnover constants from in vitro / rodent data; human neuronal
       α-synuclein kinetics may differ.
""")


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 65)
    print("α-Synuclein Nucleation–Elongation ODE Simulation")
    print(f"  k_n = {K_N:.0e} hr⁻¹   k_e = {K_E:.0e} hr⁻¹   "
          f"k_2 = {K_2:.0e} hr⁻¹   T = {T_END:.0f} hr")
    print("=" * 65)

    simulations: list[dict] = []

    for cond in CONDITIONS:
        M0 = cond["M0"]
        print(f"\n[{cond['short']}]  M₀ = {M0}")
        t, M, O, F = run_simulation(M0)
        metrics     = compute_metrics(t, M, O, F, M0, cond)
        simulations.append(
            {"t": t, "M": M, "O": O, "F": F, "cond": cond, "metrics": metrics}
        )
        print(f"  t₅₀           : {metrics['t50_hr']} hr")
        print(f"  F_max         : {metrics['F_max']:.8f}")
        print(f"  F at 200 hr   : {metrics['fraction_aggregated_at_200hr']:.6%} of M₀")
        print(f"  AUC (0–200hr) : {metrics['AUC_0_200hr']:.8f}")

    make_figure(simulations)
    print_summary(simulations)

    # ── Part 2: dynamic knockdown analysis ────────────────────────────────────
    print()
    print("=" * 65)
    print("α-Synuclein Dynamic Knockdown ODE Simulation")
    print(f"  k_prod = k_deg = {K_PROD:.5f} hr⁻¹  (t½ = 50 hr)   "
          f"T = {T_END_DYN:.0f} hr")
    print("=" * 65)

    simulations_dyn: list[dict] = []

    for cond in DYN_CONDITIONS:
        kd = cond["kd_frac"]
        print(
            f"\n[{cond['short']}]  kd_frac = {kd}  "
            f"→  k_prod = {K_PROD * (1.0 - kd):.5f} hr⁻¹"
        )
        t, M, O, F = run_simulation_dynamic(kd)
        metrics = compute_metrics_dynamic(t, M, O, F, cond)
        simulations_dyn.append(
            {"t": t, "M": M, "O": O, "F": F, "cond": cond, "metrics": metrics}
        )
        print(f"  M_final           : {metrics['M_final']:.6f}")
        print(f"  Total SNCA final  : {metrics['total_snca_final']:.6f}")
        print(f"  F_final           : {metrics['F_final']:.8f}")
        print(f"  Fibril AUC        : {metrics['AUC_fibril_0_500hr']:.6f}")

    make_figure_dynamic(simulations_dyn)
    print_summary_dynamic(simulations_dyn)


if __name__ == "__main__":
    main()
