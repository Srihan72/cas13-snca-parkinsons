# SNCA-Targeting Cas13 gRNA Design Pipeline for Parkinson's Disease

A computational pipeline for the design, scoring, off-target risk assessment,
and disease-level modelling of CRISPR-Cas13 guide RNAs (gRNAs) targeting the
*SNCA* mRNA — which encodes alpha-synuclein, the primary pathological protein
in Parkinson's disease.

---

## Biological Background

Alpha-synuclein aggregation is the hallmark of Parkinson's disease pathology.
Overexpression of *SNCA* — driven by gene duplication, triplication, or
promoter variants — accelerates aggregation and neurodegeneration. RNA
knockdown of *SNCA* via CRISPR-Cas13, a programmable RNA-targeting effector,
is a promising therapeutic strategy that avoids permanent genomic modification.

Cas13 requires a 23-nt spacer sequence flanked by a protospacer-flanking site
(PFS) and is guided exclusively to RNA, making transcriptome-wide off-target
profiling essential before any experimental work.

---

## Pipeline Overview

The pipeline is implemented as one setup script plus ten sequential Python
scripts in `scripts/`, covering gRNA design, machine-learning efficacy
prediction, off-target risk assessment (sequence-, rule-, and
expression-based), statistical analysis, and disease-level aggregation
modelling.

| Step | Script | Description |
|------|--------|-------------|
| 0 | `fetch_snca.py` | Downloads the SNCA mRNA sequence (NM_000345.4) from NCBI and saves it as `data/snca.fasta`. |
| 1 | `01_grna_generation.py` | Slides a 23-nt window across both strands of the SNCA mRNA and filters by PFS compatibility (no 3′ G), GC content (30–70 %), and absence of homopolymer runs (≥4 identical bases). Produces **2,500 candidate spacers** from an initial **6,310** 23-mers. |
| 2 | `02_rnafold_accessibility.py` | For each candidate, folds a 50-nt window centred on the target site using RNAfold (ViennaRNA). Parses minimum free energy (MFE) and calculates the **accessibility score** — the fraction of unpaired bases within the 23-nt target region. |
| 3 | `03_xgboost_model.py` | Trains an XGBoost regression model on the Wessels et al. 2020 HEK293T guide-efficacy dataset (120 guides, `standardizedGuideScores` as target). Features: GC content, mono- and dinucleotide frequencies, normalised transcript position, MFE, and accessibility (24 features total). Validated by 5-fold cross-validation; applies the model to all 2,500 SNCA candidates. |
| 4 | `04_blast_offtarget.py` | BLAST-screens the top 100 predicted candidates against the human RefSeq RNA database (NCBI `refseq_rna`). Flags off-target hits with ≤3 mismatches over ≥20 aligned nucleotides, excluding SNCA self-matches. Supports local BLAST (auto-download) or remote qblast fallback. |
| 5 | `05_composite_scoring.py` | Combines predicted efficiency, transcriptome specificity, and an off-target penalty into a single composite score and produces the final ranked shortlist of 100 candidates. |
| 6 | `06_statistics_viz.py` | One-way ANOVA comparing predicted efficiency across transcript regions (5′UTR / CDS / 3′UTR). Generates four publication-quality figures (300 DPI): composite-score bar chart, accessibility scatter, feature-importance chart, and off-target hit-profile histogram. |
| 7 | `07_random_forest_offtarget.py` | *(Filename retained for pipeline continuity, but the script no longer trains a Random Forest.)* Replaces an earlier proxy-label RF classifier — which was circular, since it re-learned the exact exponential mismatch-decay formula used to generate its own training labels — with a **literature-grounded rule-based classifier**. Off-target alignments are classified `FUNCTIONAL_RISK` or `NON_FUNCTIONAL` from total- and seed-region mismatch counts, using thresholds from Abudayyeh et al. 2017 and Wessels et al. 2020. Also documents and corrects a bug in script 04: every remote-BLAST query shared the FASTA header `>query`, so the original post-hoc summary never matched spacer sequences back to their hits, undercounting off-targets for 3 of the 100 top candidates. |
| 8 | `08_gtex_expression_weighting.py` | Weights each recovered off-target alignment by the target gene's expression in the substantia nigra (GTEx v8 median TPM, auto-downloaded), combining the rule-based functional cleavage probability with tissue expression into a `weighted_risk = P_cleavage × log10(TPM + 1)` score and risk tier. Classifies all 100 top candidates into SAFE / CAUTION / EXCLUDE therapeutic-risk categories. |
| 9 | `09_snca_aggregation_ode.py` | Two-part nucleation–elongation ODE model (Buell et al. 2014; Cremades et al. 2012; Knowles et al. 2009) simulating how Cas13-mediated SNCA knockdown suppresses α-synuclein fibril formation. Part 1 sets the initial monomer pool directly for 0 % / 50 % / 82 % knockdown; Part 2 adds explicit production and first-order degradation kinetics (turnover t½ ≈ 50 h) so knockdown acts as a fractional reduction in production rate from a physiological steady state. |
| 10 | `10_statistical_analysis.py` | Consolidated statistical report: bootstrap confidence intervals (10,000 resamples) on the top 10 candidates' predicted efficiency, Mann–Whitney U test + rank-biserial correlation and Cohen's *d* for CDS vs 3′UTR, Pearson/Spearman correlations between accessibility and efficiency across all 2,500 candidates, one-sample t-tests per region vs. the overall mean, and pairwise Bonferroni-corrected Welch t-tests between regions. Written to `output/statistical_analysis_report.txt`. |

---

## Key Results

| Metric | Value |
|--------|-------|
| Candidates generated (post-filter) | 2,500 / 6,310 |
| XGBoost 5-fold CV R² | **0.9722 ± 0.0075** |
| XGBoost 5-fold CV MAE | 0.0287 ± 0.0036 |
| Top-100 candidates with zero BLAST off-targets (≤3 mismatches) | **97 / 100** |
| Top-100 candidates cleared as SAFE (rule-based + GTEx-weighted risk) | 98 / 100 (2 CAUTION, 0 EXCLUDE) |
| ANOVA F-statistic (5′UTR / CDS / 3′UTR, n = 2,500) | 18.30 |
| ANOVA p-value | **1.28 × 10⁻⁸** |
| CDS vs 3′UTR Welch t-test, raw p | 4.04 × 10⁻⁶ |
| CDS vs 3′UTR Welch t-test, **Bonferroni-corrected** p (3 pairwise comparisons) | **1.21 × 10⁻⁵** |
| 82 % knockdown → fibril-burden AUC reduction (dynamic ODE model) | **≈ 92.2 %** |

The one-way ANOVA (script 06) shows that predicted efficiency differs
significantly across transcript regions, with 3′UTR-targeting spacers scoring
higher on average than CDS-targeting ones (mean 0.685 vs 0.644). The pairwise
Bonferroni-corrected Welch t-test in script 10 confirms this specific
comparison is significant at the corrected p = 1.21 × 10⁻⁵ (raw p = 4.04 ×
10⁻⁶ before correction — the two are easy to conflate, so both are reported
above). Across all 2,500 candidates, accessibility correlates only weakly with
predicted efficiency (Pearson r = +0.012, not significant; Spearman ρ =
+0.085, p = 2.3 × 10⁻⁵), so this region effect should not be over-interpreted
as an accessibility effect.

The top-ranked candidate (`CAGCATTTCGGTGCTTCCCTTTC`, position 811, **3′UTR**,
composite score 0.8104) has 52 % GC, MFE −10.6 kcal/mol, predicted knockdown
efficiency 82.1 %, and zero BLAST off-targets. Of the 3 (of 100) top
candidates with recovered off-target hits, all resolved alignments carry ≥1
seed-region mismatch or ≥3 total mismatches — non-functional by the Cas13
mismatch-tolerance rules in script 07 — so none were excluded on functional
grounds; 2 remain flagged CAUTION only because their off-target alignments
could not be re-recovered from NCBI (database drift between BLAST runs), not
because of elevated risk.

---

## Disease-Level Aggregation Modelling (script 09)

Beyond gRNA design, the pipeline models the downstream disease consequence of
SNCA knockdown: how a reduced α-synuclein monomer pool affects fibril
formation kinetics in the substantia nigra, using a nucleation–elongation ODE
system (`dM/dt`, `dO/dt`, `dF/dt` for monomer, oligomeric seeds, and fibril
mass) with rate constants adapted from Buell et al. 2014 and Cremades et al.
2012.

**Part 1 — static knockdown (initial-condition model).** The monomer pool is
set directly to M₀ = 1.00 / 0.50 / 0.18 (0 % / 50 % / 82 % knockdown) and
simulated over 200 h (`output/aggregation_kinetics_results.csv`,
`fig5_aggregation_kinetics.png`).

**Part 2 — dynamic knockdown (production–degradation model).** Cas13
knockdown is instead modelled mechanistically, as a fractional reduction in
monomer *production* rate (`k_prod`), with constitutive first-order
degradation (`k_deg = ln 2 / 50 h`, from measured α-synuclein turnover in
dopaminergic neurons; Mak et al. 2010, Cuervo et al. 2004), simulated over 500
h from a physiological steady state (`output/dynamic_knockdown_results.csv`,
`fig6_dynamic_knockdown.png`).

The key finding, from the dynamic model:

| Condition | Total α-synuclein reduction | Monomer reduction | **Fibril-burden AUC reduction** |
|---|---|---|---|
| 50 % knockdown | −57.3 % | −45.1 % | −74.3 % |
| **82 % knockdown (top gRNA)** | **−85.0 %** | **−79.5 %** | **−92.2 %** |

An 82 % SNCA knockdown — the predicted efficiency of the top-ranked gRNA
candidate — produces a disproportionate **≈92.2 % reduction in cumulative
fibril burden**, not merely an 82 % one. This is a **supralinear** effect:
primary nucleation scales with M², and secondary (fibril-catalysed)
nucleation scales with M × F, so both source terms of new fibril seeds are
doubly suppressed as the monomer pool falls. This nonlinearity is the
mechanistic rationale for RNA knockdown as a disease-modifying strategy —
even partial knockdown yields outsized protection.

⚠ This is a simplified 3-species phenomenological model; rate constants are
adapted from in vitro data, and the 82 % knockdown efficiency is itself a
model prediction requiring experimental validation (see Limitations).

---

## Data Sources

| Dataset | Reference | Location |
|---------|-----------|----------|
| SNCA mRNA sequence | NM_000345.4, NCBI RefSeq | Auto-downloaded by `fetch_snca.py` |
| Cas13 guide efficacy | Wessels et al., *Nature Biotechnology* 2020 ([doi:10.1038/s41587-020-0456-9](https://doi.org/10.1038/s41587-020-0456-9)) | `data/41587_2020_456_MOESM3_ESM.xlsx` |
| Off-target database | NCBI RefSeq RNA (`refseq_rna`) | Queried live via BLAST |
| RNA secondary structure | ViennaRNA / RNAfold | Local tool |
| Tissue expression | GTEx v8 median gene TPM, substantia nigra | Auto-downloaded by `08_gtex_expression_weighting.py` |
| Cas13 mismatch tolerance | Abudayyeh et al. 2017, *Science*; Wessels et al. 2020, *Nat Biotechnol* | Encoded as classification rules in `07_random_forest_offtarget.py` |
| α-Synuclein aggregation kinetics | Buell et al. 2014, *PNAS*; Cremades et al. 2012, *Cell*; Knowles et al. 2009, *Science* | Encoded as rate constants in `09_snca_aggregation_ode.py` |
| α-Synuclein protein turnover | Mak et al. 2010, *J Biol Chem*; Cuervo et al. 2004, *Science* | Encoded as `k_deg` in `09_snca_aggregation_ode.py` |

> **Not included in this repository** (too large for version control, or
> auto-downloaded on first run): the human RefSeq RNA transcriptome FASTA
> (`data/human.rna.fna`) and the GTEx v8 median-TPM GCT file
> (`data/gtex_median_tpm.gct.gz`). Both are fetched automatically by scripts
> 04 and 08 respectively.

---

## Installation

### System dependencies

```bash
brew install blast
brew tap brewsci/bio && brew install brewsci/bio/viennarna
```

### Python dependencies

Python 3.13+ required.

```bash
pip install biopython numpy pandas scikit-learn xgboost \
            matplotlib seaborn scipy requests openpyxl
```

---

## Running the Pipeline

Run scripts in order from the project root. Each script reads from `output/`
files produced by the previous step.

```bash
# 0. Download the SNCA mRNA sequence (run once)
python scripts/fetch_snca.py

# 1. Generate and filter gRNA candidates
python scripts/01_grna_generation.py

# 2. Score RNA secondary structure accessibility
python scripts/02_rnafold_accessibility.py

# 3. Train XGBoost model and predict knockdown efficiency
python scripts/03_xgboost_model.py

# 4. BLAST off-target screen (top 100; ~2 hr via remote NCBI BLAST)
#    Place human.rna.fna in data/ and re-run to use local BLAST instead.
python scripts/04_blast_offtarget.py

# 5. Composite scoring and final ranking
python scripts/05_composite_scoring.py

# 6. Statistical analysis (ANOVA) and figures 1-4
python scripts/06_statistics_viz.py

# 7. Rule-based off-target mismatch classification
python scripts/07_random_forest_offtarget.py

# 8. GTEx substantia-nigra expression-weighted off-target risk
python scripts/08_gtex_expression_weighting.py

# 9. Disease-level SNCA aggregation ODE model and figures 5-6
python scripts/09_snca_aggregation_ode.py

# 10. Consolidated statistical analysis report
python scripts/10_statistical_analysis.py
```

### Outputs

| File | Description |
|------|-------------|
| `output/grna_candidates.csv` | 2,500 filtered spacer candidates |
| `output/grna_candidates_scored.csv` | + MFE and accessibility scores |
| `output/grna_candidates_predicted.csv` | + XGBoost predicted efficiency |
| `output/top_candidates.fasta` | Top 100 candidates by predicted efficiency, FASTA |
| `output/blast_offtarget_results.csv` | BLAST off-target counts for the top 100 |
| `output/final_ranked_candidates.csv` | Final composite-scored ranking of the top 100 |
| `output/blast_offtarget_classified.csv` | Alignment-level mismatch features for recovered off-target hits |
| `output/blast_offtarget_rulebased.csv` | Rule-based FUNCTIONAL_RISK / NON_FUNCTIONAL classification per hit |
| `output/blast_offtarget_gtex_weighted.csv` | Off-target hits weighted by substantia-nigra GTEx expression |
| `output/aggregation_kinetics_results.csv` | Static ODE model metrics (t₅₀, F_max, AUC) per knockdown condition |
| `output/dynamic_knockdown_results.csv` | Dynamic (production–degradation) ODE model metrics per knockdown condition |
| `output/statistical_analysis_report.txt` | Bootstrap CIs, Mann–Whitney, Cohen's d, correlations, and pairwise t-tests |
| `output/fig1_composite_scores.png` | Top 10 candidates bar chart |
| `output/fig2_accessibility_scatter.png` | Accessibility vs efficiency scatter |
| `output/fig3_feature_importance.png` | XGBoost feature importances |
| `output/fig4_offtarget_distribution.png` | Off-target hit profile |
| `output/fig5_aggregation_kinetics.png` | Static ODE model: fibril accumulation by knockdown condition |
| `output/fig6_dynamic_knockdown.png` | Dynamic ODE model: total α-synuclein burden and fibril accumulation |
| `models/xgboost_efficiency.json` | Trained XGBoost model |

---

## Project Structure

```
.
├── data/
│   ├── snca.fasta                        # SNCA mRNA (NM_000345.4)
│   ├── 41587_2020_456_MOESM3_ESM.xlsx    # Wessels et al. 2020 training data
│   └── gtex_median_tpm.gct.gz            # GTEx v8 median TPM (auto-downloaded, gitignored)
├── models/
│   └── xgboost_efficiency.json           # Trained XGBoost model
├── output/                               # Generated by pipeline scripts
├── scripts/
│   ├── fetch_snca.py                     # Download SNCA from NCBI
│   ├── 01_grna_generation.py
│   ├── 02_rnafold_accessibility.py
│   ├── 03_xgboost_model.py
│   ├── 04_blast_offtarget.py
│   ├── 05_composite_scoring.py
│   ├── 06_statistics_viz.py
│   ├── 07_random_forest_offtarget.py     # rule-based off-target classifier
│   ├── 08_gtex_expression_weighting.py
│   ├── 09_snca_aggregation_ode.py
│   └── 10_statistical_analysis.py
└── README.md
```

---

## Limitations

This is a **computational prediction pipeline**. All results require
experimental validation before any therapeutic or research conclusions can
be drawn. Specific caveats:

- **Training data size.** The XGBoost model is trained on 120 guides from
  Wessels et al. 2020 — a relatively small dataset. Predictions should be
  interpreted as a ranking signal rather than absolute efficiency values.
  The high CV R² (0.97) likely reflects the structured nature of this
  training set; generalisation to novel sequences may be more limited.

- **Off-target sensitivity.** NCBI BLAST may not capture all low-abundance
  transcripts, and — as documented in script 07 — a query-ID collision in the
  original remote-BLAST run (script 04) silently dropped off-target counts
  for 3 of the top 100 candidates until it was diagnosed and corrected. A
  local BLAST against a comprehensive transcriptome (GENCODE + all
  GTEx-expressed isoforms) would further improve specificity assessment.

- **Rule-based off-target classification is conservative.** The
  `FUNCTIONAL_RISK` / `NON_FUNCTIONAL` rules (script 07) are derived from
  HEK293T and in vitro Cas13 assays; cellular context (RNA-binding proteins,
  local RNA structure, concentration effects) is not modelled.

- **GTEx expression is bulk, post-mortem tissue.** The substantia-nigra
  weighting (script 08) uses GTEx v8 bulk RNA-seq (n = 114 donors); expression
  in living iPSC-derived dopaminergic neurons or patient tissue may differ,
  and cell-type heterogeneity in bulk sequencing may underestimate
  dopaminergic-neuron-specific transcripts.

- **Secondary structure modelling.** RNAfold predicts the minimum free energy
  structure of an isolated 50-nt window in isolation. Cellular context
  (RNA-binding proteins, co-transcriptional folding, local ribosome density)
  will alter true accessibility — consistent with the weak accessibility–efficiency
  correlation observed across all 2,500 candidates (script 10).

- **No experimental Cas13 efficiency data for SNCA.** The model was trained
  on non-SNCA targets. SNCA-specific validation in relevant cell models
  (e.g., iPSC-derived dopaminergic neurons) is required.

- **Isoform specificity.** NM_000345.4 is the canonical SNCA transcript.
  Spacers targeting the CDS will also affect the minor isoform NM_007308.4;
  isoform-selective targeting via the 3′UTR should be evaluated separately.

- **Aggregation ODE model is phenomenological.** The 3-species
  nucleation–elongation model (script 09) does not capture chaperones,
  clearance pathways, membrane interactions, or post-translational
  modifications, and its rate constants are drawn from in vitro data. It also
  does not model clearance of pre-existing aggregates — only suppression of
  new fibril formation. The 82 % knockdown efficiency it uses as an input is
  itself a model prediction, not a measured value.

---

## Citation

If you use or adapt this pipeline, please cite the Wessels et al. training
dataset:

> Wessels, H.H., Méndez-Mancilla, A., Guo, X. *et al.* Massively parallel
> Cas13 screens reveal principles for guide RNA design. *Nature Biotechnology*
> **39**, 506–516 (2021). https://doi.org/10.1038/s41587-020-0456-9

The off-target mismatch-tolerance rules and aggregation kinetics model draw
on:

> Abudayyeh, O.O., Gootenberg, J.S., Konermann, S. *et al.* C2c2 is a
> single-component programmable RNA-guided RNA-targeting CRISPR effector.
> *Science* **353**, aaf5573 (2017).

> Buell, A.K., Galvagnion, C., Gaspar, R. *et al.* Solution conditions
> determine the relative importance of nucleation and growth processes in
> α-synuclein aggregation. *Proc Natl Acad Sci USA* **111**, 7671–7676 (2014).

> Cremades, N., Cohen, S.I.A., Deas, E. *et al.* Direct observation of the
> interconversion of normal and toxic forms of α-synuclein. *Cell* **149**,
> 1048–1059 (2012).

> Knowles, T.P.J., Waudby, C.A., Devlin, G.L. *et al.* An analytical solution
> to the kinetics of breakable filament assembly. *Science* **326**,
> 1533–1537 (2009).
