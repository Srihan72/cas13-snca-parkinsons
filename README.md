# SNCA-Targeting Cas13 gRNA Design Pipeline for Parkinson's Disease

A computational pipeline for the design, scoring, and prioritisation of
CRISPR-Cas13 guide RNAs (gRNAs) targeting the *SNCA* mRNA — which encodes
alpha-synuclein, the primary pathological protein in Parkinson's disease.

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

The pipeline is implemented as six sequential Python scripts in `scripts/`.

| Step | Script | Description |
|------|--------|-------------|
| 1 | `01_grna_generation.py` | Slide a 23-nt window across both strands of the SNCA mRNA (NM_000345.4) and filter by PFS compatibility (no 3′ G), GC content (30–70 %), and absence of homopolymer runs (≥4 identical bases). Produces **2,500 candidate spacers** from an initial 6,310. |
| 2 | `02_rnafold_accessibility.py` | For each candidate, fold a 50-nt window centred on the target site using RNAfold (ViennaRNA 2.7). Parses minimum free energy (MFE) and calculates the **accessibility score** — the fraction of unpaired bases within the 23-nt target region. |
| 3 | `03_xgboost_model.py` | Trains an XGBoost regression model on the Wessels et al. 2020 HEK293T guide-efficacy dataset (120 guides, `standardizedGuideScores` as target). Features: GC content, mono- and dinucleotide frequencies, normalised transcript position, MFE, and accessibility. Validated by 5-fold cross-validation; applies the model to all 2,500 SNCA candidates. |
| 4 | `04_blast_offtarget.py` | BLAST screens the top 100 predicted candidates against the human RefSeq RNA database (NCBI `refseq_rna`). Flags off-target hits with ≤3 mismatches over ≥20 aligned nucleotides, excluding SNCA self-matches. Supports local BLAST (auto-download) or remote qblast fallback. |
| 5 | `05_composite_scoring.py` | Combines predicted efficiency, transcriptome specificity, and off-target penalty into a single composite score and produces the final ranked shortlist. |
| 6 | `06_statistics_viz.py` | One-way ANOVA comparing predicted efficiency across transcript regions (5′UTR / CDS / 3′UTR). Generates four publication-quality figures (300 DPI). |

---

## Key Results

| Metric | Value |
|--------|-------|
| Candidates generated (post-filter) | 2,500 / 6,310 |
| XGBoost 5-fold CV R² | **0.972 ± 0.008** |
| XGBoost 5-fold CV MAE | 0.029 ± 0.004 |
| Top-100 candidates with zero off-targets | **100 / 100** |
| ANOVA F-statistic (region effect) | 18.30 |
| ANOVA p-value | **1.28 × 10⁻⁸** |
| CDS vs 3′UTR (Bonferroni-corrected) | p = 4.04 × 10⁻⁶ |

The ANOVA reveals that 3′UTR-targeting spacers have significantly higher
predicted efficiency than CDS-targeting ones (mean 0.685 vs 0.644), consistent
with reduced secondary structure in the 3′UTR. The top candidate
(`CAGCATTTCGGTGCTTCCCTTTC`, position 811 in CDS, composite score 0.8104) has
52 % GC, MFE −10.6 kcal/mol, and zero transcriptome off-targets.

---

## Data Sources

| Dataset | Reference | Location |
|---------|-----------|----------|
| SNCA mRNA sequence | NM_000345.4, NCBI RefSeq | Auto-downloaded by `fetch_snca.py` |
| Cas13 guide efficacy | Wessels et al., *Nature Biotechnology* 2020 ([doi:10.1038/s41587-020-0456-9](https://doi.org/10.1038/s41587-020-0456-9)) | `data/41587_2020_456_MOESM3_ESM.xlsx` |
| Off-target database | NCBI RefSeq RNA (`refseq_rna`) | Queried live via BLAST |
| RNA secondary structure | ViennaRNA 2.7 / RNAfold | Local tool |

> **Not included in this repository** (too large for version control):
> GENCODE transcriptome FASTA, GTEx expression data. See the Limitations
> section for how these would improve the pipeline.

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

# 6. Statistical analysis and figures
python scripts/06_statistics_viz.py
```

### Outputs

| File | Description |
|------|-------------|
| `output/grna_candidates.csv` | 2,500 filtered spacer candidates |
| `output/grna_candidates_scored.csv` | + MFE and accessibility scores |
| `output/grna_candidates_predicted.csv` | + XGBoost predicted efficiency |
| `output/blast_offtarget_results.csv` | BLAST results for top 100 |
| `output/final_ranked_candidates.csv` | Final composite-scored ranking |
| `output/fig1_composite_scores.png` | Top 10 candidates bar chart |
| `output/fig2_accessibility_scatter.png` | Accessibility vs efficiency scatter |
| `output/fig3_feature_importance.png` | XGBoost feature importances |
| `output/fig4_offtarget_distribution.png` | Off-target hit profile |
| `models/xgboost_efficiency.json` | Trained XGBoost model |

---

## Project Structure

```
.
├── data/
│   ├── snca.fasta                        # SNCA mRNA (NM_000345.4)
│   └── 41587_2020_456_MOESM3_ESM.xlsx   # Wessels et al. 2020 training data
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
│   └── 06_statistics_viz.py
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

- **Off-target sensitivity.** NCBI remote BLAST with `hitlist_size=200` may
  not capture all low-abundance transcripts. A local BLAST against a
  comprehensive transcriptome (GENCODE v47 + GTEx-expressed isoforms) would
  improve specificity assessment.

- **Secondary structure modelling.** RNAfold predicts the minimum free energy
  structure of an isolated 50-nt window in isolation. Cellular context
  (RNA-binding proteins, co-transcriptional folding, local ribosome density)
  will alter true accessibility.

- **No experimental Cas13 efficiency data for SNCA.** The model was trained
  on non-SNCA targets. SNCA-specific validation in relevant cell models
  (e.g., iPSC-derived dopaminergic neurons) is required.

- **Isoform specificity.** NM_000345.4 is the canonical SNCA transcript.
  Spacers targeting the CDS will also affect the minor isoform NM_007308.4;
  isoform-selective targeting via the 3′UTR should be evaluated separately.

---

## Citation

If you use or adapt this pipeline, please cite the Wessels et al. training
dataset:

> Wessels, H.H., Méndez-Mancilla, A., Guo, X. *et al.* Massively parallel
> Cas13 screens reveal principles for guide RNA design. *Nature Biotechnology*
> **39**, 506–516 (2021). https://doi.org/10.1038/s41587-020-0456-9
