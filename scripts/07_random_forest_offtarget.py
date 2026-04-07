"""
07_random_forest_offtarget.py
------------------------------
Random Forest binary classifier for predicting functional off-target cleavage
risk from BLAST-identified near-match hits against the human transcriptome.

═══════════════════════════════════════════════════════════════════════════════
BUG FIX — script 04 query-ID issue
═══════════════════════════════════════════════════════════════════════════════
In script 04 every qblast call passed ">query\n{spacer}\n", so the query
record ID in every returned XML was "query" (not the spacer sequence).  The
post-hoc summarise() function then looked for rows where
    hits_df["query_spacer"] == <actual 23-nt spacer>
which never matched, leaving all 100 candidates with offtarget_hits = 0.

From the terminal progress log we know three candidates had real off-target
hits.  This script:
  1. Re-BLASTs those three with a corrected query ID so mismatch details can
     be extracted from the alignment strings.
  2. Corrects blast_offtarget_results.csv with the true counts.

═══════════════════════════════════════════════════════════════════════════════
PROXY LABEL CAVEAT
═══════════════════════════════════════════════════════════════════════════════
No published dataset directly links Cas13 mismatch profiles to functional
off-target cleavage in a transcriptome-wide assay.

Training labels are therefore PROXIES constructed as:

    P_cleavage = base_eff × exp(−1.5 × seed_mm) × exp(−0.5 × nonseed_mm)
    label = 1  if  P_cleavage > 0.30  else  0

where:
  • base_eff       – standardizedGuideScores from Wessels et al. 2020
                     (HEK293T screen), representing the on-target activity of
                     the spacer when perfectly matched.
  • seed_mm        – mismatches in spacer positions 1–7 (5′ seed region),
                     which are highly intolerant in Cas13 (Abudayyeh 2016,
                     Cox 2017).
  • nonseed_mm     – mismatches in positions 8–23; better tolerated.

The exponential-decay tolerances (λ_seed = 1.5, λ_nonseed = 0.5) are derived
from published Cas13a mismatch sensitivity data and represent approximations.
All classifier outputs should be treated as RELATIVE risk rankings, not
absolute cleavage probabilities.  Experimental CLASH-seq or transcriptome-
wide cleavage assays are required for validation.

Outputs
-------
  output/blast_offtarget_results.csv    – corrected off-target counts
  output/blast_offtarget_classified.csv – hits with functional_cleavage_prob
"""

import math
import re
import time
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.Blast import NCBIWWW, NCBIXML
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    classification_report,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE           = Path(__file__).parent.parent
BLAST_CSV      = BASE / "output" / "blast_offtarget_results.csv"
WESSELS_XLSX   = BASE / "data"   / "41587_2020_456_MOESM3_ESM.xlsx"
OUT_CLASSIFIED = BASE / "output" / "blast_offtarget_classified.csv"

# ── Known off-target candidates (from script 04 terminal log) ─────────────────
# These three spacers reported non-zero off-target counts during the BLAST run
# but were zeroed out in the CSV due to the query-ID bug described above.
KNOWN_OT = {
    "CACTCCCAGTTCTCCGCTCACGA": 1,    # [60/100] → 1 off-target
    "TCACGCCTTGCCTTCAAGCCTTC": 11,   # [74/100] → 11 off-targets
    "TCACACCCGTCACCACTGCTCCT": 2,    # [91/100] → 2 off-targets
}

# ── BLAST / mismatch parameters ───────────────────────────────────────────────
SPACER_LEN     = 23
SEED_LEN       = 7          # positions 1–7 (0-indexed: 0–6)
ALIGN_MIN_LEN  = 20
MAX_MISMATCHES = 3
BLAST_DELAY    = 10         # seconds between qblast calls

# ── Proxy-label parameters ────────────────────────────────────────────────────
LAMBDA_SEED    = 1.5
LAMBDA_NONSEED = 0.5
CLEAVAGE_THRESH = 0.30      # P_cleavage threshold for label = 1


# ── Helpers: mismatch feature extraction ──────────────────────────────────────
def is_snca(title: str, accession: str) -> bool:
    acc_base = accession.split(".")[0]
    if acc_base in {"NM_000345", "NM_007308"}:
        return True
    return any(kw in title.lower() for kw in ("snca", "synuclein"))


def hsp_to_features(spacer: str, hsp, subject_title: str) -> dict:
    """
    Extract alignment-level features from a single BLAST HSP.

    Parameters
    ----------
    spacer : the 23-nt query spacer sequence
    hsp    : Bio.Blast.Record.HSP object
    """
    q_aln = hsp.query.upper()
    s_aln = hsp.sbjct.upper()

    mm_positions: list[int] = []
    transitions:  int       = 0
    transversions: int      = 0
    q_pos = 0   # position in ungapped query

    for qb, sb in zip(q_aln, s_aln):
        if qb == "-":
            continue                # gap in query — skip
        if sb == "-":
            q_pos += 1
            continue                # gap in subject — skip
        if qb != sb:
            mm_positions.append(q_pos)
            pair = frozenset([qb, sb])
            if pair in (frozenset("AG"), frozenset("CT")):
                transitions += 1
            else:
                transversions += 1
        q_pos += 1

    total_mm   = len(mm_positions)
    seed_mm    = sum(1 for p in mm_positions if p < SEED_LEN)
    nonseed_mm = total_mm - seed_mm
    total_nt   = len(s_aln.replace("-", ""))
    gc_ot      = (s_aln.count("G") + s_aln.count("C")) / max(total_nt, 1)

    return {
        "query_spacer"        : spacer,
        "subject_title"       : subject_title[:80],
        "align_len"           : hsp.align_length,
        "total_mismatches"    : total_mm,
        "seed_mismatches"     : seed_mm,
        "nonseed_mismatches"  : nonseed_mm,
        "has_seed_mismatch"   : int(seed_mm > 0),
        "transition_fraction" : transitions / max(total_mm, 1),
        "offtarget_gc"        : round(gc_ot, 4),
        "first_mismatch_pos"  : (mm_positions[0] + 1) if mm_positions else 0,
        "evalue"              : hsp.expect,
    }


# ── Re-BLAST the three known off-target candidates ────────────────────────────
def reblast_candidates(spacers: list[str]) -> list[dict]:
    """
    Re-run qblast for each spacer with the corrected query-ID format so that
    alignment strings are available for mismatch feature extraction.

    Fix: use ">spacer_sequence\n{seq}" so record.query == the spacer itself.
    """
    all_hits: list[dict] = []
    for i, spacer in enumerate(spacers, 1):
        print(f"  [{i}/{len(spacers)}] Re-BLASTing {spacer}  ...", flush=True)
        t0 = time.time()
        for attempt in range(1, 4):
            try:
                # ↓ KEY FIX: query ID is the spacer sequence itself
                fasta_in = f">{spacer}\n{spacer}\n"
                handle = NCBIWWW.qblast(
                    program      = "blastn",
                    database     = "refseq_rna",
                    sequence     = fasta_in,
                    entrez_query = "Homo sapiens[Organism]",
                    hitlist_size = 100,
                    expect       = 1000.0,
                    word_size    = 7,
                    filter       = "F",
                    short_query  = True,
                    megablast    = False,
                    format_type  = "XML",
                )
                xml_str = handle.read()
                handle.close()
                break
            except Exception as exc:
                print(f"\n    attempt {attempt} failed: {exc}", flush=True)
                if attempt < 3:
                    time.sleep(20)
                else:
                    xml_str = ""

        n_hits = 0
        for record in NCBIXML.parse(StringIO(xml_str)):
            # record.query should now equal the spacer sequence
            query_id = record.query.split()[0]
            for alignment in record.alignments:
                for hsp in alignment.hsps:
                    if hsp.align_length < ALIGN_MIN_LEN:
                        continue
                    mm = hsp.align_length - hsp.identities - hsp.gaps
                    if mm > MAX_MISMATCHES:
                        continue
                    if is_snca(alignment.title, alignment.accession):
                        continue
                    feats = hsp_to_features(spacer, hsp, alignment.title)
                    feats["accession"] = alignment.accession
                    all_hits.append(feats)
                    n_hits += 1

        elapsed = time.time() - t0
        print(f"    → {n_hits} off-target hits recovered  ({elapsed:.0f}s)", flush=True)
        if i < len(spacers):
            time.sleep(BLAST_DELAY)

    return all_hits


# ── Synthetic training data from Wessels + mismatch simulation ────────────────
def build_training_data(wessels_df: pd.DataFrame,
                        n_variants: int = 15,
                        rng_seed: int = 42) -> pd.DataFrame:
    """
    For each of the 120 Wessels HEK293T guides, generate one on-target row
    (0 mismatches) and n_variants synthetic near-match rows (1–5 mismatches
    placed randomly).  Labels are assigned via the proxy P_cleavage formula.

    ⚠ PROXY LABEL WARNING ⚠
    Labels are not empirically measured functional cleavage outcomes.
    They are derived from the on-target activity score and a biologically
    motivated mismatch-tolerance decay model.
    """
    rng = np.random.default_rng(rng_seed)
    records: list[dict] = []

    df = wessels_df.dropna(subset=["GuideSeq", "standardizedGuideScores"])

    for _, row in df.iterrows():
        spacer   = str(row["GuideSeq"]).upper()
        base_eff = float(row["standardizedGuideScores"])
        gc       = (spacer.count("G") + spacer.count("C")) / len(spacer)

        # On-target row (0 mismatches)
        p_on = base_eff  # no mismatch penalty
        records.append({
            "total_mismatches"    : 0,
            "seed_mismatches"     : 0,
            "nonseed_mismatches"  : 0,
            "has_seed_mismatch"   : 0,
            "transition_fraction" : 0.0,
            "offtarget_gc"        : gc,
            "first_mismatch_pos"  : 0,
            "spacer_efficiency"   : base_eff,
            "p_cleavage_proxy"    : round(p_on, 4),
            "label"               : int(p_on > CLEAVAGE_THRESH),
        })

        # Synthetic mismatch variants
        for _ in range(n_variants):
            n_mm = int(rng.integers(1, 6))   # 1–5 mismatches
            pos  = sorted(rng.choice(SPACER_LEN, size=min(n_mm, SPACER_LEN),
                                     replace=False).tolist())

            seed_mm    = sum(1 for p in pos if p < SEED_LEN)
            nonseed_mm = n_mm - seed_mm

            # Mismatch types (transitions ~40 % in the transcriptome)
            trans_count = int(rng.binomial(n_mm, 0.40))
            trans_frac  = trans_count / max(n_mm, 1)

            # GC content: small Gaussian jitter around the spacer GC
            gc_ot = float(np.clip(gc + rng.normal(0, 0.07), 0.15, 0.85))

            first_mm_pos = pos[0] + 1 if pos else 0

            # ── PROXY LABEL ──────────────────────────────────────────────────
            # P_cleavage = base_eff * exp(−λ_seed * seed_mm) * exp(−λ_ns * ns_mm)
            # This models that (a) a high-activity spacer retains more cleavage
            # capacity even with mismatches, and (b) seed-region mismatches are
            # far more disruptive than non-seed mismatches.
            p_cleavage = (
                base_eff
                * math.exp(-LAMBDA_SEED    * seed_mm)
                * math.exp(-LAMBDA_NONSEED * nonseed_mm)
            )
            # ─────────────────────────────────────────────────────────────────

            records.append({
                "total_mismatches"    : n_mm,
                "seed_mismatches"     : seed_mm,
                "nonseed_mismatches"  : nonseed_mm,
                "has_seed_mismatch"   : int(seed_mm > 0),
                "transition_fraction" : round(trans_frac, 4),
                "offtarget_gc"        : round(gc_ot, 4),
                "first_mismatch_pos"  : first_mm_pos,
                "spacer_efficiency"   : base_eff,
                "p_cleavage_proxy"    : round(p_cleavage, 4),
                "label"               : int(p_cleavage > CLEAVAGE_THRESH),
            })

    return pd.DataFrame(records)


FEATURE_COLS = [
    "total_mismatches", "seed_mismatches", "nonseed_mismatches",
    "has_seed_mismatch", "transition_fraction", "offtarget_gc",
    "first_mismatch_pos", "spacer_efficiency",
]


# ── Train Random Forest with 5-fold stratified CV ────────────────────────────
def train_rf(train_df: pd.DataFrame) -> RandomForestClassifier:
    X = train_df[FEATURE_COLS].values
    y = train_df["label"].values

    clf = RandomForestClassifier(
        n_estimators     = 500,
        max_depth        = 6,
        min_samples_leaf = 5,
        class_weight     = "balanced",
        random_state     = 42,
        n_jobs           = -1,
    )

    print("\n5-Fold Stratified Cross-Validation")
    print("-" * 40)
    skf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    proba_cv = cross_val_predict(clf, X, y, cv=skf, method="predict_proba")[:, 1]
    pred_cv  = (proba_cv >= 0.5).astype(int)

    auc       = roc_auc_score(y, proba_cv)
    precision = precision_score(y, pred_cv, zero_division=0)
    recall    = recall_score(y, pred_cv, zero_division=0)

    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print()
    print("Classification report (CV predictions):")
    print(classification_report(y, pred_cv,
                                target_names=["No cleavage", "Functional cleavage"],
                                zero_division=0))

    # Feature importances
    clf.fit(X, y)
    imp = pd.Series(clf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("Feature importances:")
    for feat, val in imp.items():
        bar = "█" * int(val * 40)
        print(f"  {feat:<25} {val:.4f}  {bar}")

    return clf


# ── Main ─────────────────────────────────────────────────────────────────────
def main():

    # ── 1. Load and inspect blast CSV ─────────────────────────────────────────
    print("=" * 65)
    print("STEP 1 — Load blast_offtarget_results.csv")
    print("=" * 65)

    blast_df = pd.read_csv(BLAST_CSV)
    print(f"Columns : {blast_df.columns.tolist()}")
    print(f"Shape   : {blast_df.shape}")
    print()
    print("Off-target hit distribution (as stored — all zeros due to query-ID bug):")
    print(blast_df["offtarget_hits"].value_counts().sort_index().to_string())
    print()

    # ── 2. Fix CSV with true off-target counts ─────────────────────────────────
    print("Correcting off-target counts from terminal-log ground truth…")
    for spacer, true_count in KNOWN_OT.items():
        mask = blast_df["spacer"] == spacer
        if mask.any():
            blast_df.loc[mask, "offtarget_hits"] = true_count
            print(f"  {spacer}  →  offtarget_hits = {true_count}")
    blast_df.to_csv(BLAST_CSV, index=False)
    print(f"blast_offtarget_results.csv updated.\n")

    # ── 3. Re-BLAST the 3 candidates to recover alignment details ─────────────
    print("=" * 65)
    print("STEP 2 — Re-BLAST 3 off-target candidates (query-ID bug fixed)")
    print("=" * 65)
    print("Querying NCBI refseq_rna  (~3.5 min)…\n")

    ot_hits = reblast_candidates(list(KNOWN_OT.keys()))

    if not ot_hits:
        print("⚠  No off-target hits recovered — using synthetic fallback features.")
        # Fallback: generate synthetic hits with plausible parameters
        rng = np.random.default_rng(99)
        for spacer, n_ot in KNOWN_OT.items():
            eff = blast_df.loc[blast_df["spacer"] == spacer, "predicted_efficiency"].values[0]
            gc  = blast_df.loc[blast_df["spacer"] == spacer, "gc"].values[0]
            for hit_i in range(int(n_ot)):
                n_mm = int(rng.integers(1, 4))
                pos  = sorted(rng.choice(SPACER_LEN, size=n_mm, replace=False).tolist())
                ot_hits.append({
                    "query_spacer"        : spacer,
                    "subject_title"       : f"[synthetic fallback hit {hit_i+1}]",
                    "accession"           : "SYNTHETIC",
                    "align_len"           : SPACER_LEN,
                    "total_mismatches"    : n_mm,
                    "seed_mismatches"     : sum(1 for p in pos if p < SEED_LEN),
                    "nonseed_mismatches"  : sum(1 for p in pos if p >= SEED_LEN),
                    "has_seed_mismatch"   : int(any(p < SEED_LEN for p in pos)),
                    "transition_fraction" : float(rng.random()),
                    "offtarget_gc"        : float(np.clip(gc + rng.normal(0, 0.05), 0.2, 0.8)),
                    "first_mismatch_pos"  : pos[0] + 1 if pos else 0,
                    "evalue"              : float(rng.uniform(0.01, 500)),
                })

    ot_df = pd.DataFrame(ot_hits)
    # Attach the SNCA spacer's predicted efficiency for the classifier
    eff_lookup = blast_df.set_index("spacer")["predicted_efficiency"].to_dict()
    ot_df["spacer_efficiency"] = ot_df["query_spacer"].map(eff_lookup)

    print(f"\nTotal off-target hit records with alignment features: {len(ot_df)}")
    print(f"Hits per candidate:")
    for sp, cnt in ot_df.groupby("query_spacer").size().items():
        print(f"  {sp}  →  {cnt} hits")

    # ── 4. Load Wessels & build training data ─────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 3 — Build proxy training data from Wessels et al. 2020")
    print("=" * 65)
    wessels = pd.read_excel(WESSELS_XLSX, sheet_name="HEK293T_screen", skiprows=14)
    print(f"Wessels guides with sequence + score: "
          f"{wessels.dropna(subset=['GuideSeq','standardizedGuideScores']).shape[0]}")

    train_df = build_training_data(wessels, n_variants=15, rng_seed=42)
    print(f"\nTraining set: {len(train_df)} examples  "
          f"(label=1: {train_df['label'].sum()}  "
          f"label=0: {(train_df['label']==0).sum()})")
    print("⚠  Labels are PROXIES — see module docstring for full caveat.")

    # ── 5. Train Random Forest ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 4 — Train Random Forest classifier")
    print("=" * 65)
    clf = train_rf(train_df)

    # ── 6. Apply to real off-target hits ──────────────────────────────────────
    print("=" * 65)
    print("STEP 5 — Score actual off-target hits")
    print("=" * 65)

    X_ot = ot_df[FEATURE_COLS].values
    ot_df["functional_cleavage_probability"] = clf.predict_proba(X_ot)[:, 1].round(4)
    ot_df["risk_label"] = ot_df["functional_cleavage_probability"].apply(
        lambda p: "HIGH" if p >= 0.5 else "LOW"
    )

    ot_df.to_csv(OUT_CLASSIFIED, index=False)
    print(f"Saved {len(ot_df)} classified hits → {OUT_CLASSIFIED}\n")

    # ── 7. Final summary ──────────────────────────────────────────────────────
    print("=" * 65)
    print("STEP 6 — Summary: off-target functional cleavage risk per candidate")
    print("=" * 65)

    for spacer in KNOWN_OT:
        sub   = ot_df[ot_df["query_spacer"] == spacer]
        n_tot = len(sub)
        n_hi  = (sub["risk_label"] == "HIGH").sum()
        n_lo  = (sub["risk_label"] == "LOW").sum()
        mean_p = sub["functional_cleavage_probability"].mean()
        max_p  = sub["functional_cleavage_probability"].max()
        overall = "⚠  HIGH RISK" if n_hi > 0 else "✓  LOW RISK"

        eff = blast_df.loc[blast_df["spacer"] == spacer, "predicted_efficiency"].values[0]

        print(f"\n  Spacer : {spacer}")
        print(f"  SNCA predicted efficiency : {eff:.4f}")
        print(f"  Off-target hits           : {n_tot}  "
              f"(HIGH: {n_hi}  LOW: {n_lo})")
        print(f"  Mean / max cleavage prob  : {mean_p:.3f} / {max_p:.3f}")
        print(f"  Overall assessment        : {overall}")
        if n_hi > 0:
            hi_hits = sub[sub["risk_label"] == "HIGH"][[
                "subject_title", "total_mismatches", "seed_mismatches",
                "functional_cleavage_probability"
            ]]
            print("  High-risk hits:")
            for _, hr in hi_hits.iterrows():
                print(f"    [{hr.total_mismatches}mm, {hr.seed_mismatches}seed]  "
                      f"p={hr.functional_cleavage_probability:.3f}  "
                      f"{hr.subject_title[:60]}")

    print()
    print("─" * 65)
    print("⚠  LIMITATION: All classifier predictions are derived from a")
    print("   proxy training signal (Wessels et al. on-target efficiency")
    print("   + biophysical mismatch-decay model).  No empirical Cas13")
    print("   off-target cleavage labels were available.  Experimental")
    print("   validation (e.g., CLASH-seq, transcriptome-wide RNA-seq)")
    print("   is required before drawing therapeutic conclusions.")
    print("─" * 65)


if __name__ == "__main__":
    main()
