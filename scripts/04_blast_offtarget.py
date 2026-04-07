"""
04_blast_offtarget.py
---------------------
Off-target analysis for the top 100 SNCA gRNA candidates via BLAST.

Strategy (in priority order):
  1. Local BLAST DB  – use directly if found in data/blastdb/
  2. Local FASTA     – build DB from data/human.rna.fna(.gz) if present
  3. Download        – fetch human RefSeq RNA from NCBI FTP, build DB,
                       run local blastn (fast, single batch)
  4. Remote qblast   – fallback when disk space < MIN_FREE_GB or download
                       fails; queries NCBI RefSeq RNA one sequence at a time
                       with a 10-second delay between requests

Off-target criteria  : alignment ≥ 20 nt  AND  mismatches ≤ 3
SNCA self-matches    : excluded by accession (NM_000345, NM_007308) and
                       title keywords ("SNCA", "synuclein")

Outputs:
  output/top_candidates.fasta
  output/blast_offtarget_results.csv
"""

import gzip
import shutil
import subprocess
import sys
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from Bio import SeqIO
from Bio.Blast import NCBIXML, NCBIWWW
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

# ── Paths & constants ─────────────────────────────────────────────────────────
BASE          = Path(__file__).parent.parent
PREDICTED_CSV = BASE / "output" / "grna_candidates_predicted.csv"
TOP_FASTA     = BASE / "output" / "top_candidates.fasta"
OUT_CSV       = BASE / "output" / "blast_offtarget_results.csv"
DB_DIR        = BASE / "data"   / "blastdb"
RNA_GZ        = BASE / "data"   / "human.rna.fna.gz"
RNA_FASTA     = BASE / "data"   / "human.rna.fna"
DB_PREFIX     = str(DB_DIR / "human_rna")

TRANSCRIPTOME_URL = (
    "https://ftp.ncbi.nlm.nih.gov/refseq/H_sapiens/mRNA_Prot/human.rna.fna.gz"
)

TOP_N           = 100
MIN_FREE_GB     = 6      # minimum free disk space to attempt download
ALIGN_MIN_LEN   = 20     # minimum HSP length to consider
MAX_MISMATCHES  = 3      # ≤ this many mismatches = off-target
BLAST_DELAY     = 10     # seconds between remote qblast calls
SNCA_ACCESSIONS = {"NM_000345", "NM_007308"}   # known SNCA RefSeq accessions
SNCA_KEYWORDS   = {"snca", "synuclein"}


# ── Helpers ───────────────────────────────────────────────────────────────────
def free_gb() -> float:
    total, used, free = shutil.disk_usage(BASE)
    return free / 1e9


def is_snca(title: str, accession: str) -> bool:
    acc_base = accession.split(".")[0]
    if acc_base in SNCA_ACCESSIONS:
        return True
    return any(kw in title.lower() for kw in SNCA_KEYWORDS)


def hsp_mismatches(hsp) -> int:
    return hsp.align_length - hsp.identities - hsp.gaps


def parse_blast_records(xml_handle) -> list[dict]:
    """Parse BLAST XML into a flat list of off-target dicts."""
    hits = []
    for record in NCBIXML.parse(xml_handle):
        query = record.query.split()[0]
        for alignment in record.alignments:
            accession = alignment.accession
            title     = alignment.title
            for hsp in alignment.hsps:
                if hsp.align_length < ALIGN_MIN_LEN:
                    continue
                mm = hsp_mismatches(hsp)
                if mm > MAX_MISMATCHES:
                    continue
                hits.append({
                    "query_spacer": query,
                    "subject_acc" : accession,
                    "subject_title": title[:80],
                    "align_len"   : hsp.align_length,
                    "mismatches"  : mm,
                    "identities"  : hsp.identities,
                    "evalue"      : hsp.expect,
                    "is_snca"     : is_snca(title, accession),
                })
    return hits


# ── BLAST mode detection ──────────────────────────────────────────────────────
def detect_local_db() -> bool:
    """Return True if a usable local BLAST DB exists."""
    nhr = Path(DB_PREFIX + ".nhr")
    nsq = Path(DB_PREFIX + ".nsq")
    return nhr.exists() and nsq.exists()


def download_transcriptome() -> bool:
    """Download and optionally decompress the human RefSeq RNA FASTA.
    Returns True on success."""
    free = free_gb()
    if free < MIN_FREE_GB:
        print(f"  Skipping download: only {free:.1f} GB free "
              f"(need ≥ {MIN_FREE_GB} GB)")
        return False

    print(f"  Downloading human RefSeq RNA (~500 MB compressed)…")
    print(f"  URL: {TRANSCRIPTOME_URL}")
    try:
        with requests.get(TRANSCRIPTOME_URL, stream=True, timeout=30) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            RNA_GZ.parent.mkdir(parents=True, exist_ok=True)
            with open(RNA_GZ, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB chunks
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        print(f"\r  {downloaded/1e6:.0f} / {total/1e6:.0f} MB  "
                              f"({pct:.1f}%)", end="", flush=True)
        print()
        print(f"  Download complete → {RNA_GZ}")
    except Exception as exc:
        print(f"  Download failed: {exc}")
        return False

    # Decompress
    print(f"  Decompressing…")
    try:
        with gzip.open(RNA_GZ, "rb") as f_in, open(RNA_FASTA, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        print(f"  Decompressed → {RNA_FASTA}  ({RNA_FASTA.stat().st_size/1e9:.2f} GB)")
        RNA_GZ.unlink()   # remove gz to recover space
    except Exception as exc:
        print(f"  Decompression failed: {exc}")
        return False
    return True


def build_blast_db() -> bool:
    """Run makeblastdb on the local FASTA. Returns True on success."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    fasta_src = RNA_FASTA if RNA_FASTA.exists() else None
    if fasta_src is None:
        return False
    print(f"  Building BLAST DB from {fasta_src.name}  (this may take ~10 min)…")
    cmd = [
        "makeblastdb",
        "-in",    str(fasta_src),
        "-dbtype","nucl",
        "-out",   DB_PREFIX,
        "-title", "Human_RefSeq_RNA",
        "-parse_seqids",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  makeblastdb error:\n{result.stderr}")
        return False
    print(f"  DB built → {DB_DIR}/")
    return True


# ── BLAST runners ─────────────────────────────────────────────────────────────
def run_local_blast(fasta_path: Path) -> list[dict]:
    """Run blastn locally and parse XML results."""
    out_xml = fasta_path.with_suffix(".xml")
    cmd = [
        "blastn",
        "-query",         str(fasta_path),
        "-db",            DB_PREFIX,
        "-out",           str(out_xml),
        "-outfmt",        "5",            # XML
        "-task",          "blastn-short",
        "-word_size",     "7",
        "-evalue",        "1000",
        "-perc_identity", "78",           # ≥18/23 identities → catches ≤5 mm
        "-num_alignments","200",
        "-num_threads",   "4",
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  blastn error:\n{result.stderr}")
        return []
    with open(out_xml) as fh:
        hits = parse_blast_records(fh)
    return hits


def run_remote_blast(records: list[SeqRecord]) -> list[dict]:
    """Run qblast against NCBI refseq_rna one sequence at a time."""
    all_hits: list[dict] = []
    n = len(records)
    for i, rec in enumerate(records, 1):
        spacer = str(rec.seq)
        print(f"  [{i:>3}/{n}] {spacer}  ", end="", flush=True)
        t0 = time.time()

        for attempt in range(1, 4):
            try:
                handle = NCBIWWW.qblast(
                    program      = "blastn",
                    database     = "refseq_rna",
                    sequence     = f">query\n{spacer}\n",
                    entrez_query = "Homo sapiens[Organism]",
                    hitlist_size = 200,
                    expect       = 1000.0,
                    word_size    = 7,
                    filter       = "F",
                    short_query  = True,
                    megablast    = False,
                    format_type  = "XML",
                )
                xml_str = handle.read()
                handle.close()
                hits = parse_blast_records(StringIO(xml_str))
                break
            except Exception as exc:
                print(f"\n    attempt {attempt} failed: {exc}", end="")
                if attempt < 3:
                    time.sleep(20)
                else:
                    hits = []

        offtargets = [h for h in hits if not h["is_snca"]]
        elapsed    = time.time() - t0
        print(f"→ {len(hits)} hits  {len(offtargets)} off-targets  ({elapsed:.0f}s)")
        all_hits.extend(hits)

        if i < n:
            time.sleep(BLAST_DELAY)

    return all_hits


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("STEP 1 — Load top 100 candidates")
    print("=" * 65)

    df = pd.read_csv(PREDICTED_CSV)
    top = (
        df.nlargest(TOP_N, "predicted_efficiency")
        .reset_index(drop=True)
    )
    print(f"Loaded {len(df)} candidates; keeping top {len(top)}")
    print(f"Predicted efficiency range: "
          f"{top['predicted_efficiency'].min():.4f} – "
          f"{top['predicted_efficiency'].max():.4f}")

    # ── Write FASTA ───────────────────────────────────────────────────────────
    print(f"\nWriting {TOP_FASTA.name}…")
    records = [
        SeqRecord(
            Seq(row.spacer),
            id   = f"{row.spacer}",
            description = f"strand={row.strand} start={row.start}",
        )
        for row in top.itertuples()
    ]
    SeqIO.write(records, TOP_FASTA, "fasta")
    print(f"  Saved → {TOP_FASTA}")

    # ── BLAST mode selection ──────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 2 — Select BLAST mode")
    print("=" * 65)
    print(f"  Free disk space : {free_gb():.1f} GB")

    use_remote = False

    if detect_local_db():
        mode = "local_existing"
        print(f"  Mode: LOCAL (existing DB at {DB_DIR}/)")
    elif RNA_FASTA.exists():
        mode = "local_build"
        print(f"  Mode: LOCAL (building DB from {RNA_FASTA.name})")
    elif free_gb() >= MIN_FREE_GB:
        print(f"  Mode: DOWNLOAD → local blastn")
        ok = download_transcriptome()
        if ok:
            ok = build_blast_db()
        if ok:
            mode = "local_built"
        else:
            print("  Download/build failed — falling back to remote qblast")
            mode = "remote"
            use_remote = True
    else:
        print(f"  Mode: REMOTE qblast  (insufficient free space for download)")
        mode = "remote"
        use_remote = True

    if mode == "local_build":
        if not build_blast_db():
            print("  DB build failed — falling back to remote qblast")
            mode = "remote"
            use_remote = True

    # ── Run BLAST ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 3 — Run BLAST")
    print("=" * 65)

    if use_remote:
        print(f"Running remote qblast for {len(records)} sequences")
        print(f"Estimated time: {len(records) * (BLAST_DELAY + 45) // 60}–"
              f"{len(records) * (BLAST_DELAY + 90) // 60} min\n")
        all_hits = run_remote_blast(records)
    else:
        print(f"Running local blastn against {DB_PREFIX}")
        all_hits = run_local_blast(TOP_FASTA)
        n_with_hits = len({h["query_spacer"] for h in all_hits})
        print(f"  Total HSPs passing length/mismatch filter: {len(all_hits)}")
        print(f"  Candidates with ≥1 hit                  : {n_with_hits}")

    # ── Aggregate per-candidate ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 4 — Aggregate off-target counts")
    print("=" * 65)

    hits_df = pd.DataFrame(all_hits) if all_hits else pd.DataFrame(
        columns=["query_spacer","subject_acc","subject_title",
                 "align_len","mismatches","identities","evalue","is_snca"]
    )

    def summarise(spacer: str) -> dict:
        sub = hits_df[hits_df["query_spacer"] == spacer] if len(hits_df) else pd.DataFrame()
        snca_hits  = sub[sub["is_snca"]].shape[0]  if len(sub) else 0
        offt_hits  = sub[~sub["is_snca"]].shape[0] if len(sub) else 0
        genes = (
            sub[~sub["is_snca"]]["subject_title"]
            .str.extract(r"(?:NM_\d+\.\d+\s+)?(.+?)(?:,|\s+mRNA|\s+transcript|$)")[0]
            .dropna()
            .unique()[:5]
            .tolist()
        ) if len(sub) else []
        return {
            "snca_self_hits"  : snca_hits,
            "offtarget_hits"  : offt_hits,
            "offtarget_genes" : "; ".join(genes),
        }

    summaries = [summarise(row.spacer) for row in top.itertuples()]
    result_df = pd.concat(
        [top.reset_index(drop=True), pd.DataFrame(summaries)], axis=1
    )
    result_df.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(result_df)} rows → {OUT_CSV}")

    # ── Summary table: top 20 ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("TOP 20 CANDIDATES — predicted knockdown vs off-target hits")
    print("=" * 65)

    disp = result_df.head(20)[[
        "spacer", "strand", "start", "gc",
        "mfe", "accessibility", "predicted_efficiency",
        "snca_self_hits", "offtarget_hits",
    ]].copy()

    disp["gc"]                   = (disp["gc"] * 100).round(1).astype(str) + "%"
    disp["mfe"]                  = disp["mfe"].map("{:.1f}".format)
    disp["accessibility"]        = disp["accessibility"].map("{:.3f}".format)
    disp["predicted_efficiency"] = disp["predicted_efficiency"].map("{:.4f}".format)
    disp.index = range(1, len(disp) + 1)

    print(disp.to_string())

    # Off-target summary stats
    zero_offt = (result_df["offtarget_hits"] == 0).sum()
    print(f"\nOf the top {TOP_N} candidates:")
    print(f"  {zero_offt} ({zero_offt}%) have zero off-targets (≤3 mismatches)")
    print(f"  Mean off-target hits : {result_df['offtarget_hits'].mean():.2f}")
    print(f"  Max  off-target hits : {result_df['offtarget_hits'].max()}")


if __name__ == "__main__":
    main()
