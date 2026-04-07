"""
01_grna_generation.py
---------------------
Generate and filter Cas13 gRNA candidates from the SNCA mRNA sequence (NM_000345.4).

Filters applied (in order):
  1. PFS compatibility  — no G at the 3' end of the 23-nt spacer (Cas13a requirement)
  2. GC content        — spacer GC must be between 30 % and 70 %
  3. Homopolymer runs  — no run of 4+ identical consecutive nucleotides

Output: output/grna_candidates.csv
"""

import csv
import re
from pathlib import Path
from Bio import SeqIO

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE    = Path(__file__).parent.parent
FASTA   = BASE / "data"   / "snca.fasta"
OUT_CSV = BASE / "output" / "grna_candidates.csv"
OUT_CSV.parent.mkdir(exist_ok=True)

# ── Parameters ────────────────────────────────────────────────────────────────
SPACER_LEN   = 23
GC_MIN, GC_MAX = 0.30, 0.70
HOMOPOLY_RUN = 4          # reject if any nucleotide repeats >= this many times


# ── Filter functions ──────────────────────────────────────────────────────────
def passes_pfs(spacer: str) -> bool:
    """Cas13a PFS: the nucleotide immediately 3' of the spacer must not be G.
    Here we encode that as: the last nt of the 23-mer itself must not be G,
    matching the convention that the protospacer 3' end faces the PFS."""
    return spacer[-1].upper() != "G"


def passes_gc(spacer: str) -> bool:
    s = spacer.upper()
    gc = (s.count("G") + s.count("C")) / len(s)
    return GC_MIN <= gc <= GC_MAX


def passes_homopolymer(spacer: str) -> bool:
    return not re.search(r"(.)\1{%d,}" % (HOMOPOLY_RUN - 1), spacer, re.IGNORECASE)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    record = SeqIO.read(FASTA, "fasta")
    seq    = str(record.seq).upper()
    total  = len(seq)

    print(f"Sequence : {record.id}  ({total} nt)")
    print(f"Spacer   : {SPACER_LEN} nt")
    print(f"GC range : {GC_MIN*100:.0f}–{GC_MAX*100:.0f}%")
    print(f"Homopoly : reject runs ≥ {HOMOPOLY_RUN}")
    print()

    # Slide across both strands
    candidates = []
    for strand, s in [("+", seq), ("-", str(record.seq.reverse_complement()).upper())]:
        for i in range(len(s) - SPACER_LEN + 1):
            spacer = s[i : i + SPACER_LEN]
            if "N" in spacer:
                continue
            pos = i if strand == "+" else total - i - SPACER_LEN
            candidates.append({"spacer": spacer, "strand": strand, "start": pos})

    print(f"Total 23-mers (both strands, no N): {len(candidates):>6}")

    # Filter 1: PFS
    after_pfs = [c for c in candidates if passes_pfs(c["spacer"])]
    print(f"After PFS filter (no 3′ G)        : {len(after_pfs):>6}"
          f"  ({len(candidates)-len(after_pfs):>5} removed)")

    # Filter 2: GC content
    after_gc = [c for c in after_pfs if passes_gc(c["spacer"])]
    print(f"After GC filter (30–70 %)         : {len(after_gc):>6}"
          f"  ({len(after_pfs)-len(after_gc):>5} removed)")

    # Filter 3: Homopolymer
    after_hp = [c for c in after_gc if passes_homopolymer(c["spacer"])]
    print(f"After homopolymer filter (no ≥4)  : {len(after_hp):>6}"
          f"  ({len(after_gc)-len(after_hp):>5} removed)")

    print(f"\nFinal candidates: {len(after_hp)}")

    # Add GC value for downstream use
    for c in after_hp:
        s = c["spacer"]
        c["gc"] = round((s.count("G") + s.count("C")) / len(s), 4)

    # Save
    fields = ["spacer", "strand", "start", "gc"]
    with open(OUT_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(after_hp)

    print(f"Saved  → {OUT_CSV}")


if __name__ == "__main__":
    main()
