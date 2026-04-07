"""
02_rnafold_accessibility.py
---------------------------
Score each gRNA candidate in grna_candidates.csv for RNA secondary structure
accessibility at its target site in the SNCA mRNA.

For each candidate:
  • Extract a 50-nt window centred on the 23-nt target site (clamped at
    sequence boundaries).
  • Run RNAfold (ViennaRNA) on the RNA version of that window.
  • Parse the minimum free energy (MFE) from the dot-bracket output.
  • Compute accessibility = fraction of unpaired bases (dots) within the
    23-nt target region of the dot-bracket structure.

Columns added: mfe, accessibility
Output: output/grna_candidates_scored.csv
Progress printed every 250 sequences.
"""

import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
from Bio import SeqIO

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE       = Path(__file__).parent.parent
FASTA      = BASE / "data"   / "snca.fasta"
IN_CSV     = BASE / "output" / "grna_candidates.csv"
OUT_CSV    = BASE / "output" / "grna_candidates_scored.csv"

# ── Parameters ────────────────────────────────────────────────────────────────
SPACER_LEN  = 23
WINDOW_LEN  = 50                       # total context window for RNAfold
FLANK       = (WINDOW_LEN - SPACER_LEN) // 2   # 13 nt on each side
BATCH_SIZE  = 250
MFE_RE      = re.compile(r"\(\s*(-?\d+\.\d+)\s*\)$")


# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_window(seq: str, start: int) -> tuple[str, int]:
    """Return (rna_window, target_offset_within_window).

    target_offset is the index inside the returned window where the
    23-nt spacer target begins.
    """
    total     = len(seq)
    win_start = max(0, start - FLANK)
    win_end   = min(total, start + SPACER_LEN + FLANK)
    window    = seq[win_start:win_end].replace("T", "U")
    offset    = start - win_start
    return window, offset


def run_rnafold_batch(seqs: list[str]) -> list[tuple[str, float]]:
    """Submit *seqs* to RNAfold in FASTA format; return list of (structure, mfe)."""
    fasta_in = "".join(f">s{i}\n{s}\n" for i, s in enumerate(seqs))

    result = subprocess.run(
        ["RNAfold", "--noPS"],
        input=fasta_in,
        capture_output=True,
        text=True,
        check=True,
    )

    # Output lines per entry: ">header", "SEQUENCE", "STRUCTURE (MFE)"
    outputs: list[tuple[str, float]] = []
    lines = [l.rstrip() for l in result.stdout.splitlines() if l.strip()]
    i = 0
    while i < len(lines):
        if lines[i].startswith(">"):
            struct_line = lines[i + 2]          # third line is structure
            struct      = struct_line.split()[0]
            m           = MFE_RE.search(struct_line)
            mfe         = float(m.group(1)) if m else float("nan")
            outputs.append((struct, mfe))
            i += 3
        else:
            i += 1
    return outputs


def accessibility(struct: str, offset: int) -> float:
    """Fraction of unpaired ('.' ) bases in the spacer region of *struct*."""
    target = struct[offset : offset + SPACER_LEN]
    if not target:
        return float("nan")
    return target.count(".") / len(target)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    record = SeqIO.read(FASTA, "fasta")
    seq    = str(record.seq).upper()

    df = pd.read_csv(IN_CSV)
    n  = len(df)
    print(f"Loaded {n} candidates from {IN_CSV.name}")
    print(f"Sequence: {record.id} ({len(seq)} nt)")
    print(f"Window: {WINDOW_LEN} nt  |  Flank: {FLANK} nt each side")
    print(f"Batch size: {BATCH_SIZE}\n")

    # Pre-compute windows and offsets
    windows: list[str] = []
    offsets: list[int] = []
    for start in df["start"]:
        w, off = extract_window(seq, int(start))
        windows.append(w)
        offsets.append(off)

    mfe_list:    list[float] = []
    access_list: list[float] = []
    processed = 0

    for batch_start in range(0, n, BATCH_SIZE):
        batch_end  = min(batch_start + BATCH_SIZE, n)
        batch_wins = windows[batch_start:batch_end]
        batch_offs = offsets[batch_start:batch_end]

        results = run_rnafold_batch(batch_wins)

        for (struct, mfe), off in zip(results, batch_offs):
            mfe_list.append(mfe)
            access_list.append(accessibility(struct, off))

        processed = batch_end
        pct = processed / n * 100
        print(f"  Processed {processed:>5} / {n}  ({pct:5.1f}%)  "
              f"last MFE: {mfe_list[-1]:6.2f} kcal/mol  "
              f"last acc: {access_list[-1]:.3f}")
        sys.stdout.flush()

    df["mfe"]           = mfe_list
    df["accessibility"] = access_list

    df.to_csv(OUT_CSV, index=False)

    # Summary
    print(f"\n── Summary ──────────────────────────────────────")
    print(f"MFE          mean: {df['mfe'].mean():.2f}  "
          f"min: {df['mfe'].min():.2f}  max: {df['mfe'].max():.2f} kcal/mol")
    print(f"Accessibility mean: {df['accessibility'].mean():.3f}  "
          f"min: {df['accessibility'].min():.3f}  max: {df['accessibility'].max():.3f}")
    high_acc = (df["accessibility"] >= 0.7).sum()
    print(f"Candidates with accessibility ≥ 0.70: {high_acc} ({high_acc/n*100:.1f}%)")
    print(f"\nSaved → {OUT_CSV}")


if __name__ == "__main__":
    main()
