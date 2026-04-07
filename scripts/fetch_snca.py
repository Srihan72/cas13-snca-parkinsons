"""Download SNCA mRNA sequence NM_000345.4 from NCBI and save as data/snca.fasta."""

from pathlib import Path
from Bio import Entrez, SeqIO

Entrez.email = "hyperionovax@gmail.com"

OUT = Path(__file__).parent.parent / "data" / "snca.fasta"

def fetch_snca():
    print("Fetching NM_000345.4 from NCBI...")
    handle = Entrez.efetch(db="nucleotide", id="NM_000345.4", rettype="fasta", retmode="text")
    record = SeqIO.read(handle, "fasta")
    handle.close()

    SeqIO.write(record, OUT, "fasta")
    print(f"Saved: {OUT}")
    print(f"ID:     {record.id}")
    print(f"Length: {len(record.seq)} nt")
    print(f"Seq preview: {record.seq[:60]}...")

if __name__ == "__main__":
    fetch_snca()
