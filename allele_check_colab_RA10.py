"""
RA10 (Colab) — reference-allele audit of every analysed variant against GRCh38.

No model inference: this is a FASTA lookup for ~6,800 variants and takes seconds
once the chromosome files are present. Run it in the same Colab session as the
Enformer job if that session is still alive (the .fa files will already be there).

For each variant it classifies what the recorded ref/alt actually are, relative
to the GRCh38 base at that position:

    match             ref == genome            -> correct
    swapped           alt == genome            -> ref/alt are reversed; the
                                                  "ref" column is really the
                                                  alternate allele
    strand_flip       complement(ref) == genome-> lifted to the minus strand
                                                  without complementing alleles
    strand_flip_swap  complement(alt) == genome-> both problems together
    unresolved        none of the above        -> wrong position, indel, or a
                                                  genuinely different variant

The distribution across these categories identifies the cause. A swap-dominated
profile means allele ORDER was taken from a GWAS ID that lists A1/A2 rather than
ref/alt. A strand-flip profile means the liftover crossed an inverted segment.
"unresolved" would point at a positional error, which is the most serious.

Upload:  RA10_variants_for_allele_check.tsv   (from run_RA10a_export_for_allele_check.py)
Run:     %run allele_check_colab_RA10.py
Download: RA10_allele_check_results.tsv  ->  work/step_RA10_allele_audit/
"""

import importlib
import subprocess
import sys

for _mod, _pkg in [("pyfaidx", "pyfaidx")]:
    try:
        importlib.import_module(_mod)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", _pkg], check=True)

import gzip
import os
import shutil
import time
import urllib.request

import pandas as pd

VARIANT_TSV = "RA10_variants_for_allele_check.tsv"
OUT_TSV = "RA10_allele_check_results.tsv"

variants = pd.read_csv(VARIANT_TSV, sep="\t")
print(f"variants: {len(variants):,}")
print(variants["chr"].value_counts().to_string())

# ------------------------------------------------------------------
# Reference genome (reuses files already downloaded by the Enformer run)
# ------------------------------------------------------------------
SOURCES = [
    ("UCSC-soe", "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/{c}.fa.gz"),
    ("UCSC-cse", "https://hgdownload.cse.ucsc.edu/goldenPath/hg38/chromosomes/{c}.fa.gz"),
    ("Ensembl", "https://ftp.ensembl.org/pub/release-110/fasta/homo_sapiens/dna/"
                "Homo_sapiens.GRCh38.dna.chromosome.{n}.fa.gz"),
]


def download_chrom(c):
    fa, gz = f"{c}.fa", f"{c}.fa.gz"
    if os.path.exists(fa) and os.path.getsize(fa) > 1_000_000:
        print(f"{c}: already present")
        return
    if not (os.path.exists(gz) and os.path.getsize(gz) > 1_000_000):
        n = c.replace("chr", "")
        last = None
        for name, tmpl in SOURCES:
            for attempt in range(1, 4):
                try:
                    print(f"{c}: {name} attempt {attempt} ...", flush=True)
                    req = urllib.request.Request(tmpl.format(c=c, n=n),
                                                 headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(req, timeout=120) as r, open(gz, "wb") as f:
                        shutil.copyfileobj(r, f)
                    last = None
                    break
                except Exception as e:
                    last = e
                    print(f"   failed: {type(e).__name__}: {e}", flush=True)
                    time.sleep(5 * attempt)
            if last is None:
                break
        if last is not None:
            raise SystemExit(f"could not download {c}: {last}")
    with gzip.open(gz, "rt") as fin, open(fa, "w") as fout:
        first = fin.readline()
        fout.write(f">{c}\n" if first.startswith(">") else first)
        shutil.copyfileobj(fin, fout)
    os.remove(gz)


from pyfaidx import Fasta

for c in sorted(variants["chr"].astype(str).unique()):
    download_chrom(c)
genomes = {c: Fasta(f"{c}.fa", as_raw=True, sequence_always_upper=True)
           for c in sorted(variants["chr"].astype(str).unique())}

# ------------------------------------------------------------------
# Classify
# ------------------------------------------------------------------
COMP = str.maketrans("ACGT", "TGCA")


def revcomp_base(s):
    return s.translate(COMP)[::-1]


rows = []
for r in variants.itertuples():
    chrom, pos = str(r.chr), int(r.pos)
    ref, alt = str(r.ref).upper(), str(r.alt).upper()
    g = genomes[chrom][chrom]
    if pos < 1 or pos + len(ref) - 1 > len(g):
        rows.append({"variant_key": r.variant_key, "genome_base": None,
                     "allele_status": "out_of_bounds"})
        continue

    obs = g[pos - 1: pos - 1 + max(len(ref), 1)].upper()

    if obs == ref:
        status = "match"
    elif obs == alt[:len(obs)]:
        status = "swapped"
    elif obs == revcomp_base(ref):
        status = "strand_flip"
    elif obs == revcomp_base(alt)[:len(obs)]:
        status = "strand_flip_swap"
    else:
        status = "unresolved"

    rows.append({"variant_key": r.variant_key, "genome_base": obs,
                 "allele_status": status})

res = variants.merge(pd.DataFrame(rows), on="variant_key", how="left")
res.to_csv(OUT_TSV, sep="\t", index=False)

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("ALLELE STATUS, OVERALL")
print("=" * 70)
print(res["allele_status"].value_counts().to_string())
print(f"\nmatch rate: {100.0 * res['allele_status'].eq('match').mean():.2f}%")

print("\n" + "=" * 70)
print("BY CHROMOSOME")
print("=" * 70)
print(pd.crosstab(res["chr"], res["allele_status"]).to_string())

if "signal_id" in res.columns:
    print("\n" + "=" * 70)
    print("BY SIGNAL")
    print("=" * 70)
    print(pd.crosstab(res["signal_id"], res["allele_status"]).to_string())

bad = res[~res["allele_status"].isin(["match"])]
if len(bad) and "final_score" in bad.columns:
    print("\n" + "=" * 70)
    print("HIGHEST-SCORING AFFECTED VARIANTS")
    print("=" * 70)
    cols = [c for c in ["variant_key", "signal_id", "p", "final_score",
                        "alphagenome_score", "ref", "alt", "genome_base",
                        "allele_status"] if c in bad.columns]
    print(bad.nlargest(15, "final_score")[cols].to_string(index=False))

print(f"\nwrote {OUT_TSV} — download to work/step_RA10_allele_audit/")
