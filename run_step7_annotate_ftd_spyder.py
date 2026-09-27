# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 13:09:38 2026

@author: nmehta22
"""

# run_step7_annotate_ftd_spyder.py
# Spyder-friendly: edit PATHS below, then Run.

import os
import gzip
import re
import math
import numpy as np
import pandas as pd
from pathlib import Path
import requests
import gzip

# -----------------------
# PATHS (EDIT THESE)
# -----------------------
IN_ALL = r"work\step6_prioritize_ftd\FTD_all_variants_with_ranks.tsv.gz"
OUT_ANNOT = r"work\step7_annotate_ftd\FTD_all_variants_annotated.tsv.gz"

RES_DIR = r"data\annotations"
os.makedirs(RES_DIR, exist_ok=True)

# GENCODE (choose a release you want; using release 49 basic as example)
GENCODE_GTF_GZ = os.path.join(RES_DIR, "gencode.v49.basic.annotation.gtf.gz")
GENCODE_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_49/"
    "gencode.v49.basic.annotation.gtf.gz"
)

# --- SCREEN / Registry v4 cCREs (GRCh38/hg38) ---
CCRE_BASE = "https://downloads.wenglab.org/Registry-V4"
CCRE_PLS_URL = f"{CCRE_BASE}/GRCh38-cCREs.PLS.bed"      # promoter-like
# Optional if you want enhancers too:
CCRE_ELS_URL = f"{CCRE_BASE}/GRCh38-cCREs.ELS.bed"      # all enhancers (pELS+dELS)
CCRE_pELS_URL = f"{CCRE_BASE}/GRCh38-cCREs.pELS.bed"
CCRE_dELS_URL = f"{CCRE_BASE}/GRCh38-cCREs.dELS.bed"

CCRE_PLS_BED = Path("data/annotations/GRCh38-cCREs.PLS.bed")
# (optional)
CCRE_ELS_BED = Path("data/annotations/GRCh38-cCREs.ELS.bed")

CCRE_PLS_BED = Path(r"data\annotations\GRCh38-cCREs.PLS.bed")
CCRE_ELS_BED = Path(r"data\annotations\GRCh38-cCREs.ELS.bed")  # enhancers (ELS = pELS + dELS)

# Toggle cCRE overlap (requires intervaltree)
DO_CCRE = True

# -----------------------
# Helpers: download
# -----------------------


def download_if_missing(url, outpath, chunk_size=1024*1024):
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    if outpath.exists() and outpath.stat().st_size > 0:
        print(f"Exists: {outpath}")
        return

    print(f"Downloading: {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(outpath, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
    print(f"Saved: {outpath}")


# -----------------------
# Helpers: chr normalize
# -----------------------
def normalize_chr(x):
    s = str(x)
    if not s.startswith("chr"):
        s = "chr" + s
    return s

# -----------------------
# Parse GENCODE genes
# -----------------------
def parse_gtf_genes(gtf_gz_path):
    """
    Return DataFrame with columns:
    chr, start, end, strand, gene_id, gene_name, gene_type, tss
    """
    genes = []
    attr_re = re.compile(r'(\S+)\s+"([^"]+)"')
    with gzip.open(gtf_gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line or line[0] == "#":
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            chrom, source, feature, start, end, score, strand, frame, attrs = parts
            if feature != "gene":
                continue
            chrom = normalize_chr(chrom)
            start = int(start)
            end = int(end)

            d = dict(attr_re.findall(attrs))
            gene_id = d.get("gene_id", "")
            gene_name = d.get("gene_name", "")
            gene_type = d.get("gene_type", d.get("gene_biotype", ""))

            tss = start if strand == "+" else end
            genes.append((chrom, start, end, strand, gene_id, gene_name, gene_type, tss))

    gdf = pd.DataFrame(
        genes,
        columns=["chr","start","end","strand","gene_id","gene_name","gene_type","tss"]
    )
    return gdf

# -----------------------
# Nearest gene by TSS (fast, numpy)
# -----------------------
def build_tss_index(gdf):
    """
    For each chr: sorted arrays of tss and matching row indices.
    """
    idx = {}
    for chrom, sub in gdf.groupby("chr", sort=False):
        sub2 = sub.sort_values("tss").reset_index(drop=False)  # keep original row index
        idx[chrom] = {
            "tss": sub2["tss"].to_numpy(dtype=np.int64),
            "row_i": sub2["index"].to_numpy(dtype=np.int64),
        }
    return idx

def nearest_gene_for_variant(chrom, pos, tss_index, gdf):
    if chrom not in tss_index:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)
    tss_arr = tss_index[chrom]["tss"]
    row_i = tss_index[chrom]["row_i"]

    j = np.searchsorted(tss_arr, pos)
    candidates = []
    if j > 0:
        candidates.append(j-1)
    if j < len(tss_arr):
        candidates.append(j)
    if not candidates:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)

    best = None
    best_dist = None
    for c in candidates:
        dist = abs(int(pos) - int(tss_arr[c]))
        if best is None or dist < best_dist:
            best = c
            best_dist = dist

    gene = gdf.loc[row_i[best]]
    within_gene = (gene["start"] <= pos <= gene["end"])
    signed_dist = (pos - gene["tss"])  # + means downstream of TSS
    return (
        gene["gene_name"],
        gene["gene_id"],
        gene["gene_type"],
        signed_dist,
        bool(within_gene),
    )

# -----------------------
# cCRE overlap (optional)
# -----------------------
import gzip

def open_maybe_gz(path, mode="rt"):
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, mode)
    return open(path, mode, encoding="utf-8")

def load_bed_as_intervals(bed_path, label):
    """
    Read a BED (or BED.GZ) and return a dict:
        {chrom: [(start0, end0, label), ...]}
    Coordinates stay BED-style: 0-based, half-open [start0, end0).
    """
    intervals = {}
    with open_maybe_gz(bed_path, "rt") as f:
        for line in f:
            if (not line) or line.startswith("#") or line.startswith("track") or line.startswith("browser"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue

            chrom = normalize_chr(parts[0])
            try:
                start0 = int(parts[1])
                end0 = int(parts[2])
            except ValueError:
                continue

            if end0 <= start0:
                continue

            intervals.setdefault(chrom, []).append((start0, end0, label))
    return intervals



def build_intervaltrees(interval_dict):
    from intervaltree import IntervalTree
    trees = {}
    for chrom, ivs in interval_dict.items():
        t = IntervalTree()
        for s,e,label in ivs:
            t.addi(s, e, label)
        trees[chrom] = t
    return trees

def query_ccre(chrom, pos, trees):
    # convert 1-based to 0-based coordinate
    pos0 = int(pos) - 1
    if chrom not in trees:
        return ""
    hits = trees[chrom].at(pos0)
    if not hits:
        return ""
    # join unique labels
    labs = sorted({iv.data for iv in hits})
    return ",".join(labs)

# -----------------------
# Score normalization (optional but recommended)
# -----------------------
def add_within_locus_percentiles(df):
    """
    Makes a score that's robust to scale: per-locus percentile rank per scorer,
    then takes max percentile across scorers.
    """
    scorer_cols = [c for c in df.columns if c.endswith("_max_abs") and c != "alphagenome_max_abs"]

    # drop scorers that are 100% NaN
    keep = []
    for c in scorer_cols:
        if df[c].notna().any():
            keep.append(c)
    scorer_cols = keep

    if not scorer_cols:
        df["alphagenome_pct_max"] = np.nan
        return df

    # percentile within locus for each scorer
    pct_cols = []
    for c in scorer_cols:
        pc = c.replace("_max_abs", "_pct")
        pct_cols.append(pc)
        df[pc] = (
            df.groupby("locus_id")[c]
              .rank(pct=True, method="average")
        )

    df["alphagenome_pct_max"] = df[pct_cols].max(axis=1)
    return df

# -----------------------
# MAIN
# -----------------------
def main():
    # 1) Load your Step 6 table
    df = pd.read_csv(IN_ALL, sep="\t", compression="gzip")
    df["chr"] = df["chr"].map(normalize_chr)

    print("Loaded:", df.shape)
    print("Columns:", list(df.columns)[:12], "...")

    # 2) Download annotation resources
    download_if_missing(GENCODE_URL, GENCODE_GTF_GZ)
    if DO_CCRE:
        download_if_missing(CCRE_PLS_URL, CCRE_PLS_BED)
        # optional:
        download_if_missing(CCRE_ELS_URL, CCRE_ELS_BED)

    # 3) Parse genes
    print("Parsing GENCODE genes...")
    gdf = parse_gtf_genes(GENCODE_GTF_GZ)
    print("Genes:", gdf.shape)

    tss_index = build_tss_index(gdf)

    # 4) Nearest gene per variant
    print("Annotating nearest genes...")
    gene_name = []
    gene_id = []
    gene_type = []
    dist_tss = []
    within_gene = []

    for r in df.itertuples(index=False):
        gn, gid, gt, d, wg = nearest_gene_for_variant(r.chr, int(r.pos), tss_index, gdf)
        gene_name.append(gn)
        gene_id.append(gid)
        gene_type.append(gt)
        dist_tss.append(d)
        within_gene.append(wg)

    df["nearest_gene"] = gene_name
    df["nearest_gene_id"] = gene_id
    df["nearest_gene_type"] = gene_type
    df["dist_to_tss_bp"] = dist_tss
    df["within_nearest_gene"] = within_gene

    # 5) cCRE overlap
    if DO_CCRE:
        try:
            from intervaltree import IntervalTree  # noqa: F401
        except Exception:
            raise RuntimeError(
                "intervaltree not installed. In Anaconda Prompt:\n"
                "  pip install intervaltree\n"
                "Then re-run."
            )

        print("Loading cCRE BEDs...")
        pls = load_bed_as_intervals(CCRE_PLS_BED, "PLS")
        els = load_bed_as_intervals(CCRE_ELS_BED, "ELS")

        # merge dicts
        merged = {}
        for chrom, ivs in pls.items():
            merged.setdefault(chrom, []).extend(ivs)
        for chrom, ivs in els.items():
            merged.setdefault(chrom, []).extend(ivs)

        print("Building interval trees...")
        trees = build_intervaltrees(merged)

        print("Annotating cCRE overlap...")
        df["ccre_overlap"] = [query_ccre(c, p, trees) for c,p in zip(df["chr"], df["pos"])]

    # 6) Add robust within-locus percentile summary score
    df = add_within_locus_percentiles(df)

    # 7) Save
    os.makedirs(os.path.dirname(OUT_ANNOT), exist_ok=True)
    df.to_csv(OUT_ANNOT, sep="\t", index=False, compression="gzip")
    print("Wrote:", OUT_ANNOT)

    # Quick sanity:
    print("alphagenome_max_abs NaN fraction:", df["alphagenome_max_abs"].isna().mean())
    if "alphagenome_pct_max" in df.columns:
        print("alphagenome_pct_max NaN fraction:", df["alphagenome_pct_max"].isna().mean())
    if DO_CCRE:
        print("cCRE overlap non-empty fraction:", (df["ccre_overlap"].astype(str) != "").mean())

if __name__ == "__main__":
    main()
