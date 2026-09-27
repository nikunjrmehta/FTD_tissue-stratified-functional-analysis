# -*- coding: utf-8 -*-
"""
Created on Sun Feb 15 17:59:28 2026

@author: nmehta22
"""

# -*- coding: utf-8 -*-
"""
Step 4 (Spyder): Prepare AlphaGenome inputs per FTD locus

Inputs:
  - loci_windows.tsv  (FTD-only; from Step 3; you attached this)
  - work\\ftd_sumstats_std_hg38.parquet  (from Step 2)

Outputs:
  work\\step4_alphagenome_ftd\\
    loci_manifest.tsv
    locus_FTD_001\\
      window.bed
      index_variant.tsv
      variants.tsv
    ...
"""

import os
import pandas as pd
import numpy as np

# ============================================================
# USER SETTINGS (EDIT THESE PATHS)
# ============================================================

# Option A (your local file from Step 3)
LOCI_WINDOWS_TSV = r"work\step3_pchr\loci_windows.tsv"

# Option B (if you want to point to the exact attached file you saved somewhere else)
# LOCI_WINDOWS_TSV = r"YOUR\FULL\PATH\TO\loci_windows.tsv"

FTD_SUMSTATS_PARQ = r"work\ftd_sumstats_std_hg38.parquet"

OUTDIR = r"work\step4_alphagenome_ftd"

# AlphaGenome sequence context around index SNP
SEQ_FLANK_BP = 1_000_000   # +/- 1Mb

# Variant filtering inside locus window (start..end)
P_MAX_IN_LOCUS = 1.0       # keep all; set 1e-4 to reduce size

# Safety cap (avoid gigantic per-locus variant sets)
MAX_VARIANTS_PER_LOCUS = 20000

# ============================================================

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def standardize_chr(x):
    return str(x).replace("chr", "").replace("CHR", "").replace("Chr", "")

def write_bed_window(path, chr_, start, end, locus_id):
    # BED: 0-based, half-open intervals. We'll enforce start>=0.
    start = max(0, int(start))
    end = int(end)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"chr{chr_}\t{start}\t{end}\t{locus_id}\n")

def main():
    ensure_dir(OUTDIR)

    loci = pd.read_csv(LOCI_WINDOWS_TSV, sep="\t")
    loci = loci[loci["trait"] == "FTD"].copy()

    if len(loci) == 0:
        raise ValueError("No FTD rows found in loci_windows.tsv")

    # Load FTD sumstats (hg38)
    ftd = pd.read_parquet(FTD_SUMSTATS_PARQ).copy()
    ftd["chr"] = ftd["chr"].map(standardize_chr)
    ftd["pos"] = ftd["pos"].astype(int)

    # Standardize locus chr too
    loci["chr"] = loci["chr"].map(standardize_chr)
    loci["pos"] = loci["pos"].astype(int)
    loci["start"] = loci["start"].astype(int)
    loci["end"] = loci["end"].astype(int)

    manifest_rows = []

    # Ensure deterministic ordering
    loci = loci.sort_values(["chr", "pos"]).reset_index(drop=True)

    for i, row in enumerate(loci.itertuples(index=False), start=1):
        locus_id = f"FTD_{i:03d}"
        chr_ = row.chr
        pos = int(row.pos)
        snpid = row.snpid

        # AlphaGenome window around index SNP
        seq_start = max(0, pos - SEQ_FLANK_BP)
        seq_end = pos + SEQ_FLANK_BP

        # GWAS locus window from Step 3 (clump window)
        loc_start = int(row.start)
        loc_end = int(row.end)

        # subset variants in locus window
        v = ftd[(ftd["chr"] == chr_) & (ftd["pos"] >= loc_start) & (ftd["pos"] <= loc_end)].copy()
        v = v[v["p"] <= P_MAX_IN_LOCUS].copy()
        v = v.sort_values("p", ascending=True)

        if len(v) > MAX_VARIANTS_PER_LOCUS:
            v = v.head(MAX_VARIANTS_PER_LOCUS).copy()
            print(f"Note: capped {locus_id} variants to {MAX_VARIANTS_PER_LOCUS}")

        # Write locus folder
        locus_dir = os.path.join(OUTDIR, f"locus_{locus_id}")
        ensure_dir(locus_dir)

        # window.bed
        write_bed_window(
            os.path.join(locus_dir, "window.bed"),
            chr_=chr_,
            start=seq_start,
            end=seq_end,
            locus_id=locus_id
        )

        # index_variant.tsv
        index_df = pd.DataFrame([{
            "locus_id": locus_id,
            "trait": "FTD",
            "snpid": snpid,
            "chr": chr_,
            "pos": pos,
            "locus_start": loc_start,
            "locus_end": loc_end,
            "seq_start": seq_start,
            "seq_end": seq_end
        }])
        index_df.to_csv(os.path.join(locus_dir, "index_variant.tsv"), sep="\t", index=False)

        # variants.tsv (keep allele columns for AlphaGenome allele-effect scoring)
        keep_cols = ["chr","pos","rsid","ref","alt","effect_allele","other_allele","beta","se","p","or","test"]
        keep_cols = [c for c in keep_cols if c in v.columns]
        v_out = v[keep_cols].copy()
        v_out.insert(0, "locus_id", locus_id)
        v_out.to_csv(os.path.join(locus_dir, "variants.tsv"), sep="\t", index=False)

        manifest_rows.append({
            "locus_id": locus_id,
            "index_snpid": snpid,
            "chr": chr_,
            "index_pos": pos,
            "locus_start": loc_start,
            "locus_end": loc_end,
            "seq_start": seq_start,
            "seq_end": seq_end,
            "n_variants_in_locus": len(v_out),
            "folder": locus_dir
        })

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(os.path.join(OUTDIR, "loci_manifest.tsv"), sep="\t", index=False)

    print("\nStep 4 complete.")
    print("Wrote:", os.path.join(OUTDIR, "loci_manifest.tsv"))
    print("Example folder:", os.path.join(OUTDIR, "locus_FTD_001"))

if __name__ == "__main__":
    main()
