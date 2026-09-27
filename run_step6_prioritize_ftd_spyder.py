# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 11:41:45 2026

@author: nmehta22
"""

import os, glob
import pandas as pd
import numpy as np
from pathlib import Path

# -----------------------
# USER SETTINGS
# -----------------------
OUTDIR_STEP5 = Path(r"work\step5_alphagenome_ftd")
OUT_STEP6 = Path(r"work\step6_prioritize_ftd")
OUT_STEP6.mkdir(parents=True, exist_ok=True)

# How many top variants to keep per locus + overall
TOP_PER_LOCUS = 200
TOP_GLOBAL = 500

# -----------------------
# LOAD ALL COMPACT FILES
# -----------------------
files = sorted(glob.glob(str(OUTDIR_STEP5 / "FTD_*_variant_compact.tsv.gz")))
print("Found compact files:", len(files))
if len(files) == 0:
    raise FileNotFoundError(f"No compact files found in {OUTDIR_STEP5}")

dfs = []
for f in files:
    df = pd.read_csv(f, sep="\t", compression="gzip")
    # locus_id should already exist, but ensure
    if "locus_id" not in df.columns:
        # infer from filename
        locus = Path(f).name.split("_variant_compact")[0]
        df["locus_id"] = locus
    dfs.append(df)

allv = pd.concat(dfs, ignore_index=True)
print("All variants table:", allv.shape)

# -----------------------
# BASIC SANITY CHECKS
# -----------------------
if "alphagenome_max_abs" not in allv.columns:
    raise ValueError("alphagenome_max_abs column missing. Check Step 5 output.")

nan_frac = allv["alphagenome_max_abs"].isna().mean()
print("alphagenome_max_abs NaN fraction:", nan_frac)

# Build a unique key if not present
if "variant_key" not in allv.columns:
    # expect chr/pos/ref/alt
    allv["variant_key"] = (
        allv["chr"].astype(str) + ":" +
        allv["pos"].astype(int).astype(str) + ":" +
        allv["ref"].astype(str) + ":" +
        allv["alt"].astype(str)
    )

# -----------------------
# PER-LOCUS RANKING
# -----------------------
# rank 1 = highest score
allv["rank_within_locus"] = (
    allv.groupby("locus_id")["alphagenome_max_abs"]
        .rank(method="min", ascending=False)
)

# percentile: 1.0 = best, 0.0 = worst
allv["pct_within_locus"] = (
    1.0 - (allv["rank_within_locus"] - 1) /
    allv.groupby("locus_id")["rank_within_locus"].transform("max").clip(lower=1)
)

# -----------------------
# GLOBAL RANKING (optional but useful)
# -----------------------
allv["rank_global"] = allv["alphagenome_max_abs"].rank(method="min", ascending=False)
allv["pct_global"] = 1.0 - (allv["rank_global"] - 1) / allv["rank_global"].max()

# -----------------------
# OUTPUTS
# -----------------------
# Full combined (can be large)
full_out = OUT_STEP6 / "FTD_all_variants_with_ranks.tsv.gz"
allv.to_csv(full_out, sep="\t", index=False, compression="gzip")
print("Wrote:", full_out)

# Per-locus top variants
per_locus_top = (allv.sort_values(["locus_id", "alphagenome_max_abs"], ascending=[True, False])
                   .groupby("locus_id", as_index=False)
                   .head(TOP_PER_LOCUS))
per_locus_out = OUT_STEP6 / f"FTD_top_{TOP_PER_LOCUS}_per_locus.tsv"
per_locus_top.to_csv(per_locus_out, sep="\t", index=False)
print("Wrote:", per_locus_out)

# Global top variants (across all loci)
global_top = allv.sort_values("alphagenome_max_abs", ascending=False).head(TOP_GLOBAL)
global_out = OUT_STEP6 / f"FTD_top_{TOP_GLOBAL}_global.tsv"
global_top.to_csv(global_out, sep="\t", index=False)
print("Wrote:", global_out)

# Lead variant summary per locus
lead = (allv.sort_values(["locus_id", "alphagenome_max_abs"], ascending=[True, False])
          .groupby("locus_id", as_index=False)
          .head(1))

lead_cols = [c for c in [
    "locus_id","chr","index_pos","variant_key","rsid","pos","ref","alt",
    "alphagenome_max_abs","rank_within_locus","pct_within_locus",
    "p","beta","se","or"
] if c in lead.columns]

lead_out = OUT_STEP6 / "FTD_lead_variant_per_locus.tsv"
lead[lead_cols].to_csv(lead_out, sep="\t", index=False)
print("Wrote:", lead_out)

print("\nDone Step 6.")
