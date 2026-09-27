#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 21 12:15:27 2026

@author: nmehta22
"""
import pandas as pd
import numpy as np

FINAL = r"work\step12_final_ftd\FTD_all_variants_FINAL.tsv.gz"
outdir = r"work\step13_gwas_filtered_views"
import os
os.makedirs(outdir, exist_ok=True)

df = pd.read_csv(FINAL, sep="\t", compression="gzip", low_memory=False)
df["p_num"] = pd.to_numeric(df["p"], errors="coerce")

# Option 1: GWAS p threshold
P_THRESH = 1e-4

# Option 2: top N by GWAS p within locus (useful if p threshold too strict)
TOPN_BY_P = 500

rows_brain = []
rows_blood = []

for lid, g in df.groupby("locus_id"):
    g = g.dropna(subset=["p_num"]).copy()
    if g.empty:
        continue

    # choose one filter strategy
    g_f = g[g["p_num"] <= P_THRESH].copy()
    if g_f.empty:
        g_f = g.sort_values("p_num").head(TOPN_BY_P).copy()

    # now rank within filtered set
    gb = g_f.sort_values("brain_evidence_score", ascending=False).head(20)
    gl = g_f.sort_values("blood_evidence_score", ascending=False).head(20)

    rows_brain.append(gb)
    rows_blood.append(gl)

brain_f = pd.concat(rows_brain, ignore_index=True)
blood_f = pd.concat(rows_blood, ignore_index=True)

brain_f.to_csv(os.path.join(outdir, "FTD_top20_per_locus_BRAIN_GWAS_FILTERED.tsv"), sep="\t", index=False)
blood_f.to_csv(os.path.join(outdir, "FTD_top20_per_locus_BLOOD_GWAS_FILTERED.tsv"), sep="\t", index=False)

print("Wrote GWAS-filtered brain/blood tables to", outdir)