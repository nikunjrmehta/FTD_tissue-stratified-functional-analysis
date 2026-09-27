# -*- coding: utf-8 -*-
"""
Created on Sat Feb 14 22:33:05 2026

@author: nmehta22
"""

# -*- coding: utf-8 -*-
"""
Step 3 (Spyder-compatible): Clumping + locus window definition for ALS and FTD

Inputs:
  - work/als_sumstats_std.parquet
  - work/ftd_sumstats_std_hg38.parquet

Requires:
  - PLINK2 executable (set PLINK2_EXE)
  - LD reference in PLINK bed/bim/fam (set LD_REF_PREFIX)
    IMPORTANT: LD reference variant IDs should match the IDs you pass in.
    This script uses SNP IDs formatted as "chr:pos" (e.g., "1:12345").
    When building LD reference from VCF, use: --set-all-var-ids @:#

Outputs (written to OUTDIR):
  - als_index_variants.tsv
  - ftd_index_variants.tsv
  - loci_windows.tsv
  - als_clumps_full.tsv
  - ftd_clumps_full.tsv
"""

import os
import pandas as pd
import numpy as np
import subprocess

# ============================================================
# USER SETTINGS (EDIT THESE)
# ============================================================
ALS_STD_PARQUET = r"work\als_sumstats_std.parquet"
FTD_STD_PARQUET = r"work\ftd_sumstats_std_hg38.parquet"

# Prefix without extension: must have .bed/.bim/.fam
LD_REF_PREFIX   = r"data\ldref\plink\1kg_hg38_chrpos_merged"

# If plink2 is on PATH, keep as "plink2".
# Otherwise set full path, e.g. r"C:\tools\plink2.exe"
PLINK2_EXE      = r"plink2"

OUTDIR          = r"work\step3"

# Clumping parameters
P_GWS           = 5e-8
CLUMP_P2        = 1e-4
CLUMP_R2        = 0.1
CLUMP_KB        = 1000

# Locus window around each clumped index SNP
LOCUS_FLANK_BP  = 500_000   # change to 1_000_000 for +/- 1Mb
# ============================================================


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def preflight_checks():
    # Input files
    if not os.path.exists(ALS_STD_PARQUET):
        raise FileNotFoundError(f"Missing ALS parquet: {ALS_STD_PARQUET}")
    if not os.path.exists(FTD_STD_PARQUET):
        raise FileNotFoundError(f"Missing FTD hg38 parquet: {FTD_STD_PARQUET}")

    # LD ref files
    for ext in [".bed", ".bim", ".fam"]:
        fp = LD_REF_PREFIX + ext
        if not os.path.exists(fp):
            raise FileNotFoundError(f"Missing LD reference file: {fp}")

    # PLINK2 availability
    try:
        res = subprocess.run([PLINK2_EXE, "--version"], capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(res.stderr.strip() or "plink2 returned non-zero exit code")
        print("PLINK2 OK:", (res.stdout.splitlines() or [""])[0])
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Could not run PLINK2 ('{PLINK2_EXE}'). "
            f"Set PLINK2_EXE to the full path of plink2.exe or add it to PATH."
        )

def run_cmd(cmd):
    # Run PLINK2 in a way that Spyder can display output
    print("\nRunning command:")
    print(" ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.stdout:
        print(res.stdout)
    if res.returncode != 0:
        if res.stderr:
            print(res.stderr)
        raise RuntimeError(f"Command failed: exit code {res.returncode}")

def make_snpid_chrpos(df):
    df = df.copy()
    df["chr"] = df["chr"].astype(str).str.replace("^chr", "", regex=True)
    df["pos"] = df["pos"].astype(int)
    df["snpid"] = df["chr"] + ":" + df["pos"].astype(str)
    return df

def write_clump_input(gws_df, out_path):
    # PLINK2 expects columns named SNP and P (case-insensitive usually, but be safe)
    tmp = gws_df[["snpid", "p"]].rename(columns={"snpid": "SNP", "p": "P"}).copy()
    tmp.to_csv(out_path, sep="\t", index=False)
    return out_path

def find_clump_output(out_prefix):
    # PLINK2 usually writes .clumps; PLINK1.9 writes .clumped
    for ext in [".clumps", ".clumped"]:
        fp = out_prefix + ext
        if os.path.exists(fp):
            return fp
    raise FileNotFoundError(f"Could not find clump output: {out_prefix}.clumps/.clumped")

def load_clump_file(path):
    # whitespace-delimited
    return pd.read_csv(path, sep=r"\s+", engine="python")

def clump_trait(trait, gws_df):
    ensure_dir(OUTDIR)
    clump_in = os.path.join(OUTDIR, f"{trait}_clump_input.tsv")
    write_clump_input(gws_df, clump_in)

    out_prefix = os.path.join(OUTDIR, f"{trait}_clumped")
    cmd = [
        PLINK2_EXE,
        "--bfile", LD_REF_PREFIX,
        "--clump", clump_in,
        "--clump-p1", str(P_GWS),
        "--clump-p2", str(CLUMP_P2),
        "--clump-r2", str(CLUMP_R2),
        "--clump-kb", str(CLUMP_KB),
        "--out", out_prefix
    ]
    run_cmd(cmd)

    clump_path = find_clump_output(out_prefix)
    cl = load_clump_file(clump_path)

    # Identify index SNP column
    idx_col = None
    for c in ["ID", "SNP"]:
        if c in cl.columns:
            idx_col = c
            break
    if idx_col is None:
        raise ValueError(f"Cannot find index SNP column in clump output. Columns: {cl.columns.tolist()}")

    idx = cl[[idx_col]].rename(columns={idx_col: "snpid"}).drop_duplicates()
    idx["trait"] = trait
    return idx, cl

def make_loci(index_df):
    loci = index_df.copy()
    loci[["chr", "pos"]] = loci["snpid"].str.split(":", expand=True)
    loci["pos"] = loci["pos"].astype(int)
    loci["start"] = (loci["pos"] - LOCUS_FLANK_BP).clip(lower=0).astype(int)
    loci["end"] = (loci["pos"] + LOCUS_FLANK_BP).astype(int)
    return loci[["trait", "snpid", "chr", "pos", "start", "end"]]


# ============================================================
# RUN
# ============================================================
ensure_dir(OUTDIR)
preflight_checks()

als = pd.read_parquet(ALS_STD_PARQUET)
ftd = pd.read_parquet(FTD_STD_PARQUET)

als = make_snpid_chrpos(als)
ftd = make_snpid_chrpos(ftd)

als_gws = als.loc[als["p"] < P_GWS, ["snpid", "p"]].copy()
ftd_gws = ftd.loc[ftd["p"] < P_GWS, ["snpid", "p"]].copy()

print("\nALS genome-wide significant variants:", len(als_gws))
print("FTD genome-wide significant variants:", len(ftd_gws))

# Clump
als_idx, als_cl = clump_trait("ALS", als_gws)
ftd_idx, ftd_cl = clump_trait("FTD", ftd_gws)

print("\nALS clumped index signals:", len(als_idx))
print("FTD clumped index signals:", len(ftd_idx))

# Save index + full clump tables
als_idx.to_csv(os.path.join(OUTDIR, "als_index_variants.tsv"), sep="\t", index=False)
ftd_idx.to_csv(os.path.join(OUTDIR, "ftd_index_variants.tsv"), sep="\t", index=False)

als_cl.to_csv(os.path.join(OUTDIR, "als_clumps_full.tsv"), sep="\t", index=False)
ftd_cl.to_csv(os.path.join(OUTDIR, "ftd_clumps_full.tsv"), sep="\t", index=False)

# Locus windows
loci = pd.concat([make_loci(als_idx), make_loci(ftd_idx)], ignore_index=True)
loci.to_csv(os.path.join(OUTDIR, "loci_windows.tsv"), sep="\t", index=False)

print("\nWrote outputs to:", OUTDIR)
print(" - als_index_variants.tsv")
print(" - ftd_index_variants.tsv")
print(" - loci_windows.tsv")
print(" - als_clumps_full.tsv")
print(" - ftd_clumps_full.tsv")
print("\nStep 3 complete.")
