# -*- coding: utf-8 -*-
"""
Step 3 (Spyder, Windows): clumping + locus windows WITHOUT merging LD reference.
Runs PLINK2 clumping per chromosome using per-chr PGEN references.

Inputs:
  work\\als_sumstats_std.parquet
  work\\ftd_sumstats_std_hg38.parquet

Requires:
  Per-chr LD refs:
    data\\ldref\\pgen\\1kg_chr{1..22}_hg38_chrpos.(pgen/pvar/psam)
  PLINK2 executable
"""

import os
import pandas as pd
import numpy as np
import subprocess

# ============================================================
# USER SETTINGS
# ============================================================
ALS_STD_PARQUET = r"work\als_sumstats_std.parquet"
FTD_STD_PARQUET = r"work\ftd_sumstats_std_hg38.parquet"

# Machine-specific. Resolved from a candidate list then PATH, so the deposited
# code does not fail on the first line for anyone re-running it. Add your own
# path to the front of this list if plink2 lives elsewhere.
import shutil as _shutil

_PLINK2_CANDIDATES = [
    r"C:\Users\nikun\Downloads\plink2_win64_20260808\plink2.exe",
    r"C:\tools\plink2\plink2.exe",
    "plink2",
]


def _resolve_plink2():
    for _c in _PLINK2_CANDIDATES:
        if os.path.exists(_c):
            return _c
        _f = _shutil.which(_c)
        if _f:
            return _f
    raise RuntimeError(
        "plink2 not found. Searched: " + ", ".join(_PLINK2_CANDIDATES) +
        ". Add the correct path to _PLINK2_CANDIDATES at the top of this script."
    )


PLINK2_EXE = _resolve_plink2()

LDREF_DIR       = r"data\ldref\pgen"
LDREF_PREFIX_TMPL = r"1kg_chr{CHR}_hg38_chrpos"   # inside LDREF_DIR

OUTDIR          = r"work\step3_pchr"

# Clumping params
#P_GWS           = 5e-8
P_ALS = 1e-6   # suggestive since ALS has no hits at 5e-8
P_FTD = 5e-8   # keep strict
CLUMP_P2        = 1e-4
CLUMP_R2        = 0.1
CLUMP_KB        = 1000

# Locus window around each clumped index SNP
LOCUS_FLANK_BP  = 500_000  # +/- 500kb
# ============================================================


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def check_plink2():
    res = subprocess.run([PLINK2_EXE, "--version"], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(res.stderr.strip() or "plink2 failed to run")
    print("PLINK2 OK:", (res.stdout.splitlines() or [""])[0])

def ldref_prefix_for_chr(ch):
    return os.path.join(LDREF_DIR, LDREF_PREFIX_TMPL.format(CHR=ch))

def check_ldref_chr(ch):
    prefix = ldref_prefix_for_chr(ch)
    for ext in [".pgen", ".pvar", ".psam"]:
        fp = prefix + ext
        if not os.path.exists(fp):
            raise FileNotFoundError(f"Missing LD ref file for chr{ch}: {fp}")

def run_cmd(cmd):
    print("\nRunning:\n", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.stdout:
        print(res.stdout)
    if res.returncode != 0:
        if res.stderr:
            print(res.stderr)
        raise RuntimeError(f"Command failed with exit code {res.returncode}")

def make_snpid(df):
    df = df.copy()
    df["chr"] = df["chr"].astype(str).str.replace("^chr", "", regex=True)
    df["pos"] = df["pos"].astype(int)
    df["snpid"] = df["chr"] + ":" + df["pos"].astype(str)
    return df

def write_clump_input(df_chr, out_path):
    # PLINK2 expects SNP and P columns
    tmp = df_chr[["snpid", "p"]].rename(columns={"snpid": "SNP", "p": "P"}).copy()
    tmp.to_csv(out_path, sep="\t", index=False)
    return out_path

def find_clump_output(prefix):
    # common outputs: .clumps (plink2)
    fp = prefix + ".clumps"
    if os.path.exists(fp):
        return fp
    # fallback
    fp2 = prefix + ".clumped"
    if os.path.exists(fp2):
        return fp2
    return None

def load_clumps(path):
    return pd.read_csv(path, sep=r"\s+", engine="python")

def snp_exists_in_ldref(ch, snpid):
    # Check membership by scanning pvar IDs (fast enough for a single SNP)
    pvar_path = ldref_prefix_for_chr(ch) + ".pvar"
    # pvar is tab-delimited; ID column is usually 3rd (after #CHROM and POS) in PLINK2 pvar
    # We'll do a simple substring scan for '\t<snpid>\t' which is reliable.
    needle = "\t" + snpid + "\t"
    with open(pvar_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("#"):
                continue
            if needle in line:
                return True
    return False

def clump_trait_per_chr(trait, gws_df, p1):
    """
    gws_df columns: chr(str), snpid, p
    Returns: index_df, clumps_df_all
    """
    all_idx = []
    all_clumps = []

    for ch in range(1, 23):
        df_chr = gws_df[gws_df["chr"].astype(str) == str(ch)].copy()
        if df_chr.empty:
            continue

        check_ldref_chr(ch)
        ldref = ldref_prefix_for_chr(ch)

        clump_in = os.path.join(OUTDIR, f"{trait}_chr{ch}_clump_input.tsv")
        write_clump_input(df_chr, clump_in)

        out_prefix = os.path.join(OUTDIR, f"{trait}_chr{ch}_clumped")
        
            
        if len(df_chr) <= 5:
            # For small cases like ALS, check presence
            test_snp = df_chr["snpid"].iloc[0]
            exists = snp_exists_in_ldref(ch, test_snp)
            print(f"LDref contains {test_snp} on chr{ch}? {exists}")
        
        cmd = [
            PLINK2_EXE,
            "--pfile", ldref,
            "--clump", clump_in,
            "--clump-p1", str(p1),
            "--clump-p2", str(CLUMP_P2),
            "--clump-r2", str(CLUMP_R2),
            "--clump-kb", str(CLUMP_KB),
            "--out", out_prefix
        ]
        run_cmd(cmd)
        
        clump_path = find_clump_output(out_prefix)
        if clump_path is None:
            print(f"Note: No clump output for {trait} chr{ch}. This usually means SNP IDs were not found in LD reference or no index candidates.")
            continue

        cl = load_clumps(clump_path)
        all_clumps.append(cl.assign(trait=trait, chr=str(ch)))

        # index SNP column differs across versions; usually ID or SNP
        idx_col = None
        for c in ["ID", "SNP"]:
            if c in cl.columns:
                idx_col = c
                break
        if idx_col is None:
            raise ValueError(f"Cannot find index SNP column in clump output for chr{ch}. Columns: {cl.columns.tolist()}")

        idx = cl[[idx_col]].rename(columns={idx_col: "snpid"}).drop_duplicates()
        idx["trait"] = trait
        idx["chr"] = str(ch)
        all_idx.append(idx)

    if all_idx:
        index_df = pd.concat(all_idx, ignore_index=True)
    else:
        index_df = pd.DataFrame(columns=["snpid", "trait", "chr"])

    if all_clumps:
        clumps_df = pd.concat(all_clumps, ignore_index=True)
    else:
        clumps_df = pd.DataFrame()

    return index_df, clumps_df

def make_loci(index_df):
    """
    index_df: columns trait, snpid (format 'chr:pos')
    Returns loci table with chr, pos, start, end.
    Handles empty inputs and malformed snpid safely.
    """
    cols_out = ["trait","snpid","chr","pos","start","end"]

    if index_df is None or len(index_df) == 0:
        return pd.DataFrame(columns=cols_out)

    loci = index_df.copy()

    # Ensure snpid exists and is string
    if "snpid" not in loci.columns:
        return pd.DataFrame(columns=cols_out)

    loci["snpid"] = loci["snpid"].astype(str)

    # Split safely
    parts = loci["snpid"].str.split(":", n=1, expand=True)

    # If split didn't produce 2 columns, return empty (or drop all)
    if parts.shape[1] < 2:
        print("Warning: snpid did not split into chr:pos for any rows.")
        return pd.DataFrame(columns=cols_out)

    loci["chr"] = parts[0]
    loci["pos"] = pd.to_numeric(parts[1], errors="coerce")

    # Drop malformed
    before = len(loci)
    loci = loci.dropna(subset=["chr","pos"])
    loci["pos"] = loci["pos"].astype(int)
    dropped = before - len(loci)
    if dropped > 0:
        print(f"Warning: dropped {dropped} malformed snpid rows (not chr:pos).")

    loci["start"] = (loci["pos"] - LOCUS_FLANK_BP).clip(lower=0).astype(int)
    loci["end"] = (loci["pos"] + LOCUS_FLANK_BP).astype(int)

    return loci[cols_out]


# =========================
# RUN
# =========================
ensure_dir(OUTDIR)
check_plink2()

als = pd.read_parquet(ALS_STD_PARQUET)
ftd = pd.read_parquet(FTD_STD_PARQUET)

als = make_snpid(als)
ftd = make_snpid(ftd)

als_gws = als.loc[als["p"] < P_ALS, ["chr","snpid","p"]].copy()
ftd_gws = ftd.loc[ftd["p"] < P_FTD, ["chr","snpid","p"]].copy()

print("\nALS variants p<1e-6:", len(als_gws))
print("FTD variants p<5e-8:", len(ftd_gws))

als_idx, als_cl = clump_trait_per_chr("ALS", als_gws, P_ALS)
ftd_idx, ftd_cl = clump_trait_per_chr("FTD", ftd_gws, P_FTD)

print("\nALS clumped index signals:", len(als_idx))
print("FTD clumped index signals:", len(ftd_idx))

# Save
als_idx.to_csv(os.path.join(OUTDIR, "als_index_variants.tsv"), sep="\t", index=False)
ftd_idx.to_csv(os.path.join(OUTDIR, "ftd_index_variants.tsv"), sep="\t", index=False)

if not als_cl.empty:
    als_cl.to_csv(os.path.join(OUTDIR, "als_clumps_full.tsv"), sep="\t", index=False)
if not ftd_cl.empty:
    ftd_cl.to_csv(os.path.join(OUTDIR, "ftd_clumps_full.tsv"), sep="\t", index=False)

loci_als = make_loci(als_idx)
loci_ftd = make_loci(ftd_idx)
loci = pd.concat([loci_als, loci_ftd], ignore_index=True)

loci.to_csv(os.path.join(OUTDIR, "loci_windows.tsv"), sep="\t", index=False)

print("\nWrote outputs to:", OUTDIR)
print(" - als_index_variants.tsv")
print(" - ftd_index_variants.tsv")
print(" - loci_windows.tsv")
print("\nStep 3 complete (per-chromosome clumping, no merge needed).")
