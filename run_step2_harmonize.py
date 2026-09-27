# -*- coding: utf-8 -*-
"""
Step 2: Harmonize ALS (CSV) + FTD (txt with SNP=chr:pos:ref:alt)
Spyder-friendly (no command-line arguments).

Outputs written to OUTDIR:
- als_sumstats_std.parquet
- als_sumstats_std.tsv.gz
- ftd_sumstats_std_hg19_assumed.parquet
- ftd_sumstats_std_hg19_assumed.tsv.gz
Optional liftover outputs (if enabled):
- ftd_sumstats_std_hg38.parquet
- ftd_sumstats_std_hg38.tsv.gz
"""

import os
import numpy as np
import pandas as pd

# ============================================================
# USER SETTINGS (EDIT THESE PATHS)
# ============================================================
ALS_PATH = r"data\gwas\als.csv"
FTD_PATH = r"data\gwas\ftd.txt"
OUTDIR   = r"work"

# If your FTD positions are hg19, AlphaGenome needs hg38 -> set to True
RUN_LIFTOVER = True

# Only needed if RUN_LIFTOVER = True
CHAIN_PATH = r"data\liftover\hg19ToHg38.over.chain.gz"
# ============================================================


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _clean_chr(x) -> str:
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    s = s.replace("chr", "").replace("Chr", "").replace("CHR", "")
    return s

def report_basic(name: str, df: pd.DataFrame, show_cols, n=5):
    print(f"\n=== {name} ===")
    print("Rows:", len(df), "Cols:", df.shape[1])
    print("Columns:", df.columns.tolist())
    print("Head:")
    cols = [c for c in show_cols if c in df.columns]
    print(df[cols].head(n).to_string(index=False))

def load_als_sumstats_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    expected = [
        "rsid", "chromosome", "base_pair_location",
        "effect_allele", "other_allele",
        "effect_allele_frequency", "beta", "standard_error",
        "p_value", "N_effective"
    ]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"ALS file missing expected columns: {missing}")

    df = df.rename(columns={
        "chromosome": "chr",
        "base_pair_location": "pos",
        "effect_allele_frequency": "eaf",
        "standard_error": "se",
        "p_value": "p",
        "N_effective": "n_effective",
    })

    df["trait"] = "ALS"
    df["chr"] = df["chr"].map(_clean_chr)

    for c in ["pos", "beta", "se", "p", "n_effective", "eaf"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["chr","pos","effect_allele","other_allele","beta","se","p"])
    df["pos"] = df["pos"].astype(int)
    df["effect_allele"] = df["effect_allele"].astype(str).str.upper()
    df["other_allele"] = df["other_allele"].astype(str).str.upper()
    df["rsid"] = df["rsid"].astype(str)

    # SNPs only
    df = df[df["effect_allele"].str.len().eq(1) & df["other_allele"].str.len().eq(1)]
    df = df[(df["p"] > 0) & (df["p"] <= 1)]

    keep = ["trait","chr","pos","rsid","effect_allele","other_allele","beta","se","p","n_effective","eaf"]
    return df[keep].copy()

def load_ftd_sumstats_txt(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")

    expected = ["CHR","BP","SNP","PVALUE","A1","OR","SE","T"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"FTD file missing expected columns: {missing}")

    df = df.rename(columns={
        "CHR": "chr",
        "BP": "pos",
        "SNP": "snp_str",
        "PVALUE": "p",
        "A1": "effect_allele",
        "OR": "or",
        "SE": "se",
        "T": "test",
    })

    df["trait"] = "FTD"
    df["chr"] = df["chr"].map(_clean_chr)

    # Parse SNP field like "1:768448:G:A"
    parts = df["snp_str"].astype(str).str.split(":", expand=True)
    if parts.shape[1] < 4:
        raise ValueError("FTD SNP column is not 'chr:pos:ref:alt' format.")

    df["snp_chr"] = parts[0].map(_clean_chr)
    df["snp_pos"] = pd.to_numeric(parts[1], errors="coerce")
    df["ref"] = parts[2].astype(str).str.upper()
    df["alt"] = parts[3].astype(str).str.upper()

    for c in ["pos","p","or","se"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # fallback if CHR/BP missing
    df.loc[df["chr"].isna(), "chr"] = df.loc[df["chr"].isna(), "snp_chr"]
    df.loc[df["pos"].isna(), "pos"] = df.loc[df["pos"].isna(), "snp_pos"]

    # OR -> beta
    df["beta"] = np.log(df["or"])

    # stable ID
    df["rsid"] = df["snp_str"].astype(str)

    df["effect_allele"] = df["effect_allele"].astype(str).str.upper()

    df["other_allele"] = np.where(
        df["effect_allele"].eq(df["alt"]), df["ref"],
        np.where(df["effect_allele"].eq(df["ref"]), df["alt"], np.nan)
    )

    df = df.dropna(subset=["chr","pos","ref","alt","effect_allele","beta","se","p"])
    df["pos"] = df["pos"].astype(int)

    # SNPs only
    df = df[df["ref"].str.len().eq(1) & df["alt"].str.len().eq(1)]
    df = df[(df["p"] > 0) & (df["p"] <= 1)]

    keep = ["trait","chr","pos","rsid","ref","alt","effect_allele","other_allele","beta","se","p","or","test"]
    return df[keep].copy()

def liftover_hg19_to_hg38(df: pd.DataFrame, chain_path: str) -> pd.DataFrame:
    from pyliftover import LiftOver

    if not os.path.exists(chain_path):
        raise FileNotFoundError(f"Chain file not found: {chain_path}")

    lo = LiftOver(chain_path)

    new_chr = []
    new_pos = []
    for c, p in zip(df["chr"].astype(str), df["pos"].astype(int)):
        hits = lo.convert_coordinate(f"chr{c}", p)
        if not hits:
            new_chr.append(np.nan)
            new_pos.append(np.nan)
        else:
            c2, p2, _, _ = hits[0]
            new_chr.append(c2.replace("chr", ""))
            new_pos.append(int(round(p2)))

    out = df.copy()
    out["chr_hg38"] = new_chr
    out["pos_hg38"] = new_pos
    out = out.dropna(subset=["chr_hg38","pos_hg38"])
    out["chr_hg38"] = out["chr_hg38"].astype(str)
    out["pos_hg38"] = out["pos_hg38"].astype(int)

    out = out.drop(columns=["chr","pos"]).rename(columns={"chr_hg38":"chr","pos_hg38":"pos"})
    return out


# ============================================================
# RUN
# ============================================================
ensure_dir(OUTDIR)

als = load_als_sumstats_csv(ALS_PATH)
ftd = load_ftd_sumstats_txt(FTD_PATH)

report_basic(
    "ALS standardized",
    als,
    ["trait","chr","pos","rsid","effect_allele","other_allele","beta","se","p","n_effective","eaf"]
)

bad_a1 = ftd[(ftd["effect_allele"] != ftd["ref"]) & (ftd["effect_allele"] != ftd["alt"])]
print(f"\nFTD rows where A1 not in {{REF,ALT}}: {len(bad_a1)}")
if len(bad_a1) > 0:
    print("Example bad rows:")
    print(bad_a1[["chr","pos","rsid","ref","alt","effect_allele","p"]].head(5).to_string(index=False))

report_basic(
    "FTD standardized (pre-liftover, hg19 assumed)",
    ftd,
    ["trait","chr","pos","rsid","ref","alt","effect_allele","other_allele","beta","se","p","or","test"]
)

# Save outputs
als_parq = os.path.join(OUTDIR, "als_sumstats_std.parquet")
als_tsv  = os.path.join(OUTDIR, "als_sumstats_std.tsv.gz")
ftd_parq = os.path.join(OUTDIR, "ftd_sumstats_std_hg19_assumed.parquet")
ftd_tsv  = os.path.join(OUTDIR, "ftd_sumstats_std_hg19_assumed.tsv.gz")

als.to_parquet(als_parq, index=False)
als.to_csv(als_tsv, sep="\t", index=False, compression="gzip", encoding="utf-8")

ftd.to_parquet(ftd_parq, index=False)
ftd.to_csv(ftd_tsv, sep="\t", index=False, compression="gzip", encoding="utf-8")

print(f"\nSaved ALS -> {als_parq} and {als_tsv}")
print(f"Saved FTD (hg19 assumed) -> {ftd_parq} and {ftd_tsv}")

# Optional liftover
if RUN_LIFTOVER:
    print("\nRunning liftover for FTD (hg19 -> hg38)...")
    ftd_hg38 = liftover_hg19_to_hg38(ftd, CHAIN_PATH)

    report_basic(
        "FTD standardized (post-liftover, hg38)",
        ftd_hg38,
        ["trait","chr","pos","rsid","ref","alt","effect_allele","other_allele","beta","se","p","or","test"]
    )

    ftd38_parq = os.path.join(OUTDIR, "ftd_sumstats_std_hg38.parquet")
    ftd38_tsv  = os.path.join(OUTDIR, "ftd_sumstats_std_hg38.tsv.gz")

    ftd_hg38.to_parquet(ftd38_parq, index=False)
    ftd_hg38.to_csv(ftd38_tsv, sep="\t", index=False, compression="gzip", encoding="utf-8")

    print(f"\nSaved FTD (hg38) -> {ftd38_parq} and {ftd38_tsv}")
    print("\nStep 2 complete. Next: Step 3 (clumping + loci definition).")
else:
    print("\nLiftover not run. AlphaGenome expects hg38. Set RUN_LIFTOVER=True once chain file is available.")
