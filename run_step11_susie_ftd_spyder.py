# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 12:12:58 2026

@author: nmehta22
"""

# -*- coding: utf-8 -*-
"""
Step 11: Integrate GTEx v11 SuSiE_summary.parquet (eQTL + sQTL) into master table
- Spyder-friendly (no argparse)
- Memory-safe: filters each tissue parquet to only your variant_ids
- Restricts to all Brain_* + 4 immune/blood tissues

Inputs:
  MASTER_IN: work/step10_expression_ftd/FTD_all_variants_WITH_EXPR.tsv.gz
  EQTL_SUSIE_DIR: data/qtl/gtex/GTEx_Analysis_v11_eQTL_SuSiE
  SQTL_SUSIE_DIR: data/qtl/gtex/GTEx_Analysis_v11_sQTL_SuSiE

Outputs:
  work/step11_susie_ftd/FTD_all_variants_WITH_EXPR_WITH_SUSIE.tsv.gz
  work/step11_susie_ftd/SuSiE_eqtl_tissue_counts.tsv
  work/step11_susie_ftd/SuSiE_sqtl_tissue_counts.tsv
"""

from pathlib import Path
import re
from collections import defaultdict, Counter

import numpy as np
import pandas as pd


# -----------------------------
# USER SETTINGS
# -----------------------------
MASTER_IN = r"work/step10_expression_ftd/FTD_all_variants_WITH_EXPR.tsv.gz"

EQTL_SUSIE_DIR = r"data/qtl/gtex/GTEx_Analysis_v11_eQTL_SuSiE"
SQTL_SUSIE_DIR = r"data/qtl/gtex/GTEx_Analysis_v11_sQTL_SuSiE"

OUTDIR = r"work/step11_susie_ftd"

INCLUDE_ALL_BRAIN = True
BLOOD_IMMUNE_TISSUES = [
    "Whole_Blood",
    "Spleen",
    "Cells_EBV-transformed_lymphocytes",
    "Cells_Cultured_fibroblasts",
]

# Optional thresholds for reporting counts
PIP_THRESHOLDS = [0.1, 0.5]


# -----------------------------
# Helpers
# -----------------------------
def read_table_auto(path: str) -> pd.DataFrame:
    path = str(path)
    if path.endswith(".gz"):
        return pd.read_csv(path, sep="\t", compression="gzip", low_memory=False)
    return pd.read_csv(path, sep="\t", low_memory=False)


def normalize_chr(ch):
    s = str(ch)
    return s if s.startswith("chr") else ("chr" + s)


def build_variant_id_set(df: pd.DataFrame):
    """
    Build GTEx variant_id strings: chr_pos_ref_alt_b38
    Include both ref/alt and flipped.
    """
    need = ["chr", "pos", "ref", "alt"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in master table.")
    chr_norm = df["chr"].map(normalize_chr)
    pos = df["pos"].astype(int)
    ref = df["ref"].astype(str).str.upper()
    alt = df["alt"].astype(str).str.upper()
    vid = chr_norm + "_" + pos.astype(str) + "_" + ref + "_" + alt + "_b38"
    vid_flip = chr_norm + "_" + pos.astype(str) + "_" + alt + "_" + ref + "_b38"
    return set(vid.dropna().unique().tolist()) | set(vid_flip.dropna().unique().tolist())


def list_susie_parquets(folder: str, kind: str):
    """
    kind: 'e' or 's'
    expected filenames like:
      Brain_Cortex.v11.eQTLs.SuSiE_summary.parquet
      Brain_Cortex.v11.sQTLs.SuSiE_summary.parquet
    """
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"SuSiE folder not found: {folder}")

    pat = re.compile(r"^(?P<tissue>.+)\.v11\.(?P<qt>eQTLs|sQTLs)\.SuSiE_summary\.parquet$", re.I)
    out = []
    for p in folder.glob("*.parquet"):
        m = pat.match(p.name)
        if not m:
            continue
        qt = m.group("qt").lower()
        if kind == "e" and qt != "eqtls":
            continue
        if kind == "s" and qt != "sqtls":
            continue
        tissue = m.group("tissue")
        out.append((tissue, str(p)))
    return sorted(out)


def tissue_selected(tname: str) -> bool:
    if INCLUDE_ALL_BRAIN and tname.startswith("Brain_"):
        return True
    if tname in BLOOD_IMMUNE_TISSUES:
        return True
    return False


def read_susie_filtered(parquet_path: str, target_vids: set):
    """
    Read only needed columns and filter to target variant_ids.

    Note: eQTL SuSiE summary files and sQTL SuSiE summary files don't always share the same columns.
    - eQTL SuSiE summary commonly has: phenotype_id, gene_name, biotype, variant_id, pip, af, cs_id, cs_size, afc, afc_se
      (often NO gene_id column)
    - sQTL SuSiE summary commonly has: phenotype_id, gene_id, gene_name, biotype, variant_id, pip, af, cs_id, cs_size
      (often NO afc/afc_se columns)

    We therefore inspect the parquet schema and only request columns that exist.
    """
    # Desired columns (we'll take the intersection with what's actually present)
    desired = [
        "variant_id",
        "phenotype_id",
        "gene_id",
        "gene_name",
        "biotype",
        "pip",
        "af",
        "cs_id",
        "cs_size",
        "afc",
        "afc_se",
    ]

    # Fast schema check (no full read)
    cols = None
    try:
        import pyarrow.parquet as pq
        cols = pq.ParquetFile(parquet_path).schema.names
    except Exception:
        cols = None

    if cols is None:
        # Fallback: may be slower
        df = pd.read_parquet(parquet_path)
    else:
        usecols = [c for c in desired if c in cols]
        # Minimal required columns
        required = {"variant_id", "pip", "cs_id", "cs_size"}
        missing_req = [c for c in required if c not in usecols]
        if missing_req:
            raise ValueError(
                f"Parquet {parquet_path} is missing required columns {missing_req}. "
                f"Available columns: {cols}"
            )
        df = pd.read_parquet(parquet_path, columns=usecols)

    # Filter to our variants
    df = df[df["variant_id"].isin(target_vids)].copy()
    if df.empty:
        return df

    # Ensure numeric
    df["pip"] = pd.to_numeric(df["pip"], errors="coerce")
    if "cs_size" in df.columns:
        df["cs_size"] = pd.to_numeric(df["cs_size"], errors="coerce")
    if "cs_id" in df.columns:
        df["cs_id"] = pd.to_numeric(df["cs_id"], errors="coerce")
    if "af" in df.columns:
        df["af"] = pd.to_numeric(df["af"], errors="coerce")
    if "afc" in df.columns:
        df["afc"] = pd.to_numeric(df["afc"], errors="coerce")
    if "afc_se" in df.columns:
        df["afc_se"] = pd.to_numeric(df["afc_se"], errors="coerce")

    return df
def summarize_susie_across_tissues(susie_dir: str, kind: str, target_vids: set):
    """
    For each variant_id, find the best (max) pip across selected tissues.
    Returns:
      best_pip[vid], best_gene[vid], best_gene_id[vid], best_tissue[vid], best_cs_id[vid], best_cs_size[vid]
      n_tissues_hit[vid]
      tissue_counts (how many of your variants appeared in that tissue's SuSiE file)
      pip_threshold_tissue_counts (optional, counts of variants with pip>=thr per tissue)
    """
    files = list_susie_parquets(susie_dir, kind)

    tissues_all = [t for t, _ in files]
    tissues_sel = [t for t in tissues_all if tissue_selected(t)]
    files_sel = [(t, p) for t, p in files if t in tissues_sel]

    print(f"SuSiE {'eQTL' if kind=='e' else 'sQTL'}: tissues available = {len(tissues_all)}")
    print(f"SuSiE {'eQTL' if kind=='e' else 'sQTL'}: tissues selected = {len(tissues_sel)}")
    if tissues_sel:
        print("  Example tissues:", ", ".join(tissues_sel[:10]))
        extra = [x for x in tissues_sel[10:]]
        if extra:
            print("  (also includes):", ", ".join(extra[:10]), "..." if len(extra) > 10 else "")

    best_pip = {}
    best_gene = {}
    best_gene_id = {}
    best_tissue = {}
    best_cs_id = {}
    best_cs_size = {}
    best_afc = {}
    best_afc_se = {}
    n_tissues_hit = Counter()

    tissue_counts = Counter()
    pip_tissue_counts = {thr: Counter() for thr in PIP_THRESHOLDS}

    for tissue, pq in files_sel:
        df = read_susie_filtered(pq, target_vids)
        if df.empty:
            continue

        # For each variant, keep the row with max pip within this tissue
        df = df.sort_values(["variant_id", "pip"], ascending=[True, False])
        df_best = df.drop_duplicates("variant_id", keep="first")

        tissue_counts[tissue] += df_best.shape[0]
        for thr in PIP_THRESHOLDS:
            pip_tissue_counts[thr][tissue] += int((df_best["pip"].fillna(0.0) >= thr).sum())

        for r in df_best.itertuples(index=False):
            vid = r.variant_id
            pip = r.pip
            if pd.isna(pip):
                continue

            n_tissues_hit[vid] += 1

            prev = best_pip.get(vid, -1.0)
            if pip > prev:
                best_pip[vid] = float(pip)
                best_gene[vid] = getattr(r, "gene_name", np.nan)
                best_gene_id[vid] = getattr(r, "phenotype_id", np.nan)
                best_tissue[vid] = tissue
                best_cs_id[vid] = getattr(r, "cs_id", np.nan)
                best_cs_size[vid] = getattr(r, "cs_size", np.nan)
                best_afc[vid] = getattr(r, "afc", np.nan)
                best_afc_se[vid] = getattr(r, "afc_se", np.nan)

    return (
        best_pip, best_gene, best_gene_id, best_tissue, best_cs_id, best_cs_size, best_afc, best_afc_se,
        n_tissues_hit, tissue_counts, pip_tissue_counts, tissues_sel
    )


def add_susie_columns(v: pd.DataFrame, prefix: str, best_pip, best_gene, best_gene_id, best_tissue,
                      best_cs_id, best_cs_size, best_afc, best_afc_se, n_tissues_hit):
    """
    Add SuSiE summary columns to the master table, joining by variant_id computed from chr/pos/ref/alt (+ flipped)
    """
    chr_norm = v["chr"].map(normalize_chr)
    pos = v["pos"].astype(int).astype(str)
    ref = v["ref"].astype(str).str.upper()
    alt = v["alt"].astype(str).str.upper()

    vid = chr_norm + "_" + pos + "_" + ref + "_" + alt + "_b38"
    vid_flip = chr_norm + "_" + pos + "_" + alt + "_" + ref + "_b38"

    out_pip = []
    out_gene = []
    out_gene_id = []
    out_tissue = []
    out_cs_id = []
    out_cs_size = []
    out_afc = []
    out_afc_se = []
    out_ntiss = []

    for a, b in zip(vid.tolist(), vid_flip.tolist()):
        pa = best_pip.get(a, -1.0)
        pb = best_pip.get(b, -1.0)

        if pa < 0 and pb < 0:
            out_pip.append(np.nan)
            out_gene.append(np.nan)
            out_gene_id.append(np.nan)
            out_tissue.append(np.nan)
            out_cs_id.append(np.nan)
            out_cs_size.append(np.nan)
            out_afc.append(np.nan)
            out_afc_se.append(np.nan)
            out_ntiss.append(0)
            continue

        chosen = a if pa >= pb else b
        out_pip.append(best_pip.get(chosen, np.nan))
        out_gene.append(best_gene.get(chosen, np.nan))
        out_gene_id.append(best_gene_id.get(chosen, np.nan))
        out_tissue.append(best_tissue.get(chosen, np.nan))
        out_cs_id.append(best_cs_id.get(chosen, np.nan))
        out_cs_size.append(best_cs_size.get(chosen, np.nan))
        out_afc.append(best_afc.get(chosen, np.nan))
        out_afc_se.append(best_afc_se.get(chosen, np.nan))
        out_ntiss.append(int(n_tissues_hit.get(chosen, 0)))

    v[f"{prefix}_max_pip"] = out_pip
    v[f"{prefix}_best_gene"] = out_gene
    v[f"{prefix}_best_gene_id"] = out_gene_id
    v[f"{prefix}_best_tissue"] = out_tissue
    v[f"{prefix}_best_cs_id"] = out_cs_id
    v[f"{prefix}_best_cs_size"] = out_cs_size
    v[f"{prefix}_best_afc"] = out_afc
    v[f"{prefix}_best_afc_se"] = out_afc_se
    v[f"{prefix}_n_tissues_hit"] = out_ntiss

    return v


# -----------------------------
# Main
# -----------------------------
def main():
    outdir = Path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading master table:", MASTER_IN)
    v = read_table_auto(MASTER_IN)
    print("Loaded:", v.shape)

    print("Building target variant_id set...")
    target_vids = build_variant_id_set(v)
    print("Target variant_ids:", len(target_vids))

    # eQTL SuSiE
    (
        eq_best_pip, eq_best_gene, eq_best_gene_id, eq_best_tissue, eq_best_cs_id, eq_best_cs_size, eq_best_afc, eq_best_afc_se,
        eq_n_tiss, eq_tissue_counts, eq_pip_tissue_counts, eq_tissues_sel
    ) = summarize_susie_across_tissues(EQTL_SUSIE_DIR, "e", target_vids)

    # sQTL SuSiE
    (
        sq_best_pip, sq_best_gene, sq_best_gene_id, sq_best_tissue, sq_best_cs_id, sq_best_cs_size, sq_best_afc, sq_best_afc_se,
        sq_n_tiss, sq_tissue_counts, sq_pip_tissue_counts, sq_tissues_sel
    ) = summarize_susie_across_tissues(SQTL_SUSIE_DIR, "s", target_vids)

    # Add to master
    v = add_susie_columns(v, "susie_eqtl", eq_best_pip, eq_best_gene, eq_best_gene_id, eq_best_tissue,
                          eq_best_cs_id, eq_best_cs_size, eq_best_afc, eq_best_afc_se, eq_n_tiss)

    v = add_susie_columns(v, "susie_sqtl", sq_best_pip, sq_best_gene, sq_best_gene_id, sq_best_tissue,
                          sq_best_cs_id, sq_best_cs_size, sq_best_afc, sq_best_afc_se, sq_n_tiss)

    # Write master
    out_main = outdir / "FTD_all_variants_WITH_EXPR_WITH_SUSIE.tsv.gz"
    v.to_csv(out_main, sep="\t", index=False, compression="gzip")
    print("Wrote:", out_main)

    # Tissue count summaries
    eq_df = pd.DataFrame({
        "tissue": list(eq_tissue_counts.keys()),
        "n_variants_in_susie": list(eq_tissue_counts.values())
    }).sort_values("n_variants_in_susie", ascending=False)
    eq_out = outdir / "SuSiE_eqtl_tissue_counts.tsv"
    eq_df.to_csv(eq_out, sep="\t", index=False)
    print("Wrote:", eq_out)

    sq_df = pd.DataFrame({
        "tissue": list(sq_tissue_counts.keys()),
        "n_variants_in_susie": list(sq_tissue_counts.values())
    }).sort_values("n_variants_in_susie", ascending=False)
    sq_out = outdir / "SuSiE_sqtl_tissue_counts.tsv"
    sq_df.to_csv(sq_out, sep="\t", index=False)
    print("Wrote:", sq_out)

    # Optional: pip threshold summaries
    for thr in PIP_THRESHOLDS:
        eq_thr = pd.DataFrame({
            "tissue": list(eq_pip_tissue_counts[thr].keys()),
            f"n_variants_pip_ge_{thr}": list(eq_pip_tissue_counts[thr].values())
        }).sort_values(f"n_variants_pip_ge_{thr}", ascending=False)
        eq_thr.to_csv(outdir / f"SuSiE_eqtl_tissue_counts_pip_ge_{thr}.tsv", sep="\t", index=False)

        sq_thr = pd.DataFrame({
            "tissue": list(sq_pip_tissue_counts[thr].keys()),
            f"n_variants_pip_ge_{thr}": list(sq_pip_tissue_counts[thr].values())
        }).sort_values(f"n_variants_pip_ge_{thr}", ascending=False)
        sq_thr.to_csv(outdir / f"SuSiE_sqtl_tissue_counts_pip_ge_{thr}.tsv", sep="\t", index=False)

    print("\nDone Step 11 (SuSiE summary integration).")


if __name__ == "__main__":
    main()