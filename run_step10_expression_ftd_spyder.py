# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 11:45:03 2026

@author: nmehta22
"""
# -*- coding: utf-8 -*-
"""
Step 10: Add GTEx expression context (median TPM) for implicated genes
- Spyder-friendly (no argparse)
- Uses GTEx_Analysis_*_gene_median_tpm.gct(.gz)
- Adds expression summaries for genes referenced in the master variant table

Inputs:
  MASTER_PATH: work/step9_gtex_sigpairs_ftd/FTD_all_variants_WITH_GTEx_sigpairs.tsv.gz
  GENE_MEDIAN_TPM_GCT: data/qtl/gtex/GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_median_tpm.gct.gz

Outputs:
  work/step10_expression_ftd/FTD_all_variants_WITH_EXPR.tsv.gz
  work/step10_expression_ftd/FTD_gene_expression_summary.tsv
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd


# -----------------------------
# USER SETTINGS (edit here)
# -----------------------------
MASTER_PATH = r"work/step9_gtex_sigpairs_ftd/FTD_all_variants_WITH_GTEx_sigpairs.tsv.gz"
GENE_MEDIAN_TPM_GCT = r"data/qtl/gtex/GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_median_tpm.gct.gz"
OUTDIR = r"work/step10_expression_ftd"

# tissues to include
INCLUDE_ALL_BRAIN = True
BLOOD_IMMUNE_TISSUES = [
    "Whole_Blood",
    "Spleen",
    "Cells_EBV-transformed_lymphocytes",
    "Cells_Cultured_fibroblasts",
]

# Which master-table gene columns should get expression annotation?
# If empty, we auto-detect columns containing "best_gene" or ending with "_gene"
EXPLICIT_GENE_COLS = []  # e.g. ["gtex_eqtl_best_gene", "gtex_sqtl_best_gene"]


# -----------------------------
# Helpers
# -----------------------------
def read_table_auto(path: str) -> pd.DataFrame:
    path = str(path)
    if path.endswith(".gz"):
        return pd.read_csv(path, sep="\t", compression="gzip", low_memory=False)
    return pd.read_csv(path, sep="\t", low_memory=False)


def read_gct_gene_median_tpm(path: str) -> pd.DataFrame:
    """
    GTEx GCT format:
      line1: #1.2
      line2: <nrows>\t<ncols>
      then header row: Name Description <tissue1> <tissue2> ...
    """
    # Skip first 2 metadata lines
    df = pd.read_csv(path, sep="\t", compression="infer", skiprows=2, low_memory=False)
    # Expect Name + Description
    if "Name" not in df.columns or "Description" not in df.columns:
        raise ValueError(f"GCT missing Name/Description columns. Columns: {df.columns.tolist()[:10]}")
    return df


def strip_ensg_version(x: str) -> str:
    if not isinstance(x, str):
        return x
    # ENSG00000....(.version)
    return x.split(".")[0]


def choose_relevant_tissues(all_tissues):
    all_tissues = list(all_tissues)
    brain = [t for t in all_tissues if t.startswith("Brain_")] if INCLUDE_ALL_BRAIN else []
    blood = [t for t in all_tissues if t in BLOOD_IMMUNE_TISSUES]
    selected = sorted(set(brain + blood))
    return selected


def autodetect_gene_cols(cols):
    gene_cols = []
    for c in cols:
        lc = c.lower()
        if "best_gene" in lc:
            gene_cols.append(c)
        elif re.search(r"(^gene$|_gene$|gene_name$)", lc):
            gene_cols.append(c)
    # remove obvious non-gene columns
    drop = set(["gene_id", "nearest_gene_id", "nearest_gene_name"])
    gene_cols = [c for c in gene_cols if c.lower() not in drop]
    return sorted(set(gene_cols))


def build_gene_maps(gct_df: pd.DataFrame):
    """
    Returns:
      geneid_nover -> gene_name
      gene_name -> geneid_nover (only for names mapping uniquely)
    """
    gene_id = gct_df["Name"].astype(str).map(strip_ensg_version)
    gene_name = gct_df["Description"].astype(str)

    id2name = dict(zip(gene_id, gene_name))

    # gene_name -> gene_id: keep only unique mappings
    tmp = pd.DataFrame({"gene_id": gene_id, "gene_name": gene_name})
    # drop duplicates by gene_name; keep first, but also record uniqueness
    counts = tmp["gene_name"].value_counts()
    unique_names = set(counts[counts == 1].index.tolist())
    name2id = dict(tmp[tmp["gene_name"].isin(unique_names)][["gene_name", "gene_id"]].values.tolist())

    return id2name, name2id


def summarize_expression_for_gene(gene_key: str, tpm_mat: pd.DataFrame, tissues_brain, tissues_blood):
    """
    gene_key must be in tpm_mat.index (ENSG without version)
    Returns dict of summary values.
    """
    row = tpm_mat.loc[gene_key]

    out = {}

    if tissues_brain:
        vals = row[tissues_brain].astype(float)
        out["expr_brain_median_tpm"] = float(np.nanmedian(vals.values))
        out["expr_brain_max_tpm"] = float(np.nanmax(vals.values))
    else:
        out["expr_brain_median_tpm"] = np.nan
        out["expr_brain_max_tpm"] = np.nan

    if tissues_blood:
        vals = row[tissues_blood].astype(float)
        out["expr_blood_median_tpm"] = float(np.nanmedian(vals.values))
        out["expr_blood_max_tpm"] = float(np.nanmax(vals.values))
    else:
        out["expr_blood_median_tpm"] = np.nan
        out["expr_blood_max_tpm"] = np.nan

    # best tissue among selected
    sel = tissues_brain + tissues_blood
    if sel:
        vals = row[sel].astype(float)
        best_idx = vals.values.argmax() if len(vals) else None
        if best_idx is None:
            out["expr_top_tissue"] = ""
            out["expr_top_tpm"] = np.nan
        else:
            out["expr_top_tissue"] = str(vals.index[best_idx])
            out["expr_top_tpm"] = float(vals.iloc[best_idx])
    else:
        out["expr_top_tissue"] = ""
        out["expr_top_tpm"] = np.nan

    return out


# -----------------------------
# Main
# -----------------------------
def main():
    outdir = Path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading master table:", MASTER_PATH)
    v = read_table_auto(MASTER_PATH)
    print("Loaded:", v.shape)

    print("Loading GTEx gene median TPM:", GENE_MEDIAN_TPM_GCT)
    gct = read_gct_gene_median_tpm(GENE_MEDIAN_TPM_GCT)
    print("GCT loaded:", gct.shape)

    tissue_cols = [c for c in gct.columns if c not in ("Name", "Description")]
    sel_tissues = choose_relevant_tissues(tissue_cols)
    tissues_brain = [t for t in sel_tissues if t.startswith("Brain_")]
    tissues_blood = [t for t in sel_tissues if t in BLOOD_IMMUNE_TISSUES]

    print("Selected tissues:", len(sel_tissues))
    print("  Brain tissues:", len(tissues_brain))
    print("  Blood/immune:", tissues_blood)

    # Build TPM matrix indexed by ENSG (no version)
    gct["gene_id"] = gct["Name"].astype(str).map(strip_ensg_version)
    # numeric matrix
    tpm_mat = gct.set_index("gene_id")[sel_tissues].apply(pd.to_numeric, errors="coerce")

    id2name, name2id = build_gene_maps(gct)

    # Which gene columns to annotate
    gene_cols = EXPLICIT_GENE_COLS[:] if EXPLICIT_GENE_COLS else autodetect_gene_cols(v.columns)
    if not gene_cols:
        raise ValueError("No gene columns found to annotate. Set EXPLICIT_GENE_COLS manually.")
    print("Gene columns to annotate:", gene_cols)

    # Collect unique gene tokens from those columns
    gene_tokens = set()
    for c in gene_cols:
        gene_tokens |= set(v[c].dropna().astype(str).unique().tolist())
    gene_tokens = {g for g in gene_tokens if g and g.lower() not in ("nan", "none")}
    print("Unique gene tokens found:", len(gene_tokens))

    # Map each gene token to gene_id (ENSG no version) when possible
    token2geneid = {}
    for tok in gene_tokens:
        if tok.startswith("ENSG"):
            token2geneid[tok] = strip_ensg_version(tok)
        else:
            # assume it's a gene symbol; map if unique
            gid = name2id.get(tok, None)
            if gid is not None:
                token2geneid[tok] = gid

    print("Tokens mapped to ENSG IDs:", len(token2geneid))

    # Precompute expression summaries per mapped ENSG
    geneid2expr = {}
    missing_geneids = 0
    for tok, gid in token2geneid.items():
        if gid not in tpm_mat.index:
            missing_geneids += 1
            continue
        geneid2expr[gid] = summarize_expression_for_gene(gid, tpm_mat, tissues_brain, tissues_blood)

    print("Gene IDs with expression summaries:", len(geneid2expr))
    print("Mapped gene IDs missing from TPM matrix:", missing_geneids)

    # Add columns for each gene column
    for c in gene_cols:
        # create a standardized prefix per source column
        pref = c

        gene_id_col = f"{pref}__expr_gene_id"
        gene_name_col = f"{pref}__expr_gene_name"

        v[gene_id_col] = v[c].astype(str).map(lambda x: token2geneid.get(x, np.nan))
        v[gene_name_col] = v[gene_id_col].map(lambda gid: id2name.get(gid, np.nan))

        # attach numeric summaries
        for k in ["expr_brain_median_tpm", "expr_brain_max_tpm", "expr_blood_median_tpm", "expr_blood_max_tpm", "expr_top_tissue", "expr_top_tpm"]:
            outcol = f"{pref}__{k}"
            v[outcol] = v[gene_id_col].map(lambda gid: geneid2expr.get(gid, {}).get(k, np.nan if "tissue" not in k else ""))

    out_main = outdir / "FTD_all_variants_WITH_EXPR.tsv.gz"
    v.to_csv(out_main, sep="\t", index=False, compression="gzip")
    print("Wrote:", out_main)

    # Also write a compact gene summary table for paper
    # Use genes appearing in your master table that we could map + summarize
    gene_rows = []
    for gid, expr in geneid2expr.items():
        gene_rows.append({
            "gene_id": gid,
            "gene_name": id2name.get(gid, ""),
            **expr
        })
    gsum = pd.DataFrame(gene_rows)
    if not gsum.empty:
        gsum = gsum.sort_values(["expr_brain_max_tpm", "expr_brain_median_tpm"], ascending=[False, False])
    out_gsum = outdir / "FTD_gene_expression_summary.tsv"
    gsum.to_csv(out_gsum, sep="\t", index=False)
    print("Wrote:", out_gsum)

    print("\nDone Step 10 (expression context).")


if __name__ == "__main__":
    main()