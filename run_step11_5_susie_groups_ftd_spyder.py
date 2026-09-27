# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 18:05:32 2026

@author: nmehta22
"""

# -*- coding: utf-8 -*-
"""
Step 11.5: Compute group-specific SuSiE maxima (Brain vs Blood/Immune)
- Spyder-friendly (no argparse)
- Memory-safe: per tissue parquet read -> filter to target variant_ids
- Handles schema differences (eQTL has afc/afc_se; sQTL often doesn't)

Inputs:
  MASTER_IN: work/step11_susie_ftd/FTD_all_variants_WITH_EXPR_WITH_SUSIE.tsv.gz
  EQTL_SUSIE_DIR: data/qtl/gtex/GTEx_Analysis_v11_eQTL_SuSiE
  SQTL_SUSIE_DIR: data/qtl/gtex/GTEx_Analysis_v11_sQTL_SuSiE

Outputs (work/step11_5_susie_groups_ftd/):
  FTD_all_variants_WITH_EXPR_WITH_SUSIE_GROUPS.tsv.gz
  SuSiE_eqtl_group_tissue_counts.tsv
  SuSiE_sqtl_group_tissue_counts.tsv
"""

from pathlib import Path
import re
from collections import Counter

import numpy as np
import pandas as pd


# -----------------------------
# USER SETTINGS
# -----------------------------
MASTER_IN = r"work/step11_susie_ftd/FTD_all_variants_WITH_EXPR_WITH_SUSIE.tsv.gz"

EQTL_SUSIE_DIR = r"data/qtl/gtex/GTEx_Analysis_v11_eQTL_SuSiE"
SQTL_SUSIE_DIR = r"data/qtl/gtex/GTEx_Analysis_v11_sQTL_SuSiE"

OUTDIR = r"work/step11_5_susie_groups_ftd"

BLOOD_IMMUNE_TISSUES = [
    "Whole_Blood",
    "Spleen",
    "Cells_EBV-transformed_lymphocytes",
    "Cells_Cultured_fibroblasts",
]

# Optional thresholds if you want extra summary counts later
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


def is_brain(tname: str) -> bool:
    return isinstance(tname, str) and tname.startswith("Brain_")


def is_blood(tname: str) -> bool:
    return isinstance(tname, str) and tname in BLOOD_IMMUNE_TISSUES


def tissue_group(tname: str) -> str:
    if is_brain(tname):
        return "brain"
    if is_blood(tname):
        return "blood"
    return "other"


def list_susie_parquets(folder: str, kind: str):
    """
    kind: 'e' or 's'
    expected filenames:
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


def read_susie_filtered(parquet_path: str, target_vids: set) -> pd.DataFrame:
    """
    Schema-aware: read only available columns.
    """
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

    cols = None
    try:
        import pyarrow.parquet as pq
        cols = pq.ParquetFile(parquet_path).schema.names
    except Exception:
        cols = None

    if cols is None:
        df = pd.read_parquet(parquet_path)
    else:
        usecols = [c for c in desired if c in cols]
        required = {"variant_id", "pip", "cs_id", "cs_size"}
        missing_req = [c for c in required if c not in usecols]
        if missing_req:
            raise ValueError(
                f"Parquet {parquet_path} missing required columns {missing_req}. Available: {cols}"
            )
        df = pd.read_parquet(parquet_path, columns=usecols)

    df = df[df["variant_id"].isin(target_vids)].copy()
    if df.empty:
        return df

    df["pip"] = pd.to_numeric(df["pip"], errors="coerce")
    if "cs_id" in df.columns:
        df["cs_id"] = pd.to_numeric(df["cs_id"], errors="coerce")
    if "cs_size" in df.columns:
        df["cs_size"] = pd.to_numeric(df["cs_size"], errors="coerce")
    if "afc" in df.columns:
        df["afc"] = pd.to_numeric(df["afc"], errors="coerce")
    if "afc_se" in df.columns:
        df["afc_se"] = pd.to_numeric(df["afc_se"], errors="coerce")

    return df


def summarize_susie_by_group(susie_dir: str, kind: str, target_vids: set):
    """
    Compute best (max) PIP per variant, separately within brain and blood groups.
    We keep: best_pip, best_gene_name, best_feature_id (phenotype_id/gene_id),
             best_tissue, best_cs_id, best_cs_size, best_afc, best_afc_se,
             n_tissues_hit per group.

    Returns dicts for brain and blood, plus tissue counts.
    """
    files = list_susie_parquets(susie_dir, kind)

    tissues_all = [t for t, _ in files]
    # selected = all brain + the 4 blood/immune
    tissues_sel = [t for t in tissues_all if is_brain(t) or is_blood(t)]
    files_sel = [(t, p) for t, p in files if t in tissues_sel]

    print(f"SuSiE {'eQTL' if kind=='e' else 'sQTL'}: tissues available = {len(tissues_all)}")
    print(f"SuSiE {'eQTL' if kind=='e' else 'sQTL'}: tissues selected = {len(tissues_sel)}")
    if tissues_sel:
        print("  Example tissues:", ", ".join(tissues_sel[:10]))
        if len(tissues_sel) > 10:
            print("  (also includes):", ", ".join(tissues_sel[10:17]))

    # group dict containers
    groups = ["brain", "blood"]
    best = {g: {} for g in groups}              # best_pip
    best_gene = {g: {} for g in groups}
    best_feat = {g: {} for g in groups}         # phenotype_id if present, else gene_id if present
    best_tiss = {g: {} for g in groups}
    best_cs_id = {g: {} for g in groups}
    best_cs_size = {g: {} for g in groups}
    best_afc = {g: {} for g in groups}
    best_afc_se = {g: {} for g in groups}
    n_tiss_hit = {g: Counter() for g in groups}

    # tissue-level summary counts
    tissue_counts = Counter()
    tissue_counts_pip = {thr: Counter() for thr in PIP_THRESHOLDS}

    for tissue, pq in files_sel:
        grp = tissue_group(tissue)
        if grp not in ("brain", "blood"):
            continue

        df = read_susie_filtered(pq, target_vids)
        if df.empty:
            continue

        # within this tissue, keep max pip per variant
        df = df.sort_values(["variant_id", "pip"], ascending=[True, False])
        df_best = df.drop_duplicates("variant_id", keep="first")

        tissue_counts[tissue] += df_best.shape[0]
        for thr in PIP_THRESHOLDS:
            tissue_counts_pip[thr][tissue] += int((df_best["pip"].fillna(0.0) >= thr).sum())

        for r in df_best.itertuples(index=False):
            vid = r.variant_id
            pip = r.pip
            if pd.isna(pip):
                continue

            n_tiss_hit[grp][vid] += 1

            prev = best[grp].get(vid, -1.0)
            if pip > prev:
                best[grp][vid] = float(pip)
                # gene_name may exist for both e/s
                best_gene[grp][vid] = getattr(r, "gene_name", np.nan)

                # feature id: prefer phenotype_id if present, else gene_id
                feat = np.nan
                if hasattr(r, "phenotype_id"):
                    feat = getattr(r, "phenotype_id", np.nan)
                elif hasattr(r, "gene_id"):
                    feat = getattr(r, "gene_id", np.nan)
                best_feat[grp][vid] = feat

                best_tiss[grp][vid] = tissue
                best_cs_id[grp][vid] = getattr(r, "cs_id", np.nan)
                best_cs_size[grp][vid] = getattr(r, "cs_size", np.nan)
                best_afc[grp][vid] = getattr(r, "afc", np.nan) if hasattr(r, "afc") else np.nan
                best_afc_se[grp][vid] = getattr(r, "afc_se", np.nan) if hasattr(r, "afc_se") else np.nan

    return (best, best_gene, best_feat, best_tiss, best_cs_id, best_cs_size, best_afc, best_afc_se,
            n_tiss_hit, tissue_counts, tissue_counts_pip, tissues_sel)


def add_group_columns(v: pd.DataFrame, prefix: str, group: str,
                      best_pip, best_gene, best_feat, best_tiss, best_cs_id, best_cs_size, best_afc, best_afc_se,
                      n_tiss_hit):
    """
    Join group dicts onto master by variant_id (and flipped).
    Adds columns:
      {prefix}_{group}_max_pip, ..._best_gene, ..._best_feature_id, ..._best_tissue, ..._best_cs_id, ..._best_cs_size, ..._best_afc, ..._best_afc_se, ..._n_tissues_hit
    """
    chr_norm = v["chr"].map(normalize_chr)
    pos = v["pos"].astype(int).astype(str)
    ref = v["ref"].astype(str).str.upper()
    alt = v["alt"].astype(str).str.upper()

    vid = chr_norm + "_" + pos + "_" + ref + "_" + alt + "_b38"
    vid_flip = chr_norm + "_" + pos + "_" + alt + "_" + ref + "_b38"

    out_pip, out_gene, out_feat, out_tiss = [], [], [], []
    out_cs_id, out_cs_size, out_afc, out_afc_se, out_nt = [], [], [], [], []

    for a, b in zip(vid.tolist(), vid_flip.tolist()):
        pa = best_pip.get(a, -1.0)
        pb = best_pip.get(b, -1.0)

        if pa < 0 and pb < 0:
            out_pip.append(np.nan)
            out_gene.append(np.nan)
            out_feat.append(np.nan)
            out_tiss.append(np.nan)
            out_cs_id.append(np.nan)
            out_cs_size.append(np.nan)
            out_afc.append(np.nan)
            out_afc_se.append(np.nan)
            out_nt.append(0)
            continue

        chosen = a if pa >= pb else b
        out_pip.append(best_pip.get(chosen, np.nan))
        out_gene.append(best_gene.get(chosen, np.nan))
        out_feat.append(best_feat.get(chosen, np.nan))
        out_tiss.append(best_tiss.get(chosen, np.nan))
        out_cs_id.append(best_cs_id.get(chosen, np.nan))
        out_cs_size.append(best_cs_size.get(chosen, np.nan))
        out_afc.append(best_afc.get(chosen, np.nan))
        out_afc_se.append(best_afc_se.get(chosen, np.nan))
        out_nt.append(int(n_tiss_hit.get(chosen, 0)))

    v[f"{prefix}_{group}_max_pip"] = out_pip
    v[f"{prefix}_{group}_best_gene"] = out_gene
    v[f"{prefix}_{group}_best_feature_id"] = out_feat
    v[f"{prefix}_{group}_best_tissue"] = out_tiss
    v[f"{prefix}_{group}_best_cs_id"] = out_cs_id
    v[f"{prefix}_{group}_best_cs_size"] = out_cs_size
    v[f"{prefix}_{group}_best_afc"] = out_afc
    v[f"{prefix}_{group}_best_afc_se"] = out_afc_se
    v[f"{prefix}_{group}_n_tissues_hit"] = out_nt

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

    # ---- eQTL ----
    (eq_best, eq_gene, eq_feat, eq_tiss, eq_cs_id, eq_cs_size, eq_afc, eq_afc_se,
     eq_ntiss, eq_tissue_counts, eq_tissue_counts_pip, eq_sel) = summarize_susie_by_group(EQTL_SUSIE_DIR, "e", target_vids)

    # ---- sQTL ----
    (sq_best, sq_gene, sq_feat, sq_tiss, sq_cs_id, sq_cs_size, sq_afc, sq_afc_se,
     sq_ntiss, sq_tissue_counts, sq_tissue_counts_pip, sq_sel) = summarize_susie_by_group(SQTL_SUSIE_DIR, "s", target_vids)

    # Add group columns to master
    for grp in ["brain", "blood"]:
        v = add_group_columns(
            v, "susie_eqtl", grp,
            eq_best[grp], eq_gene[grp], eq_feat[grp], eq_tiss[grp], eq_cs_id[grp], eq_cs_size[grp], eq_afc[grp], eq_afc_se[grp],
            eq_ntiss[grp]
        )
        v = add_group_columns(
            v, "susie_sqtl", grp,
            sq_best[grp], sq_gene[grp], sq_feat[grp], sq_tiss[grp], sq_cs_id[grp], sq_cs_size[grp], sq_afc[grp], sq_afc_se[grp],
            sq_ntiss[grp]
        )

    # Write updated master
    out_main = outdir / "FTD_all_variants_WITH_EXPR_WITH_SUSIE_GROUPS.tsv.gz"
    v.to_csv(out_main, sep="\t", index=False, compression="gzip")
    print("Wrote:", out_main)

    # Tissue summaries
    def write_counts(counts: Counter, outpath: Path, label: str):
        df = pd.DataFrame({"tissue": list(counts.keys()), label: list(counts.values())})
        df = df.sort_values(label, ascending=False)
        df.to_csv(outpath, sep="\t", index=False)

    write_counts(eq_tissue_counts, outdir / "SuSiE_eqtl_group_tissue_counts.tsv", "n_variants_in_susie")
    write_counts(sq_tissue_counts, outdir / "SuSiE_sqtl_group_tissue_counts.tsv", "n_variants_in_susie")

    # Optional PIP threshold counts
    for thr in PIP_THRESHOLDS:
        write_counts(eq_tissue_counts_pip[thr], outdir / f"SuSiE_eqtl_tissue_counts_pip_ge_{thr}.tsv", f"n_variants_pip_ge_{thr}")
        write_counts(sq_tissue_counts_pip[thr], outdir / f"SuSiE_sqtl_tissue_counts_pip_ge_{thr}.tsv", f"n_variants_pip_ge_{thr}")

    print("\nDone Step 11.5 (group-specific SuSiE maxima).")


if __name__ == "__main__":
    main()