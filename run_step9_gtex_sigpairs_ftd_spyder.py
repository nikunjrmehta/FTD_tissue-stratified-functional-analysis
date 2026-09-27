# -*- coding: utf-8 -*-
"""
Step 9 (GTEx v11 signif-pairs parquet): annotate FTD master table with GTEx eQTL/sQTL
significant variant-feature pairs from per-tissue parquet files.

This step DOES NOT do true colocalization (that is Step 9C later). Here we:
  - For each variant in the master table, check if it appears in any selected GTEx tissue
    "signif_pairs" parquet for eQTL and sQTL.
  - Summarize per-variant evidence: best (minimum) pval_nominal across all matching pairs,
    the corresponding feature (phenotype_id), tissue, and the list/count of tissues with any match.

Tissue selection:
  - All Brain_* tissues
  - Plus these 4 blood/immune contexts: Whole_Blood, Spleen, Cells_EBV-transformed_lymphocytes, Cells_Cultured_fibroblasts

Inputs:
  - Master table from Step 8:
      work\\step8_gtex_ftd\\FTD_all_variants_annotated_WITH_GTEx.tsv.gz
  - Parquet directories containing per-tissue files:
      <TISSUE>.v11.eQTLs.signif_pairs.parquet
      <TISSUE>.v11.sQTLs.signif_pairs.parquet

Outputs:
  work\\step9_gtex_sigpairs_ftd\\FTD_all_variants_WITH_GTEx_sigpairs.tsv.gz
  work\\step9_gtex_sigpairs_ftd\\GTEx_eqtl_sigpairs_tissue_counts.tsv
  work\\step9_gtex_sigpairs_ftd\\GTEx_sqtl_sigpairs_tissue_counts.tsv
"""

import os
import re
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd


# -----------------------------
# USER SETTINGS (edit here)
# -----------------------------
MASTER_PATH = r"work\step8_gtex_ftd\FTD_all_variants_annotated_WITH_GTEx.tsv.gz"

# Directories containing the per-tissue parquet files.
# You can point both to the same directory if you store e and s together.
EQTL_PARQUET_DIR = r"data/qtl/gtex/GTEx_Analysis_v11_eQTL/GTEx_Analysis_v11_eQTL"   # <-- change to your folder
SQTL_PARQUET_DIR = r"data/qtl/gtex/GTEx_Analysis_v11_sQTL/GTEx_Analysis_v11_sQTL"   # <-- change to your folder

OUTDIR = r"work\step9_gtex_sigpairs_ftd"

# Tissue inclusion
INCLUDE_ALL_BRAIN = True
INCLUDE_BLOOD_IMMUNE = True
BLOOD_IMMUNE_TISSUES = [
    "Whole_Blood",
    "Spleen",
    "Cells_EBV-transformed_lymphocytes",
    "Cells_Cultured_fibroblasts",
]

# In parquet signif_pairs, rows are already significant by GTEx pipeline.
# We still record pval_nominal as a strength-of-evidence score.
PVAL_COL = "pval_nominal"   # present in both eQTL and sQTL signif_pairs parquet

# -----------------------------
# Helpers
# -----------------------------
def read_table_auto(path: str) -> pd.DataFrame:
    path = str(path)
    if path.endswith(".gz"):
        return pd.read_csv(path, sep="\t", compression="gzip", low_memory=False)
    return pd.read_csv(path, sep="\t", low_memory=False)

def normalize_chr(ch):
    if pd.isna(ch):
        return ch
    s = str(ch)
    if not s.startswith("chr"):
        s = "chr" + s
    return s

def build_variant_id_set(df: pd.DataFrame):
    """
    Build GTEx variant_id strings: chr_pos_ref_alt_b38
    Include both ref/alt and flipped alt/ref.
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

_TISSUE_RE = re.compile(r"^(?P<tissue>.+?)\.v11\.(?P<kind>[es])QTLs\.signif_pairs\.parquet$", re.IGNORECASE)

def list_sigpair_parquets(parquet_dir: str, kind: str):
    """
    List (tissue, path) for eQTL/sQTL signif_pairs parquet files in a directory.
    kind: 'e' or 's'
    """
    parquet_dir = Path(parquet_dir)
    if not parquet_dir.exists():
        raise FileNotFoundError(f"Parquet directory not found: {parquet_dir}")

    out = []
    for p in parquet_dir.glob("*.parquet"):
        m = _TISSUE_RE.match(p.name)
        if not m:
            continue
        tissue = m.group("tissue")
        k = m.group("kind").lower()
        if k != kind:
            continue
        out.append((tissue, p))
    return sorted(out, key=lambda x: x[0])

def tissue_is_selected(tname: str) -> bool:
    return (
        (INCLUDE_ALL_BRAIN and tname.startswith("Brain_"))
        or (INCLUDE_BLOOD_IMMUNE and tname in BLOOD_IMMUNE_TISSUES)
    )

def summarize_sigpairs(kind: str, parquet_dir: str, target_vids: set):
    """
    Read selected tissue parquet files and summarize per-variant best pval + tissues.

    Returns:
      best_p: dict[variant_id] -> float (min pval_nominal)
      best_feature: dict[variant_id] -> phenotype_id (string)
      best_tissue: dict[variant_id] -> tissue (string)
      hit_tissues: dict[variant_id] -> list[tissue]
      tissue_counts: DataFrame with tissue-level counts
      tissues_selected: list[str]
    """
    assert kind in ("e", "s")
    files = list_sigpair_parquets(parquet_dir, kind)

    tissues_all = [t for t, _ in files]
    tissues_selected = [t for t in tissues_all if tissue_is_selected(t)]

    print(f"GTEx {'eQTL' if kind=='e' else 'sQTL'} signif_pairs: tissues available = {len(tissues_all)}")
    print(f"GTEx {'eQTL' if kind=='e' else 'sQTL'} signif_pairs: tissues selected = {len(tissues_selected)}")
    if tissues_selected:
        print("  Example tissues:", ", ".join(tissues_selected[:10]))
        extra = tissues_selected[10:20]
        if extra:
            print("  (also includes):", ", ".join(extra))

    best_p = {}
    best_feature = {}
    best_tissue = {}
    hit_tissues = defaultdict(list)

    # tissue-level summaries
    tissue_n_rows = Counter()
    tissue_n_variants = Counter()
    tissue_n_features = Counter()

    for tissue, p in files:
        if tissue not in tissues_selected:
            continue

        try:
            tdf = pd.read_parquet(p)
        except Exception as e:
            raise RuntimeError(f"Failed to read parquet: {p} ({e})")

        # Expected columns (from GTEx v11 signif_pairs parquet)
        need = ["variant_id", "phenotype_id", PVAL_COL]
        missing = [c for c in need if c not in tdf.columns]
        if missing:
            raise ValueError(f"{p.name}: missing columns {missing}. Columns found: {list(tdf.columns)}")

        # Filter to your variants
        tdf = tdf[tdf["variant_id"].isin(target_vids)].copy()
        if tdf.empty:
            continue

        # Make p numeric
        tdf[PVAL_COL] = pd.to_numeric(tdf[PVAL_COL], errors="coerce")

        tissue_n_rows[tissue] += int(tdf.shape[0])
        tissue_n_variants[tissue] += int(tdf["variant_id"].nunique(dropna=True))
        tissue_n_features[tissue] += int(tdf["phenotype_id"].nunique(dropna=True))

        # Best hit per variant in this tissue
        tdf = tdf.sort_values(["variant_id", PVAL_COL], ascending=[True, True])
        best_in_tissue = tdf.drop_duplicates("variant_id", keep="first")

        for r in best_in_tissue.itertuples(index=False):
            vid = r.variant_id
            pval = getattr(r, PVAL_COL)
            if pd.isna(pval):
                continue

            # record tissue hit
            hit_tissues[vid].append(tissue)

            prev = best_p.get(vid, np.inf)
            if pval < prev:
                best_p[vid] = float(pval)
                best_feature[vid] = str(r.phenotype_id)
                best_tissue[vid] = tissue

    tissue_counts = pd.DataFrame({
        "tissue": list(tissue_n_rows.keys()),
        "n_rows_matched": [tissue_n_rows[t] for t in tissue_n_rows.keys()],
        "n_variants_matched": [tissue_n_variants[t] for t in tissue_n_rows.keys()],
        "n_features_matched": [tissue_n_features[t] for t in tissue_n_rows.keys()],
    }).sort_values(["n_variants_matched", "n_rows_matched"], ascending=[False, False])

    return best_p, best_feature, best_tissue, hit_tissues, tissue_counts, tissues_selected

def add_sigpair_columns(v: pd.DataFrame, prefix: str, best_p, best_feature, best_tissue, hit_tissues):
    """
    Add per-row columns by mapping through GTEx variant_id for both orientations.
    Choose orientation with smaller best_p if both exist.
    """
    chr_norm = v["chr"].map(normalize_chr)
    pos = v["pos"].astype(int).astype(str)
    ref = v["ref"].astype(str).str.upper()
    alt = v["alt"].astype(str).str.upper()

    vid = chr_norm + "_" + pos + "_" + ref + "_" + alt + "_b38"
    vid_flip = chr_norm + "_" + pos + "_" + alt + "_" + ref + "_b38"

    out_p = []
    out_feat = []
    out_best_tiss = []
    out_tiss_list = []
    out_tiss_ct = []

    for a, b in zip(vid.tolist(), vid_flip.tolist()):
        pa = best_p.get(a, np.inf)
        pb = best_p.get(b, np.inf)

        if pa == np.inf and pb == np.inf:
            out_p.append(np.nan)
            out_feat.append(np.nan)
            out_best_tiss.append(np.nan)
            out_tiss_list.append("")
            out_tiss_ct.append(0)
            continue

        chosen = a if pa <= pb else b
        out_p.append(best_p.get(chosen, np.nan))
        out_feat.append(best_feature.get(chosen, np.nan))
        out_best_tiss.append(best_tissue.get(chosen, np.nan))

        ts = hit_tissues.get(chosen, [])
        ts_u = sorted(set(ts))
        out_tiss_list.append(";".join(ts_u))
        out_tiss_ct.append(len(ts_u))

    v[f"{prefix}_best_p"] = out_p
    v[f"{prefix}_best_feature_id"] = out_feat
    v[f"{prefix}_best_tissue"] = out_best_tiss
    v[f"{prefix}_hit_tissues"] = out_tiss_list
    v[f"{prefix}_hit_tissue_count"] = out_tiss_ct
    return v


# -----------------------------
# Main
# -----------------------------
def main():
    outdir = Path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading master table: {MASTER_PATH}")
    v = read_table_auto(MASTER_PATH)
    print("Loaded:", v.shape)

    print("Building GTEx variant_id set (ref/alt and flipped)...")
    target_vids = build_variant_id_set(v)
    print("Target variant_ids:", len(target_vids))

    # eQTL signif_pairs
    eq_best_p, eq_best_feat, eq_best_tiss, eq_hit_tiss, eq_counts, eq_sel = summarize_sigpairs(
        kind="e", parquet_dir=EQTL_PARQUET_DIR, target_vids=target_vids
    )

    # sQTL signif_pairs
    sq_best_p, sq_best_feat, sq_best_tiss, sq_hit_tiss, sq_counts, sq_sel = summarize_sigpairs(
        kind="s", parquet_dir=SQTL_PARQUET_DIR, target_vids=target_vids
    )

    # Add columns
    v = add_sigpair_columns(v, "gtex_eqtl_sigpair", eq_best_p, eq_best_feat, eq_best_tiss, eq_hit_tiss)
    v = add_sigpair_columns(v, "gtex_sqtl_sigpair", sq_best_p, sq_best_feat, sq_best_tiss, sq_hit_tiss)

    # Quick stats
    print("Variants with any eQTL signif-pair hit in selected tissues:", float(np.mean(~v["gtex_eqtl_sigpair_best_p"].isna())))
    print("Variants with any sQTL signif-pair hit in selected tissues:", float(np.mean(~v["gtex_sqtl_sigpair_best_p"].isna())))

    # Write outputs
    out_main = outdir / "FTD_all_variants_WITH_GTEx_sigpairs.tsv.gz"
    v.to_csv(out_main, sep="\t", index=False, compression="gzip")
    print("Wrote:", out_main)

    eq_out = outdir / "GTEx_eqtl_sigpairs_tissue_counts.tsv"
    eq_counts.to_csv(eq_out, sep="\t", index=False)
    print("Wrote:", eq_out)

    sq_out = outdir / "GTEx_sqtl_sigpairs_tissue_counts.tsv"
    sq_counts.to_csv(sq_out, sep="\t", index=False)
    print("Wrote:", sq_out)

    print("\nDone Step 9 (GTEx signif_pairs parquet annotation).")


if __name__ == "__main__":
    main()
