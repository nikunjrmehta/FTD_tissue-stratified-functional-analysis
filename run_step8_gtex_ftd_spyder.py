# -*- coding: utf-8 -*-
"""
Step 8 (GTEx v11): Annotate FTD prioritized variants with GTEx eGenes/sGenes
- Spyder-friendly (no argparse)
- Works with GTEx_Analysis_v11_eQTL.tar and GTEx_Analysis_v11_sQTL.tar
- Uses *.v11.eGenes.txt.gz and *.v11.sGenes.txt.gz inside the tarballs

Outputs:
  work\step8_gtex_ftd\FTD_all_variants_annotated_WITH_GTEx.tsv.gz
  work\step8_gtex_ftd\GTEx_eqtl_sig_tissue_counts.tsv
  work\step8_gtex_ftd\GTEx_sqtl_sig_tissue_counts.tsv
"""

import os
import re
import tarfile
import gzip
from io import BytesIO
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd


# -----------------------------
# USER SETTINGS (edit here)
# -----------------------------
VARIANTS_PATH = r"work\step7_annotate_ftd\FTD_all_variants_annotated.tsv.gz"  # or .tsv
GTEX_EQTL_TAR = r"data\qtl\gtex\GTEx_Analysis_v11_eQTL.tar"
GTEX_SQTL_TAR = r"data\qtl\gtex\GTEx_Analysis_v11_sQTL.tar"
OUTDIR = r"work\step8_gtex_ftd"

Q_SIG = 0.05

# Which tissues?
INCLUDE_ALL_BRAIN = True  # all Brain_* tissues in GTEx
INCLUDE_BLOOD_IMMUNE = True

# GTEx tissue names (prefixes) to include for blood/immune-ish context.
# Adjust if you want to be stricter.
BLOOD_IMMUNE_TISSUES= [
    "Whole_Blood",
    "Spleen",
    "Cells_EBV-transformed_lymphocytes",
    "Cells_Cultured_fibroblasts",
]

def tissue_is_selected(tname: str) -> bool:
    return (
        (INCLUDE_ALL_BRAIN and tname.startswith("Brain_"))
        or (INCLUDE_BLOOD_IMMUNE and tname in BLOOD_IMMUNE_TISSUES)
        or (INCLUDE_ALL_CELLS_PREFIX and tname.startswith("Cells_"))
    )


# If you want to *also* include all "Cells_*" automatically, set True:
INCLUDE_ALL_CELLS_PREFIX = False

# -----------------------------
# Helpers
# -----------------------------
def read_table_auto(path: str) -> pd.DataFrame:
    """Read TSV or TSV.GZ with robust separator."""
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
    Include both ref/alt and flipped alt/ref (because your table may use effect allele orientation).
    """
    need = ["chr", "pos", "ref", "alt"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in variants table.")
    chr_norm = df["chr"].map(normalize_chr)
    pos = df["pos"].astype(int)
    ref = df["ref"].astype(str).str.upper()
    alt = df["alt"].astype(str).str.upper()

    vid = chr_norm + "_" + pos.astype(str) + "_" + ref + "_" + alt + "_b38"
    vid_flip = chr_norm + "_" + pos.astype(str) + "_" + alt + "_" + ref + "_b38"
    return set(vid.dropna().unique().tolist()) | set(vid_flip.dropna().unique().tolist())

def list_gtex_members(tar_path: str, kind: str):
    """
    Return list of (tissue, member_name) for eGenes/sGenes inside tar.
    kind in {"e", "s"}
    """
    suffix = ".v11.eGenes.txt.gz" if kind == "e" else ".v11.sGenes.txt.gz"
    members = []
    with tarfile.open(tar_path, "r") as tf:
        for m in tf.getmembers():
            name = m.name
            if not name.endswith(suffix):
                continue
            base = os.path.basename(name)
            tissue = base.replace(suffix, "")
            members.append((tissue, name))
    return members

def choose_tissues(tissues_all):
    tissues_all = list(tissues_all)

    brain = [t for t in tissues_all if t.startswith("Brain_")] if INCLUDE_ALL_BRAIN else []
    blood_immune = [t for t in tissues_all if t in BLOOD_IMMUNE_TISSUES] if INCLUDE_BLOOD_IMMUNE else []
    cells = [t for t in tissues_all if t.startswith("Cells_")] if INCLUDE_ALL_CELLS_PREFIX else []

    selected = sorted(set(brain + blood_immune + cells))

    non_brain = sorted([t for t in tissues_all if not t.startswith("Brain_")])
    print(f"DEBUG: non-brain tissues available (first 20): {non_brain[:20]}")
    print(f"DEBUG: blood/immune matched: {blood_immune}")
    print(f"DEBUG: cells matched (first 10): {cells[:10]}")

    return selected


def read_gtex_member_df(tf: tarfile.TarFile, member_name: str, usecols):
    """
    Read a gzipped TSV inside a tar member into a DataFrame, selecting columns.
    """
    f = tf.extractfile(member_name)
    if f is None:
        raise FileNotFoundError(member_name)
    raw = f.read()
    with gzip.open(BytesIO(raw), "rt") as gz:
        # use python engine for safety with odd headers
        return pd.read_csv(gz, sep="\t", usecols=usecols, low_memory=False)

def annotate_gtex(df_variants: pd.DataFrame, tar_path: str, kind: str, target_vids: set):
    """
    Returns dicts mapping variant_id -> best hit and significant tissues.
    """
    assert kind in ("e", "s")
    members = list_gtex_members(tar_path, kind)
    tissues_all = [t for t, _ in members]
    tissues_sel = choose_tissues(tissues_all)

    # quick lookup member name by tissue
    tissue2member = {t: name for (t, name) in members}

    print(f"GTEx {'eQTL' if kind=='e' else 'sQTL'}: tissues available = {len(tissues_all)}")
    print(f"GTEx {'eQTL' if kind=='e' else 'sQTL'}: tissues selected = {len(tissues_sel)}")
    if tissues_sel:
        print("  Example tissues:", ", ".join(tissues_sel[:10]))
        print("  (also includes):", ", ".join(tissues_sel[-6:]))

    # Store best (lowest qval) per variant across selected tissues
    best_q = {}
    best_gene = {}
    best_tissue = {}

    # Store list of significant tissues per variant
    sig_tissues = defaultdict(list)

    # Tissue-level counts for summary
    tissue_sig_counts = Counter()

    # Columns present in both eGenes and sGenes:
    base_cols = ["variant_id", "gene_id", "gene_name", "qval"]
    # Some files may not include gene_name (rare); handle fallback
    fallback_cols = ["variant_id", "gene_id", "qval"]

    with tarfile.open(tar_path, "r") as tf:
        for tissue in tissues_sel:
            member = tissue2member[tissue]
            try:
                tdf = read_gtex_member_df(tf, member, usecols=base_cols)
                has_gene_name = True
            except Exception:
                tdf = read_gtex_member_df(tf, member, usecols=fallback_cols)
                tdf["gene_name"] = np.nan
                has_gene_name = False

            # Filter to target variants only (huge speed-up)
            tdf = tdf[tdf["variant_id"].isin(target_vids)].copy()
            if tdf.empty:
                continue

            # Ensure qval numeric
            tdf["qval"] = pd.to_numeric(tdf["qval"], errors="coerce")

            # Best hit per variant in this tissue
            # (lowest qval; if ties, first row)
            tdf = tdf.sort_values(["variant_id", "qval"], ascending=[True, True])
            best_in_tissue = tdf.drop_duplicates("variant_id", keep="first")

            # Update global best
            for r in best_in_tissue.itertuples(index=False):
                vid = r.variant_id
                q = r.qval
                if pd.isna(q):
                    continue
                prev = best_q.get(vid, np.inf)
                if q < prev:
                    best_q[vid] = float(q)
                    best_gene[vid] = getattr(r, "gene_name", np.nan)
                    best_tissue[vid] = tissue

            # Significant hits (q <= Q_SIG)
            sig = best_in_tissue[best_in_tissue["qval"] <= Q_SIG]
            if not sig.empty:
                tissue_sig_counts[tissue] += sig.shape[0]
                for vid in sig["variant_id"].tolist():
                    sig_tissues[vid].append(tissue)

    return best_q, best_gene, best_tissue, sig_tissues, tissue_sig_counts, tissues_sel

def add_gtex_columns(v: pd.DataFrame, prefix: str, best_q, best_gene, best_tissue, sig_tissues):
    """
    Adds columns to v using the GTEx variant_id computed from chr/pos/ref/alt.
    We map by both ref/alt and flipped.
    """
    # Build per-row variant_ids (both orientations)
    chr_norm = v["chr"].map(normalize_chr)
    pos = v["pos"].astype(int).astype(str)
    ref = v["ref"].astype(str).str.upper()
    alt = v["alt"].astype(str).str.upper()

    vid = chr_norm + "_" + pos + "_" + ref + "_" + alt + "_b38"
    vid_flip = chr_norm + "_" + pos + "_" + alt + "_" + ref + "_b38"

    # pick best among direct and flipped if both exist
    out_q = []
    out_gene = []
    out_tissue = []
    out_sig_list = []
    out_sig_ct = []

    for a, b in zip(vid.tolist(), vid_flip.tolist()):
        # choose which vid has smaller q (if both)
        qa = best_q.get(a, np.inf)
        qb = best_q.get(b, np.inf)
        if qa == np.inf and qb == np.inf:
            out_q.append(np.nan)
            out_gene.append(np.nan)
            out_tissue.append(np.nan)
            out_sig_list.append("")
            out_sig_ct.append(0)
            continue

        if qa <= qb:
            chosen = a
        else:
            chosen = b

        out_q.append(best_q.get(chosen, np.nan))
        out_gene.append(best_gene.get(chosen, np.nan))
        out_tissue.append(best_tissue.get(chosen, np.nan))

        sigs = sig_tissues.get(chosen, [])
        sigs_u = sorted(set(sigs))
        out_sig_list.append(";".join(sigs_u))
        out_sig_ct.append(len(sigs_u))

    v[f"{prefix}_best_q"] = out_q
    v[f"{prefix}_best_gene"] = out_gene
    v[f"{prefix}_best_tissue"] = out_tissue
    v[f"{prefix}_sig_tissues"] = out_sig_list
    v[f"{prefix}_sig_tissue_count"] = out_sig_ct

    return v

# -----------------------------
# Main
# -----------------------------
def main():
    outdir = Path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading annotated variants...")
    v = read_table_auto(VARIANTS_PATH)
    print("Loaded:", v.shape)
    print("Columns:", v.columns.tolist()[:12], "...")

    # Build target variant_id set
    print("Building GTEx variant_id set (ref/alt and flipped)...")
    target_vids = build_variant_id_set(v)
    print("Target variant_ids:", len(target_vids))

    # eGenes (eQTL)
    eq_best_q, eq_best_gene, eq_best_tissue, eq_sig_tissues, eq_tissue_counts, eq_tissues_sel = annotate_gtex(
        v, GTEX_EQTL_TAR, "e", target_vids
    )

    # sGenes (sQTL)
    sq_best_q, sq_best_gene, sq_best_tissue, sq_sig_tissues, sq_tissue_counts, sq_tissues_sel = annotate_gtex(
        v, GTEX_SQTL_TAR, "s", target_vids
    )

    # Add columns
    v = add_gtex_columns(v, "gtex_eqtl", eq_best_q, eq_best_gene, eq_best_tissue, eq_sig_tissues)
    v = add_gtex_columns(v, "gtex_sqtl", sq_best_q, sq_best_gene, sq_best_tissue, sq_sig_tissues)

    # Quick stats
    print("Variants with any eQTL match in selected tissues:", np.mean(~v["gtex_eqtl_best_q"].isna()))
    print("Variants with any sQTL match in selected tissues:", np.mean(~v["gtex_sqtl_best_q"].isna()))
    print(f"Variants with significant eQTL (q<= {Q_SIG}):", np.mean(v["gtex_eqtl_best_q"].fillna(1.0) <= Q_SIG))
    print(f"Variants with significant sQTL (q<= {Q_SIG}):", np.mean(v["gtex_sqtl_best_q"].fillna(1.0) <= Q_SIG))

    # Write main output
    out_main = outdir / "FTD_all_variants_annotated_WITH_GTEx.tsv.gz"
    v.to_csv(out_main, sep="\t", index=False, compression="gzip")
    print("Wrote:", out_main)

    # Tissue summaries
    if eq_tissue_counts:
        eq_df = pd.DataFrame({"tissue": list(eq_tissue_counts.keys()), "n_sig_variants": list(eq_tissue_counts.values())})
        eq_df = eq_df.sort_values("n_sig_variants", ascending=False)
    else:
        eq_df = pd.DataFrame({"tissue": eq_tissues_sel, "n_sig_variants": [0]*len(eq_tissues_sel)})
    eq_out = outdir / "GTEx_eqtl_sig_tissue_counts.tsv"
    eq_df.to_csv(eq_out, sep="\t", index=False)
    print("Wrote:", eq_out)

    if sq_tissue_counts:
        sq_df = pd.DataFrame({"tissue": list(sq_tissue_counts.keys()), "n_sig_variants": list(sq_tissue_counts.values())})
        sq_df = sq_df.sort_values("n_sig_variants", ascending=False)
    else:
        sq_df = pd.DataFrame({"tissue": sq_tissues_sel, "n_sig_variants": [0]*len(sq_tissues_sel)})
    sq_out = outdir / "GTEx_sqtl_sig_tissue_counts.tsv"
    sq_df.to_csv(sq_out, sep="\t", index=False)
    print("Wrote:", sq_out)

    print("\nDone Step 8 (GTEx eGenes/sGenes).")


if __name__ == "__main__":
    main()
