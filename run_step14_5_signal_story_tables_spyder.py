# -*- coding: utf-8 -*-
"""
Created on Sat Feb 21 16:03:52 2026

@author: nmehta22
"""

# -*- coding: utf-8 -*-
"""
Step 14: Build final 3-signal story tables (Spyder-friendly)

Inputs:
  work/step12_final_ftd/FTD_all_variants_FINAL.tsv.gz
  work/step12_final_ftd/FTD_locus_summary_FINAL.tsv

Outputs (work/step14_signal_story/):
  FTD_signal_summary_3signals_FINAL.tsv
  FTD_signal_lead_vs_topfunctional.tsv
  FTD_top50_variants_per_signal_OVERALL.tsv
  FTD_top50_variants_per_signal_BRAIN_GWAS_FILTERED.tsv
  FTD_top50_variants_per_signal_BLOOD_GWAS_FILTERED.tsv
  FTD_top_genes_per_signal.tsv
"""

from pathlib import Path
import numpy as np
import pandas as pd


# -----------------------------
# USER SETTINGS
# -----------------------------
FINAL_TABLE = r"work/step12_final_ftd/FTD_all_variants_FINAL.tsv.gz"
LOCUS_SUMMARY = r"work/step12_final_ftd/FTD_locus_summary_FINAL.tsv"
OUTDIR = r"work/step14_signal_story"

TOPK = 50

# GWAS filter settings (same as your Step 13)
P_THRESH = 1e-4
TOPN_BY_P = 500


# -----------------------------
# Helpers
# -----------------------------
def read_table(path: str) -> pd.DataFrame:
    if path.endswith(".gz"):
        return pd.read_csv(path, sep="\t", compression="gzip", low_memory=False)
    return pd.read_csv(path, sep="\t", low_memory=False)

def to_numeric(s):
    return pd.to_numeric(s, errors="coerce")

def safe_log10p(p):
    p = to_numeric(p)
    p = p.clip(lower=1e-300)
    return -np.log10(p)

def gwas_filter_per_locus(df_locus: pd.DataFrame) -> pd.DataFrame:
    g = df_locus.copy()
    g["p_num"] = to_numeric(g["p"])
    g = g.dropna(subset=["p_num"])
    if g.empty:
        return g
    g_f = g[g["p_num"] <= P_THRESH].copy()
    if g_f.empty:
        g_f = g.sort_values("p_num").head(TOPN_BY_P).copy()
    return g_f

def pick_lead_by_p(df_signal: pd.DataFrame) -> pd.Series:
    g = df_signal.copy()
    g["p_num"] = to_numeric(g["p"])
    g = g.dropna(subset=["p_num"])
    if g.empty:
        return df_signal.iloc[0]
    return g.sort_values("p_num").iloc[0]

def pick_top_by_col(df_signal: pd.DataFrame, col: str) -> pd.Series:
    g = df_signal.copy()
    g[col] = to_numeric(g[col])
    g = g.dropna(subset=[col])
    if g.empty:
        return df_signal.iloc[0]
    return g.sort_values(col, ascending=False).iloc[0]

def first_nonempty(*vals):
    for v in vals:
        if pd.isna(v):
            continue
        s = str(v).strip()
        if s and s.lower() != "nan":
            return s
    return ""


# -----------------------------
# Main
# -----------------------------
def main():
    outdir = Path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading final variant table:", FINAL_TABLE)
    df = read_table(FINAL_TABLE)
    print("Loaded:", df.shape)

    print("Loading locus summary:", LOCUS_SUMMARY)
    loc = read_table(LOCUS_SUMMARY)
    print("Loaded:", loc.shape)

    # Build a mapping from locus_id -> signal_id (lead variant key by p)
    # This is the cleanest way to collapse 25 loci -> 3 signals
    if "locus_id" not in loc.columns or "lead_variant_key_by_p" not in loc.columns:
        raise ValueError("LOCUS_SUMMARY must contain locus_id and lead_variant_key_by_p")

    locus2signal = dict(zip(loc["locus_id"].astype(str), loc["lead_variant_key_by_p"].astype(str)))
    df["signal_id"] = df["locus_id"].astype(str).map(locus2signal)

    # Drop rows that didn't map (should not happen)
    df = df.dropna(subset=["signal_id"]).copy()

    # Add a handy GWAS log10p column
    df["gwas_log10p"] = safe_log10p(df["p"])

    # -----------------------------
    # (1) Signal summary (3 rows)
    # -----------------------------
    rows = []
    for sig, gsig in df.groupby("signal_id"):
        lead = pick_lead_by_p(gsig)
        top_overall = pick_top_by_col(gsig, "final_score")

        # Brain/Blood “best evidence” among GWAS-supported variants, within the signal
        # Apply GWAS filter per locus, then combine to signal-level pool
        pooled = []
        for lid, gloc in gsig.groupby("locus_id"):
            gf = gwas_filter_per_locus(gloc)
            if not gf.empty:
                pooled.append(gf)
        g_pool = pd.concat(pooled, ignore_index=True) if pooled else gsig.copy()

        top_brain = pick_top_by_col(g_pool, "brain_evidence_score") if "brain_evidence_score" in g_pool.columns else top_overall
        top_blood = pick_top_by_col(g_pool, "blood_evidence_score") if "blood_evidence_score" in g_pool.columns else top_overall

        rows.append({
            "signal_id": sig,

            # Lead GWAS variant
            "lead_variant_key": lead.get("variant_key", ""),
            "lead_chr": lead.get("chr", ""),
            "lead_pos": lead.get("pos", np.nan),
            "lead_p": lead.get("p", np.nan),
            "lead_gwas_log10p": lead.get("gwas_log10p", np.nan),

            # Top overall functional candidate
            "top_overall_variant_key": top_overall.get("variant_key", ""),
            "top_overall_p": top_overall.get("p", np.nan),
            "top_overall_final_score": top_overall.get("final_score", np.nan),
            "top_overall_alphagenome_pct": top_overall.get("alphagenome_pct_max", np.nan),
            "top_overall_ccre_overlap": top_overall.get("ccre_overlap", ""),

            # Brain/Blood evidence summaries (group-specific SuSiE from Step 11.5 already merged)
            "top_overall_susie_brain_pip": top_overall.get("susie_brain_max_pip_any", np.nan),
            "top_overall_susie_blood_pip": top_overall.get("susie_blood_max_pip_any", np.nan),
            "top_overall_susie_best_group": top_overall.get("susie_best_group_any", ""),

            # GTEx “best genes”
            "top_overall_gtex_eqtl_gene": top_overall.get("gtex_eqtl_best_gene", ""),
            "top_overall_gtex_eqtl_tissue": top_overall.get("gtex_eqtl_best_tissue", ""),
            "top_overall_gtex_sqtl_gene": top_overall.get("gtex_sqtl_best_gene", ""),
            "top_overall_gtex_sqtl_tissue": top_overall.get("gtex_sqtl_best_tissue", ""),

            # Expression context
            "top_overall_expr_brain_max_tpm_any": top_overall.get("expr_brain_max_tpm_any", np.nan),
            "top_overall_expr_blood_max_tpm_any": top_overall.get("expr_blood_max_tpm_any", np.nan),

            # Best GWAS-supported brain/blood candidates (for narrative)
            "top_brain_variant_key_gwas_filtered": top_brain.get("variant_key", ""),
            "top_brain_evidence_score": top_brain.get("brain_evidence_score", np.nan),
            "top_brain_p": top_brain.get("p", np.nan),
            "top_brain_susie_brain_pip": top_brain.get("susie_brain_max_pip_any", np.nan),

            "top_blood_variant_key_gwas_filtered": top_blood.get("variant_key", ""),
            "top_blood_evidence_score": top_blood.get("blood_evidence_score", np.nan),
            "top_blood_p": top_blood.get("p", np.nan),
            "top_blood_susie_blood_pip": top_blood.get("susie_blood_max_pip_any", np.nan),
        })

    sig_summary = pd.DataFrame(rows).sort_values("lead_gwas_log10p", ascending=False)
    out1 = outdir / "FTD_signal_summary_3signals_FINAL.tsv"
    sig_summary.to_csv(out1, sep="\t", index=False)
    print("Wrote:", out1)

    # -----------------------------
    # (2) Lead vs Top functional (3 rows) – reviewer-friendly
    # -----------------------------
    lv = sig_summary[[
        "signal_id",
        "lead_variant_key","lead_p","lead_gwas_log10p",
        "top_overall_variant_key","top_overall_p","top_overall_final_score",
        "top_overall_susie_best_group",
        "top_overall_susie_brain_pip","top_overall_susie_blood_pip",
        "top_overall_ccre_overlap",
        "top_overall_gtex_eqtl_gene","top_overall_gtex_eqtl_tissue",
        "top_overall_gtex_sqtl_gene","top_overall_gtex_sqtl_tissue",
    ]].copy()
    lv["lead_equals_top_overall"] = (lv["lead_variant_key"] == lv["top_overall_variant_key"])
    out2 = outdir / "FTD_signal_lead_vs_topfunctional.tsv"
    lv.to_csv(out2, sep="\t", index=False)
    print("Wrote:", out2)

    # -----------------------------
    # (3) Top variants per signal – OVERALL
    # -----------------------------
    df_overall = (df.sort_values(["signal_id","final_score"], ascending=[True, False])
                    .groupby("signal_id")
                    .head(TOPK)
                    .copy())
    out3 = outdir / f"FTD_top{TOPK}_variants_per_signal_OVERALL.tsv"
    df_overall.to_csv(out3, sep="\t", index=False)
    print("Wrote:", out3)

    # -----------------------------
    # (4) Top variants per signal – BRAIN evidence, GWAS-filtered (per locus first)
    # -----------------------------
    chunks = []
    for sig, gsig in df.groupby("signal_id"):
        pooled = []
        for lid, gloc in gsig.groupby("locus_id"):
            gf = gwas_filter_per_locus(gloc)
            if not gf.empty:
                pooled.append(gf)
        g_pool = pd.concat(pooled, ignore_index=True) if pooled else gsig.copy()

        if "brain_evidence_score" in g_pool.columns:
            top_sig = g_pool.sort_values("brain_evidence_score", ascending=False).head(TOPK).copy()
        else:
            top_sig = g_pool.sort_values("final_score", ascending=False).head(TOPK).copy()

        top_sig["signal_id"] = sig
        chunks.append(top_sig)

    df_brain = pd.concat(chunks, ignore_index=True)
    out4 = outdir / f"FTD_top{TOPK}_variants_per_signal_BRAIN_GWAS_FILTERED.tsv"
    df_brain.to_csv(out4, sep="\t", index=False)
    print("Wrote:", out4)

    # -----------------------------
    # (5) Top variants per signal – BLOOD evidence, GWAS-filtered
    # -----------------------------
    chunks = []
    for sig, gsig in df.groupby("signal_id"):
        pooled = []
        for lid, gloc in gsig.groupby("locus_id"):
            gf = gwas_filter_per_locus(gloc)
            if not gf.empty:
                pooled.append(gf)
        g_pool = pd.concat(pooled, ignore_index=True) if pooled else gsig.copy()

        if "blood_evidence_score" in g_pool.columns:
            top_sig = g_pool.sort_values("blood_evidence_score", ascending=False).head(TOPK).copy()
        else:
            top_sig = g_pool.sort_values("final_score", ascending=False).head(TOPK).copy()

        top_sig["signal_id"] = sig
        chunks.append(top_sig)

    df_blood = pd.concat(chunks, ignore_index=True)
    out5 = outdir / f"FTD_top{TOPK}_variants_per_signal_BLOOD_GWAS_FILTERED.tsv"
    df_blood.to_csv(out5, sep="\t", index=False)
    print("Wrote:", out5)

    # -----------------------------
    # (6) Signal-level gene summary (top genes implicated by GTEx/SuSiE)
    # -----------------------------
    gene_rows = []
    for sig, gsig in df.groupby("signal_id"):
        # collect candidate gene strings
        cols = [c for c in [
            "gtex_eqtl_best_gene","gtex_sqtl_best_gene",
            "susie_eqtl_brain_best_gene","susie_eqtl_blood_best_gene",
            "susie_sqtl_brain_best_gene","susie_sqtl_blood_best_gene",
            "susie_eqtl_best_gene","susie_sqtl_best_gene"
        ] if c in gsig.columns]

        genes = []
        for c in cols:
            genes += gsig[c].dropna().astype(str).tolist()
        genes = [g for g in genes if g and g.lower() != "nan"]

        if not genes:
            continue

        s = pd.Series(genes).value_counts().head(20)
        for gene, ct in s.items():
            gene_rows.append({
                "signal_id": sig,
                "gene": gene,
                "n_mentions_in_table": int(ct)
            })

    gene_sum = pd.DataFrame(gene_rows)
    if not gene_sum.empty:
        gene_sum = gene_sum.sort_values(["signal_id","n_mentions_in_table"], ascending=[True, False])
    out6 = outdir / "FTD_top_genes_per_signal.tsv"
    gene_sum.to_csv(out6, sep="\t", index=False)
    print("Wrote:", out6)

    print("\nDone Step 14 (3-signal story tables).")


if __name__ == "__main__":
    main()