# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 15:11:52 2026

@author: nmehta22
"""

# -*- coding: utf-8 -*-
"""
Step 12: Final prioritization + brain vs blood/immune views
- Spyder-friendly (no argparse)
- Uses Step 11 master with SuSiE + expression context

Input:
  work/step11_susie_ftd/FTD_all_variants_WITH_EXPR_WITH_SUSIE.tsv.gz

Outputs (work/step12_final_ftd/):
  FTD_all_variants_FINAL.tsv.gz
  FTD_top20_per_locus_OVERALL.tsv
  FTD_top20_per_locus_BRAIN.tsv
  FTD_top20_per_locus_BLOOD.tsv
  FTD_top500_global_OVERALL.tsv
  FTD_top500_global_BRAIN.tsv
  FTD_top500_global_BLOOD.tsv
  FTD_locus_summary_FINAL.tsv
"""

from pathlib import Path
import numpy as np
import pandas as pd

# -----------------------------
# USER SETTINGS
# -----------------------------
MASTER_IN = r"work/step11_5_susie_groups_ftd/FTD_all_variants_WITH_EXPR_WITH_SUSIE_GROUPS.tsv.gz"
OUTDIR = r"work/step12_final_ftd"

BLOOD_IMMUNE_TISSUES = [
    "Whole_Blood",
    "Spleen",
    "Cells_EBV-transformed_lymphocytes",
    "Cells_Cultured_fibroblasts",
]

TOP_PER_LOCUS = 20
TOP_GLOBAL = 500

# Score weights (overall)
W_GWAS = 0.25
W_AG   = 0.25
W_SUSIE= 0.15
W_GTEX = 0.15
W_CCRE = 0.10
W_EXPR = 0.10

# Brain/blood score weights (presentation views)
WB_SUSIE = 0.40
WB_GTEX  = 0.35
WB_EXPR  = 0.25


# -----------------------------
# Helpers
# -----------------------------
def read_table(path: str) -> pd.DataFrame:
    if path.endswith(".gz"):
        return pd.read_csv(path, sep="\t", compression="gzip", low_memory=False)
    return pd.read_csv(path, sep="\t", low_memory=False)

def is_brain_tissue(t: str) -> bool:
    return isinstance(t, str) and t.startswith("Brain_")

def is_blood_tissue(t: str) -> bool:
    return isinstance(t, str) and t in BLOOD_IMMUNE_TISSUES

def split_tissue_list(x) -> list:
    if not isinstance(x, str) or x.strip() == "" or x.lower() == "nan":
        return []
    return [t.strip() for t in x.split(";") if t.strip()]

def count_brain_blood(tissues: list):
    b = sum(is_brain_tissue(t) for t in tissues)
    bl = sum(is_blood_tissue(t) for t in tissues)
    return b, bl

def nonempty_ccre(x) -> int:
    if pd.isna(x):
        return 0
    s = str(x).strip()
    return 1 if s != "" and s.lower() != "nan" else 0

def safe_log10p(p):
    # avoid -inf for p=0
    p = pd.to_numeric(p, errors="coerce")
    p = p.clip(lower=1e-300)
    return -np.log10(p)

def cap01(x):
    x = pd.to_numeric(x, errors="coerce")
    return x.clip(lower=0.0, upper=1.0)

def max_across_cols(df, cols):
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.Series(np.nan, index=df.index)
    return df[cols].apply(pd.to_numeric, errors="coerce").max(axis=1)

def group_from_best_tissue(t):
    if is_brain_tissue(t):
        return "Brain"
    if is_blood_tissue(t):
        return "BloodImmune"
    if pd.isna(t):
        return ""
    return "Other"


# -----------------------------
# Main
# -----------------------------
def main():
    outdir = Path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading:", MASTER_IN)
    v = read_table(MASTER_IN)
    print("Loaded:", v.shape)

    # ---------
    # Brain vs blood GTEx sigpairs counts
    # ---------
    # Use the sigpair hit tissue lists (Step 9 output)
    for qtl in ["eqtl", "sqtl"]:
        col_list = f"gtex_{qtl}_sigpair_hit_tissues"
        if col_list not in v.columns:
            continue

        brain_ct = []
        blood_ct = []
        any_brain = []
        any_blood = []

        for x in v[col_list].tolist():
            tis = split_tissue_list(x)
            b, bl = count_brain_blood(tis)
            brain_ct.append(b)
            blood_ct.append(bl)
            any_brain.append(b > 0)
            any_blood.append(bl > 0)

        v[f"gtex_{qtl}_sigpair_hit_brain_count"] = brain_ct
        v[f"gtex_{qtl}_sigpair_hit_blood_count"] = blood_ct
        v[f"gtex_{qtl}_sigpair_hit_any_brain"] = any_brain
        v[f"gtex_{qtl}_sigpair_hit_any_blood"] = any_blood

    # Combined GTEx support (brain/blood)
    v["gtex_sigpair_hit_brain_count"] = v.get("gtex_eqtl_sigpair_hit_brain_count", 0) + v.get("gtex_sqtl_sigpair_hit_brain_count", 0)
    v["gtex_sigpair_hit_blood_count"] = v.get("gtex_eqtl_sigpair_hit_blood_count", 0) + v.get("gtex_sqtl_sigpair_hit_blood_count", 0)
    v["gtex_sigpair_hit_any_brain"] = (v["gtex_sigpair_hit_brain_count"] > 0)
    v["gtex_sigpair_hit_any_blood"] = (v["gtex_sigpair_hit_blood_count"] > 0)

    # Normalize GTEx tissue count evidence to 0..1 (17 max tissues used)
    v["gtex_brain_score"] = (v["gtex_sigpair_hit_brain_count"] / 13.0).clip(0, 1)   # 13 brain tissues
    v["gtex_blood_score"] = (v["gtex_sigpair_hit_blood_count"] / 4.0).clip(0, 1)   # 4 blood/immune

    # ---------
    # SuSiE (Step 11.5): use group-specific max PIP columns (Brain vs Blood/Immune)
    # ---------

    # Step 11.5 produces (canonical) columns:
    #   susie_eqtl_brain_max_pip, susie_sqtl_brain_max_pip
    #   susie_eqtl_blood_max_pip, susie_sqtl_blood_max_pip
    # We'll compute group maxima robustly (and fall back gracefully if columns are missing).

    # Detect group columns (robust to small naming differences)
    susie_brain_cols = [
        c for c in v.columns
        if c.lower().startswith("susie_") and "brain" in c.lower() and "pip" in c.lower() and "max" in c.lower()
    ]
    susie_blood_cols = [
        c for c in v.columns
        if c.lower().startswith("susie_") and ("blood" in c.lower() or "immune" in c.lower()) and "pip" in c.lower() and "max" in c.lower()
    ]

    # Prefer canonical names when present
    preferred_brain = [c for c in ["susie_eqtl_brain_max_pip", "susie_sqtl_brain_max_pip"] if c in v.columns]
    preferred_blood = [c for c in ["susie_eqtl_blood_max_pip", "susie_sqtl_blood_max_pip"] if c in v.columns]
    if preferred_brain:
        susie_brain_cols = preferred_brain
    if preferred_blood:
        susie_blood_cols = preferred_blood

    v["susie_brain_max_pip_any"] = max_across_cols(v, susie_brain_cols)
    v["susie_blood_max_pip_any"] = max_across_cols(v, susie_blood_cols)

    # Overall max PIP from any available SuSiE columns
    v["susie_max_pip_any"] = max_across_cols(
        v,
        ["susie_eqtl_max_pip", "susie_sqtl_max_pip", "susie_brain_max_pip_any", "susie_blood_max_pip_any"],
    )

    # Scores (0..1)
    v["susie_brain_score"] = cap01(v["susie_brain_max_pip_any"])
    v["susie_blood_score"] = cap01(v["susie_blood_max_pip_any"])
    v["susie_score_any"] = cap01(v["susie_max_pip_any"])

    # Best group label for reporting
    b = v["susie_brain_max_pip_any"].fillna(0.0)
    d = v["susie_blood_max_pip_any"].fillna(0.0)
    v["susie_best_group_any"] = np.where(
        (b > 0) & (b >= d), "Brain",
        np.where(d > 0, "BloodImmune", "Other")
    )
# ---------
    # Expression evidence (brain vs blood): use max across gene columns we already annotated in Step 10
    # ---------
    expr_brain_cols = [
        "gtex_eqtl_best_gene__expr_brain_max_tpm",
        "gtex_sqtl_best_gene__expr_brain_max_tpm",
        "nearest_gene__expr_brain_max_tpm",
        "within_nearest_gene__expr_brain_max_tpm",
    ]
    expr_blood_cols = [
        "gtex_eqtl_best_gene__expr_blood_max_tpm",
        "gtex_sqtl_best_gene__expr_blood_max_tpm",
        "nearest_gene__expr_blood_max_tpm",
        "within_nearest_gene__expr_blood_max_tpm",
    ]
    v["expr_brain_max_tpm_any"] = max_across_cols(v, expr_brain_cols)
    v["expr_blood_max_tpm_any"] = max_across_cols(v, expr_blood_cols)

    # Normalize expression evidence 0..1 (cap at TPM 10)
    v["expr_brain_score"] = (pd.to_numeric(v["expr_brain_max_tpm_any"], errors="coerce") / 10.0).clip(0, 1)
    v["expr_blood_score"] = (pd.to_numeric(v["expr_blood_max_tpm_any"], errors="coerce") / 10.0).clip(0, 1)

    # ---------
    # Overall scoring blocks
    # ---------
    v["gwas_log10p"] = safe_log10p(v["p"]) if "p" in v.columns else np.nan
    v["gwas_score"] = (v["gwas_log10p"] / 20.0).clip(0, 1)

    v["alphagenome_score"] = cap01(v["alphagenome_pct_max"]) if "alphagenome_pct_max" in v.columns else np.nan
    v["ccre_score"] = v.get("ccre_overlap", pd.Series(np.nan, index=v.index)).map(nonempty_ccre)

    # GTEx overall score: combine brain + blood support as max
    v["gtex_score_any"] = np.maximum(v["gtex_brain_score"], v["gtex_blood_score"])

    # SuSiE overall score: max_pip_any
    v["susie_score_any"] = cap01(v["susie_max_pip_any"])

    # Expression overall score: max of brain vs blood
    v["expr_score_any"] = np.maximum(v["expr_brain_score"], v["expr_blood_score"])

    # Ensure sub-scores used in brain/blood evidence sums are numeric (avoid NaN-propagation)
    for c in ["susie_brain_score","susie_blood_score","expr_brain_score","expr_blood_score","gtex_brain_score","gtex_blood_score"]:
        if c in v.columns:
            v[c] = pd.to_numeric(v[c], errors="coerce").fillna(0.0)

    # ---------------------------------------------------------------
    # FIX (revision): the fillna above covers ONLY the brain/blood sub-scores.
    # gwas_score and alphagenome_score were never filled, so any variant missing
    # an AlphaGenome score produced final_score = NaN and was silently dropped
    # from every sort_values()-based top list.
    #
    # RA0 measured the impact on this dataset: 22,928 of 66,726 rows (34%) had
    # final_score = NaN, driven by susie_score_any (25.7% missing) and
    # expr_score_any (10.4% missing). Those variants were absent from every top
    # list. Missing SuSiE PIP means the variant is in no credible set -- evidence
    # of absence, which scores 0 -- so excluding them systematically favoured
    # variants in annotation-dense regions. That is the bias Reviewer #2
    # hypothesised in Comment 4.
    #
    # DECISION: missing components are treated as zero evidence. This must be
    # stated explicitly in Methods. run_RA0b_missingness_impact.py provides the
    # accompanying sensitivity analysis (published vs corrected scores) for the
    # supplement -- RUN IT BEFORE RE-RUNNING THIS SCRIPT, because this script
    # overwrites FTD_all_variants_FINAL.tsv.gz.
    # ---------------------------------------------------------------
    FILL_MISSING_OVERALL_COMPONENTS = True

    _overall_components = ["gwas_score", "alphagenome_score", "susie_score_any",
                           "gtex_score_any", "ccre_score", "expr_score_any"]
    _audit = {c: int(pd.to_numeric(v[c], errors="coerce").isna().sum())
              for c in _overall_components if c in v.columns}
    _n_affected = int(pd.DataFrame(
        {c: pd.to_numeric(v[c], errors="coerce").isna()
         for c in _overall_components if c in v.columns}
    ).any(axis=1).sum())
    print("[audit] missing values per overall component:", _audit)
    print(f"[audit] variants with >=1 missing component: {_n_affected:,} of {len(v):,}")
    if _n_affected and not FILL_MISSING_OVERALL_COMPONENTS:
        print("[audit] WARNING: these variants will have final_score = NaN and will "
              "be absent from all top lists. Document this in Methods or set "
              "FILL_MISSING_OVERALL_COMPONENTS = True.")

    if FILL_MISSING_OVERALL_COMPONENTS:
        for c in _overall_components:
            if c in v.columns:
                v[c] = pd.to_numeric(v[c], errors="coerce").fillna(0.0)

    # Final overall score
    v["final_score"] = (
        W_GWAS * v["gwas_score"]
        + W_AG   * v["alphagenome_score"]
        + W_SUSIE* v["susie_score_any"]
        + W_GTEX * v["gtex_score_any"]
        + W_CCRE * v["ccre_score"]
        + W_EXPR * v["expr_score_any"]
    )

    # Brain-focused and blood-focused presentation scores
    v["brain_evidence_score"] = (
        WB_SUSIE * v["susie_brain_score"]
        + WB_GTEX  * v["gtex_brain_score"]
        + WB_EXPR  * v["expr_brain_score"]
    )

    v["blood_evidence_score"] = (
        WB_SUSIE * v["susie_blood_score"]
        + WB_GTEX  * v["gtex_blood_score"]
        + WB_EXPR  * v["expr_blood_score"]
    )

    # ---------
    # Ranks
    # ---------
    if "locus_id" in v.columns:
        v["final_rank_within_locus"] = v.groupby("locus_id")["final_score"].rank(ascending=False, method="min")
        v["brain_rank_within_locus"] = v.groupby("locus_id")["brain_evidence_score"].rank(ascending=False, method="min")
        v["blood_rank_within_locus"] = v.groupby("locus_id")["blood_evidence_score"].rank(ascending=False, method="min")

    v["final_rank_global"] = v["final_score"].rank(ascending=False, method="min")
    v["brain_rank_global"] = v["brain_evidence_score"].rank(ascending=False, method="min")
    v["blood_rank_global"] = v["blood_evidence_score"].rank(ascending=False, method="min")

    # ---------
    # Write full final table
    # ---------
    out_full = outdir / "FTD_all_variants_FINAL.tsv.gz"
    v.to_csv(out_full, sep="\t", index=False, compression="gzip")
    print("Wrote:", out_full)

    # ---------
    # Top lists
    # ---------
    if "locus_id" in v.columns:
        top_overall = v.sort_values(["locus_id", "final_rank_within_locus"]).groupby("locus_id").head(TOP_PER_LOCUS)
        top_brain   = v.sort_values(["locus_id", "brain_rank_within_locus"]).groupby("locus_id").head(TOP_PER_LOCUS)
        top_blood   = v.sort_values(["locus_id", "blood_rank_within_locus"]).groupby("locus_id").head(TOP_PER_LOCUS)
    else:
        top_overall = v.sort_values("final_score", ascending=False).head(TOP_PER_LOCUS)
        top_brain   = v.sort_values("brain_evidence_score", ascending=False).head(TOP_PER_LOCUS)
        top_blood   = v.sort_values("blood_evidence_score", ascending=False).head(TOP_PER_LOCUS)

    # FIX (revision): this write was previously indented inside the `else:` branch,
    # so on a normal run (locus_id present) the OVERALL file was never written.
    top_overall.to_csv(outdir / f"FTD_top{TOP_PER_LOCUS}_per_locus_OVERALL.tsv", sep="\t", index=False)
    top_brain.to_csv(outdir / f"FTD_top{TOP_PER_LOCUS}_per_locus_BRAIN.tsv", sep="\t", index=False)
    top_blood.to_csv(outdir / f"FTD_top{TOP_PER_LOCUS}_per_locus_BLOOD.tsv", sep="\t", index=False)
    print("Wrote top-per-locus files")

    glob_overall = v.sort_values("final_score", ascending=False).head(TOP_GLOBAL)
    glob_brain   = v.sort_values("brain_evidence_score", ascending=False).head(TOP_GLOBAL)
    glob_blood   = v.sort_values("blood_evidence_score", ascending=False).head(TOP_GLOBAL)

    glob_overall.to_csv(outdir / f"FTD_top{TOP_GLOBAL}_global_OVERALL.tsv", sep="\t", index=False)
    glob_brain.to_csv(outdir / f"FTD_top{TOP_GLOBAL}_global_BRAIN.tsv", sep="\t", index=False)
    glob_blood.to_csv(outdir / f"FTD_top{TOP_GLOBAL}_global_BLOOD.tsv", sep="\t", index=False)
    print("Wrote top-global files")

    # ---------
    # Locus summary (one row per locus)
    # ---------
    if "locus_id" in v.columns:
        rows = []
        for lid, g in v.groupby("locus_id"):
            g2 = g.sort_values("final_score", ascending=False)
            best = g2.iloc[0]
            # lead by GWAS p
            gp = g.copy()
            gp["p_num"] = pd.to_numeric(gp["p"], errors="coerce")
            gp = gp.sort_values("p_num", ascending=True)
            lead = gp.iloc[0]

            rows.append({
                "locus_id": lid,
                "n_variants": int(len(g)),
                "lead_variant_key_by_p": lead.get("variant_key", ""),
                "lead_p": lead.get("p", np.nan),
                "top_variant_key_by_final": best.get("variant_key", ""),
                "top_final_score": best.get("final_score", np.nan),
                "top_gwas_log10p": best.get("gwas_log10p", np.nan),
                "top_alphagenome_pct": best.get("alphagenome_pct_max", np.nan),
                "top_ccre_overlap": best.get("ccre_overlap", ""),
                "top_gtex_eqtl_gene": best.get("gtex_eqtl_best_gene", ""),
                "top_gtex_sqtl_gene": best.get("gtex_sqtl_best_gene", ""),
                "top_susie_eqtl_pip": best.get("susie_eqtl_max_pip", np.nan),
                "top_susie_sqtl_pip": best.get("susie_sqtl_max_pip", np.nan),
                "top_susie_best_group": best.get("susie_best_group_any", ""),
                "top_susie_brain_pip": best.get("susie_brain_max_pip_any", np.nan),
                "top_susie_blood_pip": best.get("susie_blood_max_pip_any", np.nan),
                "expr_brain_max_tpm_any": best.get("expr_brain_max_tpm_any", np.nan),
                "expr_blood_max_tpm_any": best.get("expr_blood_max_tpm_any", np.nan),
                "gtex_sigpair_hit_brain_count": best.get("gtex_sigpair_hit_brain_count", 0),
                "gtex_sigpair_hit_blood_count": best.get("gtex_sigpair_hit_blood_count", 0),
            })

        locus_sum = pd.DataFrame(rows).sort_values("top_final_score", ascending=False)
        out_locus = outdir / "FTD_locus_summary_FINAL.tsv"
        locus_sum.to_csv(out_locus, sep="\t", index=False)
        print("Wrote:", out_locus)

    print("\nDone Step 12 (final prioritization).")


if __name__ == "__main__":
    main()