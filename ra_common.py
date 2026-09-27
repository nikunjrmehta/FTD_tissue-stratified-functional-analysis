"""
ra_common.py — shared helpers for the Communications Biology revision analyses (RA0-RA9).

Spyder-friendly: no argparse, paths relative to repo root (set CWD to the repo root
before running, same as the run_step*.py scripts).
"""

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------
# Paths
# ---------------------------------------------------------------
FINAL_TSV = "work/step12_final_ftd/FTD_all_variants_FINAL.tsv.gz"
LOCUS_SUMMARY = "work/step12_final_ftd/FTD_locus_summary_FINAL.tsv"

# ---------------------------------------------------------------
# The six component scores of final_score, and the current weights.
# Mirrors run_step12_finalize_ftd_spyder.py lines 49-59.
# ---------------------------------------------------------------
COMPONENTS = [
    "gwas_score",
    "alphagenome_score",
    "susie_score_any",
    "gtex_score_any",
    "ccre_score",
    "expr_score_any",
]

CURRENT_WEIGHTS = {
    "gwas_score": 0.25,
    "alphagenome_score": 0.25,
    "susie_score_any": 0.15,
    "gtex_score_any": 0.15,
    "ccre_score": 0.10,
    "expr_score_any": 0.10,
}

# Plain-language labels, reused in figures and in the rebuilt Table 1
COMPONENT_LABELS = {
    "gwas_score": "GWAS association",
    "alphagenome_score": "AlphaGenome",
    "susie_score_any": "SuSiE fine-mapping",
    "gtex_score_any": "GTEx QTL",
    "ccre_score": "ENCODE cCRE",
    "expr_score_any": "Gene expression",
}

# Underlying data source for each component -> answers Reviewer #2 Comment 2
# ("the final score will likely assign more weight to GTEx-based scores").
COMPONENT_SOURCE = {
    "gwas_score": "GWAS (Manzoni)",
    "alphagenome_score": "Sequence model (AlphaGenome)",
    "susie_score_any": "GTEx",
    "gtex_score_any": "GTEx",
    "ccre_score": "ENCODE",
    "expr_score_any": "GTEx",
}

# ---------------------------------------------------------------
# CANONICAL SIGNAL ORDER — decided and fixed here so text, tables and figures
# cannot drift apart (Reviewer #3, minor 2: "Signal numbering and ordering should
# be harmonized across the text, figures, legends, and tables").
# Ordered by chromosome then position, which is the convention a reader expects
# and which is stable regardless of how the scores change.
# ---------------------------------------------------------------
SIGNAL_ORDER = {
    "chr17:45680084:C:A": 1,
    "chr17:46751565:G:A": 2,
    "chr19:44908684:T:C": 3,
}
SIGNAL_LABEL = {
    "chr17:45680084:C:A": "Signal 1 (17q21.31, chr17:45,680,084)",
    "chr17:46751565:G:A": "Signal 2 (17q21.31, chr17:46,751,565)",
    "chr19:44908684:T:C": "Signal 3 (APOE, chr19:44,908,684)",
}


def order_signals(df, col="signal_id"):
    """Add signal_number and signal_label, sorted canonically."""
    df = df.copy()
    df["signal_number"] = df[col].map(SIGNAL_ORDER)
    df["signal_label"] = df[col].map(SIGNAL_LABEL)
    return df.sort_values(["signal_number", col], na_position="last")


# Tissue panel (mirrors steps 8-12)
BLOOD_IMMUNE_TISSUES = [
    "Whole_Blood",
    "Spleen",
    "Cells_EBV-transformed_lymphocytes",
    "Cells_Cultured_fibroblasts",
]
N_BRAIN_TISSUES = 13
N_BLOOD_TISSUES = 4


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def load_final(dedupe=True, verbose=True):
    """
    Load the step-12 master table.

    IMPORTANT: locus windows are +/-500 kb and overlap heavily, so a variant that
    falls inside several windows appears as several ROWS. Any statistic computed
    over rows is pseudo-replicated. `dedupe=True` returns one row per unique
    variant_key, keeping the row with the highest final_score.

    Returns (df, info) where info records both counts so every downstream script
    can report N honestly.
    """
    df = pd.read_csv(FINAL_TSV, sep="\t", low_memory=False)

    n_rows = len(df)
    n_unique = df["variant_key"].nunique()
    info = {
        "n_rows_variant_locus_pairs": n_rows,
        "n_unique_variants": n_unique,
        "rows_are_duplicated": bool(n_rows != n_unique),
    }

    if verbose:
        print(f"[load_final] rows (variant-locus pairs): {n_rows:,}")
        print(f"[load_final] unique variant_key        : {n_unique:,}")
        if n_rows != n_unique:
            print(
                f"[load_final] WARNING: {n_rows - n_unique:,} duplicate rows from "
                "overlapping locus windows."
            )

    if dedupe and n_rows != n_unique:
        df = (
            df.sort_values("final_score", ascending=False)
              .drop_duplicates(subset="variant_key", keep="first")
              .reset_index(drop=True)
        )
        if verbose:
            print(f"[load_final] deduplicated to {len(df):,} unique variants.")

    return df, info


def add_signal_id(df):
    """
    Attach signal_id by mapping locus_id -> lead_variant_key_by_p.
    The 25 locus windows collapse onto 3 independent signals.
    """
    loci = pd.read_csv(LOCUS_SUMMARY, sep="\t")
    mapping = dict(zip(loci["locus_id"], loci["lead_variant_key_by_p"]))
    df = df.copy()
    df["signal_id"] = df["locus_id"].map(mapping)
    return df


def component_matrix(df, dropna=True):
    """Numeric matrix of the six component scores."""
    m = df[COMPONENTS].apply(pd.to_numeric, errors="coerce")
    if dropna:
        m = m.dropna()
    return m


def weighted_score(df, weights):
    """
    Recompute a composite score under an arbitrary weight dict.
    Missing component values are treated as 0, matching run_step12 behaviour
    for the brain/blood sub-scores.
    """
    out = pd.Series(0.0, index=df.index)
    for col, w in weights.items():
        if w == 0:
            continue
        out = out + w * pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return out


def renormalise(weights):
    """Rescale a weight dict to sum to 1."""
    total = float(sum(weights.values()))
    if total == 0:
        raise ValueError("weights sum to zero")
    return {k: v / total for k, v in weights.items()}
