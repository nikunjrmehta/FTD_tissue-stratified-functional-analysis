"""
RA8b — is the GWAS risk allele associated with higher or lower expression?

Editor's priority (2):
    "Extend the downstream analyses to further comment on the biological relevance
     of the signals gleaned from this analyses, per Reviewer #3."

A ranking is not biology. The cheapest genuinely biological extension available
from data already on disk is effect-direction concordance: for each prioritised
variant, does the allele that raises FTD risk also raise or lower expression of
the nominated gene? That converts a ranked list into a DIRECTIONAL, testable
hypothesis per gene -- "the risk allele is associated with reduced expression of
X in tissue Y" -- which is what an experimental follow-up would actually test.

ALLELE ALIGNMENT IS THE HARD PART, AND IT IS NOT GLOSSED OVER
------------------------------------------------------------
The GWAS effect is reported for `effect_allele`. The GTEx allelic fold change
(`susie_*_best_afc`) is reported with respect to the ALT allele. Combining them
requires knowing that both refer to the same allele, and RA10 showed that ref/alt
in the variant key came from the GWAS marker ID rather than the reference genome
-- 278 variants (4.1%) are reversed relative to GRCh38.

So this script:
  * uses the RA10 audit to establish the true reference allele,
  * aligns the GWAS effect direction onto the ALT allele explicitly,
  * reports how many variants could be aligned with confidence, and
  * EXCLUDES anything ambiguous rather than guessing.

A concordance rate computed on silently misaligned alleles is worse than no
concordance rate at all.

Outputs -> work/step_RA8b_effect_direction/
"""

import os

import numpy as np
import pandas as pd
from scipy import stats

from ra_common import add_signal_id, ensure_dir, load_final

OUTDIR = ensure_dir("work/step_RA8b_effect_direction")
AUDIT = "work/step_RA10_allele_audit/RA10_allele_check_results.tsv"

df, info = load_final(dedupe=True)
df = add_signal_id(df)
print(f"\n[RA8b] {len(df):,} unique variants\n")

# ---------------------------------------------------------------
# 1. Establish the true reference allele
# ---------------------------------------------------------------
if os.path.exists(AUDIT):
    audit = pd.read_csv(AUDIT, sep="\t")
    df = df.merge(audit[["variant_key", "allele_status", "genome_base"]],
                  on="variant_key", how="left")
else:
    raise SystemExit(f"{AUDIT} not found -- run the RA10 allele audit first. "
                     "Without it, allele alignment cannot be verified.")

# true ALT = the allele that is NOT the genome base
df["true_ref"] = np.where(df["allele_status"] == "swapped", df["alt"], df["ref"])
df["true_alt"] = np.where(df["allele_status"] == "swapped", df["ref"], df["alt"])
print("allele status:")
print(df["allele_status"].value_counts().to_string())

# ---------------------------------------------------------------
# 2. Align the GWAS effect onto the ALT allele
# ---------------------------------------------------------------
if "effect_allele" not in df.columns:
    raise SystemExit("no effect_allele column -- cannot align directions")

ea = df["effect_allele"].astype(str).str.upper()
df["gwas_beta"] = pd.to_numeric(df.get("beta"), errors="coerce")
if df["gwas_beta"].isna().all() and "or" in df.columns:
    df["gwas_beta"] = np.log(pd.to_numeric(df["or"], errors="coerce"))

df["effect_matches_alt"] = ea == df["true_alt"].astype(str).str.upper()
df["effect_matches_ref"] = ea == df["true_ref"].astype(str).str.upper()
df["alignable"] = df["effect_matches_alt"] ^ df["effect_matches_ref"]

# beta expressed per ALT allele
df["beta_wrt_alt"] = np.where(
    df["effect_matches_alt"], df["gwas_beta"],
    np.where(df["effect_matches_ref"], -df["gwas_beta"], np.nan))

print()
print("=" * 74)
print("ALLELE ALIGNMENT")
print("=" * 74)
print(f"effect allele == ALT      : {int(df['effect_matches_alt'].sum()):,}")
print(f"effect allele == REF      : {int(df['effect_matches_ref'].sum()):,}")
print(f"unambiguously alignable   : {int(df['alignable'].sum()):,} "
      f"({100.0 * df['alignable'].mean():.1f}%)")
print(f"NOT alignable (excluded)  : {int((~df['alignable']).sum()):,}")
if df["alignable"].mean() < 0.9:
    print("  !! A large unalignable fraction means the effect allele does not")
    print("     correspond to either recorded allele for many variants. Resolve")
    print("     that before interpreting concordance.")

# ---------------------------------------------------------------
# 3. Concordance with eQTL direction
# ---------------------------------------------------------------
AFC_SETS = [
    ("eQTL", "susie_eqtl_best_afc", "susie_eqtl_best_gene", "susie_eqtl_best_tissue"),
    ("eQTL brain", "susie_eqtl_brain_best_afc", "susie_eqtl_brain_best_gene",
     "susie_eqtl_brain_best_tissue"),
    ("eQTL blood", "susie_eqtl_blood_best_afc", "susie_eqtl_blood_best_gene",
     "susie_eqtl_blood_best_tissue"),
    ("sQTL", "susie_sqtl_best_afc", "susie_sqtl_best_gene", "susie_sqtl_best_tissue"),
]

print()
print("=" * 74)
print("EFFECT-DIRECTION CONCORDANCE")
print("=" * 74)

rows = []
detail = []
for label, afc_col, gene_col, tis_col in AFC_SETS:
    if afc_col not in df.columns:
        continue
    afc = pd.to_numeric(df[afc_col], errors="coerce")
    ok = df["alignable"] & afc.notna() & df["beta_wrt_alt"].notna() & (afc != 0)
    if ok.sum() < 10:
        print(f"{label}: too few aligned variants ({int(ok.sum())})")
        continue
    sub = df[ok].copy()
    sub["afc"] = afc[ok]
    # concordant = risk-increasing allele also increases expression
    sub["concordant"] = np.sign(sub["beta_wrt_alt"]) == np.sign(sub["afc"])
    n_con = int(sub["concordant"].sum())
    binom = stats.binomtest(n_con, len(sub), 0.5)
    rows.append({
        "qtl_layer": label,
        "n_aligned_variants": len(sub),
        "n_concordant": n_con,
        "pct_concordant": round(100.0 * n_con / len(sub), 1),
        "binomial_p_vs_50pct": binom.pvalue,
    })
    d = sub[["variant_key", "signal_id", "p", "beta_wrt_alt", "afc",
             "concordant"]].copy()
    d["qtl_layer"] = label
    d["gene"] = sub[gene_col] if gene_col in sub.columns else None
    d["tissue"] = sub[tis_col] if tis_col in sub.columns else None
    d["direction_statement"] = np.where(
        sub["beta_wrt_alt"] > 0,
        np.where(sub["afc"] > 0,
                 "risk allele increases expression",
                 "risk allele decreases expression"),
        np.where(sub["afc"] > 0,
                 "risk allele decreases expression",
                 "risk allele increases expression"))
    detail.append(d)

conc = pd.DataFrame(rows)
conc.to_csv(f"{OUTDIR}/RA8b_concordance_summary.tsv", sep="\t", index=False)
print(conc.to_string(index=False) if len(conc) else "  (nothing computable)")
print()
print("  >> ~50% concordance is the null and is the expected result genome-wide.")
print("     The value here is not the aggregate rate but the PER-GENE directional")
print("     statement, which is what an experiment would test.")

if detail:
    det = pd.concat(detail, ignore_index=True)
    det.to_csv(f"{OUTDIR}/RA8b_variant_level_directions.tsv.gz", sep="\t",
               index=False, compression="gzip")

# ---------------------------------------------------------------
# 4. Directional hypotheses for the reported candidates
# ---------------------------------------------------------------
print()
print("=" * 74)
print("DIRECTIONAL HYPOTHESES FOR THE REPORTED CANDIDATES")
print("=" * 74)
try:
    cons = pd.read_csv("work/step_RA0e_consensus/RA0e_candidate_sets_tier1.tsv",
                       sep="\t")
    if detail:
        m = cons.merge(det, on=["variant_key", "signal_id"], how="left")
        show = ["signal_id", "variant_key", "qtl_layer", "gene", "tissue",
                "direction_statement"]
        show = [c for c in show if c in m.columns]
        out = m[show].dropna(subset=["direction_statement"])
        print(out.to_string(index=False) if len(out)
              else "  (no aligned QTL evidence for the consensus candidates)")
        m.to_csv(f"{OUTDIR}/RA8b_candidate_directions.tsv", sep="\t", index=False)
        print()
        print("  >> Each row is a testable hypothesis and belongs in the Discussion")
        print("     as the concrete follow-up the framework generates. This is the")
        print("     substance behind 'a foundation for experimental follow-up'.")
except FileNotFoundError:
    print("(run RA0e first)")

print(f"\n[RA8b] done. Outputs in {OUTDIR}/")
