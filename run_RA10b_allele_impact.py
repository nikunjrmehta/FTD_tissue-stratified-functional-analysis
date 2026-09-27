"""
RA10b — what does the ref/alt swap actually affect?

The audit found 278 of 6,787 variants (4.1%) where the recorded ref/alt are
reversed relative to GRCh38, with ZERO positional errors and ZERO strand flips.
So the liftover is sound; the alleles were taken from the GWAS marker ID (A1/A2)
rather than being reference-anchored.

Expected impact is near zero, because:
  * alphagenome_max_abs is a MAX OF ABSOLUTE VALUES. Swapping ref/alt negates the
    predicted effect, and the absolute value undoes that. (Assumption to be TESTED
    by RA10c, not asserted.)
  * steps 8/9/11 already query GTEx in both allele orientations.
  * cCRE overlap is positional; the GWAS p-value and expression are
    orientation-independent.

This script establishes what is at stake before any re-scoring: which variants are
affected, where they rank, and whether any reported candidate is among them.

Outputs -> work/step_RA10_allele_audit/
"""

import numpy as np
import pandas as pd

from ra_common import COMPONENTS, add_signal_id, ensure_dir, load_final

OUTDIR = ensure_dir("work/step_RA10_allele_audit")
AUDIT = f"{OUTDIR}/RA10_allele_check_results.tsv"
CONSENSUS = "work/step_RA0e_consensus/RA0e_candidate_sets_tier1.tsv"

audit = pd.read_csv(AUDIT, sep="\t")
df, info = load_final(dedupe=True)
df = add_signal_id(df)
df = df.merge(audit[["variant_key", "allele_status", "genome_base"]],
              on="variant_key", how="left")
df["is_swapped"] = df["allele_status"].eq("swapped")

print(f"\n[RA10b] {len(df):,} variants; {int(df['is_swapped'].sum()):,} swapped "
      f"({100.0 * df['is_swapped'].mean():.2f}%)\n")

# ---------------------------------------------------------------
# 1. Where are they?
# ---------------------------------------------------------------
print("=" * 74)
print("1. DISTRIBUTION")
print("=" * 74)
by_sig = pd.crosstab(df["signal_id"], df["allele_status"], margins=True)
print(by_sig.to_string())
by_sig.to_csv(f"{OUTDIR}/RA10b_status_by_signal.tsv", sep="\t")

# Are swapped variants unusual in any component score?
rows = []
for c in COMPONENTS + ["final_score"]:
    v = pd.to_numeric(df[c], errors="coerce")
    a, b = v[df["is_swapped"]].dropna(), v[~df["is_swapped"]].dropna()
    if len(a) < 10 or len(b) < 10:
        continue
    from scipy import stats
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    rows.append({
        "score": c,
        "median_swapped": round(float(a.median()), 4),
        "median_correct": round(float(b.median()), 4),
        "auc": round(float(u / (len(a) * len(b))), 4),
        "p": p,
    })
comp = pd.DataFrame(rows)
comp.to_csv(f"{OUTDIR}/RA10b_score_comparison.tsv", sep="\t", index=False)
print("\nAre swapped variants systematically different?")
print(comp.to_string(index=False))
print("\n  >> AUC near 0.50 with a non-significant p means the swap is not")
print("     concentrated in high- or low-scoring variants, i.e. it behaves like")
print("     an orientation convention rather than a scoring error.")

# ---------------------------------------------------------------
# 2. Does it touch anything we report?
# ---------------------------------------------------------------
print()
print("=" * 74)
print("2. ARE ANY REPORTED CANDIDATES AFFECTED?")
print("=" * 74)

df["rank_in_signal"] = df.groupby("signal_id")["final_score"].rank(
    ascending=False, method="min")

top50 = df[df["rank_in_signal"] <= 50]
print(f"swapped variants in the top 50 per signal: "
      f"{int(top50['is_swapped'].sum())} of {len(top50)}")

hit = top50[top50["is_swapped"]]
if len(hit):
    cols = [c for c in ["variant_key", "signal_id", "rank_in_signal", "p",
                        "final_score", "alphagenome_score", "ref", "alt",
                        "genome_base"] if c in hit.columns]
    print("\nAffected top-50 variants:")
    print(hit.sort_values("rank_in_signal")[cols].to_string(index=False))
    hit[cols].to_csv(f"{OUTDIR}/RA10b_affected_top50.tsv", sep="\t", index=False)

try:
    cons = pd.read_csv(CONSENSUS, sep="\t")
    cons = cons.merge(audit[["variant_key", "allele_status"]],
                      on="variant_key", how="left")
    print("\nRA0e consensus candidates (Tier 1) and their allele status:")
    print(cons[["signal_id", "variant_key", "p", "pct_specs_in_top1",
                "allele_status"]].to_string(index=False))
    cons.to_csv(f"{OUTDIR}/RA10b_consensus_allele_status.tsv", sep="\t", index=False)
    n_bad = int(cons["allele_status"].ne("match").sum())
    print(f"\nconsensus candidates affected: {n_bad} of {len(cons)}")
    if n_bad == 0:
        print("  >> No reported candidate is affected. The swap is a documentation")
        print("     and reproducibility issue, not a results issue.")
except FileNotFoundError:
    print("\n(RA0e consensus table not found — run RA0e first for this section.)")

# ---------------------------------------------------------------
# 3. Export the swapped variants for AlphaGenome re-scoring
# ---------------------------------------------------------------
sw = df[df["is_swapped"]].copy()
sw["ref_corrected"] = sw["alt"]
sw["alt_corrected"] = sw["ref"]
cols = [c for c in ["variant_key", "chr", "pos", "ref", "alt",
                    "ref_corrected", "alt_corrected", "locus_id", "signal_id",
                    "alphagenome_max_abs", "alphagenome_pct_max",
                    "alphagenome_score", "final_score", "rank_in_signal"]
        if c in sw.columns]
sw[cols].to_csv(f"{OUTDIR}/RA10b_swapped_for_rescoring.tsv", sep="\t", index=False)

print()
print("=" * 74)
print("3. NEXT STEP")
print("=" * 74)
print(f"exported {len(sw):,} swapped variants with corrected orientation ->")
print(f"  {OUTDIR}/RA10b_swapped_for_rescoring.tsv")
print()
print("Run run_RA10c_rescore_swapped.py to re-score these with AlphaGenome using")
print("the corrected ref/alt and compare alphagenome_max_abs. If the values are")
print("identical, the max-abs summarisation is orientation-invariant and the swap")
print("provably does not affect any result -- which is the statement to put in the")
print("response letter. That costs ~278 API calls, a bounded one-off.")
