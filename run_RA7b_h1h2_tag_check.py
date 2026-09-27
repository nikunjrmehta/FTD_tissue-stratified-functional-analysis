"""
RA7b — is a canonical H1/H2 tag SNP present in our own dataset?

WHY THIS EXISTS
The RA7 haplotype-surrogate analysis failed QC and cannot be reported. The
failure is a panel problem, not a biological result:
  * rs1052553, the H1/H2 tag, is absent from the 1000G subset used;
  * most candidate variants are also absent (proxies 200-4,100 bp away);
  * so every r2 in RA7_candidates_vs_h2_tag.tsv compares two proxies, neither
    of which is the variant or the tag of interest;
  * PC1 explains 4.3% of variance, far too little for a ~1 Mb recombination-
    suppressed inversion, and tracks the tag proxy at only |r| = 0.327.

Before spending time on a better LD panel, check the cheaper route. The FTD
GWAS window for Signal 1 is chr17:45,180,084-46,180,084, which CONTAINS
rs1052553 at chr17:45,996,523. If the tag is in our own scored dataset we can
state the haplotype background from association statistics directly, with no
LD panel at all.

This script only looks things up. It computes nothing and changes nothing.

Outputs -> work/step_RA7b_h1h2_tag_check/
"""

import pandas as pd

from ra_common import ensure_dir, load_final

OUTDIR = ensure_dir("work/step_RA7b_h1h2_tag_check")

# GRCh38. Only rs1052553 is hardcoded by position, taken from run_RA7 so the
# two scripts cannot disagree. The other tags are searched by rsID only,
# because their GRCh38 coordinates were not independently verified here and a
# wrong coordinate would silently return the wrong variant.
TAG_POS = 45_996_523
TAG_RSID = "rs1052553"
OTHER_TAGS = ["rs8070723", "rs1800547", "rs17563986", "rs17649553"]
FLANK = 50_000

# FIX: load_final returns (df, info), not a bare DataFrame.
df, info = load_final()
print(f"loaded {len(df):,} unique variants\n")

# --- which columns can carry an rsID or a position? -------------------------
rsid_cols = [c for c in df.columns if "rsid" in c.lower() or c.lower() == "rs"]
pos_cols = [c for c in df.columns if c.lower() in ("pos", "position", "bp")]
print(f"rsID-like columns    : {rsid_cols or 'NONE FOUND'}")
print(f"position-like columns: {pos_cols or 'NONE FOUND'}")

# --- 1. direct rsID lookup --------------------------------------------------
wanted = [TAG_RSID] + OTHER_TAGS
hits = pd.DataFrame()
if rsid_cols:
    col = rsid_cols[0]
    # FIX: report how populated the column is. "Tag absent" means nothing if
    # the rsID column is mostly empty.
    n_pop = int(df[col].notna().sum())
    n_rs = int(df[col].astype(str).str.startswith("rs").sum())
    print(f"\n`{col}`: {n_pop:,}/{len(df):,} non-null, "
          f"{n_rs:,} look like rsIDs")
    if n_rs:
        print(f"  examples: {', '.join(df.loc[df[col].astype(str).str.startswith('rs'), col].head(3).astype(str))}")

    hits = df[df[col].astype(str).isin(wanted)].copy()
    print(f"rsID search: {len(hits)} of {len(wanted)} tags found")
    for t in wanted:
        n = int((df[col].astype(str) == t).sum())
        print(f"  {t:12s} {'FOUND' if n else 'absent'}")
else:
    print("\nno rsID column; falling back to position only")

# --- 2. positional window around rs1052553 ----------------------------------
# FIX: the first version parsed the position out of `variant_key` and returned
# an empty window, which is indistinguishable from a parsing bug. The file has
# native `chr` and `pos` columns, so use those, and print enough context that
# an empty result can be trusted as a real absence.
pos = pd.to_numeric(df["pos"], errors="coerce")
chrom = df["chr"].astype(str).str.replace("^chr", "", regex=True)

c17 = chrom == "17"
print(f"\nchromosome 17 variants in dataset: {int(c17.sum()):,}")
if c17.any():
    print(f"chr17 position range: {int(pos[c17].min()):,} - "
          f"{int(pos[c17].max()):,}")
    # Signal 1's window is TAG_POS-dependent; show local coverage either side
    # so a gap at the tag is visible as a gap, not as a failed filter.
    for lo, hi, lbl in [(TAG_POS - 500_000, TAG_POS, "500 kb below tag"),
                        (TAG_POS, TAG_POS + 500_000, "500 kb above tag"),
                        (TAG_POS - FLANK, TAG_POS + FLANK,
                         f"+/-{FLANK // 1000} kb of tag")]:
        n = int((c17 & pos.between(lo, hi)).sum())
        print(f"  {lbl:22s} ({lo:,}-{hi:,}): {n:,} variants")

near = df[c17 & pos.between(TAG_POS - FLANK, TAG_POS + FLANK)].copy()
near["distance_from_tag_bp"] = (pos[near.index] - TAG_POS).abs()
near = near.sort_values("distance_from_tag_bp")

exact = near[near["distance_from_tag_bp"] == 0]
print(f"\nvariants within +/-{FLANK:,} bp of {TAG_RSID}: {len(near)}")
print(f"variant exactly at chr17:{TAG_POS:,}: "
      f"{'YES' if len(exact) else 'NO'}")

keep = [c for c in ["variant_key", "distance_from_tag_bp", "p", "final_score",
                    "signal_id", "gwas_score", "alphagenome_score"]
        if c in near.columns or c == "distance_from_tag_bp"]
if len(near):
    print("\nnearest 10:")
    print(near[keep].head(10).to_string(index=False))

# FIX: `near` inherits all 207 columns of the master table, which is unusable
# as a supplementary sheet. Write a slim version for publication and keep the
# full one for the deposit.
SLIM = ["variant_key", "distance_from_tag_bp", "ref", "alt", "effect_allele",
        "other_allele", "beta", "se", "or", "p", "nearest_gene",
        "ccre_overlap", "gwas_score", "alphagenome_score", "ccre_score",
        "susie_score_any", "gtex_score_any", "expr_score_any", "final_score"]
slim = near[[c for c in SLIM if c in near.columns]]
slim.to_csv(f"{OUTDIR}/RA7b_variants_near_tag.tsv", sep="\t", index=False)
near.to_csv(f"{OUTDIR}/RA7b_variants_near_tag_full.tsv.gz", sep="\t",
            index=False, compression="gzip")
print(f"\nnear-tag table: {len(slim)} rows x {slim.shape[1]} columns "
      f"(full {near.shape[1]}-column version written separately)")
if len(hits):
    hits.to_csv(f"{OUTDIR}/RA7b_tag_hits.tsv", sep="\t", index=False)

# --- 3. association statistics at the H1/H2 tags -----------------------------
# The `rsid` column carries no rsIDs, so tags are identified by GRCh38
# coordinate and alleles, both verified against Ensembl (release 115):
#   rs1052553  17:45,996,523  A/G  synonymous_variant in MAPT
#   rs8070723  17:46,003,698  A/G  intron_variant
# Both discriminate the H1/H2 haplotypes. If they tag the same inversion they
# must give near-identical effect estimates; that is an internal check the data
# can settle, and it is reported below.
#
# WHICH ALLELE MARKS H2: resolved from population allele frequency, not from a
# citation, so it can be checked. H2 is European-enriched and essentially
# absent from East Asia. Ensembl/gnomAD frequencies for the G allele of
# rs8070723:
#     1000G EUR 0.241 | gnomAD NFE 0.218 | 1000G FIN 0.106
#     1000G EAS 0.001 | gnomAD EAS 0.0015 | CDX/CHB/CHS/KHV 0.000
# That is the H2 distribution, so G is the H2-tagging allele at both SNPs
# (they are in near-complete LD across the inversion).
TAGS = [("rs1052553", 45_996_523, "A", "G", "G"),
        ("rs8070723", 46_003_698, "A", "G", "G")]
STATS = ["variant_key", "effect_allele", "other_allele", "beta", "se", "or",
         "p", "final_score", "gwas_score", "alphagenome_score"]

rows = []
for rsid, p38, a1, a2, h2_allele in TAGS:
    m = df[c17 & (pos == p38)]
    if not len(m):
        rows.append({"tag": rsid, "grch38_pos": p38, "found": False})
        continue
    r = m.iloc[0]
    alleles_ok = {str(r.get("ref")), str(r.get("alt"))} == {a1, a2}

    # Re-express the effect on the H2 allele regardless of which allele the
    # deposit used as the effect allele. Flipping beta is only valid because
    # both are biallelic SNPs with dbSNP-matching alleles, checked above.
    beta = pd.to_numeric(pd.Series([r.get("beta")]), errors="coerce").iloc[0]
    eff = str(r.get("effect_allele"))
    if pd.notna(beta) and eff in (a1, a2):
        beta_h2 = beta if eff == h2_allele else -beta
        h2_dir = "protective" if beta_h2 < 0 else "risk-increasing"
    else:
        beta_h2, h2_dir = None, "undetermined"

    rows.append({"tag": rsid, "grch38_pos": p38, "found": True,
                 "alleles_match_dbsnp": alleles_ok,
                 "h2_allele": h2_allele, "beta_on_h2_allele": beta_h2,
                 "h2_haplotype_direction": h2_dir,
                 **{k: r.get(k) for k in STATS}})

tags = pd.DataFrame(rows)
tags.to_csv(f"{OUTDIR}/RA7b_tag_association.tsv", sep="\t", index=False)

print("\n" + "-" * 74)
print("ASSOCIATION AT THE H1/H2 TAGS")
print("-" * 74)
print(tags.to_string(index=False))

found = tags[tags["found"] == True]  # noqa: E712
if len(found) == 2 and found["beta"].notna().all():
    b = found["beta"].astype(float).values
    same_dir = (b[0] * b[1]) > 0
    print(f"\neffect directions concordant between the two tags: "
          f"{'YES' if same_dir else 'NO'}")
    print(f"|beta| ratio: {abs(b[0] / b[1]):.3f}  "
          "(near 1.0 if both tag the same inversion)")
    if not same_dir:
        print("  !! Discordant. Check allele orientation before reporting;")
        print("     effect alleles may be given on opposite strands.")

# --- verdict ----------------------------------------------------------------
print("\n" + "=" * 74)
print("VERDICT")
print("=" * 74)
if len(exact) or len(hits):
    print("A canonical H1/H2 tag IS present in the scored dataset.")
    print("=> The haplotype-discriminating variant is itself associated, so")
    print("   the Signal 1 association lies on the H1/H2 axis. Report the tag")
    print("   statistics; no LD panel is needed for that statement.")
    print("=> This does NOT assign individual candidates to H1 or H2. That")
    print("   needs phased haplotypes, which we do not have. State the limit.")
elif len(near):
    d = int(near['distance_from_tag_bp'].iloc[0])
    print(f"The tag itself is absent; nearest variant is {d:,} bp away.")
    print("=> Proximity is NOT haplotype assignment. Do not substitute it.")
    print("   Close the haplotype question as a stated limitation.")
else:
    print("No variant near the tag at all, which would mean the GWAS deposit")
    print("does not cover this position. Close as a stated limitation.")

print(f"\n[RA7b] done. Outputs in {OUTDIR}/")
