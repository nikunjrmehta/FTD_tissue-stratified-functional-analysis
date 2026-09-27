"""
RA7 — the 17q21.31 inversion, LD structure, and the 22/25 window collapse.

Reviewer #3, major 3:
    "The MAPT region is especially challenging because inversion-dependent
     haplotypes and extended LD can complicate LD clumping, QTL interpretation,
     and variant-to-gene assignment. The fact that 22 of 25 locus windows collapse
     into a single chr17 signal likely reflects this structure and deserves more
     discussion. The authors should explicitly address how the inversion haplotype
     may affect candidate prioritization, and whether their conclusions are robust
     to this regional complexity."

This referee is a statistical (neuro)geneticist and knows the region. Two things
follow: the analysis has to be empirical rather than a citation, and the inversion
boundary has to be stated with a source rather than asserted.

CAUTION ON BOUNDARIES
---------------------
Published GRCh38 coordinates for the H1/H2 inversion differ between sources,
mostly in the range chr17:45.5-46.3 Mb. Signal 2 (chr17:46,751,565) lies OUTSIDE
every common definition, so the claim "the inversion explains the collapse" is
only partly true and must not be overstated. This script therefore evaluates
several published boundary definitions and reports which windows fall inside each,
rather than committing to one.

WHAT IT DOES
------------
  1. Which locus windows and consensus candidates fall inside the inversion,
     under each boundary definition.
  2. Empirical LD (plink2 --r2-unphased) between each signal lead, each consensus
     candidate, and all variants across the region -- so the LD block is measured
     rather than assumed.
  3. Whether LD decays at the inversion boundary, which is the actual test of
     whether the inversion is driving the collapse.
  4. Credible-set sizes inside versus outside the inversion, connecting this to
     the RA1 finding (median eQTL credible set = 874 variants) that explains why
     SuSiE PIPs are diluted here.

PREREQUISITE: plink2 (the path below matches run_step3_clump_and_loci_spyder_pchr.py)

Outputs -> work/step_RA7_mapt_inversion/
"""

import os
import subprocess

import numpy as np
import pandas as pd

from ra_common import add_signal_id, ensure_dir, load_final

OUTDIR = ensure_dir("work/step_RA7_mapt_inversion")
PFILE17 = "data/ldref/pgen/1kg_chr17_hg38_chrpos"

# FIX: PLINK2 was hard-coded to C:\tools\plink2\plink2.exe, which no longer
# exists on this machine. The script then took the "not found" branch and
# skipped the entire LD analysis after printing one line -- easy to miss in a
# long log. Now it searches known locations and the PATH, and STOPS LOUDLY if
# it finds nothing rather than silently skipping the analysis.
#
# NOTE for the code deposit: run_step3_clump_and_loci_spyder_pchr.py still has
# the old hard-coded path. Point it at the same resolver, or at minimum document
# that the plink2 location is machine-specific.
import shutil

PLINK2_CANDIDATES = [
    r"C:\Users\nikun\Downloads\plink2_win64_20260808\plink2.exe",
    r"C:\tools\plink2\plink2.exe",
    "plink2",
]


def resolve_plink2():
    for cand in PLINK2_CANDIDATES:
        if os.path.exists(cand):
            return cand
        found = shutil.which(cand)
        if found:
            return found
    return None


PLINK2 = resolve_plink2()
print(f"[RA7] plink2: {PLINK2}")

# ---------------------------------------------------------------
# ANCHORING ON A TAG SNP RATHER THAN ON BOUNDARY COORDINATES
#
# Zody et al. 2008 (Nat Genet, doi:10.1038/ng.193) sequenced the H1 and H2
# haplotypes directly and state that the inversion is ~970 kb with four
# breakpoints, all falling inside LRRC37 core duplications, and that the
# breakpoints CANNOT be precisely delineated because of the sequence identity
# between those duplications. So a hard coordinate boundary is not available
# from the primary literature, and asserting one would be indefensible.
#
# The tractable and more informative anchor is the H1/H2 tagging SNV. LD with
# the tag measures haplotype membership directly, which is what Reviewer #3 is
# actually asking about, and it does not depend on a boundary.
#
# rs1052553  chr17:45,996,523 (GRCh38), A>G; G tags H2
#   Reyes-Perez et al. 2025, doi:10.1101/2025.10.17.25337033
#   "17q21.3 H1/H2 haplotypes were investigated using the H2-tagging SNV
#    rs1052553 (chr17:45996523:A:G)"
#
# Boundary windows are retained only as a SECONDARY sensitivity check, and are
# deliberately labelled as approximate.
#
# IMPORTANT CONTEXT FOR THE DISCUSSION: the inversion is ~900-970 kb but the
# linkage-disequilibrium block it creates is ~1.8 Mb (Reyes-Perez et al. 2025;
# Pedicone et al. 2024, doi:10.1186/s13024-024-00731-x). The two are NOT the
# same span. Signal 2 (chr17:46,751,565) lies outside the inversion on every
# published definition but may still lie inside the LD block -- so "extended LD
# at 17q21.31" and "the inversion" must not be used interchangeably.
# ---------------------------------------------------------------
H2_TAG_SNP = {"rsid": "rs1052553", "chr": "17", "pos": 45_996_523,
              "ref": "A", "alt": "G", "h2_allele": "G",
              "source": "Reyes-Perez et al. 2025, doi:10.1101/2025.10.17.25337033"}

# Approximate windows, secondary only. ~970 kb inversion (Zody 2008) centred on
# the tag, and the ~1.8 Mb LD block.
INVERSION_DEFS = {
    "inversion_970kb_approx": (H2_TAG_SNP["pos"] - 485_000,
                               H2_TAG_SNP["pos"] + 485_000),
    "ld_block_1.8Mb_approx": (H2_TAG_SNP["pos"] - 900_000,
                              H2_TAG_SNP["pos"] + 900_000),
}

df, info = load_final(dedupe=True)
df = add_signal_id(df)
df["pos"] = pd.to_numeric(df["pos"], errors="coerce")
c17 = df[df["chr"].astype(str) == "chr17"].copy()

print(f"\n[RA7] chr17 variants: {len(c17):,}\n")

# ---------------------------------------------------------------
# 1. What falls inside the inversion?
# ---------------------------------------------------------------
print("=" * 74)
print("1. WINDOWS AND CANDIDATES RELATIVE TO THE INVERSION")
print("=" * 74)

loci = pd.read_csv("work/step12_final_ftd/FTD_locus_summary_FINAL.tsv", sep="\t")
loci["lead_pos"] = loci["lead_variant_key_by_p"].str.split(":").str[1].astype(float)
loci17 = loci[loci["lead_variant_key_by_p"].str.startswith("chr17")]

rows = []
for name, (lo, hi) in INVERSION_DEFS.items():
    inside_loci = int(((loci17["lead_pos"] >= lo) & (loci17["lead_pos"] <= hi)).sum())
    inside_vars = int(((c17["pos"] >= lo) & (c17["pos"] <= hi)).sum())
    rows.append({
        "definition": name, "start": lo, "end": hi, "width_kb": (hi - lo) / 1000,
        "n_chr17_locus_windows_inside": inside_loci,
        "n_chr17_locus_windows_total": len(loci17),
        "n_chr17_variants_inside": inside_vars,
        "pct_chr17_variants_inside": round(100.0 * inside_vars / len(c17), 1),
    })
inv = pd.DataFrame(rows)
inv.to_csv(f"{OUTDIR}/RA7_inversion_definitions.tsv", sep="\t", index=False)
print(inv.to_string(index=False))

try:
    cons = pd.read_csv("work/step_RA0e_consensus/RA0e_candidate_sets_tier1.tsv", sep="\t")
    cons = cons[cons["variant_key"].str.startswith("chr17")].copy()
    cons["pos"] = cons["variant_key"].str.split(":").str[1].astype(float)
    for name, (lo, hi) in INVERSION_DEFS.items():
        cons[f"inside__{name}"] = (cons["pos"] >= lo) & (cons["pos"] <= hi)
    print("\nConsensus candidates (chr17) vs inversion:")
    show = ["signal_id", "variant_key", "p", "pct_specs_in_top1"] + \
           [f"inside__{n}" for n in INVERSION_DEFS]
    print(cons[show].to_string(index=False))
    cons.to_csv(f"{OUTDIR}/RA7_candidates_vs_inversion.tsv", sep="\t", index=False)
except FileNotFoundError:
    print("\n(run RA0e first for the consensus-candidate section)")

# ---------------------------------------------------------------
# 2. Empirical LD
# ---------------------------------------------------------------
print()
print("=" * 74)
print("2. EMPIRICAL LD ACROSS THE REGION")
print("=" * 74)

if PLINK2 is None:
    # Loud, not a one-line note buried in the log -- this section is the answer
    # to Reviewer #3 major 3 and must not be skipped by accident.
    print("!" * 74)
    print("plink2 NOT FOUND. The LD analysis -- the primary answer to Reviewer #3")
    print("major 3 -- has been SKIPPED. Searched:")
    for c in PLINK2_CANDIDATES:
        print(f"    {c}")
    print("Add the correct path to PLINK2_CANDIDATES at the top of this script.")
    print("!" * 74)
else:
    focal = []
    for s, grp in c17.groupby("signal_id"):
        lead = s
        top = grp.loc[grp["final_score"].idxmax(), "variant_key"]
        focal += [lead, top]
    # include every consensus candidate as well
    try:
        _c = pd.read_csv("work/step_RA0e_consensus/RA0e_candidate_sets_tier1.tsv",
                         sep="\t")
        focal += list(_c.loc[_c["variant_key"].str.startswith("chr17"), "variant_key"])
    except FileNotFoundError:
        pass
    focal = sorted(set(focal))
    # pvar IDs are chr:pos in this reference panel
    focal_ids = [":".join(v.split(":")[:2]) for v in focal]
    # THE KEY ADDITION: the H1/H2 tagging SNV. r2 with this variant measures
    # inversion-haplotype membership directly, which is what Reviewer #3 asked
    # about, and needs no boundary coordinates.
    tag_id = f"{H2_TAG_SNP['chr']}:{H2_TAG_SNP['pos']}"
    focal_ids = [tag_id] + [f for f in focal_ids if f != tag_id]
    pd.Series(focal_ids).to_csv(f"{OUTDIR}/focal_ids.txt", index=False, header=False)
    print(f"H2 tag SNP  : {H2_TAG_SNP['rsid']} at {tag_id} "
          f"({H2_TAG_SNP['source']})")
    print(f"focal variants: {len(focal_ids)}")

    # FIX: the previous call used --r2-unphased with --ld-snp-list and produced
    # no output. plink2's LD flags and their names vary between builds, and
    # guessing at them wasted a run. Exporting genotypes and computing r2 in
    # numpy depends only on --export A, which is stable across every plink2
    # build, and it gives exact control over what is correlated with what.
    cmd = [
        PLINK2, "--pfile", PFILE17,
        "--chr", "17", "--from-bp", "45000000", "--to-bp", "47500000",
        "--max-alleles", "2",
        "--export", "A",
        "--out", f"{OUTDIR}/RA7_geno",
    ]
    print("\n" + " ".join(cmd) + "\n")
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-1500:])
    raw = f"{OUTDIR}/RA7_geno.raw"
    if r.returncode != 0 or not os.path.exists(raw):
        print("PLINK2 STDERR:\n", r.stderr[-2000:])
        print("\n--export A failed. Check the --pfile path and that plink2 runs.")
    else:
        # .raw layout: FID IID PAT MAT SEX PHENOTYPE then one column per variant,
        # named "<ID>_<counted allele>", coded 0/1/2 with NA for missing.
        print(f"\nreading {raw} ...")
        geno = pd.read_csv(raw, sep=r"\s+")
        meta = [c for c in ["FID", "IID", "PAT", "MAT", "SEX", "PHENOTYPE"]
                if c in geno.columns]
        G = geno.drop(columns=meta)
        ids = [c.rsplit("_", 1)[0] for c in G.columns]
        print(f"  genotypes: {G.shape[0]} samples x {G.shape[1]} variants")

        X = G.to_numpy(dtype=float)
        # mean-impute missing calls so a few NAs do not drop a whole variant
        col_mean = np.nanmean(X, axis=0)
        idx = np.where(np.isnan(X))
        X[idx] = np.take(col_mean, idx[1])

        sd = X.std(axis=0, ddof=0)
        keep = sd > 0                       # monomorphic variants have no LD
        X, sd = X[:, keep], sd[keep]
        ids = [i for i, k in zip(ids, keep) if k]
        pos_of = {i: int(i.split(":")[1]) for i in ids if ":" in i}
        Z = (X - X.mean(axis=0)) / sd
        n = Z.shape[0]

        tag = f"{H2_TAG_SNP['chr']}:{H2_TAG_SNP['pos']}"
        print()
        print("=" * 74)
        print(f"HAPLOTYPE MEMBERSHIP: r2 WITH {H2_TAG_SNP['rsid']} ({tag})")
        print("=" * 74)

        # FIX: rs1052553 is ABSENT from this 1000G subset -- the panel jumps from
        # 17:45,995,653 to 17:45,997,220. Rather than abandon the analysis, do two
        # things that are individually defensible and jointly stronger than a
        # single tag SNP:
        #   (1) use the nearest available variant as a positional proxy, and
        #   (2) derive the H1/H2 axis from the data itself.
        # The inversion suppresses recombination across ~1 Mb, so the leading
        # principal component of genotypes in that window separates H1 and H2
        # carriers. Trimodality of PC1 (H1H1 / H1H2 / H2H2) is the check that it
        # really is the inversion rather than population structure.
        if tag not in ids:
            arr = np.array([pos_of.get(i, -1) for i in ids])
            valid = arr > 0
            k = int(np.argmin(np.abs(arr[valid] - H2_TAG_SNP["pos"])))
            proxy = [i for i, v in zip(ids, valid) if v][k]
            print(f"  {tag} is NOT in the reference panel.")
            print(f"  nearest available variant: {proxy} "
                  f"({abs(pos_of[proxy] - H2_TAG_SNP['pos']):,} bp away) -- used as"
                  f" a positional proxy.")
            j = ids.index(proxy)
            tag_label = f"{proxy} (proxy for {H2_TAG_SNP['rsid']})"
        else:
            j = ids.index(tag)
            tag_label = H2_TAG_SNP["rsid"]

        # ===========================================================
        # PRIMARY ANALYSIS: LD EXTENT AROUND EACH SIGNAL LEAD
        #
        # Two attempts to anchor on the H1/H2 haplotype failed for reasons that
        # are properties of the data, not fixable by more code:
        #   * rs1052553 is absent from this 1000G subset;
        #   * the PC1 surrogate fails QC (see below), most likely because the
        #     panel is all 2,548 multi-ancestry samples while H2 is European-
        #     enriched, so regional PC1 tracks ancestry.
        #
        # But Reviewer #3's actual question is whether extended LD complicates
        # clumping and variant-to-gene assignment. That is answerable directly by
        # measuring how far LD extends from each signal lead, and comparing
        # 17q21.31 against the chr19 locus as an internal control. No haplotype
        # call is required.
        # ===========================================================
        def nearest_in_panel(target_pos, max_bp=10_000):
            arr = np.array([pos_of.get(i, -1) for i in ids])
            ok = arr > 0
            k = int(np.argmin(np.abs(arr[ok] - target_pos)))
            cand = [i for i, v in zip(ids, ok) if v][k]
            d = abs(pos_of[cand] - target_pos)
            return (cand, d) if d <= max_bp else (None, d)

        print()
        print("=" * 74)
        print("LD EXTENT AROUND EACH chr17 SIGNAL LEAD")
        print("=" * 74)
        decay_rows = []
        for sig in sorted(c17["signal_id"].dropna().unique()):
            lead_pos = int(sig.split(":")[1])
            prox, d = nearest_in_panel(lead_pos)
            if prox is None:
                print(f"  {sig}: no panel variant within 10 kb ({d:,} bp) -- skipped")
                continue
            jj = ids.index(prox)
            r2v = (Z.T @ Z[:, jj] / n) ** 2
            posv = np.array([pos_of.get(i, np.nan) for i in ids], dtype=float)
            for thr in (0.2, 0.5, 0.8):
                sel = r2v >= thr
                if sel.sum() > 1:
                    span = np.nanmax(posv[sel]) - np.nanmin(posv[sel])
                else:
                    span = 0.0
                decay_rows.append({
                    "signal_id": sig, "lead_pos": lead_pos,
                    "panel_proxy": prox, "proxy_distance_bp": d,
                    "r2_threshold": thr,
                    "n_variants_in_ld": int(sel.sum()),
                    "ld_span_kb": round(span / 1000, 1),
                })
            print(f"  {sig}  (proxy {prox}, {d:,} bp away)")
            for row in decay_rows[-3:]:
                print(f"    r2 >= {row['r2_threshold']}: {row['n_variants_in_ld']:,} "
                      f"variants spanning {row['ld_span_kb']} kb")
        if decay_rows:
            dec = pd.DataFrame(decay_rows)
            dec.to_csv(f"{OUTDIR}/RA7_ld_extent_by_signal.tsv", sep="\t", index=False)
            print()
            print("  >> An LD span of hundreds of kb at r2 >= 0.5 is the direct")
            print("     explanation for 22 of 25 windows collapsing onto one signal,")
            print("     and for why variant-to-gene assignment cannot be resolved")
            print("     within 17q21.31. Quote the span in kb, not the inversion")
            print("     boundary, since the LD block is wider than the inversion.")

        # ---- data-driven inversion axis (reported, but QC-gated) ----
        lo_i, hi_i = INVERSION_DEFS["inversion_970kb_approx"]
        inwin = np.array([lo_i <= pos_of.get(i, -1) <= hi_i for i in ids])
        print(f"\n  deriving the H1/H2 axis from {int(inwin.sum()):,} variants "
              f"inside the inversion window")
        Zi = Z[:, inwin].astype(np.float32)
        # eigendecompose the sample-by-sample covariance: 2548 x 2548 is cheap
        C = (Zi @ Zi.T) / Zi.shape[1]
        w, V = np.linalg.eigh(C)
        pc1 = V[:, -1] * np.sqrt(max(w[-1], 0))
        var1 = float(w[-1] / w.sum())
        print(f"  PC1 explains {100 * var1:.1f}% of genotypic variance in the window")

        # 1D 3-means to call H1H1 / H1H2 / H2H2
        c = np.quantile(pc1, [0.17, 0.5, 0.83])
        for _ in range(50):
            lab = np.argmin(np.abs(pc1[:, None] - c[None, :]), axis=1)
            newc = np.array([pc1[lab == g].mean() if (lab == g).any() else c[g]
                             for g in range(3)])
            if np.allclose(newc, c):
                break
            c = np.sort(newc)
        counts = [int((lab == g).sum()) for g in range(3)]
        h2_dose = lab.astype(float)                    # 0/1/2 along the PC1 axis
        if np.corrcoef(h2_dose, pc1)[0, 1] < 0:
            h2_dose = 2 - h2_dose
        print(f"  inferred genotype groups (n): {counts}")

        # VALIDITY CHECKS. 3-means always returns three clusters whether or not
        # the data is trimodal, so the group sizes alone prove nothing. Two
        # independent checks:
        #   (i) the axis should track the nearest available tag proxy;
        #  (ii) PC1 should explain a large share of variance -- the inversion
        #       suppresses recombination across ~1 Mb, which is a much stronger
        #       structure than ordinary LD.
        # CAVEAT to state in Methods: this panel is all of 1000G (2,548 samples,
        # multiple ancestries) and H2 is European-enriched, so PC1 in this region
        # can partly reflect ancestry. A high correlation with the tag proxy is
        # what distinguishes "inversion axis" from "ancestry axis".
        r_proxy = abs(float(np.corrcoef(h2_dose, X[:, j])[0, 1]))
        print(f"  |correlation| with the tag proxy genotype: {r_proxy:.3f}")
        print(f"  PC1 share of variance: {100 * var1:.1f}%")
        if r_proxy < 0.5:
            print("  !! WEAK. The inferred axis does not track the tag proxy, so it")
            print("     may be capturing ancestry rather than the inversion. Do NOT")
            print("     report the PC1-based haplotype calls; fall back to the")
            print("     proxy-SNP LD and the credible-set analysis, and say why.")
        else:
            print("  >> The inferred axis tracks the tag proxy, supporting its use")
            print("     as an H1/H2 surrogate.")
        pd.DataFrame([{
            "pc1_variance_explained": var1,
            "abs_corr_with_tag_proxy": r_proxy,
            "group_sizes": ";".join(map(str, counts)),
            "tag_snp": H2_TAG_SNP["rsid"],
            "tag_in_panel": tag in ids,
            "proxy_used": ids[j],
            "proxy_distance_bp": abs(pos_of.get(ids[j], 0) - H2_TAG_SNP["pos"]),
        }]).to_csv(f"{OUTDIR}/RA7_haplotype_axis_qc.tsv", sep="\t", index=False)

        zd = (h2_dose - h2_dose.mean()) / h2_dose.std(ddof=0)
        r2_pc = (Z.T @ zd / n) ** 2
        pd.DataFrame({
            "variant_plink_id": ids,
            "pos": [pos_of.get(i, np.nan) for i in ids],
            "r2_with_inferred_H2_dosage": r2_pc,
        }).to_csv(f"{OUTDIR}/RA7_ld_with_inferred_haplotype.tsv",
                  sep="\t", index=False)
        print(f"  variants with r2 >= 0.5 against the inferred axis: "
              f"{int((r2_pc >= 0.5).sum()):,}")

        if True:
            r2 = (Z.T @ Z[:, j] / n) ** 2       # r^2 of every variant with the tag
            ld = pd.DataFrame({
                "variant_plink_id": ids,
                "pos": [pos_of.get(i, np.nan) for i in ids],
                "r2_with_H2_tag": r2,
            })
            ld.to_csv(f"{OUTDIR}/RA7_ld_with_h2_tag.tsv", sep="\t", index=False)
            print(f"  variants correlated with the tag: {len(ld):,}")
            for thr in (0.2, 0.5, 0.8):
                print(f"    r2 >= {thr}: {int((ld['r2_with_H2_tag'] >= thr).sum()):,}")

            print("\n  median r2 with the tag, inside vs outside each window:")
            for name, (lo, hi) in INVERSION_DEFS.items():
                ins = ld[(ld["pos"] >= lo) & (ld["pos"] <= hi)]["r2_with_H2_tag"]
                out = ld[(ld["pos"] < lo) | (ld["pos"] > hi)]["r2_with_H2_tag"]
                if len(ins) and len(out):
                    print(f"    {name}: inside {ins.median():.3f} (n={len(ins)}), "
                          f"outside {out.median():.3f} (n={len(out)})")

            # where do our candidates sit?
            try:
                cc = pd.read_csv(
                    "work/step_RA0e_consensus/RA0e_candidate_sets_tier1.tsv",
                    sep="\t")
                cc = cc[cc["variant_key"].str.startswith("chr17")].copy()
                cc["variant_plink_id"] = (
                    cc["variant_key"].str.split(":").str[0].str.replace("chr", "",
                                                                        regex=False)
                    + ":" + cc["variant_key"].str.split(":").str[1])
                # FIX: candidates are joined on chr:pos, but several of them are
                # simply NOT in the 1000G panel (e.g. 17:45771602 is absent while
                # 17:45771400 and 17:45771984 are present). The previous version
                # silently filled those with r2 = 0, which reads as "no LD" when
                # it actually means "not measurable". Use the nearest panel
                # variant instead and record the distance so the substitution is
                # visible; leave r2 as NaN where no proxy is close enough.
                prox_ids, prox_d = [], []
                for vk in cc["variant_key"]:
                    pcand, pd_bp = nearest_in_panel(int(vk.split(":")[1]))
                    prox_ids.append(pcand)
                    prox_d.append(pd_bp)
                cc["panel_proxy_id"] = prox_ids
                cc["panel_proxy_distance_bp"] = prox_d
                cc["in_panel_directly"] = cc["variant_plink_id"].isin(set(ids))
                m = cc.merge(ld.rename(columns={"variant_plink_id": "panel_proxy_id"}),
                             on="panel_proxy_id", how="left")
                # also attach the data-driven axis, which does not depend on any
                # single variant being present in the panel
                m = m.merge(
                    pd.DataFrame({"panel_proxy_id": ids,
                                  "r2_with_inferred_H2_dosage": r2_pc}),
                    on="panel_proxy_id", how="left")
                # QC-gate the PC1 column: if the axis failed validation it must
                # not be reported as a haplotype call.
                if r_proxy < 0.5:
                    m["haplotype_axis_qc"] = "FAILED - do not report"
                    m["on_H2_haplotype"] = np.nan
                else:
                    m["haplotype_axis_qc"] = "passed"
                    m["on_H2_haplotype"] = m["r2_with_inferred_H2_dosage"] >= 0.5
                print(f"\n  candidate LD ({tag_label}); "
                      f"haplotype axis QC: "
                      f"{'FAILED' if r_proxy < 0.5 else 'passed'}")
                print(m[["signal_id", "variant_key", "pct_specs_in_top1",
                         "in_panel_directly", "panel_proxy_distance_bp",
                         "r2_with_H2_tag"]]
                      .round(4).to_string(index=False))
                m.to_csv(f"{OUTDIR}/RA7_candidates_vs_h2_tag.tsv",
                         sep="\t", index=False)
                print()
                print("  >> This is the direct answer to Reviewer #3 major 3.")
                print("     High r2 means the candidate tags the H1/H2 haplotype")
                print("     rather than marking an independent causal signal, so")
                print("     variant-to-gene assignment cannot be resolved within it.")
                print("     Low r2 means the candidate carries information beyond")
                print("     haplotype membership.")
            except FileNotFoundError:
                pass

            # LD decay figure
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 3.6))
            ax.scatter(ld["pos"] / 1e6, ld["r2_with_H2_tag"], s=3, alpha=0.25,
                       color="#4C72B0", rasterized=True)
            ax.axvline(H2_TAG_SNP["pos"] / 1e6, color="black", lw=1,
                       label=f"{H2_TAG_SNP['rsid']} (H2 tag)")
            for (nm, (lo, hi)), c in zip(INVERSION_DEFS.items(),
                                         ["#C44E52", "#DD8452"]):
                ax.axvspan(lo / 1e6, hi / 1e6, color=c, alpha=0.10, label=nm)
            try:
                ax.scatter(m["pos"] / 1e6, m["r2_with_H2_tag"], s=45,
                           color="#C44E52", edgecolor="black", zorder=5,
                           label="reported candidates")
            except Exception:
                pass
            ax.set_xlabel("chr17 position (Mb, GRCh38)")
            ax.set_ylabel("r$^2$ with H2 tag SNV")
            ax.legend(fontsize=7, frameon=False)
            fig.tight_layout()
            for ext in ("png", "pdf"):
                fig.savefig(f"{OUTDIR}/RA7_ld_with_h2_tag.{ext}", dpi=300,
                            bbox_inches="tight")
            plt.close(fig)

# ---------------------------------------------------------------
# 3. Credible-set dilution inside vs outside
# ---------------------------------------------------------------
print()
print("=" * 74)
print("3. CREDIBLE-SET SIZE AND PIP DILUTION")
print("=" * 74)
cs_col = "susie_eqtl_best_cs_size"
if cs_col in c17.columns:
    c17["cs_size"] = pd.to_numeric(c17[cs_col], errors="coerce")
    c17["max_pip"] = pd.to_numeric(c17.get("susie_eqtl_max_pip"), errors="coerce")
    rows = []
    for name, (lo, hi) in INVERSION_DEFS.items():
        inside = c17[(c17["pos"] >= lo) & (c17["pos"] <= hi)]
        outside = c17[(c17["pos"] < lo) | (c17["pos"] > hi)]
        rows.append({
            "definition": name,
            "median_cs_size_inside": float(inside["cs_size"].median()),
            "median_cs_size_outside": float(outside["cs_size"].median()),
            "median_max_pip_inside": float(inside["max_pip"].median()),
            "median_max_pip_outside": float(outside["max_pip"].median()),
            "n_inside": len(inside), "n_outside": len(outside),
        })
    cs = pd.DataFrame(rows)
    cs.to_csv(f"{OUTDIR}/RA7_credible_set_by_region.tsv", sep="\t", index=False)
    print(cs.to_string(index=False))
    print()
    print("  >> This is the mechanistic link to report: extended LD inflates")
    print("     credible sets, which dilutes every PIP toward zero, which is why")
    print("     the SuSiE layer contributes only 5.7% of its nominal weight")
    print("     (RA0b). One structural fact explains Reviewer #3's inversion")
    print("     concern and Reviewer #2's layer-utility question at the same time.")

print(f"\n[RA7] done. Outputs in {OUTDIR}/")
