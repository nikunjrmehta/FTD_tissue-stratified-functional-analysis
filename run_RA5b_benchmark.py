"""
RA5b — matched benchmark against fine-mapped variants, with circularity guards.

Answers, empirically:
  Reviewer #2 Comment 3 : "does combining all six scores provide more accuracy in
                           prioritizing signals than using only a subset of them?"
  Reviewer #3 major 1   : empirical justification for the weights, via
                           "previously fine-mapped loci"

THE CIRCULARITY PROBLEM, AND THE DESIGN THAT AVOIDS IT
------------------------------------------------------
The obvious positive set is high-PIP fine-mapped GTEx variants. But
susie_score_any IS the PIP, and gtex_score_any correlates with it at rho = 0.656
(RA1). Evaluating the full six-layer score against PIP-defined positives would
produce a high AUROC that measures nothing but its own definition.

So each half of the score is evaluated against a positive set it played no part
in defining:

  PRIMARY   positives = eQTL/sQTL fine-mapped (high PIP)
            evaluated = the NON-GTEx layers only
                        (gwas_score, alphagenome_score, ccre_score)
            -> Does sequence/regulatory/association evidence independently
               predict which variants fine-mapping resolves? Fully clean.

  SECONDARY positives = genome-wide significant variants
            evaluated = the NON-GWAS layers only
                        (susie, gtex, expr, alphagenome, ccre)
            -> Does molecular evidence independently predict association?
               Confounded by LD, so reported as supporting, not primary.

  REFERENCE the full six-layer score against PIP positives, reported and
            EXPLICITLY LABELLED CIRCULAR, so a referee can see we knew.

  FLOOR     a distance-to-TSS-only predictor. Fine-mapped eQTLs cluster near the
            TSS, so any score that cannot beat this is measuring TSS proximity.

Negatives are matched to positives within signal, by distance-to-TSS decile and
(if available) MAF decile.

PREREQUISITE, optional but recommended -- MAF for matching:
    plink2 --pfile data/ldref/pgen/1kg_chr17_hg38_chrpos --freq --out work/freq_chr17
    plink2 --pfile data/ldref/pgen/1kg_chr19_hg38_chrpos --freq --out work/freq_chr19
The script runs without it and reports that matching was TSS-only.

Outputs -> work/step_RA5b_benchmark/
"""

import glob

import numpy as np
import pandas as pd
from scipy import stats

from ra_common import (
    COMPONENTS, CURRENT_WEIGHTS,
    add_signal_id, ensure_dir, load_final, renormalise,
)

OUTDIR = ensure_dir("work/step_RA5b_benchmark")

PIP_PRIMARY = 0.5
PIP_SENSITIVITY = 0.3
PIP_NEGATIVE_MAX = 0.01
N_NEG_PER_POS = 3
N_BOOTSTRAP = 1000
# REPRODUCIBILITY FIX: matching and bootstrapping previously drew from one shared
# generator, so adding a score changed how much randomness the bootstrap consumed
# and therefore changed which negatives were matched in LATER benchmarks. The
# PIP>=0.3 gwas_only AUROC moved from 0.432 to 0.468 between runs for exactly this
# reason. Independent generators make each stage reproducible on its own.
RNG_MATCH = np.random.default_rng(20260808)
RNG_BOOT = np.random.default_rng(31415926)

df, info = load_final(dedupe=True)
df = add_signal_id(df)
df["p_num"] = pd.to_numeric(df["p"], errors="coerce")
df["dist_tss"] = pd.to_numeric(df.get("dist_to_tss_bp"), errors="coerce").abs()

pip_cols = [c for c in ["susie_eqtl_max_pip", "susie_sqtl_max_pip",
                        "susie_brain_max_pip_any", "susie_blood_max_pip_any"]
            if c in df.columns]
df["max_pip"] = df[pip_cols].apply(pd.to_numeric, errors="coerce").max(axis=1).fillna(0.0)

# ---------------------------------------------------------------
# MAF, if available
# ---------------------------------------------------------------
afreq_files = sorted(glob.glob("work/freq_chr*.afreq"))
if afreq_files:
    af = pd.concat([pd.read_csv(f, sep="\t") for f in afreq_files], ignore_index=True)
    id_col = "ID" if "ID" in af.columns else af.columns[1]
    frq_col = "ALT_FREQS" if "ALT_FREQS" in af.columns else af.columns[-1]
    af["maf"] = pd.to_numeric(af[frq_col], errors="coerce").apply(lambda x: min(x, 1 - x))
    df["_key"] = df["chr"].astype(str) + ":" + df["pos"].astype(str)
    af["_key"] = af[id_col].astype(str)
    df = df.merge(af[["_key", "maf"]].drop_duplicates("_key"), on="_key", how="left")
    print(f"[RA5b] MAF merged for {df['maf'].notna().sum():,} / {len(df):,} variants")
else:
    df["maf"] = np.nan
    print("[RA5b] no .afreq files -- matching on distance-to-TSS only")

USE_MAF = df["maf"].notna().mean() > 0.5

# ---------------------------------------------------------------
# Score definitions
# ---------------------------------------------------------------
GTEX_LAYERS = ["susie_score_any", "gtex_score_any", "expr_score_any"]
NON_GTEX = ["gwas_score", "alphagenome_score", "ccre_score"]
NON_GWAS = [c for c in COMPONENTS if c != "gwas_score"]


def wsum(frame, weights):
    out = pd.Series(0.0, index=frame.index)
    for c, w in weights.items():
        out = out + w * pd.to_numeric(frame[c], errors="coerce").fillna(0.0)
    return out


scores = {
    "full_6layer__CIRCULAR": wsum(df, CURRENT_WEIGHTS),
    "gtex_free_3layer": wsum(df, renormalise({c: CURRENT_WEIGHTS[c] for c in NON_GTEX})),
    # BUGFIX: susie_excluded_5layer still CONTAINS gwas_score, so it was circular
    # in the GWAS-significant benchmark (positives defined by p-value). Added a
    # genuinely GWAS-free score for that benchmark; susie_excluded_5layer is kept
    # only for the fine-mapped benchmark, where excluding SuSiE is what matters.
    "susie_excluded_5layer": wsum(
        df, renormalise({c: w for c, w in CURRENT_WEIGHTS.items() if c != "susie_score_any"})),
    "gwas_free_5layer": wsum(
        df, renormalise({c: CURRENT_WEIGHTS[c] for c in NON_GWAS})),
    "gwas_only": pd.to_numeric(df["gwas_score"], errors="coerce").fillna(0.0),
    "alphagenome_only": pd.to_numeric(df["alphagenome_score"], errors="coerce").fillna(0.0),
    "ccre_only": pd.to_numeric(df["ccre_score"], errors="coerce").fillna(0.0),
    "gwas_plus_alphagenome": wsum(df, {"gwas_score": 0.5, "alphagenome_score": 0.5}),
    "BASELINE_tss_proximity": -df["dist_tss"].fillna(df["dist_tss"].max()),
}

# ---------------------------------------------------------------
# ADDED: evaluate every AlphaGenome summarisation against the fine-mapped
# positives. RA0c showed the current max-of-percentiles summarisation compresses
# the layer to almost no dynamic range, but had no external standard to choose a
# better one -- so any replacement would have been just as arbitrary.
#
# This benchmark IS that external standard: the positives are defined by GTEx
# fine-mapping, which shares no information with AlphaGenome. Selecting the
# summarisation that best predicts fine-mapped variants is therefore empirical
# justification, not another arbitrary choice, and it is exactly what Reviewer #3
# major 1 asked for.
# ---------------------------------------------------------------
_pct_cols = sorted(
    [c for c in df.columns if c.startswith("scorer") and c.endswith("_pct")],
    key=lambda x: int(x.replace("scorer", "").replace("_pct", "")),
)
_pct_cols = [c for c in _pct_cols if pd.to_numeric(df[c], errors="coerce").notna().any()]
_P = df[_pct_cols].apply(pd.to_numeric, errors="coerce")
_max_raw = _P.max(axis=1)

AG_SUMMARISATIONS = {
    "AGsumm_max_current": _max_raw,
    "AGsumm_mean": _P.mean(axis=1),
    "AGsumm_median": _P.median(axis=1),
    "AGsumm_q75": _P.quantile(0.75, axis=1),
    "AGsumm_top3_mean": _P.apply(lambda r: r.nlargest(3).mean(), axis=1),
    "AGsumm_max_then_repercentile": _max_raw.groupby(df["locus_id"]).rank(
        pct=True, method="average"),
}
scores.update({k: v.fillna(0.0) for k, v in AG_SUMMARISATIONS.items()})

for k, v in scores.items():
    df[f"sc__{k}"] = v

CLEAN_FOR_PIP = [
    "gtex_free_3layer", "gwas_only", "alphagenome_only", "ccre_only",
    "gwas_plus_alphagenome", "BASELINE_tss_proximity",
] + list(AG_SUMMARISATIONS)
CLEAN_FOR_GWAS = [
    "gwas_free_5layer", "alphagenome_only", "ccre_only", "BASELINE_tss_proximity",
]


# ---------------------------------------------------------------
# Matching
# ---------------------------------------------------------------
def build_matched_set(pos_mask, neg_mask, label):
    d = df.copy()
    d["_tss_dec"] = d.groupby("signal_id")["dist_tss"].transform(
        lambda x: pd.qcut(x.rank(method="first"), 10, labels=False, duplicates="drop"))
    strata = ["signal_id", "_tss_dec"]
    if USE_MAF:
        d["_maf_dec"] = d.groupby("signal_id")["maf"].transform(
            lambda x: pd.qcut(x.rank(method="first"), 5, labels=False, duplicates="drop"))
        strata.append("_maf_dec")

    pos = d[pos_mask].copy()
    neg_pool = d[neg_mask].copy()

    chosen = []
    for key, grp in pos.groupby(strata, dropna=False):
        cand = neg_pool
        for col, val in zip(strata, key if isinstance(key, tuple) else (key,)):
            cand = cand[cand[col] == val]
        n_take = min(len(cand), N_NEG_PER_POS * len(grp))
        if n_take > 0:
            chosen.append(cand.sample(n=n_take, random_state=int(RNG_MATCH.integers(1e6))))
    neg = pd.concat(chosen, ignore_index=True) if chosen else neg_pool.head(0)

    print(f"\n[{label}] positives={len(pos):,}  matched negatives={len(neg):,}"
          f"  (matched on {', '.join(strata)})")
    if len(neg) == 0:
        return None, None
    # matching quality
    for c in (["dist_tss"] + (["maf"] if USE_MAF else [])):
        _, pv = stats.mannwhitneyu(pos[c].dropna(), neg[c].dropna(), alternative="two-sided")
        print(f"    {c}: pos median={pos[c].median():.4g}  neg median={neg[c].median():.4g}"
              f"  balance p={pv:.3f}" + ("  <-- IMBALANCED" if pv < 0.05 else ""))
    return pos, neg


def auroc_with_ci(pos_vals, neg_vals):
    pos_vals = np.asarray(pos_vals, dtype=float)
    neg_vals = np.asarray(neg_vals, dtype=float)
    u, _ = stats.mannwhitneyu(pos_vals, neg_vals, alternative="two-sided")
    auc = u / (len(pos_vals) * len(neg_vals))
    boots = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        a = RNG_BOOT.choice(pos_vals, len(pos_vals), replace=True)
        b = RNG_BOOT.choice(neg_vals, len(neg_vals), replace=True)
        uu, _ = stats.mannwhitneyu(a, b, alternative="two-sided")
        boots[i] = uu / (len(a) * len(b))
    return auc, np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def run_benchmark(pos, neg, score_list, label):
    rows = []
    for name in score_list:
        col = f"sc__{name}"
        a, lo, hi = auroc_with_ci(pos[col].values, neg[col].values)
        rows.append({
            "benchmark": label,
            "score": name,
            "n_pos": len(pos), "n_neg": len(neg),
            "auroc": round(a, 4),
            "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
            "beats_tss_floor": None,
        })
    out = pd.DataFrame(rows)
    floor = out.loc[out["score"] == "BASELINE_tss_proximity", "auroc"].iloc[0]
    out["beats_tss_floor"] = out["auroc"] > floor
    out = out.sort_values("auroc", ascending=False)
    print(f"\n--- {label} (TSS floor AUROC = {floor:.3f}) ---")
    print(out.to_string(index=False))
    return out


results = []
_export_sets = []

# ---------- PRIMARY: fine-mapped positives, non-GTEx scores ----------
for thresh, tag in [(PIP_PRIMARY, "primary"), (PIP_SENSITIVITY, "sensitivity")]:
    pos, neg = build_matched_set(
        df["max_pip"] >= thresh, df["max_pip"] <= PIP_NEGATIVE_MAX,
        f"fine-mapped PIP>={thresh} ({tag})")
    if pos is None:
        continue
    r = run_benchmark(pos, neg, CLEAN_FOR_PIP + ["full_6layer__CIRCULAR"],
                      f"finemapped_PIP{thresh}_{tag}")
    results.append(r)

    _p, _n = pos.copy(), neg.copy()
    _p["is_positive"], _n["is_positive"] = 1, 0
    _p["pip_threshold"] = _n["pip_threshold"] = thresh
    _export_sets += [_p, _n]

# ---------- SECONDARY: GWAS-significant positives, non-GWAS scores ----------
pos, neg = build_matched_set(
    df["p_num"] < 5e-8, df["p_num"] > 0.05, "genome-wide significant")
if pos is not None:
    r = run_benchmark(pos, neg, CLEAN_FOR_GWAS, "gwas_significant")
    results.append(r)

allres = pd.concat(results, ignore_index=True)
allres.to_csv(f"{OUTDIR}/RA5b_auroc_results.tsv", sep="\t", index=False)

# ---------------------------------------------------------------
# ADDED: export the matched benchmark set for the Enformer comparison.
#
# This is a far better -- and ~10x cheaper -- design than re-scoring the top-N
# ranked variants. Instead of asking "does swapping the model shuffle the
# ranking?", it asks "does Enformer predict fine-mapped variants better or worse
# than AlphaGenome?", against a standard neither model contributed to. That is a
# validity comparison rather than a concordance one, it needs only the benchmark
# variants rather than thousands, and it is directly answerable.
# ---------------------------------------------------------------
if _export_sets:
    exp = pd.concat(_export_sets, ignore_index=True)
    # BUGFIX: drop_duplicates() previously kept only the first occurrence, so a
    # variant present in BOTH the PIP>=0.5 and PIP>=0.3 sets was labelled with the
    # 0.5 threshold only. Grouping downstream by pip_threshold then compared
    # non-nested sets and reported the wrong n. Membership is now recorded as
    # separate flags, so each threshold's set can be reconstructed exactly.
    memb = (exp.groupby("variant_key")["pip_threshold"]
              .apply(lambda s: set(s)).to_dict())
    exp = exp.drop_duplicates("variant_key").copy()
    exp["in_pip_0.5_set"] = exp["variant_key"].map(lambda v: PIP_PRIMARY in memb[v])
    exp["in_pip_0.3_set"] = exp["variant_key"].map(lambda v: PIP_SENSITIVITY in memb[v])
    cols = [c for c in ["variant_key", "chr", "pos", "ref", "alt", "rsid",
                        "signal_id", "locus_id", "p", "max_pip", "dist_tss",
                        "alphagenome_score", "is_positive",
                        "in_pip_0.5_set", "in_pip_0.3_set"]
            if c in exp.columns]
    exp[cols].to_csv(f"{OUTDIR}/RA5b_benchmark_variants_for_enformer.tsv",
                     sep="\t", index=False)
    print(f"\n[RA5b] exported {len(exp):,} benchmark variants for Enformer scoring "
          f"-> {OUTDIR}/RA5b_benchmark_variants_for_enformer.tsv")

# ---------------------------------------------------------------
print()
print("=" * 78)
print("HOW TO READ THIS")
print("=" * 78)
print("* full_6layer__CIRCULAR is reported for transparency only. Its positives")
print("  were defined by one of its own components. Never quote it as evidence.")
print("* gtex_free_3layer is the PRIMARY result: it shares no information with the")
print("  positive set, so its AUROC is an honest estimate.")
print("* Any score not beating BASELINE_tss_proximity is measuring TSS proximity.")
print("* If gwas_plus_alphagenome ~= gtex_free_3layer, the cCRE layer adds nothing")
print("  on this benchmark -- report that; it directly answers Comment 3.")
print("* Confidence intervals are wide at these sample sizes. Report them, and do")
print("  not claim a difference that the intervals do not support.")

# ---------------------------------------------------------------
# Which AlphaGenome summarisation has the best external validity?
# ---------------------------------------------------------------
ag = allres[allres["score"].str.startswith("AGsumm_")].copy()
if len(ag):
    print()
    print("=" * 78)
    print("ALPHAGENOME SUMMARISATION, RANKED BY EXTERNAL VALIDITY")
    print("=" * 78)
    for bench, grp in ag.groupby("benchmark"):
        grp = grp.sort_values("auroc", ascending=False)
        print(f"\n{bench}:")
        print(grp[["score", "auroc", "ci95_low", "ci95_high"]].to_string(index=False))
    ag.to_csv(f"{OUTDIR}/RA5b_alphagenome_summarisation_auroc.tsv",
              sep="\t", index=False)
    print()
    print("  >> The positives here are defined by GTEx fine-mapping and share no")
    print("     information with AlphaGenome, so this is a legitimate basis for")
    print("     choosing the summarisation. If one option is clearly best AND its")
    print("     CI excludes the current max, that is empirical justification for")
    print("     changing it -- which is what Reviewer #3 major 1 asked for. If the")
    print("     CIs all overlap, keep the published max and say the choice was")
    print("     tested against an external standard and found not to matter.")

# ---------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for bench, grp in allres.groupby("benchmark"):
    grp = grp.sort_values("auroc")
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(grp) + 2))
    colors = ["#C44E52" if "CIRCULAR" in s else
              ("#B0B0B0" if s.startswith("BASELINE") else "#4C72B0")
              for s in grp["score"]]
    ax.barh(grp["score"], grp["auroc"], color=colors)
    ax.errorbar(grp["auroc"], range(len(grp)),
                xerr=[grp["auroc"] - grp["ci95_low"], grp["ci95_high"] - grp["auroc"]],
                fmt="none", ecolor="black", capsize=3, lw=1)
    ax.axvline(0.5, color="grey", ls="--", lw=1)
    ax.set_xlim(0.3, 1.0)
    ax.set_xlabel("AUROC (95% bootstrap CI)")
    ax.set_title(f"{bench}\nn_pos={grp['n_pos'].iloc[0]}, n_neg={grp['n_neg'].iloc[0]}",
                 fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUTDIR}/RA5b_{bench}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

print(f"\n[RA5b] done. Outputs in {OUTDIR}/")
