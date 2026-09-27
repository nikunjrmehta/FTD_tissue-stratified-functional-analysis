"""
RA0c — where does discriminative power actually come from, and is the AlphaGenome
layer being summarised in a way that throws its signal away?

Two questions raised by the RA0b saturation table:

  (1) RA0b measured each layer's spread across the MIDDLE 80% of variants. That is
      not the same as influence on the TOP of the ranking, which is what the paper
      reports. A layer can be flat across the bulk and still separate the winners.
      This script decomposes final_score for the top-20 per signal against the
      locus median, so the claim "layer X does/doesn't matter" is made about the
      variants the manuscript actually discusses.

  (2) alphagenome_score has median 0.911 and a 10th-90th spread of only 0.276,
      so the layer named in the title contributes ~0.07 of discriminative range.
      The cause is probably the summarisation, not the model: alphagenome_pct_max
      is the MAX across ~15 within-locus percentile ranks. Taking a maximum over
      many percentiles is strongly biased upward -- with k independent uniform
      percentiles the expected max is k/(k+1), i.e. ~0.94 for k=15. That is almost
      exactly what is observed, which means the layer may be measuring "how many
      scorers were run" rather than "how regulatory is this variant".

      This tests alternative summarisations using the per-scorer percentile columns
      ALREADY on disk. No AlphaGenome API calls are needed.

Outputs -> work/step_RA0c_layer_influence/
"""

import numpy as np
import pandas as pd
from scipy import stats

from ra_common import (
    COMPONENTS, COMPONENT_LABELS, CURRENT_WEIGHTS,
    add_signal_id, ensure_dir, load_final,
)

OUTDIR = ensure_dir("work/step_RA0c_layer_influence")

df, info = load_final(dedupe=True)
df = add_signal_id(df)
print(f"\n[RA0c] {len(df):,} unique variants\n")

# ===============================================================
# PART 1 — layer contribution at the TOP of the ranking
# ===============================================================
print("=" * 78)
print("1. WHAT SEPARATES THE TOP CANDIDATES FROM THE FIELD?")
print("=" * 78)

rows = []
for s, grp in df.groupby("signal_id"):
    grp = grp.copy()
    top20 = grp.nlargest(20, "final_score")
    top1 = grp.loc[grp["final_score"].idxmax()]
    for c in COMPONENTS:
        w = CURRENT_WEIGHTS[c]
        v = pd.to_numeric(grp[c], errors="coerce").fillna(0.0)
        vt = pd.to_numeric(top20[c], errors="coerce").fillna(0.0)
        med_all = float(v.median())
        med_top = float(vt.median())
        rows.append({
            "signal_id": s,
            "component": c,
            "label": COMPONENT_LABELS[c],
            "weight": w,
            "median_all_variants": round(med_all, 4),
            "median_top20": round(med_top, 4),
            "top1_value": round(float(pd.to_numeric(top1[c], errors="coerce") or 0.0), 4),
            # how much of the top-20's score advantage comes from this layer
            "weighted_advantage_top20_vs_median": round(w * (med_top - med_all), 4),
            "weighted_advantage_top1_vs_median": round(
                w * (float(pd.to_numeric(top1[c], errors="coerce") or 0.0) - med_all), 4),
        })
adv = pd.DataFrame(rows)
adv.to_csv(f"{OUTDIR}/RA0c_top_advantage_by_layer.tsv", sep="\t", index=False)

pivot = adv.pivot_table(index="label", columns="signal_id",
                        values="weighted_advantage_top20_vs_median")
pivot["mean"] = pivot.mean(axis=1)
pivot = pivot.sort_values("mean", ascending=False)
print("\nScore advantage of the top-20 over the signal median, by layer:")
print(pivot.round(4).to_string())
pivot.round(4).to_csv(f"{OUTDIR}/RA0c_top20_advantage_pivot.tsv", sep="\t")

print()
print("  >> This is the number to quote when asked whether a layer earns its")
print("     weight. It is about the variants the manuscript reports, unlike the")
print("     RA0b spread statistic which describes the bulk distribution.")

# ===============================================================
# PART 2 — is alphagenome_pct_max a max-of-percentiles artefact?
# ===============================================================
print()
print("=" * 78)
print("2. ALPHAGENOME SUMMARISATION")
print("=" * 78)

pct_cols = sorted(
    [c for c in df.columns if c.startswith("scorer") and c.endswith("_pct")],
    key=lambda x: int(x.replace("scorer", "").replace("_pct", "")),
)
pct_cols = [c for c in pct_cols if pd.to_numeric(df[c], errors="coerce").notna().any()]
k = len(pct_cols)
print(f"per-scorer percentile columns with data: {k}")
print(f"  {pct_cols}")

P = df[pct_cols].apply(pd.to_numeric, errors="coerce")

expected_max = k / (k + 1.0)
observed_max_median = float(P.max(axis=1).median())
print(f"\nexpected median of max over {k} independent uniform percentiles: ~{expected_max:.3f}")
print(f"observed median of alphagenome_pct_max                        : {observed_max_median:.3f}")
if abs(observed_max_median - expected_max) < 0.08:
    print("  >> These agree closely. The high median is largely a property of taking")
    print("     a MAXIMUM over many percentiles, not evidence that most variants are")
    print("     strongly regulatory. A referee could raise this.")

alts = {
    "max_current": P.max(axis=1),
    "mean": P.mean(axis=1),
    "median": P.median(axis=1),
    "q75": P.quantile(0.75, axis=1),
    "top3_mean": P.apply(lambda r: r.nlargest(3).mean(), axis=1),
}

# ---------------------------------------------------------------
# Preferred candidate: keep the MAX, then re-rank it within locus.
#
# Taking a max is scientifically motivated -- AlphaGenome's scorers cover
# different modalities (expression, splicing, chromatin), and a variant that
# affects only splicing should not be penalised for being unremarkable on the
# other scorers. Averaging destroys that. The defect is not the max, it is that
# a max over k percentiles is upward-biased towards k/(k+1), so the resulting
# values are no longer uniform and the layer loses its range.
#
# Re-percentiling the max within each locus preserves the "any modality counts"
# semantics while restoring a uniform, fully-discriminative 0-1 scale. It is also
# the smallest possible change to the published method, and it keeps the variant
# ORDERING within a locus identical to the current score -- only the spacing
# changes. Ordering-preserving means Signal 1 and 2 conclusions cannot move.
# ---------------------------------------------------------------
_max_raw = P.max(axis=1)
alts["max_then_repercentile"] = (
    _max_raw.groupby(df["locus_id"]).rank(pct=True, method="average")
)

rows = []
for name, v in alts.items():
    v = pd.to_numeric(v, errors="coerce")
    p10, p90 = v.quantile(0.10), v.quantile(0.90)
    w = CURRENT_WEIGHTS["alphagenome_score"]
    rows.append({
        "summarisation": name,
        "median": round(float(v.median()), 4),
        "p10_p90_spread": round(float(p90 - p10), 4),
        "effective_score_span": round(w * float(p90 - p10), 4),
        "pct_at_max_1.0": round(100.0 * float((v >= 0.999).mean()), 2),
        "spearman_vs_current_max": round(
            float(stats.spearmanr(v, alts["max_current"])[0]), 4),
    })
summ = pd.DataFrame(rows)
summ.to_csv(f"{OUTDIR}/RA0c_alphagenome_summarisation.tsv", sep="\t", index=False)
print("\nAlternative AlphaGenome summarisations:")
print(summ.to_string(index=False))

best = summ.sort_values("effective_score_span", ascending=False).iloc[0]
print(f"\nwidest discriminative range: '{best['summarisation']}' "
      f"({best['effective_score_span']:.4f} vs "
      f"{summ.set_index('summarisation').loc['max_current','effective_score_span']:.4f} for the current max)")

# ---------------------------------------------------------------
# Would a better summarisation change the top candidates?
# ---------------------------------------------------------------
other_w = {c: w for c, w in CURRENT_WEIGHTS.items() if c != "alphagenome_score"}
base = pd.Series(0.0, index=df.index)
for c, w in other_w.items():
    base = base + w * pd.to_numeric(df[c], errors="coerce").fillna(0.0)

rows = []
for name, v in alts.items():
    sc = base + CURRENT_WEIGHTS["alphagenome_score"] * pd.to_numeric(v, errors="coerce").fillna(0.0)
    df[f"score__ag_{name}"] = sc
    for s, grp in df.groupby("signal_id"):
        srt = grp.sort_values([f"score__ag_{name}", "variant_key"], ascending=[False, True])
        top = srt.iloc[0]
        cur = grp.loc[grp["final_score"].idxmax(), "variant_key"]
        cur20 = set(grp.nlargest(20, "final_score")["variant_key"])
        rows.append({
            "summarisation": name,
            "signal_id": s,
            "top_variant_key": top["variant_key"],
            "top_p": top["p"],
            "top_changed_vs_current": bool(top["variant_key"] != cur),
            "top20_overlap": len(set(srt.head(20)["variant_key"]) & cur20),
        })
impact = pd.DataFrame(rows)
impact.to_csv(f"{OUTDIR}/RA0c_alphagenome_summarisation_impact.tsv", sep="\t", index=False)
print("\nEffect on top candidates:")
print(impact.to_string(index=False))

n_changed = int(impact[impact["summarisation"] != "max_current"]["top_changed_vs_current"].sum())
print(f"\ntop-candidate changes across alternatives: {n_changed}")
if n_changed == 0:
    print("  >> Conclusions are robust to how AlphaGenome is summarised. Report the")
    print("     compression as a characterised limitation and move on -- no re-run.")
else:
    print("  >> Summarisation choice moves the results. This needs a decision before")
    print("     the Enformer comparison, because a like-for-like model swap requires")
    print("     a summarisation that is not throwing the signal away.")

# ---------------------------------------------------------------
# Figure
# ---------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

ax = axes[0]
pv = pivot.drop(columns=["mean"])
x = np.arange(len(pv))
width = 0.8 / pv.shape[1]
for i, col in enumerate(pv.columns):
    ax.bar(x + i * width - 0.4, pv[col], width=width, label=col)
ax.set_xticks(x)
ax.set_xticklabels(pv.index, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("weighted advantage of top-20 over median")
ax.set_title("Which layers separate the top candidates?", fontsize=10)
ax.legend(fontsize=7, frameon=False)
ax.axhline(0, color="grey", lw=0.8)

ax = axes[1]
for name, v in alts.items():
    ax.plot(np.linspace(0, 100, 101),
            np.percentile(pd.to_numeric(v, errors="coerce").dropna(), np.linspace(0, 100, 101)),
            lw=1.8 if name == "max_current" else 1.2,
            ls="-" if name == "max_current" else "--", label=name)
ax.set_xlabel("percentile of variants")
ax.set_ylabel("AlphaGenome summary score")
ax.set_title("Summarisation choice and discriminative range", fontsize=10)
ax.legend(fontsize=8, frameon=False)

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{OUTDIR}/RA0c_layer_influence.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"\n[RA0c] done. Outputs in {OUTDIR}/")
