"""
RA0b — impact of the missing-component defect, and layer saturation.

RA0 found that 22,928 of 66,726 rows have final_score = NaN, because
susie_score_any (25.7% missing) and expr_score_any (10.4% missing) were never
filled before the weighted sum. Those variants are silently absent from every
sort_values()-based top list.

This is not a neutral bug. Missing SuSiE PIP means "this variant is in no credible
set", i.e. no fine-mapping evidence -- which is informative absence and should
score 0. Dropping those variants instead means the pipeline preferentially
retained variants in annotation-dense regions. That is exactly the bias Reviewer
#2 hypothesised in Comment 4, and it needs to be measured before it is described.

This script answers three questions WITHOUT re-running the pipeline:

  1. If missing components are treated as 0, which variants re-enter the ranking,
     and does the top candidate at any signal change?
  2. Are the excluded variants systematically different (annotation-sparse)?
  3. How much discriminative power does each layer actually have? Two layers look
     near-saturated in the RA0 output, which bears on Reviewer #2 Comment 3.

Outputs -> work/step_RA0b_missingness_impact/
"""

import numpy as np
import pandas as pd
from scipy import stats

from ra_common import (
    COMPONENTS, COMPONENT_LABELS, CURRENT_WEIGHTS,
    add_signal_id, ensure_dir, load_final,
)

OUTDIR = ensure_dir("work/step_RA0b_missingness_impact")

df, info = load_final(dedupe=True)
df = add_signal_id(df)
print(f"\n[RA0b] {len(df):,} unique variants\n")

# ---------------------------------------------------------------
# 1. Published vs corrected score
# ---------------------------------------------------------------
df["final_score_published"] = pd.to_numeric(df["final_score"], errors="coerce")

corrected = pd.Series(0.0, index=df.index)
for c, w in CURRENT_WEIGHTS.items():
    corrected = corrected + w * pd.to_numeric(df[c], errors="coerce").fillna(0.0)
df["final_score_corrected"] = corrected

df["was_excluded"] = df["final_score_published"].isna()

n_excl = int(df["was_excluded"].sum())
print("=" * 78)
print("1. HOW MANY UNIQUE VARIANTS WERE EXCLUDED?")
print("=" * 78)
print(f"unique variants with NaN published final_score : {n_excl:,} "
      f"({100.0 * n_excl / len(df):.1f}%)")

ok = ~df["was_excluded"]
if ok.sum() > 10:
    r, _ = stats.spearmanr(df.loc[ok, "final_score_published"],
                           df.loc[ok, "final_score_corrected"])
    print(f"Spearman rho (published vs corrected, on retained variants): {r:.4f}")
    print("  >> Should be 1.0: for retained variants nothing changes. The issue is")
    print("     purely which variants are present, not how they are scored.")

# ---------------------------------------------------------------
# 2. Does the top candidate change at any signal?
# ---------------------------------------------------------------
print()
print("=" * 78)
print("2. DOES THE TOP CANDIDATE CHANGE?")
print("=" * 78)

rows = []
for s, grp in df.groupby("signal_id"):
    pub = grp.dropna(subset=["final_score_published"])
    top_pub = pub.loc[pub["final_score_published"].idxmax()] if len(pub) else None
    top_cor = grp.loc[grp["final_score_corrected"].idxmax()]

    # how far up do previously-excluded variants climb?
    grp2 = grp.sort_values("final_score_corrected", ascending=False)
    grp2["rank_corrected"] = np.arange(1, len(grp2) + 1)
    best_new = grp2[grp2["was_excluded"]].head(1)

    rows.append({
        "signal_id": s,
        "n_unique_variants": len(grp),
        "n_excluded": int(grp["was_excluded"].sum()),
        "top_published": top_pub["variant_key"] if top_pub is not None else None,
        "top_published_score": round(float(top_pub["final_score_published"]), 4) if top_pub is not None else None,
        "top_corrected": top_cor["variant_key"],
        "top_corrected_score": round(float(top_cor["final_score_corrected"]), 4),
        "top_changed": bool(top_pub is None or top_pub["variant_key"] != top_cor["variant_key"]),
        "best_reentering_variant": best_new["variant_key"].iloc[0] if len(best_new) else None,
        "best_reentering_rank": int(best_new["rank_corrected"].iloc[0]) if len(best_new) else None,
        "best_reentering_score": round(float(best_new["final_score_corrected"].iloc[0]), 4) if len(best_new) else None,
    })
sig = pd.DataFrame(rows).sort_values("signal_id")
sig.to_csv(f"{OUTDIR}/RA0b_top_candidate_impact.tsv", sep="\t", index=False)
print(sig.to_string(index=False))

n_changed = int(sig["top_changed"].sum())
print()
if n_changed == 0:
    print("  >> No top candidate changes. The fix is still required for correctness")
    print("     and for the deposited code, but Table 1's headline variants stand.")
else:
    print(f"  >> {n_changed} signal(s) change top candidate. Results text, Table 1")
    print("     and figures must be regenerated after re-running steps 12-17.")

# top-20 churn even where the winner is unchanged
rows = []
for s, grp in df.groupby("signal_id"):
    pub = set(grp.dropna(subset=["final_score_published"])
                 .nlargest(20, "final_score_published")["variant_key"])
    cor = set(grp.nlargest(20, "final_score_corrected")["variant_key"])
    rows.append({
        "signal_id": s,
        "top20_overlap": len(pub & cor),
        "n_new_in_top20": len(cor - pub),
        "new_entrants": ";".join(sorted(cor - pub)[:5]),
    })
churn = pd.DataFrame(rows)
churn.to_csv(f"{OUTDIR}/RA0b_top20_churn.tsv", sep="\t", index=False)
print("\nTop-20 churn per signal:")
print(churn.to_string(index=False))

# ---------------------------------------------------------------
# 3. Were the excluded variants annotation-sparse?  (Reviewer #2 Comment 4)
# ---------------------------------------------------------------
print()
print("=" * 78)
print("3. WERE EXCLUDED VARIANTS SYSTEMATICALLY ANNOTATION-SPARSE?")
print("=" * 78)

rows = []
for c in COMPONENTS:
    v = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    a = v[df["was_excluded"]]
    b = v[~df["was_excluded"]]
    if len(a) < 10 or len(b) < 10:
        continue
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    rows.append({
        "component": c,
        "label": COMPONENT_LABELS[c],
        "median_excluded": round(float(a.median()), 4),
        "median_retained": round(float(b.median()), 4),
        "auc_excluded_higher": round(float(u / (len(a) * len(b))), 4),
        "p": p,
    })
bias = pd.DataFrame(rows)
bias.to_csv(f"{OUTDIR}/RA0b_exclusion_bias.tsv", sep="\t", index=False)
print(bias.to_string(index=False))
print()
print("  >> An AUC well below 0.5 means excluded variants scored LOWER on that")
print("     layer, i.e. exclusion was concentrated in annotation-sparse variants.")
print("     This is the direct empirical answer to Reviewer #2's Comment 4.")

# ---------------------------------------------------------------
# 4. Layer saturation -- how much does each layer actually discriminate?
# ---------------------------------------------------------------
print()
print("=" * 78)
print("4. LAYER SATURATION  (Reviewer #2 Comment 3)")
print("=" * 78)

rows = []
for c in COMPONENTS:
    v = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    w = CURRENT_WEIGHTS[c]
    p10, p90 = v.quantile(0.10), v.quantile(0.90)
    rows.append({
        "component": c,
        "label": COMPONENT_LABELS[c],
        "nominal_weight": w,
        "median": round(float(v.median()), 4),
        "pct_at_max_1.0": round(100.0 * float((v >= 0.999).mean()), 2),
        "pct_at_min_0.0": round(100.0 * float((v <= 0.001).mean()), 2),
        "p10_p90_spread": round(float(p90 - p10), 4),
        "effective_score_span": round(w * float(p90 - p10), 4),
        "pct_of_nominal_weight_used": round(100.0 * float(p90 - p10), 1),
    })
sat = pd.DataFrame(rows).sort_values("effective_score_span", ascending=False)
sat.to_csv(f"{OUTDIR}/RA0b_layer_saturation.tsv", sep="\t", index=False)
print(sat.to_string(index=False))

total_span = sat["effective_score_span"].sum()
print(f"\ntotal 10th-90th percentile span of final_score: {total_span:.4f}")
print()
print("  >> 'effective_score_span' is what each layer CAN move final_score across")
print("     the middle 80% of variants. A layer with a high nominal weight but a")
print("     small effective span is not doing the work the manuscript implies.")
print("     Report this table: it is the honest, quantitative answer to Comment 3,")
print("     and it is far more informative than the nominal weights alone.")

# ---------------------------------------------------------------
# Figure
# ---------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

ax = axes[0]
x = np.arange(len(sat))
ax.bar(x - 0.2, sat["nominal_weight"], width=0.4, label="nominal weight",
       color="#B0B0B0")
ax.bar(x + 0.2, sat["effective_score_span"], width=0.4,
       label="effective span (10th-90th pct)", color="#4C72B0")
ax.set_xticks(x)
ax.set_xticklabels(sat["label"], rotation=30, ha="right", fontsize=8)
ax.set_ylabel("contribution to final_score")
ax.set_title("Nominal weight vs realised discriminative power", fontsize=10)
ax.legend(fontsize=8, frameon=False)

ax = axes[1]
for c in COMPONENTS:
    v = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    ax.plot(np.linspace(0, 100, 101), np.percentile(v, np.linspace(0, 100, 101)),
            lw=1.6, label=COMPONENT_LABELS[c])
ax.set_xlabel("percentile of variants")
ax.set_ylabel("component score")
ax.set_title("Score distributions (flat = saturated = uninformative)", fontsize=10)
ax.legend(fontsize=7, frameon=False, loc="upper left")

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{OUTDIR}/RA0b_layer_saturation.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)

keep = ["variant_key", "signal_id", "locus_id", "p", "was_excluded"] + COMPONENTS + [
    "final_score_published", "final_score_corrected",
]
df[keep].to_csv(f"{OUTDIR}/RA0b_scored_variants.tsv.gz", sep="\t",
                index=False, compression="gzip")

print(f"\n[RA0b] done. Outputs in {OUTDIR}/")
