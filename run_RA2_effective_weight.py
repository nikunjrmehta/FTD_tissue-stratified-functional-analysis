"""
RA2 — effective weight per information source.

This answers the part of Reviewer #2's Comment 2 that a correlation matrix cannot
touch, and that nothing in the submitted manuscript addresses:

    "...the final score will likely assign more weight to GTEx-based scores
     (i.e. susie_score_any, gtex_score_any, expr_score_any)."

The reviewer is not making a claim about correlation magnitude. They are pointing
out that three of the six nominal layers are all derived from GTEx, so the
composite assigns 0.40 of its total weight to a single data source while GWAS and
the sequence model get 0.25 each. That is arithmetic, not statistics, and the
honest response is to state it plainly and then show whether it matters.

Two outputs:
  (a) the weight-by-source table -- put this in the response letter verbatim;
  (b) a GTEx-collapsed scheme that merges the three GTEx layers into one, so the
      composite is balanced across sources, and a report of whether the top
      candidate at each signal survives.

Outputs -> work/step_RA2_effective_weight/
"""

import numpy as np
import pandas as pd
from scipy import stats

from ra_common import (
    COMPONENTS, COMPONENT_LABELS, COMPONENT_SOURCE, CURRENT_WEIGHTS,
    add_signal_id, ensure_dir, load_final, weighted_score,
)

OUTDIR = ensure_dir("work/step_RA2_effective_weight")

df, info = load_final(dedupe=True)
df = add_signal_id(df)

# ---------------------------------------------------------------
# 1. Weight by information source
# ---------------------------------------------------------------
rows = []
for c in COMPONENTS:
    rows.append({
        "component": c,
        "label": COMPONENT_LABELS[c],
        "nominal_weight": CURRENT_WEIGHTS[c],
        "data_source": COMPONENT_SOURCE[c],
    })
comp = pd.DataFrame(rows)

by_source = (
    comp.groupby("data_source")["nominal_weight"].agg(["sum", "count"])
        .rename(columns={"sum": "total_weight", "count": "n_component_layers"})
        .sort_values("total_weight", ascending=False)
        .reset_index()
)

comp.to_csv(f"{OUTDIR}/RA2_component_source_map.tsv", sep="\t", index=False)
by_source.to_csv(f"{OUTDIR}/RA2_weight_by_source.tsv", sep="\t", index=False)

print("=" * 70)
print("EFFECTIVE WEIGHT BY INFORMATION SOURCE")
print("=" * 70)
print(by_source.to_string(index=False))
print()
print("  >> Reviewer #2 is correct: GTEx contributes 0.40 of the composite through")
print("     three separate layers, versus 0.25 each for GWAS and the sequence model.")
print("     State this in Methods rather than letting the reviewer derive it.")

# ---------------------------------------------------------------
# 2. GTEx-collapsed scheme
#    Merge SuSiE + QTL + expression into a single GTEx layer, so that each
#    information source contributes one layer, then rebalance.
# ---------------------------------------------------------------
GTEX_LAYERS = [c for c in COMPONENTS if COMPONENT_SOURCE[c] == "GTEx"]
print(f"\nGTEx layers being collapsed: {GTEX_LAYERS}")

gtex_vals = df[GTEX_LAYERS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
df["gtex_composite"] = gtex_vals.mean(axis=1)

# Balanced-by-source weights: GWAS 0.25, sequence model 0.25, GTEx 0.40, ENCODE 0.10
# collapsed into four layers preserving the original source proportions.
collapsed_weights = {
    "gwas_score": 0.25,
    "alphagenome_score": 0.25,
    "gtex_composite": 0.40,
    "ccre_score": 0.10,
}

# An equal-by-source alternative: each of the four sources gets 0.25.
equal_source_weights = {
    "gwas_score": 0.25,
    "alphagenome_score": 0.25,
    "gtex_composite": 0.25,
    "ccre_score": 0.25,
}

df["final_score_current"] = weighted_score(df, CURRENT_WEIGHTS)
df["final_score_gtex_collapsed"] = weighted_score(df, collapsed_weights)
df["final_score_equal_source"] = weighted_score(df, equal_source_weights)

# ---------------------------------------------------------------
# 3. Does it change the ranking?
# ---------------------------------------------------------------
sch_cols = {
    "current": "final_score_current",
    "gtex_collapsed": "final_score_gtex_collapsed",
    "equal_by_source": "final_score_equal_source",
}

corr_rows = []
for name, col in sch_cols.items():
    if name == "current":
        continue
    ok = df["final_score_current"].notna() & df[col].notna()
    r, _ = stats.spearmanr(df.loc[ok, "final_score_current"], df.loc[ok, col])
    t, _ = stats.kendalltau(df.loc[ok, "final_score_current"], df.loc[ok, col])
    corr_rows.append({
        "scheme": name,
        "spearman_rho_vs_current": round(float(r), 4),
        "kendall_tau_vs_current": round(float(t), 4),
        "n": int(ok.sum()),
    })
corr = pd.DataFrame(corr_rows)
corr.to_csv(f"{OUTDIR}/RA2_rank_correlation.tsv", sep="\t", index=False)

print("\nRank correlation vs the current scheme:")
print(corr.to_string(index=False))

# Top candidate per signal under each scheme, plus top-20 overlap
top_rows = []
for s, grp in df.groupby("signal_id"):
    ref_top20 = set(
        grp.nlargest(20, "final_score_current")["variant_key"]
    )
    for name, col in sch_cols.items():
        top = grp.loc[grp[col].idxmax()]
        top20 = set(grp.nlargest(20, col)["variant_key"])
        top_rows.append({
            "signal_id": s,
            "scheme": name,
            "top_variant_key": top["variant_key"],
            "top_p": top["p"],
            "top_score": round(float(top[col]), 4),
            "top20_overlap_with_current": len(top20 & ref_top20),
        })
tops = pd.DataFrame(top_rows).sort_values(["signal_id", "scheme"])
tops.to_csv(f"{OUTDIR}/RA2_top_per_signal_per_scheme.tsv", sep="\t", index=False)

print("\nTop candidate per signal under each scheme:")
print(tops.to_string(index=False))

changed = (
    tops.pivot(index="signal_id", columns="scheme", values="top_variant_key")
)
n_changed = sum(
    changed.loc[s, "current"] != changed.loc[s, c]
    for s in changed.index for c in ["gtex_collapsed", "equal_by_source"]
)
print(f"\ntop-candidate changes across {len(changed)} signals x 2 schemes: {n_changed}")
if n_changed == 0:
    print("  >> Conclusions are robust to GTEx triple-counting. Report this.")
else:
    print("  >> At least one top candidate is sensitive to GTEx weighting.")
    print("     Report it honestly; it bounds how much the triple-counting matters.")

# ---------------------------------------------------------------
# Figure
# ---------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

ax = axes[0]
ax.barh(by_source["data_source"], by_source["total_weight"], color="#4C72B0")
for i, (w, n) in enumerate(zip(by_source["total_weight"], by_source["n_component_layers"])):
    ax.text(w + 0.01, i, f"{w:.2f}  ({n} layer{'s' if n > 1 else ''})",
            va="center", fontsize=9)
ax.set_xlabel("Total weight in final_score")
ax.set_xlim(0, 0.55)
ax.set_title("Effective weight by information source", fontsize=10)

ax = axes[1]
ok = df["final_score_current"].notna() & df["final_score_gtex_collapsed"].notna()
ax.scatter(df.loc[ok, "final_score_current"], df.loc[ok, "final_score_gtex_collapsed"],
           s=3, alpha=0.15, color="#4C72B0", rasterized=True)
lim = [0, max(df["final_score_current"].max(), df["final_score_gtex_collapsed"].max()) * 1.05]
ax.plot(lim, lim, "--", color="grey", lw=1)
ax.set_xlabel("final_score (current)")
ax.set_ylabel("final_score (GTEx-collapsed)")
ax.set_title("Effect of collapsing the three GTEx layers", fontsize=10)

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{OUTDIR}/RA2_effective_weight.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)

keep = ["variant_key", "signal_id", "locus_id", "p"] + COMPONENTS + [
    "gtex_composite", "final_score_current",
    "final_score_gtex_collapsed", "final_score_equal_source",
]
df[keep].to_csv(f"{OUTDIR}/RA2_scored_variants.tsv.gz", sep="\t",
                index=False, compression="gzip")

print(f"\n[RA2] done. Outputs in {OUTDIR}/")
