"""
RA0d — is the Tier 1 (association-constrained) result stable against EVERY
arbitrary choice the referees objected to?

WHY THIS IS THE DECISIVE ANALYSIS
---------------------------------
RA0b, RA0c, RA3 and RA4b each found the same thing from a different direction:
the unconstrained ranking is sensitive to how much range the regulatory layers
have relative to the GWAS layer. Widen AlphaGenome (RA0c max_then_repercentile),
weaken the p-value transform (RA4b cap_40), or drop the GWAS layer (RA3), and a
non-significant variant takes the top slot at Signal 2 or Signal 3.

That is a real property of an additive composite and it cannot be reweighted away.
But the revision is adopting a Tier 1 / Tier 2 split, where the PRIMARY reported
candidate at each signal is drawn only from association-supported variants.

So the question that actually matters for the manuscript is not "which
summarisation / weighting / transform is correct?" -- it is:

    Once the association constraint is applied, does ANY of these choices
    change the primary candidate?

If the answer is no, then the arbitrariness objections (R1 major 4, R2 Comments
3 and 4, R3 major 1) do not affect the primary results, and that can be stated
with evidence rather than asserted. If the answer is yes, the manuscript must
report which choices matter and why.

Outputs -> work/step_RA0d_tier1_stability/
"""

import itertools

import numpy as np
import pandas as pd
from scipy import stats

from ra_common import (
    COMPONENTS, CURRENT_WEIGHTS,
    add_signal_id, ensure_dir, load_final, renormalise,
)

OUTDIR = ensure_dir("work/step_RA0d_tier1_stability")

# Tier 1 definition. 5e-5 is the suggestive threshold used in the revision;
# 5e-8 is reported alongside as the strict alternative.
TIER1_THRESHOLDS = [5e-5, 5e-8]

df, info = load_final(dedupe=True)
df = add_signal_id(df)
df["p_num"] = pd.to_numeric(df["p"], errors="coerce")
df["neglog10p"] = -np.log10(df["p_num"].clip(lower=1e-300))

print(f"\n[RA0d] {len(df):,} unique variants")
for t in TIER1_THRESHOLDS:
    print(f"  association-supported at p < {t:g}: {int((df['p_num'] < t).sum()):,}")

# ---------------------------------------------------------------
# Build the three axes of arbitrary choice
# ---------------------------------------------------------------

# --- axis 1: AlphaGenome summarisation ---
pct_cols = sorted(
    [c for c in df.columns if c.startswith("scorer") and c.endswith("_pct")],
    key=lambda x: int(x.replace("scorer", "").replace("_pct", "")),
)
pct_cols = [c for c in pct_cols if pd.to_numeric(df[c], errors="coerce").notna().any()]
P = df[pct_cols].apply(pd.to_numeric, errors="coerce")
_max_raw = P.max(axis=1)

ag_variants = {
    "max_current": _max_raw,
    "mean": P.mean(axis=1),
    "median": P.median(axis=1),
    "q75": P.quantile(0.75, axis=1),
    "top3_mean": P.apply(lambda r: r.nlargest(3).mean(), axis=1),
    "max_then_repercentile": _max_raw.groupby(df["locus_id"]).rank(pct=True, method="average"),
}

# --- axis 2: weighting scheme ---
_rest = {c: w for c, w in CURRENT_WEIGHTS.items() if c != "gwas_score"}
_s = 0.50 / sum(_rest.values())
gwas_dom = {"gwas_score": 0.50, **{c: w * _s for c, w in _rest.items()}}
_rest = {c: w for c, w in CURRENT_WEIGHTS.items() if c != "alphagenome_score"}
_s = 0.50 / sum(_rest.values())
reg_dom = {"alphagenome_score": 0.50, **{c: w * _s for c, w in _rest.items()}}

weight_variants = {
    "current": dict(CURRENT_WEIGHTS),
    "equal": {c: 1.0 / len(COMPONENTS) for c in COMPONENTS},
    "gwas_dominant": gwas_dom,
    "regulatory_dominant": reg_dom,
}
for c in COMPONENTS:
    weight_variants[f"drop_{c}"] = renormalise(
        {k: v for k, v in CURRENT_WEIGHTS.items() if k != c}
    )

# --- axis 3: GWAS transform ---
transform_variants = {
    "cap_7.3": lambda d: (d["neglog10p"] / 7.30).clip(0, 1),
    "cap_12": lambda d: (d["neglog10p"] / 12.0).clip(0, 1),
    "cap_20_current": lambda d: (d["neglog10p"] / 20.0).clip(0, 1),
    "cap_40": lambda d: (d["neglog10p"] / 40.0).clip(0, 1),
}

print(f"\ngrid: {len(ag_variants)} summarisations x {len(weight_variants)} weightings "
      f"x {len(transform_variants)} transforms = "
      f"{len(ag_variants) * len(weight_variants) * len(transform_variants)} models")

# ---------------------------------------------------------------
# Evaluate the full grid, constrained and unconstrained
# ---------------------------------------------------------------
rows = []
for ag_name, ag_vals in ag_variants.items():
    ag = pd.to_numeric(ag_vals, errors="coerce").fillna(0.0)
    for tr_name, tr_fn in transform_variants.items():
        gw = pd.to_numeric(tr_fn(df), errors="coerce").fillna(0.0)
        for w_name, w in weight_variants.items():
            score = pd.Series(0.0, index=df.index)
            for c, wt in w.items():
                if wt == 0:
                    continue
                if c == "alphagenome_score":
                    v = ag
                elif c == "gwas_score":
                    v = gw
                else:
                    v = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
                score = score + wt * v
            df["_s"] = score

            for thresh in TIER1_THRESHOLDS + [None]:
                sub = df if thresh is None else df[df["p_num"] < thresh]
                for sig, grp in sub.groupby("signal_id"):
                    if len(grp) == 0:
                        continue
                    srt = grp.sort_values(["_s", "variant_key"], ascending=[False, True])
                    top = srt.iloc[0]
                    rows.append({
                        "tier": "unconstrained" if thresh is None else f"tier1_p<{thresh:g}",
                        "signal_id": sig,
                        "ag_summarisation": ag_name,
                        "weighting": w_name,
                        "transform": tr_name,
                        "n_candidates": len(grp),
                        "top_variant_key": top["variant_key"],
                        "top_p": top["p_num"],
                        "top_is_association_supported": bool(top["p_num"] < 5e-5),
                    })

grid = pd.DataFrame(rows)
grid.to_csv(f"{OUTDIR}/RA0d_full_grid.tsv.gz", sep="\t", index=False, compression="gzip")
print(f"\nevaluated {len(grid):,} signal-level results")

# ---------------------------------------------------------------
# Stability summary
# ---------------------------------------------------------------
print()
print("=" * 78)
print("STABILITY OF THE TOP CANDIDATE, BY TIER")
print("=" * 78)

summ = (
    grid.groupby(["tier", "signal_id"])
        .agg(n_models=("top_variant_key", "size"),
             n_distinct_top_variants=("top_variant_key", "nunique"),
             modal_top_variant=("top_variant_key", lambda x: x.mode().iloc[0]),
             modal_frequency=("top_variant_key", lambda x: int((x == x.mode().iloc[0]).sum())),
             pct_models_association_supported=("top_is_association_supported",
                                               lambda x: round(100.0 * x.mean(), 1)))
        .reset_index()
)
summ["pct_models_agreeing"] = (
    100.0 * summ["modal_frequency"] / summ["n_models"]
).round(1)
summ.to_csv(f"{OUTDIR}/RA0d_stability_summary.tsv", sep="\t", index=False)
print(summ.to_string(index=False))

print()
print("  >> Read the 'tier1' rows against the 'unconstrained' rows. If the")
print("     constrained result is invariant while the unconstrained one is not,")
print("     that is the empirical justification for the Tier 1 / Tier 2 design,")
print("     and it answers R1 major 4, R2 Comments 3-4 and R3 major 1 at once.")

# Which axis causes any instability that remains?
print()
print("=" * 78)
print("WHICH CHOICE DRIVES INSTABILITY?")
print("=" * 78)
rows = []
for tier in grid["tier"].unique():
    g = grid[grid["tier"] == tier]
    for axis in ["ag_summarisation", "weighting", "transform"]:
        others = [a for a in ["ag_summarisation", "weighting", "transform"] if a != axis]
        # for each fixed setting of the other axes, how often does varying THIS axis
        # change the top variant?
        changed = (
            g.groupby(["signal_id"] + others)["top_variant_key"]
             .nunique().gt(1).mean()
        )
        rows.append({
            "tier": tier,
            "axis_varied": axis,
            "pct_configurations_where_top_changes": round(100.0 * float(changed), 1),
        })
drivers = pd.DataFrame(rows)
drivers.to_csv(f"{OUTDIR}/RA0d_instability_drivers.tsv", sep="\t", index=False)
print(drivers.to_string(index=False))

# Per-signal detail for the manuscript
detail = (
    grid[grid["tier"] == f"tier1_p<{TIER1_THRESHOLDS[0]:g}"]
    .groupby(["signal_id", "top_variant_key"])
    .agg(n_models=("top_p", "size"), top_p=("top_p", "first"))
    .reset_index()
    .sort_values(["signal_id", "n_models"], ascending=[True, False])
)
detail.to_csv(f"{OUTDIR}/RA0d_tier1_candidates.tsv", sep="\t", index=False)
print(f"\nTier 1 (p < {TIER1_THRESHOLDS[0]:g}) candidates by model support:")
print(detail.to_string(index=False))

# ---------------------------------------------------------------
# Figure
# ---------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 4.8))
order = ["unconstrained"] + [f"tier1_p<{t:g}" for t in TIER1_THRESHOLDS]
sigs = sorted(grid["signal_id"].unique())
width = 0.8 / len(sigs)
x = np.arange(len(order))
for i, sig in enumerate(sigs):
    vals = [
        summ[(summ["tier"] == t) & (summ["signal_id"] == sig)]["pct_models_agreeing"].iloc[0]
        if len(summ[(summ["tier"] == t) & (summ["signal_id"] == sig)]) else 0
        for t in order
    ]
    ax.bar(x + i * width - 0.4, vals, width=width, label=sig)
ax.set_xticks(x)
ax.set_xticklabels(order)
ax.set_ylabel("% of models agreeing on the top candidate")
ax.set_ylim(0, 105)
ax.axhline(100, color="grey", ls=":", lw=1)
ax.set_title(
    f"Stability of the top candidate across {grid.groupby('tier').size().iloc[0] // len(sigs)} "
    "model specifications",
    fontsize=10,
)
ax.legend(fontsize=8, frameon=False)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{OUTDIR}/RA0d_tier1_stability.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"\n[RA0d] done. Outputs in {OUTDIR}/")
