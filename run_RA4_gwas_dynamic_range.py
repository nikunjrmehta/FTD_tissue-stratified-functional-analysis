"""
RA4 — dynamic range of the GWAS evidence layer.

Reviewer #1, major 4:
    "...I don't remember that the GWAS signals in Manzoni et al. differed a lot
     from each other, so would it be needed to provide a score based on this?"

That is an empirical question about how much the p-values actually vary inside
the analysed locus windows, and nothing in the submitted manuscript answers it.

Reviewer #2, Comment 6:
    "...whether the 0.25 weight assigned to GWAS results is sufficient to prevent
     prioritizing variants with no significant association as top functional targets"

Same underlying quantity. Under gwas_score = clip(-log10(p)/20, 0, 1), a variant
at genome-wide significance scores 0.365 and contributes 0.091 to final_score.
This script measures the realised spread so the response letter can state the
number rather than assert a position.

NOTE: decision taken to KEEP the /20 transform as-is rather than re-cap it. That
makes this analysis load-bearing: the manuscript must show what the transform
does, and concede the calibration point in R2-C6 explicitly, because it is no
longer being fixed by rescaling. See run_RA4 output section 3.

Outputs -> work/step_RA4_gwas_dynamic_range/
"""

import numpy as np
import pandas as pd

from ra_common import (
    CURRENT_WEIGHTS, add_signal_id, ensure_dir, load_final,
)

OUTDIR = ensure_dir("work/step_RA4_gwas_dynamic_range")
W_GWAS = CURRENT_WEIGHTS["gwas_score"]

df, info = load_final(dedupe=True)
df = add_signal_id(df)

df["p_num"] = pd.to_numeric(df["p"], errors="coerce")
df["neglog10p"] = -np.log10(df["p_num"].clip(lower=1e-300))
df["gwas_score_recomputed"] = (df["neglog10p"] / 20.0).clip(0, 1)

# ---------------------------------------------------------------
# 1. Spread of -log10(p) within each signal
# ---------------------------------------------------------------
rows = []
for s, grp in df.groupby("signal_id"):
    nl = grp["neglog10p"].dropna()
    gs = grp["gwas_score_recomputed"].dropna()
    rows.append({
        "signal_id": s,
        "n_variants": len(grp),
        "neglog10p_min": round(float(nl.min()), 3),
        "neglog10p_median": round(float(nl.median()), 3),
        "neglog10p_p95": round(float(nl.quantile(0.95)), 3),
        "neglog10p_max": round(float(nl.max()), 3),
        "gwas_score_min": round(float(gs.min()), 4),
        "gwas_score_median": round(float(gs.median()), 4),
        "gwas_score_max": round(float(gs.max()), 4),
        "gwas_score_iqr": round(float(gs.quantile(0.75) - gs.quantile(0.25)), 4),
        "final_score_span_from_gwas": round(W_GWAS * float(gs.max() - gs.min()), 4),
        "pct_genome_wide_sig": round(100.0 * float((grp["p_num"] < 5e-8).mean()), 3),
        "pct_suggestive_5e-5": round(100.0 * float((grp["p_num"] < 5e-5).mean()), 3),
    })
per_signal = pd.DataFrame(rows).sort_values("signal_id")
per_signal.to_csv(f"{OUTDIR}/RA4_dynamic_range_per_signal.tsv", sep="\t", index=False)

print("=" * 78)
print("1. GWAS EVIDENCE SPREAD WITHIN EACH SIGNAL")
print("=" * 78)
print(per_signal.to_string(index=False))

# ---------------------------------------------------------------
# 2. Same, per locus window (25 windows) -- R1 asked about the loci
# ---------------------------------------------------------------
rows = []
for lid, grp in df.groupby("locus_id"):
    gs = grp["gwas_score_recomputed"].dropna()
    rows.append({
        "locus_id": lid,
        "n_variants": len(grp),
        "gwas_score_min": round(float(gs.min()), 4),
        "gwas_score_max": round(float(gs.max()), 4),
        "gwas_score_range": round(float(gs.max() - gs.min()), 4),
        "final_score_span_from_gwas": round(W_GWAS * float(gs.max() - gs.min()), 4),
    })
per_locus = pd.DataFrame(rows).sort_values("locus_id")
per_locus.to_csv(f"{OUTDIR}/RA4_dynamic_range_per_locus.tsv", sep="\t", index=False)

# ---------------------------------------------------------------
# 3. The headline number for the response letter
# ---------------------------------------------------------------
gs_all = df["gwas_score_recomputed"].dropna()
overall_span = W_GWAS * float(gs_all.max() - gs_all.min())
p10, p90 = gs_all.quantile(0.10), gs_all.quantile(0.90)
typical_span = W_GWAS * float(p90 - p10)

print()
print("=" * 78)
print("3. THE NUMBER TO PUT IN THE RESPONSE LETTER")
print("=" * 78)
print(f"gwas_score across all variants   : {gs_all.min():.4f} to {gs_all.max():.4f}")
print(f"full span of final_score from GWAS: {overall_span:.4f} of 1.0")
print(f"10th-90th percentile span        : {typical_span:.4f} of 1.0")
print()
print("Worked example for Reviewer #2, Comment 6:")
for label, p in [
    ("chr19 top regulatory candidate", 0.155),
    ("chr19 Tier-1 candidate", 5.44e-6),
    ("genome-wide significance", 5e-8),
]:
    s = min(-np.log10(p) / 20.0, 1.0)
    print(f"  p = {p:<10.3g}  gwas_score = {s:.4f}  contributes {W_GWAS * s:.4f}")
print()
print("  >> Because the /20 transform is being retained, the manuscript must state")
print("     this explicitly and concede C6 rather than argue the weight is adequate.")
print("     The associated fix is the Tier 1 / Tier 2 split, not reweighting.")

pd.DataFrame([{
    "weight_gwas": W_GWAS,
    "gwas_score_min": float(gs_all.min()),
    "gwas_score_max": float(gs_all.max()),
    "gwas_score_p10": float(p10),
    "gwas_score_p90": float(p90),
    "final_score_full_span_from_gwas": overall_span,
    "final_score_p10_p90_span_from_gwas": typical_span,
}]).to_csv(f"{OUTDIR}/RA4_headline_numbers.tsv", sep="\t", index=False)

# ---------------------------------------------------------------
# Figure: within-signal p-value distributions and the induced score spread
# ---------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

signals = sorted(df["signal_id"].dropna().unique())
fig, axes = plt.subplots(2, len(signals), figsize=(4.2 * len(signals), 7.0))
if len(signals) == 1:
    axes = axes.reshape(2, 1)

for j, s in enumerate(signals):
    grp = df[df["signal_id"] == s]

    ax = axes[0, j]
    ax.hist(grp["neglog10p"].dropna(), bins=60, color="#4C72B0")
    ax.axvline(-np.log10(5e-8), color="#C44E52", ls="--", lw=1,
               label="genome-wide (5e-8)")
    ax.axvline(-np.log10(5e-5), color="#DD8452", ls=":", lw=1,
               label="suggestive (5e-5)")
    ax.set_yscale("log")
    ax.set_xlabel(r"$-\log_{10}(p)$")
    ax.set_ylabel("variants (log scale)")
    ax.set_title(s, fontsize=10)
    if j == 0:
        ax.legend(fontsize=7, frameon=False)

    ax = axes[1, j]
    contrib = W_GWAS * grp["gwas_score_recomputed"].dropna()
    ax.hist(contrib, bins=60, color="#55A868")
    ax.set_yscale("log")
    ax.set_xlabel("contribution to final_score")
    ax.set_ylabel("variants (log scale)")
    ax.set_xlim(0, W_GWAS)
    ax.set_title(
        f"GWAS layer uses {contrib.max():.3f} of its {W_GWAS:.2f} weight",
        fontsize=9,
    )

fig.suptitle(
    "Realised dynamic range of the GWAS evidence layer within each signal",
    fontsize=11,
)
fig.tight_layout(rect=[0, 0, 1, 0.96])
for ext in ("png", "pdf"):
    fig.savefig(f"{OUTDIR}/RA4_gwas_dynamic_range.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"\n[RA4] done. Outputs in {OUTDIR}/")
