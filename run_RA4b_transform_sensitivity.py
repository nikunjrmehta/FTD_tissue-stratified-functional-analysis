"""
RA4b — sensitivity of the results to the GWAS p-value transform.

WHY THIS EXISTS INSTEAD OF A RESCALE
------------------------------------
Reviewer #1 asked for a reference supporting `gwas_score = -log10(p)/20` and said
the weighting "sounds subjective". There is no canonical citation for that exact
transform, and replacing 20 with some other constant would simply substitute one
unjustifiable choice for another.

The defensible response is the same one used for the weights in RA3: stop
defending the specific choice and instead show that conclusions do not depend
on it. This script varies the transform across its whole plausible range --
including the two limiting cases (saturating at genome-wide significance, and a
scale-free within-locus percentile) -- and reports whether anything changes.

It also settles a question that matters for Reviewer #2 Comment 6: can ANY
monotone transform of the GWAS p-value displace the chr19 regulatory candidate?
If not, the calibration issue is not caused by the transform, and rescaling is
not the fix -- the Tier 1 / Tier 2 association constraint is.

Outputs -> work/step_RA4b_transform_sensitivity/
"""

import numpy as np
import pandas as pd
from scipy import stats

from ra_common import (
    COMPONENTS, CURRENT_WEIGHTS,
    add_signal_id, ensure_dir, load_final,
)

OUTDIR = ensure_dir("work/step_RA4b_transform_sensitivity")

df, info = load_final(dedupe=True)
df = add_signal_id(df)

df["p_num"] = pd.to_numeric(df["p"], errors="coerce")
df["neglog10p"] = -np.log10(df["p_num"].clip(lower=1e-300))

# ---------------------------------------------------------------
# Candidate transforms
# ---------------------------------------------------------------
GW_SIG = -np.log10(5e-8)          # 7.30

def cap_transform(nl, cap):
    return (nl / cap).clip(0, 1)

transforms = {
    "cap_7.3_genomewide": lambda d: cap_transform(d["neglog10p"], GW_SIG),
    "cap_12":             lambda d: cap_transform(d["neglog10p"], 12.0),
    "cap_20_current":     lambda d: cap_transform(d["neglog10p"], 20.0),
    "cap_40":             lambda d: cap_transform(d["neglog10p"], 40.0),
    # Scale-free limiting case. Included to DEMONSTRATE why it is unacceptable:
    # in a window where nothing is significant, the best variant still scores ~1.
    "within_locus_percentile": lambda d: d.groupby("locus_id")["neglog10p"].rank(pct=True),
    # Absolute floor: no GWAS evidence at all
    "gwas_layer_removed": lambda d: pd.Series(0.0, index=d.index),
}

W = CURRENT_WEIGHTS
other = {c: w for c, w in W.items() if c != "gwas_score"}

for name, fn in transforms.items():
    g = fn(df).astype(float)
    df[f"gwas__{name}"] = g
    s = W["gwas_score"] * g
    for c, w in other.items():
        s = s + w * pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df[f"score__{name}"] = s

ref_col = "score__cap_20_current"

# ---------------------------------------------------------------
# 1. Rank agreement
# ---------------------------------------------------------------
rows = []
for name in transforms:
    col = f"score__{name}"
    ok = df[ref_col].notna() & df[col].notna()
    r, _ = stats.spearmanr(df.loc[ok, ref_col], df.loc[ok, col])
    rows.append({
        "transform": name,
        "spearman_rho_vs_current": round(float(r), 4),
        "gwas_score_min": round(float(df[f"gwas__{name}"].min()), 4),
        "gwas_score_max": round(float(df[f"gwas__{name}"].max()), 4),
        "n_saturated_at_1": int((df[f"gwas__{name}"] >= 0.999).sum()),
    })
agree = pd.DataFrame(rows)
agree.to_csv(f"{OUTDIR}/RA4b_rank_agreement.tsv", sep="\t", index=False)

print("=" * 78)
print("1. RANK AGREEMENT ACROSS GWAS TRANSFORMS")
print("=" * 78)
print(agree.to_string(index=False))

# ---------------------------------------------------------------
# 2. Top candidate per signal under each transform
# ---------------------------------------------------------------
rows = []
for s, grp in df.groupby("signal_id"):
    ref_top = grp.loc[grp[ref_col].idxmax(), "variant_key"]
    for name in transforms:
        col = f"score__{name}"
        top = grp.loc[grp[col].idxmax()]
        rows.append({
            "signal_id": s,
            "transform": name,
            "top_variant_key": top["variant_key"],
            "top_p": top["p_num"],
            "top_score": round(float(top[col]), 4),
            "changed_vs_current": bool(top["variant_key"] != ref_top),
        })
tops = pd.DataFrame(rows)
tops.to_csv(f"{OUTDIR}/RA4b_top_per_signal.tsv", sep="\t", index=False)

print()
print("=" * 78)
print("2. TOP CANDIDATE PER SIGNAL UNDER EACH TRANSFORM")
print("=" * 78)
print(tops.to_string(index=False))

n_changed = int(tops["changed_vs_current"].sum())
print(f"\ntop-candidate changes across all signals x transforms: {n_changed}")

# ---------------------------------------------------------------
# 3. THE KEY RESULT for Reviewer #2 Comment 6
#    Can any transform displace a non-significant top candidate?
# ---------------------------------------------------------------
print()
print("=" * 78)
print("3. CAN ANY TRANSFORM RESCUE A NON-SIGNIFICANT TOP CANDIDATE?")
print("=" * 78)

rows = []
for s, grp in df.groupby("signal_id"):
    for name in transforms:
        col = f"score__{name}"
        top = grp.loc[grp[col].idxmax()]
        # best variant that IS association-supported
        sup = grp[grp["p_num"] < 5e-5]
        rows.append({
            "signal_id": s,
            "transform": name,
            "top_variant_p": top["p_num"],
            "top_is_association_supported": bool(top["p_num"] < 5e-5),
            "n_association_supported": len(sup),
            "best_supported_variant": (
                sup.loc[sup[col].idxmax(), "variant_key"] if len(sup) else None
            ),
            "score_gap_to_best_supported": (
                round(float(top[col] - sup[col].max()), 4) if len(sup) else None
            ),
        })
rescue = pd.DataFrame(rows)
rescue.to_csv(f"{OUTDIR}/RA4b_association_support_check.tsv", sep="\t", index=False)
print(rescue.to_string(index=False))

never_fixed = rescue.groupby("signal_id")["top_is_association_supported"].apply(
    lambda x: not x.any()
)
print()
for s, flag in never_fixed.items():
    if flag:
        print(f"  {s}: top candidate is NOT association-supported under ANY transform.")
print()
print("  >> If a signal appears above, the calibration issue in Reviewer #2's")
print("     Comment 6 is NOT caused by the choice of transform, and cannot be")
print("     fixed by rescaling. The correct remedy is the Tier 1 / Tier 2")
print("     association constraint. State this explicitly in the response.")

# ---------------------------------------------------------------
# 4. Why the scale-free percentile transform is rejected
# ---------------------------------------------------------------
print()
print("=" * 78)
print("4. WHY THE WITHIN-LOCUS PERCENTILE TRANSFORM IS NOT USED")
print("=" * 78)
pct_col = "gwas__within_locus_percentile"
nonsig = df[df["p_num"] >= 5e-5]
if len(nonsig):
    hi = nonsig[nonsig[pct_col] >= 0.99]
    print(f"variants with p >= 5e-5 that still score >= 0.99 "
          f"under the percentile transform: {len(hi):,}")
    if len(hi):
        print(f"  worst case: p = {hi['p_num'].max():.4g} scoring "
              f"{hi[pct_col].max():.3f}")
    print("  >> A scale-free transform assigns near-maximal GWAS support to the")
    print("     best variant in a window even when nothing in it is significant.")
    print("     This makes the Comment 6 pathology worse, not better.")

sat = agree.set_index("transform").loc["cap_7.3_genomewide", "n_saturated_at_1"]
print(f"\nvariants saturating at 1.0 under a genome-wide-significance cap: {sat:,}")
print("  >> An aggressive cap removes all discrimination among the strongest")
print("     variants, which is where discrimination matters most.")

keep = ["variant_key", "signal_id", "locus_id", "p_num", "neglog10p"] + \
       [f"gwas__{n}" for n in transforms] + [f"score__{n}" for n in transforms]
df[keep].to_csv(f"{OUTDIR}/RA4b_scores_all_transforms.tsv.gz", sep="\t",
                index=False, compression="gzip")

# ---------------------------------------------------------------
# Figure
# ---------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

ax = axes[0]
pgrid = np.logspace(-14, -0.3, 400)
nl = -np.log10(pgrid)
for cap, lab in [(GW_SIG, "cap 7.3 (genome-wide)"), (12, "cap 12"),
                 (20, "cap 20 (used)"), (40, "cap 40")]:
    ax.plot(nl, np.clip(nl / cap, 0, 1), lw=1.8,
            ls="-" if cap == 20 else "--", label=lab)
ax.axvline(GW_SIG, color="grey", ls=":", lw=1)
ax.text(GW_SIG + 0.15, 0.05, "p = 5e-8", fontsize=8, color="grey")
ax.set_xlabel(r"$-\log_{10}(p)$")
ax.set_ylabel("gwas_score")
ax.set_title("Candidate transforms", fontsize=10)
ax.legend(fontsize=8, frameon=False)

ax = axes[1]
sub = agree[agree["transform"] != "cap_20_current"]
ax.barh(sub["transform"], sub["spearman_rho_vs_current"], color="#4C72B0")
for i, v in enumerate(sub["spearman_rho_vs_current"]):
    ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
ax.set_xlim(0, 1.05)
ax.invert_yaxis()
ax.set_xlabel("Spearman rho vs the transform used")
ax.set_title("Ranking is insensitive to the transform", fontsize=10)

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{OUTDIR}/RA4b_transform_sensitivity.{ext}",
                dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"\n[RA4b] done. Outputs in {OUTDIR}/")
