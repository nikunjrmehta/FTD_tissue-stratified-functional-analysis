"""
RA3 — weighting sensitivity and leave-one-out ablation.

Addresses:
  Reviewer #1, major 4  : "the weights used in the score calculation ... sound subjective"
  Reviewer #2, Comment 3: "does combining all six scores provide more accuracy in
                           prioritizing signals than using only a subset of them?
                           ... would a simpler combination of GWAS + SuSiE PIP
                           produce equivalent top candidates?"
  Reviewer #2, Comment 4: "How were these specific weights determined?"
  Reviewer #3, major 1  : "include a sensitivity analysis to show how top ranked
                           candidates change across alternative weighting schemes,
                           i.e. equal weighting, association constrained models
                           and/or models that down weight or exclude individual
                           evidence layers"

TWO WARNINGS, both of which a statistical-genetics referee will apply:

 1. REPORT EVERY SCHEME, INCLUDING gwas_susie_only. That is the exact comparator
    Reviewer #2 named. If it has the lowest concordance, say so and explain what
    it means -- do not quote a concordance floor that silently excludes it.

 2. CONCORDANCE IS NOT ACCURACY. Spearman rho against the current scheme cannot
    answer "does using all six give more accuracy", because it presupposes the
    current scheme is correct. This script writes the per-scheme scores to disk so
    RA5 can evaluate each one by AUROC against an external positive set. RA3 alone
    answers STABILITY; RA3 + RA5 answers CALIBRATION.

Outputs -> work/step_RA3_weight_sensitivity/
"""

import numpy as np
import pandas as pd
from scipy import stats

from ra_common import (
    COMPONENTS, COMPONENT_LABELS, CURRENT_WEIGHTS,
    add_signal_id, ensure_dir, load_final, renormalise, weighted_score,
)

OUTDIR = ensure_dir("work/step_RA3_weight_sensitivity")

df, info = load_final(dedupe=True)
df = add_signal_id(df)

# ---------------------------------------------------------------
# Define the schemes
# ---------------------------------------------------------------
schemes = {}

schemes["current"] = dict(CURRENT_WEIGHTS)

schemes["equal"] = {c: 1.0 / len(COMPONENTS) for c in COMPONENTS}

# GWAS-dominant: GWAS at 0.50, remaining 0.50 split in the current proportions
_rest = {c: w for c, w in CURRENT_WEIGHTS.items() if c != "gwas_score"}
_scale = 0.50 / sum(_rest.values())
schemes["gwas_dominant"] = {"gwas_score": 0.50, **{c: w * _scale for c, w in _rest.items()}}

# Regulatory-dominant: AlphaGenome at 0.50, remainder in current proportions
_rest = {c: w for c, w in CURRENT_WEIGHTS.items() if c != "alphagenome_score"}
_scale = 0.50 / sum(_rest.values())
schemes["regulatory_dominant"] = {
    "alphagenome_score": 0.50, **{c: w * _scale for c, w in _rest.items()}
}

# The comparator Reviewer #2 explicitly proposed
schemes["gwas_susie_only"] = {"gwas_score": 0.50, "susie_score_any": 0.50}

# Leave-one-out: drop each layer, renormalise the rest
for c in COMPONENTS:
    dropped = {k: v for k, v in CURRENT_WEIGHTS.items() if k != c}
    schemes[f"drop_{c}"] = renormalise(dropped)

# Leave-one-in: each layer alone, as a lower bound on what a single layer achieves
for c in COMPONENTS:
    schemes[f"only_{c}"] = {c: 1.0}

print(f"[RA3] {len(schemes)} weighting schemes over {len(df):,} unique variants\n")

# ---------------------------------------------------------------
# Score under every scheme
# ---------------------------------------------------------------
score_cols = {}
for name, w in schemes.items():
    col = f"score__{name}"
    df[col] = weighted_score(df, w)
    score_cols[name] = col

# Record the weight table so the supplement can show exactly what was run
wrows = []
for name, w in schemes.items():
    row = {"scheme": name}
    for c in COMPONENTS:
        row[c] = round(w.get(c, 0.0), 4)
    row["sum"] = round(sum(w.values()), 4)
    wrows.append(row)
weight_table = pd.DataFrame(wrows)
weight_table.to_csv(f"{OUTDIR}/RA3_scheme_weights.tsv", sep="\t", index=False)
print("Scheme weights:")
print(weight_table.to_string(index=False))

# ---------------------------------------------------------------
# Rank agreement vs current
# ---------------------------------------------------------------
ref = df[score_cols["current"]]
rows = []
for name, col in score_cols.items():
    ok = ref.notna() & df[col].notna()
    r, _ = stats.spearmanr(ref[ok], df.loc[ok, col])
    t, _ = stats.kendalltau(ref[ok], df.loc[ok, col])
    rows.append({
        "scheme": name,
        "spearman_rho_vs_current": round(float(r), 4),
        "kendall_tau_vs_current": round(float(t), 4),
        "n": int(ok.sum()),
    })
agree = pd.DataFrame(rows).sort_values("spearman_rho_vs_current", ascending=False)
agree.to_csv(f"{OUTDIR}/RA3_rank_agreement.tsv", sep="\t", index=False)

print("\nRank agreement with the current scheme:")
print(agree.to_string(index=False))

lo = agree.loc[agree["scheme"] != "current", "spearman_rho_vs_current"].min()
print(f"\nlowest concordance across ALL alternative schemes: rho = {lo:.3f}")
print("  >> Report this number including gwas_susie_only. Do not quote a floor")
print("     that excludes the comparator the reviewer proposed.")

# ---------------------------------------------------------------
# Top candidate and top-20 overlap per signal
# ---------------------------------------------------------------
rows = []
for s, grp in df.groupby("signal_id"):
    ref_top20 = set(grp.nlargest(20, score_cols["current"])["variant_key"])
    ref_top = grp.loc[grp[score_cols["current"]].idxmax(), "variant_key"]
    for name, col in score_cols.items():
        # Deterministic tie-break. idxmax() silently returns whichever row happens
        # to come first, which for a near-binary layer (ccre, or gtex where 48% of
        # variants sit at 1.0) is an artefact of row order, not a result. Sorting
        # by variant_key makes it reproducible, and n_tied_at_top makes it visible.
        srt = grp.sort_values([col, "variant_key"], ascending=[False, True])
        top = srt.iloc[0]
        n_tied = int((grp[col] == top[col]).sum())
        top20 = set(srt.head(20)["variant_key"])
        rows.append({
            "signal_id": s,
            "scheme": name,
            "top_variant_key": top["variant_key"],
            "top_p": top["p"],
            "top_score": round(float(top[col]), 4),
            "n_tied_at_top": n_tied,
            "top_is_arbitrary_tie": bool(n_tied > 1),
            "top_changed_vs_current": bool(top["variant_key"] != ref_top),
            "top20_overlap_with_current": len(top20 & ref_top20),
        })
tops = pd.DataFrame(rows)
tops.to_csv(f"{OUTDIR}/RA3_top_per_signal_per_scheme.tsv", sep="\t", index=False)

summary = (
    tops[tops["scheme"] != "current"]
    .groupby("scheme")
    .agg(n_signals_top_changed=("top_changed_vs_current", "sum"),
         n_signals_top_is_tie=("top_is_arbitrary_tie", "sum"),
         mean_top20_overlap=("top20_overlap_with_current", "mean"))
    .reset_index()
    .sort_values(["n_signals_top_changed", "mean_top20_overlap"],
                 ascending=[True, False])
)
# Single-layer schemes are diagnostics, not candidate models. Where the layer is
# binary or heavily tied, the "top variant" is meaningless -- do not report it.
summary["report_top_candidate"] = ~(
    summary["scheme"].str.startswith("only_") & (summary["n_signals_top_is_tie"] > 0)
)
summary.to_csv(f"{OUTDIR}/RA3_scheme_summary.tsv", sep="\t", index=False)

print("\nPer-scheme summary (3 signals):")
print(summary.to_string(index=False))

# ---------------------------------------------------------------
# Persist per-scheme scores for RA5 (AUROC evaluation)
# ---------------------------------------------------------------
keep = ["variant_key", "signal_id", "locus_id", "p"] + COMPONENTS + list(score_cols.values())
df[keep].to_csv(f"{OUTDIR}/RA3_scores_all_schemes.tsv.gz", sep="\t",
                index=False, compression="gzip")
print("\n[RA3] per-scheme scores written for RA5 to evaluate by AUROC.")

# ---------------------------------------------------------------
# Figure: primary schemes only (LOO/only_* go to the supplement table)
# ---------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

primary = ["equal", "regulatory_dominant", "gwas_dominant", "gwas_susie_only"]
loo = [f"drop_{c}" for c in COMPONENTS]
show = primary + loo
sub = agree[agree["scheme"].isin(show)].set_index("scheme").loc[show].reset_index()

nice = {f"drop_{c}": f"without {COMPONENT_LABELS[c]}" for c in COMPONENTS}
nice.update({
    "equal": "Equal weights",
    "regulatory_dominant": "AlphaGenome-dominant",
    "gwas_dominant": "GWAS-dominant",
    "gwas_susie_only": "GWAS + SuSiE only",
})
sub["label"] = sub["scheme"].map(nice)

fig, ax = plt.subplots(figsize=(8.2, 5.2))
colors = ["#C44E52" if s == "gwas_susie_only" else "#4C72B0" for s in sub["scheme"]]
ax.barh(sub["label"], sub["spearman_rho_vs_current"], color=colors)
for i, v in enumerate(sub["spearman_rho_vs_current"]):
    ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
ax.set_xlabel("Spearman rho of variant ranking vs the primary scheme")
ax.set_xlim(0, 1.05)
ax.invert_yaxis()
ax.set_title(
    "Sensitivity of variant ranking to the weighting scheme\n"
    f"({len(df):,} unique variants; red = the comparator proposed by Reviewer #2)",
    fontsize=10,
)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{OUTDIR}/RA3_rank_agreement.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"\n[RA3] done. Outputs in {OUTDIR}/")
