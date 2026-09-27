"""
RA8a — how separated are brain and blood evidence, and what does it mean?

Reviewer #3, major 5:
    "Figures 3 and Supplementary Figure 1 appear to show only modest differences
     between brain and blood-derived scores for many variants, making it difficult
     to understand how tissue-specific information contributes to prioritization.
     It would be good to expand the discussion to how brain and blood evidence
     should be interpreted biologically."

Two halves, and only one of them is a writing task. The first half -- "the
differences appear modest" -- is a claim about the numbers, and the referee can
check it. So quantify it first and let the answer determine what the Discussion
can honestly say. If the separation really is modest, say so; overclaiming
tissue specificity that the data does not support is the failure mode here.

Also produces the distribution figure that Communications Biology requires
(individual points, not summaries) -- editorial item E4.

Outputs -> work/step_RA8a_brain_blood/
"""

import numpy as np
import pandas as pd
from scipy import stats

from ra_common import add_signal_id, ensure_dir, load_final

OUTDIR = ensure_dir("work/step_RA8a_brain_blood")
MEANINGFUL_DIFF = 0.10          # what counts as a real difference, stated up front

df, info = load_final(dedupe=True)
df = add_signal_id(df)
for c in ["brain_evidence_score", "blood_evidence_score",
          "susie_brain_score", "susie_blood_score",
          "gtex_brain_score", "gtex_blood_score",
          "expr_brain_score", "expr_blood_score"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

df["tissue_diff"] = df["brain_evidence_score"] - df["blood_evidence_score"]
print(f"\n[RA8a] {len(df):,} unique variants\n")

# ---------------------------------------------------------------
# 1. Is the separation real, and how large?
# ---------------------------------------------------------------
print("=" * 74)
print("1. BRAIN vs BLOOD SEPARATION")
print("=" * 74)

rows = []
for s, grp in df.groupby("signal_id"):
    b, d = grp["brain_evidence_score"], grp["blood_evidence_score"]
    try:
        w, p = stats.wilcoxon(b, d, zero_method="zsplit")
    except ValueError:
        w, p = np.nan, np.nan
    diff = grp["tissue_diff"]
    # rank-biserial correlation: interpretable paired effect size
    n_pos, n_neg = int((diff > 0).sum()), int((diff < 0).sum())
    rbc = (n_pos - n_neg) / max(n_pos + n_neg, 1)
    rows.append({
        "signal_id": s,
        "n": len(grp),
        "median_brain": round(float(b.median()), 4),
        "median_blood": round(float(d.median()), 4),
        "median_difference": round(float(diff.median()), 4),
        "wilcoxon_p": p,
        "rank_biserial_effect_size": round(rbc, 4),
        "pct_brain_biased": round(100.0 * n_pos / len(grp), 1),
        "pct_blood_biased": round(100.0 * n_neg / len(grp), 1),
        f"pct_diff_gt_{MEANINGFUL_DIFF}": round(
            100.0 * float((diff.abs() > MEANINGFUL_DIFF).mean()), 1),
    })
sep = pd.DataFrame(rows).sort_values("signal_id")
sep.to_csv(f"{OUTDIR}/RA8a_separation_by_signal.tsv", sep="\t", index=False)
print(sep.to_string(index=False))
print()
print(f"  >> 'pct_diff_gt_{MEANINGFUL_DIFF}' is the honest answer to the referee.")
print("     A low value means the differences ARE modest for most variants, and")
print("     the Discussion must say so rather than claim tissue specificity the")
print("     data does not support.")

# ---------------------------------------------------------------
# 2. Which layer drives the separation?
# ---------------------------------------------------------------
print()
print("=" * 74)
print("2. WHICH LAYER CREATES THE TISSUE DIFFERENCE?")
print("=" * 74)
rows = []
for s, grp in df.groupby("signal_id"):
    for lab, bcol, dcol, w in [
        ("SuSiE fine-mapping", "susie_brain_score", "susie_blood_score", 0.40),
        ("GTEx QTL", "gtex_brain_score", "gtex_blood_score", 0.35),
        ("Expression", "expr_brain_score", "expr_blood_score", 0.25),
    ]:
        d = grp[bcol] - grp[dcol]
        rows.append({
            "signal_id": s, "layer": lab, "weight": w,
            "median_abs_difference": round(float(d.abs().median()), 4),
            "weighted_median_abs_difference": round(w * float(d.abs().median()), 4),
            "pct_nonzero_difference": round(100.0 * float((d.abs() > 1e-9).mean()), 1),
        })
lay = pd.DataFrame(rows)
lay.to_csv(f"{OUTDIR}/RA8a_layer_contribution.tsv", sep="\t", index=False)
print(lay.to_string(index=False))

# ---------------------------------------------------------------
# 3. Top candidates specifically
# ---------------------------------------------------------------
print()
print("=" * 74)
print("3. TISSUE ASSIGNMENT OF THE REPORTED CANDIDATES")
print("=" * 74)
try:
    cons = pd.read_csv("work/step_RA0e_consensus/RA0e_candidate_sets_tier1.tsv",
                       sep="\t")
    cols = ["variant_key", "brain_evidence_score", "blood_evidence_score",
            "tissue_diff", "susie_best_group_any"]
    cols = [c for c in cols if c in df.columns]
    m = cons.merge(df[cols], on="variant_key", how="left")
    m["tissue_call"] = np.where(
        m["tissue_diff"].abs() <= MEANINGFUL_DIFF, "ambiguous",
        np.where(m["tissue_diff"] > 0, "brain", "blood/immune"))
    show = ["signal_id", "variant_key", "pct_specs_in_top1",
            "brain_evidence_score", "blood_evidence_score", "tissue_call"]
    print(m[show].round(4).to_string(index=False))
    m.to_csv(f"{OUTDIR}/RA8a_candidate_tissue_assignment.tsv", sep="\t", index=False)
    n_amb = int((m["tissue_call"] == "ambiguous").sum())
    print(f"\ncandidates with ambiguous tissue assignment: {n_amb} of {len(m)}")
    if n_amb:
        print("  >> Report these as tissue-ambiguous. Assigning a tissue to a")
        print("     variant whose brain and blood scores differ by <0.1 is not")
        print("     supportable, and the referee has already noticed.")
except FileNotFoundError:
    print("(run RA0e first)")

# ---------------------------------------------------------------
# 4. Figure — distributions with individual points (editorial item E4)
# ---------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

signals = sorted(df["signal_id"].dropna().unique())
fig, axes = plt.subplots(1, len(signals) + 1, figsize=(4.3 * (len(signals) + 1), 4.6))
rng = np.random.default_rng(0)

for ax, s in zip(axes[:-1], signals):
    grp = df[df["signal_id"] == s]
    data = [grp["brain_evidence_score"].values, grp["blood_evidence_score"].values]
    # matplotlib renamed `labels` to `tick_labels` in 3.9 and removes the old
    # name in 3.11; try the new one first so this keeps working after an upgrade
    try:
        bp = ax.boxplot(data, tick_labels=["brain", "blood/immune"], widths=0.5,
                        showfliers=False, patch_artist=True)
    except TypeError:
        bp = ax.boxplot(data, labels=["brain", "blood/immune"], widths=0.5,
                        showfliers=False, patch_artist=True)
    for patch, col in zip(bp["boxes"], ["#4C72B0", "#DD8452"]):
        patch.set_facecolor(col)
        patch.set_alpha(0.35)
    # individual points, jittered -- required by the journal
    for i, v in enumerate(data, start=1):
        idx = rng.choice(len(v), size=min(len(v), 400), replace=False)
        ax.scatter(np.full(len(idx), i) + rng.normal(0, 0.06, len(idx)),
                   v[idx], s=5, alpha=0.30,
                   color=["#4C72B0", "#DD8452"][i - 1], rasterized=True)
    ax.set_ylabel("tissue evidence score")
    ax.set_title(s, fontsize=9)

ax = axes[-1]
for s, col in zip(signals, ["#4C72B0", "#DD8452", "#55A868"]):
    ax.hist(df.loc[df["signal_id"] == s, "tissue_diff"], bins=60,
            histtype="step", lw=1.6, label=s, color=col)
ax.axvline(0, color="grey", lw=1)
ax.axvspan(-MEANINGFUL_DIFF, MEANINGFUL_DIFF, color="grey", alpha=0.15)
ax.set_xlabel("brain − blood evidence")
ax.set_ylabel("variants")
ax.set_title(f"shaded = |difference| < {MEANINGFUL_DIFF}\n(tissue-ambiguous)",
             fontsize=9)
ax.legend(fontsize=7, frameon=False)

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{OUTDIR}/RA8a_brain_blood.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)

df[["variant_key", "signal_id", "brain_evidence_score", "blood_evidence_score",
    "tissue_diff"]].to_csv(f"{OUTDIR}/RA8a_source_data.tsv.gz", sep="\t",
                           index=False, compression="gzip")

print(f"\n[RA8a] done. Outputs in {OUTDIR}/")
print("Figure uses boxplots WITH individual points, satisfying editorial item E4.")
