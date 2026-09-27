"""
RA9 — AlphaGenome vs Enformer, head to head.

Run AFTER enformer_colab_RA9.py has produced RA9_enformer_scores.tsv and you have
placed it in work/step_RA5b_benchmark/.

Reviewer #2, Comment 1 asks whether replacing AlphaGenome with an alternative
model changes the identified candidates. This answers it two ways:

  1. PREDICTIVE VALIDITY (primary). Both models are scored on the same matched
     fine-mapped benchmark from RA5b, with the same positives and negatives.
     Neither model contributed to the positive set, so the AUROCs are directly
     comparable. If the confidence intervals overlap, the honest conclusion is
     that the two models are indistinguishable on this task -- which IS the
     answer to the reviewer's question.

  2. AGREEMENT. Spearman correlation between the two models' variant scores, so
     the answer is not resting on AUROC alone.

Outputs -> work/step_RA9_enformer/
"""

import os

import numpy as np
import pandas as pd
from scipy import stats

from ra_common import ensure_dir

OUTDIR = ensure_dir("work/step_RA9_enformer")

BENCH = "work/step_RA5b_benchmark/RA5b_benchmark_variants_for_enformer.tsv"
ENF = "work/step_RA5b_benchmark/RA9_enformer_scores.tsv"
N_BOOTSTRAP = 1000
RNG = np.random.default_rng(31415926)

if not os.path.exists(ENF):
    raise SystemExit(
        f"{ENF} not found.\n"
        "Run enformer_colab_RA9.py in Colab first, then place the downloaded\n"
        "RA9_enformer_scores.tsv in work/step_RA5b_benchmark/."
    )

bench = pd.read_csv(BENCH, sep="\t")
enf = pd.read_csv(ENF, sep="\t")
df = bench.merge(enf, on="variant_key", how="left")

print(f"benchmark variants : {len(bench):,}")
print(f"Enformer scored    : {enf['enformer_status'].eq('ok').sum():,} ok, "
      f"{enf['enformer_status'].ne('ok').sum():,} flagged")

# ---------------------------------------------------------------
# QC: reference-allele mismatches are a liftover signal, not just noise
# ---------------------------------------------------------------
if "enformer_status" in df.columns:
    mism = df[df["enformer_status"] == "ref_mismatch"]
    if len(mism):
        print()
        print("!" * 78)
        print(f"WARNING: {len(mism):,} variants have a ref allele that does not match")
        print("GRCh38. That points at a liftover or allele-orientation problem in the")
        print("main pipeline and is worth investigating on its own -- a referee running")
        print("the deposited code would hit the same thing. Excluded from the")
        print("comparison below, but do not simply discard the finding.")
        print("!" * 78)
        mism.to_csv(f"{OUTDIR}/RA9_ref_mismatch_variants.tsv", sep="\t", index=False)

df = df[df["enformer_status"] == "ok"].copy()

ENF_COLS = [c for c in df.columns if c.startswith("enformer_") and c != "enformer_status"]


def auroc_ci(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    u, _ = stats.mannwhitneyu(pos, neg, alternative="two-sided")
    auc = u / (len(pos) * len(neg))
    b = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        a = RNG.choice(pos, len(pos), replace=True)
        c = RNG.choice(neg, len(neg), replace=True)
        uu, _ = stats.mannwhitneyu(a, c, alternative="two-sided")
        b[i] = uu / (len(a) * len(c))
    return auc, np.percentile(b, 2.5), np.percentile(b, 97.5)


# ---------------------------------------------------------------
# 1. Head-to-head AUROC, per PIP threshold
# ---------------------------------------------------------------
rows = []
for thresh, grp in df.groupby("pip_threshold"):
    pos = grp[grp["is_positive"] == 1]
    neg = grp[grp["is_positive"] == 0]
    if len(pos) < 20 or len(neg) < 20:
        continue
    for name, col in [("AlphaGenome", "alphagenome_score")] + \
                     [(c.replace("enformer_", "Enformer "), c) for c in ENF_COLS]:
        p = pos[col].dropna()
        n = neg[col].dropna()
        if len(p) < 20 or len(n) < 20:
            continue
        a, lo, hi = auroc_ci(p, n)
        rows.append({
            "pip_threshold": thresh, "model": name,
            "n_pos": len(p), "n_neg": len(n),
            "auroc": round(a, 4), "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
        })

res = pd.DataFrame(rows).sort_values(["pip_threshold", "auroc"], ascending=[True, False])
res.to_csv(f"{OUTDIR}/RA9_auroc_comparison.tsv", sep="\t", index=False)

print()
print("=" * 78)
print("1. PREDICTIVE VALIDITY — AlphaGenome vs Enformer")
print("=" * 78)
print(res.to_string(index=False))

# Do the intervals overlap?
print()
for thresh, grp in res.groupby("pip_threshold"):
    ag = grp[grp["model"] == "AlphaGenome"]
    if not len(ag):
        continue
    ag = ag.iloc[0]
    best_enf = grp[grp["model"].str.startswith("Enformer")].head(1)
    if not len(best_enf):
        continue
    be = best_enf.iloc[0]
    overlap = not (ag["ci95_high"] < be["ci95_low"] or be["ci95_high"] < ag["ci95_low"])
    print(f"PIP>={thresh}: AlphaGenome {ag['auroc']:.3f} "
          f"[{ag['ci95_low']:.3f}-{ag['ci95_high']:.3f}]  vs  "
          f"best Enformer {be['auroc']:.3f} "
          f"[{be['ci95_low']:.3f}-{be['ci95_high']:.3f}]  -> "
          f"{'INDISTINGUISHABLE' if overlap else 'DIFFERENT'}")

print()
print("  >> If indistinguishable, that is a complete and defensible answer to")
print("     Comment 1: the two models perform equivalently on an external")
print("     standard, so the choice between them does not drive the results.")
print("     Report it as a positive finding, not as a null result.")

# ---------------------------------------------------------------
# 2. Agreement between the two models
# ---------------------------------------------------------------
rows = []
for col in ENF_COLS:
    ok = df["alphagenome_score"].notna() & df[col].notna()
    if ok.sum() < 30:
        continue
    r, p = stats.spearmanr(df.loc[ok, "alphagenome_score"], df.loc[ok, col])
    rows.append({"enformer_summarisation": col, "n": int(ok.sum()),
                 "spearman_rho_vs_alphagenome": round(float(r), 4), "p": p})
agree = pd.DataFrame(rows).sort_values("spearman_rho_vs_alphagenome", ascending=False)
agree.to_csv(f"{OUTDIR}/RA9_model_agreement.tsv", sep="\t", index=False)

print()
print("=" * 78)
print("2. AGREEMENT BETWEEN MODELS")
print("=" * 78)
print(agree.to_string(index=False))
print()
print("  >> Low correlation with equivalent AUROC means the models capture")
print("     different but similarly predictive signal -- worth stating, and a")
print("     natural argument for why an ensemble would be future work.")

# ---------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

ax = axes[0]
sub = res[res["pip_threshold"] == res["pip_threshold"].min()].sort_values("auroc")
colors = ["#DD8452" if m == "AlphaGenome" else "#4C72B0" for m in sub["model"]]
ax.barh(sub["model"], sub["auroc"], color=colors)
ax.errorbar(sub["auroc"], range(len(sub)),
            xerr=[sub["auroc"] - sub["ci95_low"], sub["ci95_high"] - sub["auroc"]],
            fmt="none", ecolor="black", capsize=3, lw=1)
ax.axvline(0.5, color="grey", ls="--", lw=1)
ax.set_xlim(0.35, 0.85)
ax.set_xlabel("AUROC (95% bootstrap CI)")
ax.set_title("Predicting fine-mapped variants", fontsize=10)

ax = axes[1]
best = agree.iloc[0]["enformer_summarisation"] if len(agree) else ENF_COLS[0]
ok = df["alphagenome_score"].notna() & df[best].notna()
ax.scatter(df.loc[ok & (df["is_positive"] == 0), "alphagenome_score"],
           df.loc[ok & (df["is_positive"] == 0), best],
           s=12, alpha=0.4, label="negatives", color="#B0B0B0")
ax.scatter(df.loc[ok & (df["is_positive"] == 1), "alphagenome_score"],
           df.loc[ok & (df["is_positive"] == 1), best],
           s=18, alpha=0.8, label="fine-mapped positives", color="#C44E52")
ax.set_xlabel("AlphaGenome score")
ax.set_ylabel(best)
ax.set_yscale("log")
ax.legend(fontsize=8, frameon=False)
ax.set_title("Model agreement", fontsize=10)

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{OUTDIR}/RA9_enformer_comparison.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"\n[RA9] done. Outputs in {OUTDIR}/")
