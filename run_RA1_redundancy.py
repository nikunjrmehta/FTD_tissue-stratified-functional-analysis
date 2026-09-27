"""
RA1 — redundancy among the six component scores.

Reviewer #2, Comment 2: "Have you assessed pairwise correlations among the six
component scores for all variants? If strong correlations exist, how could this
potentially affect the reliability of variant ranking and selection?"

Three things this does that a naive correlation matrix does not:

  * Computes statistics on UNIQUE variants, not variant-locus pairs. Locus windows
    overlap, so a row-wise correlation is pseudo-replicated.
  * Handles the binary cCRE score with BOTH Spearman and a Mann-Whitney AUC.
    Reporting only one, chosen after seeing which is larger, is metric-shopping.
  * Reports VIF and PCA effective dimensionality, which is what "redundancy"
    actually means -- pairwise rho can be low while the set is still collinear.

Outputs -> work/step_RA1_redundancy/

NOTE ON INTERPRETING THE OUTPUT: if gtex_score_any and susie_score_any come out
NEGATIVELY correlated, do not put that number in a supplementary table until you
have confirmed it is not a merge artefact. The plausible genuine explanation is
that SuSiE PIP is diluted across large credible sets in high-LD regions -- which
is exactly the MAPT situation -- so strong QTL evidence and high PIP become
anti-correlated. That is a real and interesting finding, but it must be stated as
such, with the credible-set-size evidence alongside it (this script computes it).
"""

import numpy as np
import pandas as pd
from scipy import stats

from ra_common import (
    COMPONENTS, COMPONENT_LABELS,
    add_signal_id, component_matrix, ensure_dir, load_final,
)

OUTDIR = ensure_dir("work/step_RA1_redundancy")

df, info = load_final(dedupe=True)
df = add_signal_id(df)
M = component_matrix(df, dropna=False)

print(f"\n[RA1] analysing {len(M):,} unique variants")

# ---------------------------------------------------------------
# 1. Pairwise Spearman
# ---------------------------------------------------------------
rho = pd.DataFrame(index=COMPONENTS, columns=COMPONENTS, dtype=float)
pval = pd.DataFrame(index=COMPONENTS, columns=COMPONENTS, dtype=float)

for a in COMPONENTS:
    for b in COMPONENTS:
        ok = M[a].notna() & M[b].notna()
        if ok.sum() < 10:
            rho.loc[a, b], pval.loc[a, b] = np.nan, np.nan
            continue
        r, p = stats.spearmanr(M.loc[ok, a], M.loc[ok, b])
        rho.loc[a, b], pval.loc[a, b] = r, p

rho.round(4).to_csv(f"{OUTDIR}/RA1_spearman_matrix.tsv", sep="\t")
pval.to_csv(f"{OUTDIR}/RA1_spearman_pvalues.tsv", sep="\t")

print("\nPairwise Spearman rho:")
print(rho.astype(float).round(3).to_string())

off = rho.where(~np.eye(len(COMPONENTS), dtype=bool)).abs()
print(f"\nmax |rho| off-diagonal: {np.nanmax(off.values):.3f}")

# ---------------------------------------------------------------
# 2. Binary cCRE handled honestly: Mann-Whitney AUC alongside Spearman
# ---------------------------------------------------------------
ccre = pd.to_numeric(df["ccre_score"], errors="coerce")
is_ccre = ccre >= 0.5

auc_rows = []
for c in COMPONENTS:
    if c == "ccre_score":
        continue
    v = pd.to_numeric(df[c], errors="coerce")
    a = v[is_ccre & v.notna()]
    b = v[(~is_ccre) & v.notna()]
    if len(a) < 10 or len(b) < 10:
        continue
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    auc = u / (len(a) * len(b))          # probability a cCRE variant scores higher
    auc_rows.append({
        "component": c,
        "label": COMPONENT_LABELS[c],
        "n_in_ccre": len(a),
        "n_not_in_ccre": len(b),
        "median_in_ccre": round(float(a.median()), 4),
        "median_not_in_ccre": round(float(b.median()), 4),
        "mannwhitney_auc": round(float(auc), 4),
        "mannwhitney_p": p,
        "spearman_rho_vs_ccre": round(float(rho.loc[c, "ccre_score"]), 4),
    })

auc_df = pd.DataFrame(auc_rows)
auc_df.to_csv(f"{OUTDIR}/RA1_ccre_binary_association.tsv", sep="\t", index=False)
print("\ncCRE (binary) association, Spearman vs Mann-Whitney AUC:")
print(auc_df.to_string(index=False))

# ---------------------------------------------------------------
# 3. VIF -- pairwise rho can be low while the set is collinear
# ---------------------------------------------------------------
X = M.dropna()
Xs = (X - X.mean()) / X.std(ddof=0)
Xs = Xs.loc[:, Xs.std(ddof=0) > 0]

vif_rows = []
for c in Xs.columns:
    y = Xs[c].values
    A = np.column_stack([np.ones(len(Xs)), Xs.drop(columns=[c]).values])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else 0.0
    vif_rows.append({
        "component": c,
        "label": COMPONENT_LABELS[c],
        "r2_on_other_five": round(r2, 4),
        "vif": round(1.0 / (1.0 - r2), 3) if r2 < 0.999 else np.inf,
    })
vif = pd.DataFrame(vif_rows)
vif.to_csv(f"{OUTDIR}/RA1_vif.tsv", sep="\t", index=False)
print("\nVariance inflation factors (VIF > 5 would indicate real collinearity):")
print(vif.to_string(index=False))

# ---------------------------------------------------------------
# 4. PCA effective dimensionality
# ---------------------------------------------------------------
C = np.corrcoef(Xs.values, rowvar=False)
eigvals = np.sort(np.linalg.eigvalsh(C))[::-1]
expl = eigvals / eigvals.sum()
cum = np.cumsum(expl)

# Participation-ratio style effective dimension
eff_dim = (eigvals.sum() ** 2) / (eigvals ** 2).sum()

pca = pd.DataFrame({
    "PC": [f"PC{i+1}" for i in range(len(eigvals))],
    "eigenvalue": np.round(eigvals, 4),
    "prop_variance": np.round(expl, 4),
    "cumulative_variance": np.round(cum, 4),
})
pca.to_csv(f"{OUTDIR}/RA1_pca.tsv", sep="\t", index=False)
print("\nPCA of the six standardised scores:")
print(pca.to_string(index=False))
print(f"\neffective number of independent dimensions: {eff_dim:.2f} of 6")
print(f"PCs needed for 90% of variance            : {int(np.searchsorted(cum, 0.90) + 1)}")

# ---------------------------------------------------------------
# 5. Credible-set-size context for the GTEx/SuSiE relationship
#    (pre-empts the "why is this negative?" query -- see module docstring)
# ---------------------------------------------------------------
cs_rows = []
for cs_col, pip_col, lab in [
    ("susie_eqtl_best_cs_size", "susie_eqtl_max_pip", "eQTL"),
    ("susie_sqtl_best_cs_size", "susie_sqtl_max_pip", "sQTL"),
]:
    if cs_col not in df.columns:
        continue
    size = pd.to_numeric(df[cs_col], errors="coerce")
    pip = pd.to_numeric(df[pip_col], errors="coerce")
    ok = size.notna() & pip.notna() & (size > 0)
    if ok.sum() < 10:
        continue
    r, p = stats.spearmanr(size[ok], pip[ok])
    cs_rows.append({
        "qtl_type": lab,
        "n_variants": int(ok.sum()),
        "median_credible_set_size": float(size[ok].median()),
        "spearman_rho_cs_size_vs_pip": round(float(r), 4),
        "p": p,
    })
if cs_rows:
    cs = pd.DataFrame(cs_rows)
    cs.to_csv(f"{OUTDIR}/RA1_credible_set_size_vs_pip.tsv", sep="\t", index=False)
    print("\nCredible-set size vs PIP (explains any negative GTEx/SuSiE correlation):")
    print(cs.to_string(index=False))

# ---------------------------------------------------------------
# 6. Per-signal correlations -- regional structure differs
# ---------------------------------------------------------------
per_signal = []
for s, grp in df.groupby("signal_id"):
    ms = component_matrix(grp, dropna=False)
    for i, a in enumerate(COMPONENTS):
        for b in COMPONENTS[i + 1:]:
            ok = ms[a].notna() & ms[b].notna()
            if ok.sum() < 30:
                continue
            r, p = stats.spearmanr(ms.loc[ok, a], ms.loc[ok, b])
            per_signal.append({
                "signal_id": s, "component_a": a, "component_b": b,
                "n": int(ok.sum()), "spearman_rho": round(float(r), 4), "p": p,
            })
pd.DataFrame(per_signal).to_csv(
    f"{OUTDIR}/RA1_spearman_per_signal.tsv", sep="\t", index=False
)

# ---------------------------------------------------------------
# Figure: correlation heatmap
# ---------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

labels = [COMPONENT_LABELS[c] for c in COMPONENTS]
fig, ax = plt.subplots(figsize=(7.2, 6.0))
im = ax.imshow(rho.astype(float).values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=9)
for i in range(len(COMPONENTS)):
    for j in range(len(COMPONENTS)):
        val = rho.astype(float).values[i, j]
        if np.isfinite(val):
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color="black" if abs(val) < 0.6 else "white")
ax.set_title(
    f"Spearman correlation among evidence components\n({len(M):,} unique variants)",
    fontsize=10,
)
fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman rho")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{OUTDIR}/RA1_spearman_heatmap.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"\n[RA1] done. Outputs in {OUTDIR}/")
