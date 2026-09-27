"""
RA11 — does annotation density inflate the integrated score?

Reviewer #2, Comment 4:
    "...variants that happen to fall in a gene-dense, highly expressed, eQTL-rich
     region could receive inflated scores simply because the same information is
     counted multiple times. Could this inflated score mask those equally
     important signals from variants that happen to be in less annotation-dense
     regions?"

This is the sharpest statistical point in the letter, and all three signals sit in
gene-dense regions, so it cannot be waved away. RA0b already established one half
of it empirically: variants excluded by the NaN defect were annotation-sparse
(AUC 0.03-0.25 across the GTEx layers) while cCRE overlap showed no difference at
all (AUC 0.497, p = 0.54) -- a clean negative control, because cCRE coverage does
not depend on GTEx annotation.

This script does the direct test the comment asks for:

  1. Three independent density measures per variant.
  2. Correlation of final_score with density, and PARTIAL correlation controlling
     for GWAS evidence (otherwise density and association strength are confounded).
  3. Whether the reported candidates are density outliers.
  4. A density-ADJUSTED score, and whether the top candidate at any signal changes.

Step 4 is the one that matters: if candidates survive density adjustment, the
comment is answered with evidence rather than argument.

Outputs -> work/step_RA11_annotation_density/
"""

import gzip
import os

import numpy as np
import pandas as pd
from scipy import stats

from ra_common import add_signal_id, ensure_dir, load_final

OUTDIR = ensure_dir("work/step_RA11_annotation_density")
GENCODE = "data/annotations/gencode.v49.basic.annotation.gtf.gz"
CCRE_BEDS = ["data/annotations/GRCh38-cCREs.PLS.bed",
             "data/annotations/GRCh38-cCREs.ELS.bed"]

CCRE_WINDOW = 50_000        # +/- for cCRE density
GENE_WINDOW = 500_000       # +/- for gene density
QTL_WINDOW = 100_000        # +/- for local QTL richness

df, info = load_final(dedupe=True)
df = add_signal_id(df)
df["pos"] = pd.to_numeric(df["pos"], errors="coerce")
df["chr"] = df["chr"].astype(str)
print(f"\n[RA11] {len(df):,} unique variants\n")

# ---------------------------------------------------------------
# 1a. cCRE density
# ---------------------------------------------------------------
print("computing cCRE density ...")
ccre = []
for bed in CCRE_BEDS:
    if not os.path.exists(bed):
        print(f"  missing {bed} -- skipping")
        continue
    b = pd.read_csv(bed, sep="\t", header=None, usecols=[0, 1, 2],
                    names=["chr", "start", "end"], dtype={0: str})
    ccre.append(b)
if ccre:
    ccre = pd.concat(ccre, ignore_index=True)
    ccre = ccre[ccre["chr"].isin(df["chr"].unique())]
    dens = np.zeros(len(df))
    for c, grp in df.groupby("chr"):
        starts = np.sort(ccre.loc[ccre["chr"] == c, "start"].to_numpy())
        if len(starts) == 0:
            continue
        lo = np.searchsorted(starts, grp["pos"].to_numpy() - CCRE_WINDOW, "left")
        hi = np.searchsorted(starts, grp["pos"].to_numpy() + CCRE_WINDOW, "right")
        dens[grp.index.to_numpy()] = hi - lo
    df["density_ccre"] = dens
else:
    df["density_ccre"] = np.nan

# ---------------------------------------------------------------
# 1b. Protein-coding gene density from GENCODE
# ---------------------------------------------------------------
print("computing gene density ...")
if os.path.exists(GENCODE):
    rows = []
    keep_chr = set(df["chr"].unique())
    with gzip.open(GENCODE, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.split("\t")
            if len(p) < 9 or p[2] != "gene" or p[0] not in keep_chr:
                continue
            if 'gene_type "protein_coding"' not in p[8]:
                continue
            rows.append((p[0], int(p[3]), int(p[4])))
    genes = pd.DataFrame(rows, columns=["chr", "start", "end"])
    print(f"  protein-coding genes on analysed chromosomes: {len(genes):,}")
    dens = np.zeros(len(df))
    for c, grp in df.groupby("chr"):
        g = genes[genes["chr"] == c]
        if not len(g):
            continue
        gs, ge = g["start"].to_numpy(), g["end"].to_numpy()
        for i, pos in zip(grp.index.to_numpy(), grp["pos"].to_numpy()):
            dens[i] = int(((ge >= pos - GENE_WINDOW) & (gs <= pos + GENE_WINDOW)).sum())
    df["density_genes"] = dens
else:
    print(f"  {GENCODE} not found -- skipping gene density")
    df["density_genes"] = np.nan

# ---------------------------------------------------------------
# 1c. Local QTL richness, computed from our own data
# ---------------------------------------------------------------
print("computing local QTL richness ...")
hit = pd.to_numeric(df.get("gtex_sigpair_hit_brain_count"), errors="coerce").fillna(0) \
    + pd.to_numeric(df.get("gtex_sigpair_hit_blood_count"), errors="coerce").fillna(0)
df["_has_qtl"] = (hit > 0).astype(float)
qdens = np.zeros(len(df))
for c, grp in df.groupby("chr"):
    order = grp.sort_values("pos")
    pos = order["pos"].to_numpy()
    val = order["_has_qtl"].to_numpy()
    cum = np.concatenate([[0.0], np.cumsum(val)])
    lo = np.searchsorted(pos, pos - QTL_WINDOW, "left")
    hi = np.searchsorted(pos, pos + QTL_WINDOW, "right")
    frac = (cum[hi] - cum[lo]) / np.maximum(hi - lo, 1)
    qdens[order.index.to_numpy()] = frac
df["density_local_qtl"] = qdens

DENSITY_COLS = ["density_ccre", "density_genes", "density_local_qtl"]
print("\ndensity summary:")
print(df[DENSITY_COLS].describe().T.to_string())

# ---------------------------------------------------------------
# 2. Does score track density? Partial correlation controls for GWAS evidence
# ---------------------------------------------------------------
def partial_spearman(x, y, z):
    """Spearman(x, y) controlling for z, via residuals of rank regression."""
    ok = x.notna() & y.notna() & z.notna()
    xr, yr, zr = (stats.rankdata(v[ok]) for v in (x, y, z))
    def resid(a, b):
        A = np.column_stack([np.ones(len(b)), b])
        coef, *_ = np.linalg.lstsq(A, a, rcond=None)
        return a - A @ coef
    return stats.spearmanr(resid(xr, zr), resid(yr, zr))


print()
print("=" * 74)
print("2. IS final_score DRIVEN BY ANNOTATION DENSITY?")
print("=" * 74)
rows = []
for d in DENSITY_COLS:
    if df[d].notna().sum() < 100:
        continue
    r, p = stats.spearmanr(df["final_score"], df[d], nan_policy="omit")
    pr, pp = partial_spearman(df["final_score"], df[d], df["gwas_score"])
    rows.append({
        "density_measure": d,
        "spearman_rho": round(float(r), 4), "p": p,
        "partial_rho_given_gwas": round(float(pr), 4), "partial_p": pp,
    })
corr = pd.DataFrame(rows)
corr.to_csv(f"{OUTDIR}/RA11_density_correlations.tsv", sep="\t", index=False)
print(corr.to_string(index=False))
print()
print("  >> A high raw rho with a much smaller partial rho means the apparent")
print("     density effect is mostly association strength, not double-counting.")

# ---------------------------------------------------------------
# 3. Are the reported candidates density outliers?
# ---------------------------------------------------------------
print()
print("=" * 74)
print("3. ARE OUR CANDIDATES IN UNUSUALLY DENSE REGIONS?")
print("=" * 74)
for d in DENSITY_COLS:
    df[f"pct_{d}"] = df.groupby("signal_id")[d].rank(pct=True)

try:
    cons = pd.read_csv("work/step_RA0e_consensus/RA0e_candidate_sets_tier1.tsv",
                       sep="\t")
    m = cons.merge(df[["variant_key"] + [f"pct_{d}" for d in DENSITY_COLS]],
                   on="variant_key", how="left")
    print(m[["signal_id", "variant_key", "pct_specs_in_top1"]
            + [f"pct_{d}" for d in DENSITY_COLS]].round(3).to_string(index=False))
    m.to_csv(f"{OUTDIR}/RA11_candidate_density_percentiles.tsv",
             sep="\t", index=False)
    print()
    print("  >> Percentiles near 0.5 mean the candidates are NOT in unusually")
    print("     annotation-dense parts of their own locus.")
except FileNotFoundError:
    print("(run RA0e first)")

# ---------------------------------------------------------------
# 4. Density-adjusted score
# ---------------------------------------------------------------
print()
print("=" * 74)
print("4. DENSITY-ADJUSTED RANKING")
print("=" * 74)

use = [d for d in DENSITY_COLS if df[d].notna().sum() > 100]
X = df[use].apply(lambda s: s.fillna(s.median()))
X = (X - X.mean()) / X.std(ddof=0).replace(0, 1)
A = np.column_stack([np.ones(len(df)), X.to_numpy()])
y = df["final_score"].fillna(0).to_numpy()
coef, *_ = np.linalg.lstsq(A, y, rcond=None)
df["final_score_density_adjusted"] = y - (A @ coef) + float(np.mean(y))

# within-decile ranking as a second, non-parametric adjustment
df["_dec"] = pd.qcut(X.mean(axis=1).rank(method="first"), 10,
                     labels=False, duplicates="drop")
df["rank_within_density_decile"] = df.groupby(["signal_id", "_dec"])["final_score"] \
                                     .rank(ascending=False, method="min")

rows = []
for s, grp in df.groupby("signal_id"):
    top_raw = grp.loc[grp["final_score"].idxmax()]
    top_adj = grp.loc[grp["final_score_density_adjusted"].idxmax()]
    top20_raw = set(grp.nlargest(20, "final_score")["variant_key"])
    top20_adj = set(grp.nlargest(20, "final_score_density_adjusted")["variant_key"])
    rows.append({
        "signal_id": s,
        "top_raw": top_raw["variant_key"], "top_raw_p": top_raw["p"],
        "top_density_adjusted": top_adj["variant_key"],
        "top_adjusted_p": top_adj["p"],
        "top_changed": bool(top_raw["variant_key"] != top_adj["variant_key"]),
        "top20_overlap": len(top20_raw & top20_adj),
    })
adj = pd.DataFrame(rows)
adj.to_csv(f"{OUTDIR}/RA11_density_adjusted_top.tsv", sep="\t", index=False)
print(adj.to_string(index=False))
print()
if not adj["top_changed"].any():
    print("  >> No top candidate changes after density adjustment. This is the")
    print("     direct answer to Comment 4: the candidates are not artefacts of")
    print("     annotation-dense regions.")
else:
    print("  >> At least one candidate is density-sensitive. Report it honestly;")
    print("     it bounds how far the concern extends.")

keep = ["variant_key", "signal_id", "locus_id", "p", "final_score",
        "final_score_density_adjusted", "rank_within_density_decile"] + \
       DENSITY_COLS + [f"pct_{d}" for d in DENSITY_COLS]
df[keep].to_csv(f"{OUTDIR}/RA11_density_scored.tsv.gz", sep="\t",
                index=False, compression="gzip")

# ---------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, len(use) + 1, figsize=(4.2 * (len(use) + 1), 4.0))
for ax, d in zip(axes[:-1], use):
    ax.scatter(df[d], df["final_score"], s=4, alpha=0.12,
               color="#4C72B0", rasterized=True)
    try:
        c = cons.merge(df[["variant_key", d, "final_score"]],
                       on="variant_key", how="left")
        ax.scatter(c[d], c["final_score"], s=45, color="#C44E52",
                   edgecolor="black", zorder=5, label="reported candidates")
        ax.legend(fontsize=7, frameon=False)
    except Exception:
        pass
    ax.set_xlabel(d.replace("density_", "").replace("_", " "))
    ax.set_ylabel("final_score")
    ax.set_title(d, fontsize=9)

ax = axes[-1]
ax.scatter(df["final_score"], df["final_score_density_adjusted"], s=4,
           alpha=0.12, color="#55A868", rasterized=True)
lim = [0, max(df["final_score"].max(), df["final_score_density_adjusted"].max()) * 1.05]
ax.plot(lim, lim, "--", color="grey", lw=1)
ax.set_xlabel("final_score")
ax.set_ylabel("density-adjusted")
ax.set_title("Effect of density adjustment", fontsize=9)

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{OUTDIR}/RA11_annotation_density.{ext}", dpi=300,
                bbox_inches="tight")
plt.close(fig)

print(f"\n[RA11] done. Outputs in {OUTDIR}/")
