"""
RA0e — consensus ranking and specification-curve analysis.

RA0d established that no single model specification is privileged: 240 defensible
combinations of AlphaGenome summarisation, weighting scheme and p-value transform
nominate 5-6 different top candidates per signal, all association-supported once
the Tier 1 constraint is applied.

This script turns that from a problem into the reported result. For every variant
it records, across all 240 specifications:

    * how often it is ranked first within its signal
    * how often it appears in the top 5 / 10 / 20
    * its mean, median, best and worst rank
    * a CONSENSUS RANK, reported alongside final_score

Two reporting modes are produced, so the choice can be made later without re-running:

    MODE A (primary)  -> RA0e_candidate_sets_*.tsv
        A candidate SET per signal with model-support frequencies. This is a
        specification-curve analysis: robustness is reported rather than a single
        arbitrary specification being privileged.

    MODE C (fallback) -> RA0e_modal_plus_set_*.tsv
        The modal candidate named as primary, with the supported set and support
        counts alongside it, formatted for Table 1.

Everything is computed under both the unconstrained ranking and the Tier 1
association constraint, so the two can be contrasted directly.

Outputs -> work/step_RA0e_consensus/
"""

import numpy as np
import pandas as pd

from ra_common import (
    COMPONENTS, CURRENT_WEIGHTS,
    add_signal_id, ensure_dir, load_final, renormalise,
)

OUTDIR = ensure_dir("work/step_RA0e_consensus")

TIER1_P = 5e-5          # association-supported threshold for Tier 1
TOPN_TRACKED = [1, 5, 10, 20]

df, info = load_final(dedupe=True)
df = add_signal_id(df)
df["p_num"] = pd.to_numeric(df["p"], errors="coerce")
df["neglog10p"] = -np.log10(df["p_num"].clip(lower=1e-300))
df = df.reset_index(drop=True)

print(f"\n[RA0e] {len(df):,} unique variants across "
      f"{df['signal_id'].nunique()} signals")

# ---------------------------------------------------------------
# The same 240-specification grid as RA0d
# ---------------------------------------------------------------
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

transform_variants = {
    "cap_7.3": lambda d: (d["neglog10p"] / 7.30).clip(0, 1),
    "cap_12": lambda d: (d["neglog10p"] / 12.0).clip(0, 1),
    "cap_20_current": lambda d: (d["neglog10p"] / 20.0).clip(0, 1),
    "cap_40": lambda d: (d["neglog10p"] / 40.0).clip(0, 1),
}

n_spec = len(ag_variants) * len(weight_variants) * len(transform_variants)
print(f"specifications: {n_spec}")

# Pre-extract the non-AG, non-GWAS component columns once
fixed_cols = {c: pd.to_numeric(df[c], errors="coerce").fillna(0.0).to_numpy()
              for c in COMPONENTS if c not in ("alphagenome_score", "gwas_score")}

signals = sorted(df["signal_id"].dropna().unique())
sig_idx = {s: np.where(df["signal_id"].to_numpy() == s)[0] for s in signals}
is_supported = (df["p_num"] < TIER1_P).to_numpy()


def new_accumulator(n):
    return {
        "sum_rank": np.zeros(n),
        "sum_rank_sq": np.zeros(n),
        "best_rank": np.full(n, np.inf),
        "worst_rank": np.zeros(n),
        "n_spec": 0,
        **{f"n_top{k}": np.zeros(n) for k in TOPN_TRACKED},
    }


acc = {
    "unconstrained": {s: new_accumulator(len(sig_idx[s])) for s in signals},
    "tier1": {s: new_accumulator(int(is_supported[sig_idx[s]].sum())) for s in signals},
}
tier1_pos = {s: np.where(is_supported[sig_idx[s]])[0] for s in signals}

spec_rows = []      # per-specification top-1, for the specification curve

for ag_name, ag_vals in ag_variants.items():
    ag = pd.to_numeric(ag_vals, errors="coerce").fillna(0.0).to_numpy()
    for tr_name, tr_fn in transform_variants.items():
        gw = pd.to_numeric(tr_fn(df), errors="coerce").fillna(0.0).to_numpy()
        for w_name, w in weight_variants.items():
            score = np.zeros(len(df))
            for c, wt in w.items():
                if wt == 0:
                    continue
                v = ag if c == "alphagenome_score" else (gw if c == "gwas_score" else fixed_cols[c])
                score = score + wt * v

            for s in signals:
                idx = sig_idx[s]
                sc_full = score[idx]

                for tier, sc, keep in (
                    ("unconstrained", sc_full, None),
                    ("tier1", sc_full[tier1_pos[s]], tier1_pos[s]),
                ):
                    if len(sc) == 0:
                        continue
                    order = np.argsort(-sc, kind="stable")
                    ranks = np.empty(len(sc), dtype=float)
                    ranks[order] = np.arange(1, len(sc) + 1)

                    a = acc[tier][s]
                    a["sum_rank"] += ranks
                    a["sum_rank_sq"] += ranks ** 2
                    a["best_rank"] = np.minimum(a["best_rank"], ranks)
                    a["worst_rank"] = np.maximum(a["worst_rank"], ranks)
                    a["n_spec"] += 1
                    for k in TOPN_TRACKED:
                        a[f"n_top{k}"] += (ranks <= k)

                    if tier == "tier1":
                        win = idx[keep][order[0]]
                        spec_rows.append({
                            "signal_id": s,
                            "ag_summarisation": ag_name,
                            "transform": tr_name,
                            "weighting": w_name,
                            "top_variant_key": df.at[win, "variant_key"],
                            "top_p": df.at[win, "p_num"],
                        })

spec_curve = pd.DataFrame(spec_rows)
spec_curve.to_csv(f"{OUTDIR}/RA0e_specification_curve_data.tsv", sep="\t", index=False)

# ---------------------------------------------------------------
# Build the consensus tables
# ---------------------------------------------------------------
def build_table(tier):
    out = []
    for s in signals:
        idx = sig_idx[s]
        if tier == "tier1":
            idx = idx[tier1_pos[s]]
        a = acc[tier][s]
        n = a["n_spec"]
        sub = df.iloc[idx].copy()
        sub["n_specifications"] = n
        sub["mean_rank"] = a["sum_rank"] / n
        sub["sd_rank"] = np.sqrt(np.maximum(
            a["sum_rank_sq"] / n - (a["sum_rank"] / n) ** 2, 0))
        sub["best_rank"] = a["best_rank"]
        sub["worst_rank"] = a["worst_rank"]
        for k in TOPN_TRACKED:
            sub[f"pct_specs_in_top{k}"] = 100.0 * a[f"n_top{k}"] / n
        sub["consensus_rank"] = sub["mean_rank"].rank(method="min").astype(int)
        out.append(sub)
    return pd.concat(out, ignore_index=True)


keep_cols = [
    "signal_id", "locus_id", "variant_key", "p", "final_score",
    "consensus_rank", "mean_rank", "sd_rank", "best_rank", "worst_rank",
    "n_specifications",
] + [f"pct_specs_in_top{k}" for k in TOPN_TRACKED] + COMPONENTS

for tier in ("unconstrained", "tier1"):
    tab = build_table(tier)
    tab = tab[[c for c in keep_cols if c in tab.columns]]
    tab = tab.sort_values(["signal_id", "consensus_rank"])
    tab.to_csv(f"{OUTDIR}/RA0e_consensus_full_{tier}.tsv.gz",
               sep="\t", index=False, compression="gzip")

    # ---------- MODE A: candidate sets ----------
    setA = tab[tab["pct_specs_in_top1"] > 0].copy()
    setA = setA.sort_values(["signal_id", "pct_specs_in_top1"], ascending=[True, False])
    setA_out = setA[[
        "signal_id", "variant_key", "p", "consensus_rank",
        "pct_specs_in_top1", "pct_specs_in_top5", "pct_specs_in_top20",
        "mean_rank", "final_score",
    ]].copy()
    # BUGFIX: a blanket .round(4) collapsed the p-value column to 0.0 (p = 2.5e-08
    # rounds to 0.0000). Round only the display columns and leave p untouched.
    for _c in ["pct_specs_in_top1", "pct_specs_in_top5", "pct_specs_in_top20",
               "mean_rank", "final_score"]:
        setA_out[_c] = setA_out[_c].astype(float).round(4)
    setA_out.to_csv(f"{OUTDIR}/RA0e_candidate_sets_{tier}.tsv", sep="\t", index=False)

    # ---------- MODE C: modal candidate + set, Table 1 format ----------
    rows = []
    for s, grp in setA.groupby("signal_id"):
        modal = grp.iloc[0]
        others = grp.iloc[1:]
        rows.append({
            "signal_id": s,
            "primary_candidate": modal["variant_key"],
            "primary_p": modal["p"],
            "primary_pct_specs_top1": round(float(modal["pct_specs_in_top1"]), 1),
            "primary_consensus_rank": int(modal["consensus_rank"]),
            "n_alternative_candidates": len(others),
            "alternative_candidates": "; ".join(
                f"{r.variant_key} (p={r.p:.3g}, {r.pct_specs_in_top1:.1f}%)"
                for r in others.itertuples()
            ),
            "all_candidates_association_supported": bool(
                (pd.to_numeric(grp["p"], errors="coerce") < TIER1_P).all()),
        })
    pd.DataFrame(rows).to_csv(
        f"{OUTDIR}/RA0e_modal_plus_set_{tier}.tsv", sep="\t", index=False)

    print()
    print("=" * 78)
    print(f"CANDIDATE SETS — {tier}")
    print("=" * 78)
    print(setA_out.head(30).to_string(index=False))

# ---------------------------------------------------------------
# Consensus rank merged onto the main table, for reporting alongside final_score
# ---------------------------------------------------------------
cons = pd.read_csv(f"{OUTDIR}/RA0e_consensus_full_unconstrained.tsv.gz", sep="\t")
merged = df[["variant_key", "signal_id", "locus_id", "p", "final_score"]].merge(
    cons[["variant_key", "consensus_rank", "mean_rank",
          "pct_specs_in_top1", "pct_specs_in_top20"]],
    on="variant_key", how="left",
)
merged["final_score_rank_within_signal"] = (
    merged.groupby("signal_id")["final_score"].rank(ascending=False, method="min")
)
merged.to_csv(f"{OUTDIR}/RA0e_final_score_with_consensus_rank.tsv.gz",
              sep="\t", index=False, compression="gzip")

agree = merged.dropna(subset=["consensus_rank", "final_score_rank_within_signal"])
from scipy import stats as _st
r, _ = _st.spearmanr(agree["final_score_rank_within_signal"], agree["consensus_rank"])
print()
print("=" * 78)
print("CONSENSUS RANK vs final_score RANK")
print("=" * 78)
print(f"Spearman rho: {r:.4f}")
print("  >> Reported alongside final_score in Table 1 and the supplementary tables.")

# ---------------------------------------------------------------
# Specification curve figure
# ---------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, len(signals), figsize=(5.0 * len(signals), 5.0))
if len(signals) == 1:
    axes = [axes]

for ax, s in zip(axes, signals):
    g = spec_curve[spec_curve["signal_id"] == s]
    counts = (g["top_variant_key"].value_counts(normalize=True) * 100).head(8)
    labels = [f"{k}\n(p={g.loc[g['top_variant_key'] == k, 'top_p'].iloc[0]:.2g})"
              for k in counts.index]
    ax.barh(range(len(counts)), counts.values, color="#4C72B0")
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("% of specifications ranking variant first")
    ax.set_xlim(0, 100)
    ax.set_title(s, fontsize=10)
    for i, v in enumerate(counts.values):
        ax.text(v + 1, i, f"{v:.0f}%", va="center", fontsize=8)

fig.suptitle(
    f"Consensus candidates across {n_spec} model specifications "
    f"(Tier 1, p < {TIER1_P:g})",
    fontsize=11,
)
fig.tight_layout(rect=[0, 0, 1, 0.94])
for ext in ("png", "pdf"):
    fig.savefig(f"{OUTDIR}/RA0e_consensus_candidates.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"\n[RA0e] done. Outputs in {OUTDIR}/")
