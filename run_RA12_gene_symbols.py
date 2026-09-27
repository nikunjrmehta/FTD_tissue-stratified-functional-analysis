"""
RA12 — resolve every Ensembl gene ID to a symbol, and characterise the ones that
cannot be resolved.

Reviewer #3, minor 5:
    "Several genes in Table 1 and the Supplementary Tables are reported only as
     Ensembl IDs (e.g., ENSG00000291175, ENSG00000262879). These should be
     replaced with gene symbols where possible to improve interpretability. In
     addition, ENSG00000291175 (Signal 1) and the gene associated with the top
     candidate variant at chr19:44646342:C:T (Signal 3) could not be identified
     in GTEx or gnomAD, respectively. Please confirm these annotations and clarify
     whether they represent poorly annotated or potentially novel transcripts."

The referee tried to look these up and failed. They will check the answer, so the
two IDs they named are handled explicitly, with biotype and annotation status
rather than just a symbol.

Uses the GENCODE v49 annotation already on disk -- the same release the pipeline
used, which matters: resolving against a different release could produce symbols
the analysis never actually used.

Outputs -> work/step_RA12_gene_symbols/
"""

import gzip
import os
import re

import pandas as pd

from ra_common import add_signal_id, ensure_dir, load_final

OUTDIR = ensure_dir("work/step_RA12_gene_symbols")
GENCODE = "data/annotations/gencode.v49.basic.annotation.gtf.gz"

# The two IDs the referee named, plus the Signal 3 top candidate's gene
REFEREE_IDS = ["ENSG00000291175", "ENSG00000262879"]
REFEREE_VARIANT = "chr19:44646342:C:T"

# ---------------------------------------------------------------
# 1. Build the GENCODE lookup
# ---------------------------------------------------------------
if not os.path.exists(GENCODE):
    raise SystemExit(f"{GENCODE} not found.")

print("parsing GENCODE v49 ...")
pat_id = re.compile(r'gene_id "([^"]+)"')
pat_name = re.compile(r'gene_name "([^"]+)"')
pat_type = re.compile(r'gene_type "([^"]+)"')

recs = {}
with gzip.open(GENCODE, "rt") as f:
    for line in f:
        if line.startswith("#"):
            continue
        p = line.split("\t")
        if len(p) < 9 or p[2] != "gene":
            continue
        attr = p[8]
        m = pat_id.search(attr)
        if not m:
            continue
        gid = m.group(1).split(".")[0]          # strip the version suffix
        recs[gid] = {
            "gene_id": gid,
            "gene_id_versioned": m.group(1),
            "gene_symbol": (pat_name.search(attr).group(1)
                            if pat_name.search(attr) else None),
            "gene_type": (pat_type.search(attr).group(1)
                          if pat_type.search(attr) else None),
            "chr": p[0], "start": int(p[3]), "end": int(p[4]), "strand": p[6],
        }
lookup = pd.DataFrame(recs.values())
print(f"  genes in GENCODE v49: {len(lookup):,}")
lookup.to_csv(f"{OUTDIR}/RA12_gencode_gene_index.tsv.gz", sep="\t",
              index=False, compression="gzip")

# ---------------------------------------------------------------
# 2. Collect every Ensembl ID that appears anywhere in our outputs
# ---------------------------------------------------------------
print("\ncollecting Ensembl IDs used in the analysis ...")
df, info = load_final(dedupe=True)
df = add_signal_id(df)

ID_RE = re.compile(r"ENSG\d{11}")
sources = {}

gene_cols = [c for c in df.columns
             if "gene" in c.lower() and df[c].dtype == object]
for c in gene_cols:
    for v in df[c].dropna().astype(str).unique():
        for gid in ID_RE.findall(v):
            sources.setdefault(gid, set()).add(f"FINAL:{c}")

for path, col in [
    ("work/step14_signal_story/FTD_top_genes_per_signal.tsv", "gene"),
    ("work/step12_final_ftd/FTD_locus_summary_FINAL.tsv", None),
]:
    if not os.path.exists(path):
        continue
    t = pd.read_csv(path, sep="\t")
    cols = [col] if col else [c for c in t.columns if t[c].dtype == object]
    for c in cols:
        if c not in t.columns:
            continue
        for v in t[c].dropna().astype(str).unique():
            for gid in ID_RE.findall(v):
                sources.setdefault(gid, set()).add(os.path.basename(path))

print(f"  distinct Ensembl IDs used: {len(sources):,}")

# ---------------------------------------------------------------
# 3. Resolve
# ---------------------------------------------------------------
rows = []
for gid, where in sorted(sources.items()):
    rec = recs.get(gid)
    rows.append({
        "gene_id": gid,
        "gene_symbol": (rec or {}).get("gene_symbol"),
        "gene_type": (rec or {}).get("gene_type"),
        "chr": (rec or {}).get("chr"),
        "start": (rec or {}).get("start"),
        "end": (rec or {}).get("end"),
        "resolved_in_gencode_v49": rec is not None,
        "symbol_differs_from_id": bool(
            rec and rec.get("gene_symbol")
            and not rec["gene_symbol"].startswith("ENSG")),
        "used_in": "; ".join(sorted(where)),
    })
res = pd.DataFrame(rows)
res.to_csv(f"{OUTDIR}/RA12_ensembl_to_symbol.tsv", sep="\t", index=False)

print()
print("=" * 74)
print("RESOLUTION SUMMARY")
print("=" * 74)
print(f"total IDs                       : {len(res)}")
print(f"resolved in GENCODE v49         : {int(res['resolved_in_gencode_v49'].sum())}")
print(f"with an informative symbol      : {int(res['symbol_differs_from_id'].sum())}")
print(f"symbol identical to Ensembl ID  : "
      f"{int((res['resolved_in_gencode_v49'] & ~res['symbol_differs_from_id']).sum())}")
print(f"NOT in GENCODE v49              : "
      f"{int((~res['resolved_in_gencode_v49']).sum())}")

if res["gene_type"].notna().any():
    print("\nbiotype of the IDs used:")
    print(res["gene_type"].value_counts().to_string())

# ---------------------------------------------------------------
# 4. The two IDs the referee named, plus the Signal 3 candidate gene
# ---------------------------------------------------------------
print()
print("=" * 74)
print("THE SPECIFIC IDs RAISED BY REVIEWER #3")
print("=" * 74)

targets = list(REFEREE_IDS)
row = df[df["variant_key"] == REFEREE_VARIANT]
if len(row):
    for c in ["gtex_eqtl_best_gene", "gtex_sqtl_best_gene", "nearest_gene",
              "within_nearest_gene"]:
        if c in row.columns:
            v = row.iloc[0].get(c)
            if isinstance(v, str):
                targets += ID_RE.findall(v)
                if v.startswith("ENSG"):
                    targets.append(v.split(".")[0])
    print(f"{REFEREE_VARIANT} gene annotations:")
    for c in ["gtex_eqtl_best_gene", "gtex_sqtl_best_gene", "nearest_gene",
              "within_nearest_gene"]:
        if c in row.columns:
            print(f"  {c:26s} {row.iloc[0].get(c)}")
else:
    print(f"{REFEREE_VARIANT} not present in the deduplicated table")

print()
for gid in dict.fromkeys(targets):
    rec = recs.get(gid)
    if rec:
        print(f"{gid}: symbol={rec['gene_symbol']}  biotype={rec['gene_type']}  "
              f"{rec['chr']}:{rec['start']}-{rec['end']} ({rec['strand']})")
        if rec["gene_symbol"] == gid or rec["gene_symbol"].startswith("ENSG"):
            print("    -> no approved symbol; report as an unnamed GENCODE gene "
                  "model with its biotype")
    else:
        print(f"{gid}: NOT in GENCODE v49")
        print("    -> report as unresolved. If it was present in an earlier")
        print("       release it may have been retired; state the release used.")

pd.DataFrame([
    {"gene_id": g,
     "gene_symbol": (recs.get(g) or {}).get("gene_symbol"),
     "gene_type": (recs.get(g) or {}).get("gene_type"),
     "in_gencode_v49": g in recs}
    for g in dict.fromkeys(targets)
]).to_csv(f"{OUTDIR}/RA12_referee_flagged_genes.tsv", sep="\t", index=False)

print()
print("  >> The referee could not find these in GTEx or gnomAD. A biotype such as")
print("     lncRNA, processed_pseudogene or TEC explains that directly, and is a")
print("     better answer than a symbol alone. Put the biotype in the table")
print("     footnote, not just the symbol.")

print(f"\n[RA12] done. Outputs in {OUTDIR}/")
print("Merge RA12_ensembl_to_symbol.tsv into Table 1 and every supplementary table.")
