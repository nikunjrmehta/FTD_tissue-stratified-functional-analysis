"""
RA6b — head-to-head against Open Targets Platform.

Reviewer #2, Comment 5:
    "...what does the present pipeline offer that cannot be obtained by running
     the same GWAS through FUMA or Open Targets? It'll add great value if the
     authors can perform a comparison against at least one established tool in
     terms of methodology, inputs, and outputs."
Reviewer #3, major 2: same request, plus "explain what added value this approach
provides". The editor made this priority (1).

Written against the schema recorded by RA6a (API 26.6.3, data 26.06), not against
remembered documentation. Confirmed there: credibleSets() accepts `regions`,
CredibleSet exposes l2GPredictions, and L2GPrediction carries {score, target}.

TWO COMPARISON AXES, KEPT SEPARATE
----------------------------------
Open Targets L2G prioritises GENES. This pipeline prioritises VARIANTS. Mixing
the two would be a category error a referee would catch, so they are reported
independently:

  GENE axis    OT L2G-prioritised genes vs the genes nominated here, per signal.
  VARIANT axis OT credible-set membership vs this pipeline's ranking, per signal.

The honest differentiator to claim is narrow and true: neither FUMA nor Open
Targets provides a TISSUE-STRATIFIED brain-vs-blood view, and neither integrates
a sequence-to-function model at variant resolution. Overlap in gene nominations
is therefore expected and is not a weakness -- concordance where the tools share
inputs, plus extra resolution where they do not, is the argument.

Outputs -> work/step_RA6_opentargets/
"""

import json
import time
import urllib.error
import urllib.request

import pandas as pd

from ra_common import add_signal_id, ensure_dir, load_final

OUTDIR = ensure_dir("work/step_RA6_opentargets")
API = "https://api.platform.opentargets.org/api/v4/graphql"

# ±500 kb around each lead, matching the locus windows used in this study
REGIONS = {
    "chr17:45680084:C:A": "17:45180084-46180084",
    "chr17:46751565:G:A": "17:46251565-47251565",
    "chr19:44908684:T:C": "19:44408684-45408684",
}
FTD_DISEASE = "MONDO_0017276"


def gql(query, variables=None, retries=3):
    """
    FIX: urllib raises HTTPError on a 400 and the previous version printed only
    'HTTP Error 400: Bad Request', discarding the response body -- which is where
    GraphQL puts the actual reason (e.g. an unknown field name). The body is now
    read from the error object, and a 400 is not retried because a malformed
    query will never succeed on a second attempt.
    """
    payload = json.dumps({"query": query, "variables": variables or {}}).encode()
    for attempt in range(1, retries + 1):
        req = urllib.request.Request(
            API, data=payload,
            headers={"Content-Type": "application/json",
                     "User-Agent": "Mozilla/5.0"},
        )
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                out = json.loads(r.read().decode())
            if "errors" in out:
                print("  GraphQL errors:")
                for e in out["errors"][:8]:
                    print("   ", e.get("message"))
            return out
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode()
            except Exception:
                pass
            print(f"  HTTP {e.code}: {body[:1200] or e.reason}")
            if e.code == 400:
                return {"error": f"bad request: {body[:1200]}"}
            time.sleep(4 * attempt)
        except Exception as e:
            print(f"  attempt {attempt} failed: {type(e).__name__}: {e}")
            time.sleep(4 * attempt)
    return {"error": "failed"}


# ---------------------------------------------------------------
# 1. Credible sets overlapping each locus
# ---------------------------------------------------------------
# FIX: the previous query used Study.publicationYear, which does NOT exist --
# the RA6a probe listed publicationDate, publicationFirstAuthor, publicationTitle,
# publicationJournal and pubmedId, but no publicationYear. One unknown field makes
# the whole query a 400. Every field below now appears in the RA6a introspection
# output. Also dropped `studyTypes: [gwas]` (the StudyTypeEnum literal was never
# verified) and the page argument on l2GPredictions; GWAS filtering is done
# client-side on study.studyType instead, which cannot fail.
Q_CS = """
query($regions: [String!], $size: Int!) {
  credibleSets(regions: $regions, page: {index: 0, size: $size}) {
    count
    rows {
      studyLocusId
      studyId
      chromosome
      position
      region
      finemappingMethod
      pValueMantissa
      pValueExponent
      variant { id rsIds referenceAllele alternateAllele }
      study {
        id studyType traitFromSource nCases nControls
        publicationFirstAuthor publicationDate pubmedId
        diseases { id name }
      }
      l2GPredictions {
        count
        rows { score target { id approvedSymbol } }
      }
    }
  }
}
"""

# ---------------------------------------------------------------
# FIX: the region query was syntactically VALID but returned count=0, so the
# `regions` string format is wrong rather than the query. The APOE locus
# unquestionably has GWAS credible sets in Open Targets.
#
# Two changes:
#   (a) the region format is probed across plausible spellings rather than
#       assumed, and the first that returns rows is used;
#   (b) a variant-anchored fallback via Variant.credibleSets, which needs no
#       region string at all and is anchored on exactly the variants we report.
# ---------------------------------------------------------------
REGION_FORMATS = [
    "{c}:{s}-{e}",          # 19:44408684-45408684
    "chr{c}:{s}-{e}",       # chr19:44408684-45408684
    "{c}_{s}_{e}",          # 19_44408684_45408684
    "{c}:{s}..{e}",         # 19:44408684..45408684
]


def probe_region_format(sig, chrom, start, end):
    """Return (format_string, response) for the first format that yields rows."""
    for fmt in REGION_FORMATS:
        region = fmt.format(c=chrom, s=start, e=end)
        res = gql(Q_CS, {"regions": [region], "size": 200})
        data = ((res.get("data") or {}).get("credibleSets") or {})
        n = len(data.get("rows") or [])
        print(f"    format '{fmt}' -> {region}: {n} rows "
              f"(count={data.get('count')})")
        if n:
            return region, res
    return None, None


Q_VAR_CS = """
query($id: String!) {
  variant(variantId: $id) {
    id rsIds
    credibleSets {
      count
      rows {
        studyLocusId
        studyId
        chromosome
        position
        region
        finemappingMethod
        pValueMantissa
        pValueExponent
        variant { id rsIds referenceAllele alternateAllele }
        study {
          id studyType traitFromSource nCases nControls
          publicationFirstAuthor publicationDate pubmedId
          diseases { id name }
        }
        l2GPredictions {
          count
          rows { score target { id approvedSymbol } }
        }
      }
    }
  }
}
"""

# variants to anchor on: the three leads plus every consensus candidate
anchor_ids = {}
for sig in REGIONS:
    c, p, r, a = sig.split(":")
    anchor_ids.setdefault(sig, set()).add(
        f"{c.replace('chr', '')}_{p}_{r}_{a}")
try:
    _cons = pd.read_csv("work/step_RA0e_consensus/RA0e_candidate_sets_tier1.tsv",
                        sep="\t")
    for row in _cons.itertuples():
        c, p, r, a = row.variant_key.split(":")
        anchor_ids.setdefault(row.signal_id, set()).add(
            f"{c.replace('chr', '')}_{p}_{r}_{a}")
except FileNotFoundError:
    print("(RA0e consensus table not found; anchoring on lead variants only)")

cs_rows, l2g_rows = [], []
seen_locus_ids = set()

for sig, region in REGIONS.items():
    print(f"\n=== {sig} ===")
    chrom = sig.split(":")[0].replace("chr", "")
    lead_pos = int(sig.split(":")[1])
    start, end = lead_pos - 500_000, lead_pos + 500_000

    print("  probing region formats:")
    good_region, res = probe_region_format(sig, chrom, start, end)

    rows = []
    if res is not None:
        rows = (((res.get("data") or {}).get("credibleSets") or {}).get("rows")) or []
        print(f"  region query worked with '{good_region}': {len(rows)} rows")
    else:
        print("  no region format returned rows -- using variant-anchored fallback")
        for vid in sorted(anchor_ids.get(sig, [])):
            vres = gql(Q_VAR_CS, {"id": vid})
            vdata = ((vres.get("data") or {}).get("variant") or {})
            vrows = ((vdata.get("credibleSets") or {}).get("rows")) or []
            print(f"    {vid}: {len(vrows)} credible sets")
            rows.extend(vrows)
            res = res or vres

    with open(f"{OUTDIR}/RA6b_credsets_{sig.replace(':', '_')}.json", "w") as f:
        json.dump(res if res is not None else {"note": "no data"}, f, indent=2)

    # de-duplicate: the same credible set can be reached from several variants
    rows = [r for r in rows
            if r.get("studyLocusId") not in seen_locus_ids
            and not seen_locus_ids.add(r.get("studyLocusId"))]
    print(f"  credible sets after de-duplication: {len(rows)}")

    for r in rows:
        st = r.get("study") or {}
        dis = "; ".join(d.get("name", "") for d in (st.get("diseases") or []))
        is_ftd = any(d.get("id") == FTD_DISEASE for d in (st.get("diseases") or []))
        v = r.get("variant") or {}
        cs_rows.append({
            "signal_id": sig,
            "studyLocusId": r.get("studyLocusId"),
            "studyId": r.get("studyId"),
            "studyType": st.get("studyType"),
            "trait": st.get("traitFromSource"),
            "diseases": dis,
            "is_ftd_study": is_ftd,
            "author": st.get("publicationFirstAuthor"),
            "publicationDate": st.get("publicationDate"),
            "pubmedId": st.get("pubmedId"),
            "nCases": st.get("nCases"), "nControls": st.get("nControls"),
            "lead_variant": v.get("id"),
            "lead_rsid": ";".join(v.get("rsIds") or []),
            "position": r.get("position"),
            "finemapping": r.get("finemappingMethod"),
            "n_l2g_genes": (r.get("l2GPredictions") or {}).get("count"),
        })
        for g in ((r.get("l2GPredictions") or {}).get("rows") or []):
            tgt = g.get("target") or {}
            l2g_rows.append({
                "signal_id": sig,
                "studyLocusId": r.get("studyLocusId"),
                "studyId": r.get("studyId"),
                "trait": st.get("traitFromSource"),
                "is_ftd_study": is_ftd,
                "l2g_score": g.get("score"),
                "gene_id": tgt.get("id"),
                "gene_symbol": tgt.get("approvedSymbol"),
            })

# FIX: an empty result produced a DataFrame with no columns, so the gene-axis
# section died with KeyError: 'signal_id' and masked the real problem (the 400).
# Explicit columns keep every downstream section working on an empty result.
CS_COLS = ["signal_id", "studyLocusId", "studyId", "studyType", "trait", "diseases",
           "is_ftd_study", "author", "publicationDate", "pubmedId", "nCases",
           "nControls", "lead_variant", "lead_rsid", "position", "finemapping",
           "n_l2g_genes"]
L2G_COLS = ["signal_id", "studyLocusId", "studyId", "trait", "is_ftd_study",
            "l2g_score", "gene_id", "gene_symbol"]
cs = pd.DataFrame(cs_rows, columns=CS_COLS)
l2g = pd.DataFrame(l2g_rows, columns=L2G_COLS)
cs.to_csv(f"{OUTDIR}/RA6b_credible_sets.tsv", sep="\t", index=False)
l2g.to_csv(f"{OUTDIR}/RA6b_l2g_predictions.tsv", sep="\t", index=False)

if len(cs) == 0:
    print("\n!! No credible sets returned. Read the HTTP/GraphQL error above --")
    print("   it names the offending field. Do not interpret the empty comparison")
    print("   below as 'Open Targets has no data here' until the query succeeds.")

print()
print("=" * 74)
print("1. WHAT OPEN TARGETS HAS AT THESE LOCI")
print("=" * 74)
if len(cs):
    print(cs.groupby("signal_id").agg(
        n_credible_sets=("studyLocusId", "nunique"),
        n_studies=("studyId", "nunique"),
        n_ftd_studies=("is_ftd_study", "sum"),
    ).to_string())
    print("\nFTD-specific credible sets:")
    ftd = cs[cs["is_ftd_study"]]
    if len(ftd):
        print(ftd[["signal_id", "studyId", "trait", "lead_rsid",
                   "nCases", "nControls", "n_l2g_genes"]].to_string(index=False))
    else:
        print("  NONE. The Manzoni sFTD GWAS is not ingested by Open Targets.")
        print("  >> Say this explicitly in the manuscript. It is itself part of the")
        print("     answer to Comment 5: the comparison can only be made at the")
        print("     locus/gene level because OT has no sFTD study at these loci,")
        print("     which is a limitation of the established tool, not of this work.")
else:
    print("no credible sets returned -- check the region strings and API response")

# ---------------------------------------------------------------
# 2. GENE axis
# ---------------------------------------------------------------
print()
print("=" * 74)
print("2. GENE AXIS — OT L2G vs genes nominated here")
print("=" * 74)

ours = pd.read_csv("work/step14_signal_story/FTD_top_genes_per_signal.tsv", sep="\t")
ours_by_sig = {s: set(g["gene"].dropna().astype(str))
               for s, g in ours.groupby("signal_id")}

rows = []
for sig in REGIONS:
    ot_all = set(l2g.loc[l2g["signal_id"] == sig, "gene_symbol"].dropna())
    ot_top = set(l2g[(l2g["signal_id"] == sig) & (l2g["l2g_score"] >= 0.5)]
                 ["gene_symbol"].dropna())
    mine = ours_by_sig.get(sig, set())
    rows.append({
        "signal_id": sig,
        "n_ot_l2g_genes": len(ot_all),
        "n_ot_l2g_score_ge_0.5": len(ot_top),
        "n_our_genes": len(mine),
        "n_shared_any": len(ot_all & mine),
        "n_shared_high_confidence": len(ot_top & mine),
        "shared_genes": "; ".join(sorted(ot_all & mine)),
        "ot_only": "; ".join(sorted(ot_all - mine)[:12]),
        "ours_only": "; ".join(sorted(mine - ot_all)[:12]),
    })
gene_cmp = pd.DataFrame(rows)
gene_cmp.to_csv(f"{OUTDIR}/RA6b_gene_axis_comparison.tsv", sep="\t", index=False)
print(gene_cmp.to_string(index=False))

# ---------------------------------------------------------------
# 3. VARIANT axis
# ---------------------------------------------------------------
print()
print("=" * 74)
print("3. VARIANT AXIS — are our candidates in OT credible sets?")
print("=" * 74)

df, info = load_final(dedupe=True)
df = add_signal_id(df)
df["ot_id"] = (df["chr"].astype(str).str.replace("chr", "", regex=False) + "_"
               + df["pos"].astype(str) + "_" + df["ref"].astype(str) + "_"
               + df["alt"].astype(str))

try:
    cons = pd.read_csv("work/step_RA0e_consensus/RA0e_candidate_sets_tier1.tsv", sep="\t")
    cons = cons.merge(df[["variant_key", "ot_id"]], on="variant_key", how="left")
    ot_leads = set(cs["lead_variant"].dropna())
    cons["is_ot_credible_set_lead"] = cons["ot_id"].isin(ot_leads)
    print(cons[["signal_id", "variant_key", "p", "pct_specs_in_top1",
                "is_ot_credible_set_lead"]].to_string(index=False))
    cons.to_csv(f"{OUTDIR}/RA6b_variant_axis_comparison.tsv", sep="\t", index=False)
except FileNotFoundError:
    print("(run RA0e first)")

print()
print("=" * 74)
print("HOW TO USE THIS")
print("=" * 74)
print("* Report the two axes separately. OT prioritises genes; this pipeline")
print("  prioritises variants. Presenting cross-axis disagreement as added value")
print("  is a category error.")
print("* Overlap on the gene axis is EXPECTED and is a validation, not a problem.")
print("* The defensible claim is what OT cannot produce at all: a tissue-stratified")
print("  brain-vs-blood view, and variant-resolution sequence-model scoring.")
print("* If no FTD study is ingested, that fact belongs in the manuscript.")
