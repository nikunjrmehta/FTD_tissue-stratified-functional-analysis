"""
RA15 — coding consequences, so APOE is recovered without re-running the pipeline.

WHY THIS EXISTS
---------------
Open Targets assigns APOE an L2G score of 0.94 at rs429358 -- the chr19 lead in
this study -- while this pipeline nominates no APOE at all (RA6b/RA6c). The reason
is structural: gene nomination here runs through GTEx eQTL/sQTL targets and
nearest-gene assignment, so it can only recover REGULATORY mechanisms. rs429358 is
the APOE e4-defining MISSENSE variant, and a coding mechanism is invisible to that
design.

Decision taken: include APOE. Implemented as an explicitly labelled coding-
consequence ANNOTATION rather than a change to the scoring, because:
  * adding a coding layer to the composite would require re-running steps 5-17
    and the entire 240-specification robustness analysis, which is not feasible
    in the time remaining and would invalidate the frozen numbers;
  * an annotation is more honest -- it reports what the pipeline did not capture
    rather than retrofitting the score so it appears to have captured it;
  * it directly answers Reviewer #3's request to explain what the framework adds
    and, equally, what it misses.

So the integrated score is unchanged, and Table 1 gains a column stating each
candidate's protein-level consequence and the gene affected. APOE then appears
where a reader expects it, attributed to the right evidence type.

Outputs -> work/step_RA15_coding_consequences/
"""

import json
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

from ra_common import SIGNAL_ORDER, add_signal_id, ensure_dir, load_final, order_signals

OUTDIR = ensure_dir("work/step_RA15_coding_consequences")
API = "https://api.platform.opentargets.org/api/v4/graphql"

PROTEIN_ALTERING = {
    "missense_variant", "stop_gained", "stop_lost", "start_lost",
    "frameshift_variant", "inframe_insertion", "inframe_deletion",
    "splice_acceptor_variant", "splice_donor_variant", "protein_altering_variant",
    "coding_sequence_variant", "incomplete_terminal_codon_variant",
    "start_retained_variant", "stop_retained_variant",
}


def gql(query, variables=None, retries=3):
    payload = json.dumps({"query": query, "variables": variables or {}}).encode()
    for attempt in range(1, retries + 1):
        req = urllib.request.Request(
            API, data=payload,
            headers={"Content-Type": "application/json",
                     "User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                out = json.loads(r.read().decode())
            if "errors" in out:
                for e in out["errors"][:5]:
                    print("    GraphQL error:", e.get("message"))
            return out
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode()
            except Exception:
                pass
            print(f"    HTTP {e.code}: {body[:700] or e.reason}")
            if e.code == 400:
                return {"error": body}
            time.sleep(4 * attempt)
        except Exception as e:
            print(f"    attempt {attempt}: {type(e).__name__}: {e}")
            time.sleep(4 * attempt)
    return {"error": "failed"}


# ---------------------------------------------------------------
# 1. Introspect TranscriptConsequence so the query uses real fields only.
#    Three earlier scripts failed on this API by assuming field names; this
#    builds the selection set from what the schema actually exposes.
# ---------------------------------------------------------------
print("introspecting TranscriptConsequence ...")
res = gql("""
query { __type(name: "TranscriptConsequence") {
  fields { name type { name kind ofType { name kind } } } } }
""")
avail = {f["name"] for f in
         (((res.get("data") or {}).get("__type") or {}).get("fields") or [])}
print(f"  fields available: {sorted(avail)}")

WISHLIST_SCALAR = ["aminoAcidChange", "impact", "isEnsemblCanonical",
                   "transcriptId", "consequenceScore", "codons",
                   "distanceFromFootprint", "distanceFromTss"]
sel = [f for f in WISHLIST_SCALAR if f in avail]
if "target" in avail:
    sel.append("target { id approvedSymbol biotype }")
if "variantConsequences" in avail:
    sel.append("variantConsequences { id label }")
elif "consequences" in avail:
    sel.append("consequences { id label }")
print(f"  using: {sel}")

Q = """
query($id: String!) {
  variant(variantId: $id) {
    id rsIds
    mostSevereConsequence { id label }
    transcriptConsequences { %s }
  }
}
""" % "\n      ".join(sel)

# ---------------------------------------------------------------
# 2. Which variants to annotate: every consensus candidate plus every lead
# ---------------------------------------------------------------
df, info = load_final(dedupe=True)
df = add_signal_id(df)

targets = {}
for sig in SIGNAL_ORDER:
    targets[sig] = {sig}
try:
    cons = pd.read_csv("work/step_RA0e_consensus/RA0e_candidate_sets_tier1.tsv",
                       sep="\t")
    for r in cons.itertuples():
        targets.setdefault(r.signal_id, set()).add(r.variant_key)
except FileNotFoundError:
    print("(RA0e consensus table not found -- annotating leads only)")

flat = [(s, v) for s, vs in targets.items() for v in sorted(vs)]
print(f"\nvariants to annotate: {len(flat)}")

# ---------------------------------------------------------------
# 3. Annotate
# ---------------------------------------------------------------
rows = []
for sig, vk in flat:
    c, p, r_, a = vk.split(":")
    vid = f"{c.replace('chr', '')}_{p}_{r_}_{a}"
    out = gql(Q, {"id": vid})
    with open(f"{OUTDIR}/RA15_ot_{vid}.json", "w") as f:
        json.dump(out, f, indent=2)
    v = ((out.get("data") or {}).get("variant") or {})
    msc = (v.get("mostSevereConsequence") or {})
    tcs = v.get("transcriptConsequences") or []

    # protein-altering consequences only, deduplicated by gene
    hits = {}
    for tc in tcs:
        terms = [x.get("label") or x.get("id") for x in
                 (tc.get("variantConsequences") or tc.get("consequences") or [])]
        terms = [t for t in terms if t]
        prot = [t for t in terms if t.replace(" ", "_").lower() in PROTEIN_ALTERING]
        if not prot:
            continue
        tgt = tc.get("target") or {}
        sym = tgt.get("approvedSymbol") or tgt.get("id")
        if not sym:
            continue
        prev = hits.get(sym, {"terms": set(), "aa": None, "canonical": False})
        prev["terms"].update(prot)
        prev["aa"] = prev["aa"] or tc.get("aminoAcidChange")
        prev["canonical"] = prev["canonical"] or bool(tc.get("isEnsemblCanonical"))
        hits[sym] = prev

    rows.append({
        "signal_id": sig,
        "variant_key": vk,
        "is_signal_lead": vk == sig,
        "rsid": ";".join(v.get("rsIds") or []),
        "most_severe_consequence": msc.get("label") or msc.get("id"),
        "n_transcript_consequences": len(tcs),
        "protein_altering": bool(hits),
        "coding_genes": "; ".join(sorted(hits)),
        "coding_detail": "; ".join(
            f"{g} ({', '.join(sorted(d['terms']))}"
            + (f", {d['aa']}" if d["aa"] else "")
            + (", canonical" if d["canonical"] else "") + ")"
            for g, d in sorted(hits.items())),
    })
    flag = "PROTEIN-ALTERING" if hits else ""
    print(f"  {vk:26s} {str(msc.get('label') or '-'):28s} {flag}")

ann = order_signals(pd.DataFrame(rows))
ann.to_csv(f"{OUTDIR}/RA15_coding_consequences.tsv", sep="\t", index=False)

print()
print("=" * 76)
print("PROTEIN-ALTERING CANDIDATES")
print("=" * 76)
pa = ann[ann["protein_altering"]]
if len(pa):
    print(pa[["signal_number", "variant_key", "rsid", "is_signal_lead",
              "most_severe_consequence", "coding_detail"]].to_string(index=False))
else:
    print("  none")

# ---------------------------------------------------------------
# 4. Combined gene list per signal: regulatory + coding
# ---------------------------------------------------------------
print()
print("=" * 76)
print("COMBINED GENE NOMINATIONS PER SIGNAL")
print("=" * 76)
try:
    reg = pd.read_csv("work/step14_signal_story/FTD_top_genes_per_signal.tsv",
                      sep="\t")
    rows = []
    for sig in SIGNAL_ORDER:
        rgenes = set(reg.loc[reg["signal_id"] == sig, "gene"].dropna().astype(str))
        cgenes = set()
        for s in ann.loc[ann["signal_id"] == sig, "coding_genes"].dropna():
            cgenes |= {x.strip() for x in str(s).split(";") if x.strip()}
        rows.append({
            "signal_number": SIGNAL_ORDER[sig],
            "signal_id": sig,
            "n_regulatory_genes": len(rgenes),
            "n_coding_genes": len(cgenes),
            "coding_only_genes": "; ".join(sorted(cgenes - rgenes)),
            "shared_genes": "; ".join(sorted(cgenes & rgenes)),
        })
    comb = pd.DataFrame(rows).sort_values("signal_number")
    comb.to_csv(f"{OUTDIR}/RA15_combined_gene_nominations.tsv",
                sep="\t", index=False)
    print(comb.to_string(index=False))
    print()
    print("  >> 'coding_only_genes' are genes the pipeline could NOT reach, because")
    print("     it nominates through eQTL/sQTL targets and nearest-gene assignment.")
    print("     If APOE appears there, that is the honest framing for Signal 3:")
    print("     a coding mechanism outside the framework's design, recovered by")
    print("     annotation and reported as such -- not a scoring failure to hide,")
    print("     and not a retrofit to make the score appear to have found it.")
except FileNotFoundError:
    print("(step14 gene table not found)")

print(f"\n[RA15] done. Outputs in {OUTDIR}/")
print("Add 'most_severe_consequence' and 'coding_genes' as columns in Table 1.")
