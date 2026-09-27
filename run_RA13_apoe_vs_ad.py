"""
RA13 — is the chr19 signal AD-like? Testing the misdiagnosis hypothesis.

Reviewer #1, major 2:
    "I am not convinced that APOE is a true signal for FTD, but could also be an
     effect of misdiagnosis. Since FTD has no biomarker, it is possible that
     clinical FTD cases are in fact bvAD (or even lvPPA)..."
The editor made this a named priority (3).

RA6c already established the context: rs429358 -- the chr19 lead in this study --
is the credible-set lead variant for Alzheimer's disease, Lewy body disease and
vascular dementia in Open Targets, and it is the APOE e4-defining MISSENSE variant.

This script tests the hypothesis quantitatively without downloading multi-gigabyte
summary statistics: for each chr19 candidate it pulls effect sizes from every
dementia GWAS in Open Targets that fine-maps the same variant, and compares
direction and magnitude against the FTD effect in this study.

WHAT THE POSSIBLE OUTCOMES MEAN
  * Same direction, FTD effect SMALLER than AD -> consistent with dilution of a
    true AD signal by misdiagnosed cases. Supports the referee.
  * Same direction, comparable magnitude -> either a shared mechanism or heavy
    contamination; cannot be distinguished here, and the manuscript should say so.
  * Opposite direction -> argues against simple AD contamination.

Confirming the referee's suspicion is a good outcome. It is more defensible than
asserting an independent FTD signal at the most famous AD locus in the genome.

Outputs -> work/step_RA13_apoe_vs_ad/
"""

import json
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

from ra_common import add_signal_id, ensure_dir, load_final

OUTDIR = ensure_dir("work/step_RA13_apoe_vs_ad")
API = "https://api.platform.opentargets.org/api/v4/graphql"

DEMENTIA_PATTERN = ("alzheimer|dementia|lewy|cognitive|neurodegener|"
                    "frontotemporal|amyloid|tau |braak")
SIGNAL3 = "chr19:44908684:T:C"


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
            print(f"    HTTP {e.code}: {body[:600] or e.reason}")
            if e.code == 400:
                return {"error": body}
            time.sleep(4 * attempt)
        except Exception as e:
            print(f"    attempt {attempt}: {type(e).__name__}: {e}")
            time.sleep(4 * attempt)
    return {"error": "failed"}


# ---------------------------------------------------------------
# 1. Our own chr19 effect sizes, aligned to the ALT allele
# ---------------------------------------------------------------
df, info = load_final(dedupe=True)
df = add_signal_id(df)
audit = pd.read_csv(
    "work/step_RA10_allele_audit/RA10_allele_check_results.tsv", sep="\t")
df = df.merge(audit[["variant_key", "allele_status"]], on="variant_key", how="left")

df["true_ref"] = np.where(df["allele_status"] == "swapped", df["alt"], df["ref"])
df["true_alt"] = np.where(df["allele_status"] == "swapped", df["ref"], df["alt"])
df["gwas_beta"] = pd.to_numeric(df.get("beta"), errors="coerce")
if df["gwas_beta"].isna().all() and "or" in df.columns:
    df["gwas_beta"] = np.log(pd.to_numeric(df["or"], errors="coerce"))
ea = df["effect_allele"].astype(str).str.upper()
df["beta_wrt_alt"] = np.where(
    ea == df["true_alt"].astype(str).str.upper(), df["gwas_beta"],
    np.where(ea == df["true_ref"].astype(str).str.upper(), -df["gwas_beta"], np.nan))

c19 = df[df["signal_id"] == SIGNAL3].copy()
print(f"\n[RA13] chr19 variants: {len(c19):,}")

# candidates to query: the consensus set plus the lead
targets = {SIGNAL3}
try:
    cons = pd.read_csv("work/step_RA0e_consensus/RA0e_candidate_sets_tier1.tsv",
                       sep="\t")
    targets |= set(cons.loc[cons["signal_id"] == SIGNAL3, "variant_key"])
except FileNotFoundError:
    pass
targets = sorted(targets)
print(f"variants to query in Open Targets: {len(targets)}")

# ---------------------------------------------------------------
# 2. Dementia GWAS effect sizes from Open Targets
# ---------------------------------------------------------------
Q = """
query($id: String!) {
  variant(variantId: $id) {
    id rsIds
    credibleSets {
      count
      rows {
        studyLocusId studyId position
        beta zScore standardError
        pValueMantissa pValueExponent
        effectAlleleFrequencyFromSource
        finemappingMethod
        variant { id referenceAllele alternateAllele }
        study {
          id studyType traitFromSource nCases nControls
          publicationFirstAuthor publicationDate
          diseases { id name }
        }
      }
    }
  }
}
"""

rows = []
for vk in targets:
    c, p, r, a = vk.split(":")
    vid = f"{c.replace('chr', '')}_{p}_{r}_{a}"
    print(f"\n  {vk} -> {vid}")
    res = gql(Q, {"id": vid})
    with open(f"{OUTDIR}/RA13_ot_{vid}.json", "w") as f:
        json.dump(res, f, indent=2)
    v = ((res.get("data") or {}).get("variant") or {})
    cs = ((v.get("credibleSets") or {}).get("rows")) or []
    print(f"    credible sets: {len(cs)}")
    for cset in cs:
        st = cset.get("study") or {}
        if st.get("studyType") != "gwas":
            continue
        text = ((st.get("traitFromSource") or "") + " " +
                " ".join(d.get("name", "") for d in (st.get("diseases") or []))).lower()
        if not pd.Series([text]).str.contains(DEMENTIA_PATTERN, regex=True).iloc[0]:
            continue
        lead = (cset.get("variant") or {}).get("id")
        mant = cset.get("pValueMantissa")
        expo = cset.get("pValueExponent")
        rows.append({
            "our_variant": vk,
            "ot_variant_id": vid,
            "is_credible_set_lead": lead == vid,
            "ot_lead_variant": lead,
            "studyId": cset.get("studyId"),
            "trait": st.get("traitFromSource"),
            "author": st.get("publicationFirstAuthor"),
            "nCases": st.get("nCases"), "nControls": st.get("nControls"),
            "beta": cset.get("beta"),
            "zScore": cset.get("zScore"),
            "standardError": cset.get("standardError"),
            "p": (float(mant) * 10 ** int(expo)
                  if mant is not None and expo is not None else np.nan),
            "eaf": cset.get("effectAlleleFrequencyFromSource"),
            "ref": (cset.get("variant") or {}).get("referenceAllele"),
            "alt": (cset.get("variant") or {}).get("alternateAllele"),
        })

ot = pd.DataFrame(rows)
ot.to_csv(f"{OUTDIR}/RA13_dementia_gwas_effects.tsv", sep="\t", index=False)

print()
print("=" * 76)
print("1. DEMENTIA GWAS AT THE chr19 CANDIDATES")
print("=" * 76)
if len(ot):
    print(f"records: {len(ot)}  studies: {ot['studyId'].nunique()}")
    print(ot[["our_variant", "trait", "author", "nCases", "nControls",
              "beta", "zScore", "p", "is_credible_set_lead"]]
          .sort_values(["our_variant", "p"]).to_string(index=False))
else:
    print("no dementia GWAS credible sets returned for these variants")

# ---------------------------------------------------------------
# 3. Compare with the FTD effect in this study
# ---------------------------------------------------------------
print()
print("=" * 76)
print("2. FTD vs DEMENTIA EFFECT SIZES")
print("=" * 76)

ours = c19[c19["variant_key"].isin(targets)][
    ["variant_key", "p", "beta_wrt_alt", "true_ref", "true_alt"]].copy()
ours.columns = ["our_variant", "ftd_p", "ftd_beta_wrt_alt", "ftd_ref", "ftd_alt"]
print("\nFTD effects in this study:")
print(ours.to_string(index=False))

if len(ot):
    # keep only records where the credible-set lead IS our variant, so beta refers
    # to the same variant rather than to a different lead
    same = ot[ot["is_credible_set_lead"]].copy()
    print(f"\nrecords where the OT credible-set lead is our variant: {len(same)}")
    if len(same):
        m = same.merge(ours, on="our_variant", how="left")
        # OT beta is per ALT allele; ours is aligned to ALT above
        m["allele_match"] = (m["alt"].astype(str).str.upper()
                             == m["ftd_alt"].astype(str).str.upper())
        m["same_direction"] = np.sign(m["beta"]) == np.sign(m["ftd_beta_wrt_alt"])
        m["abs_ratio_dementia_over_ftd"] = (
            m["beta"].abs() / m["ftd_beta_wrt_alt"].abs())
        show = ["our_variant", "trait", "nCases", "beta", "ftd_beta_wrt_alt",
                "allele_match", "same_direction", "abs_ratio_dementia_over_ftd"]
        print(m[show].round(4).to_string(index=False))
        m.to_csv(f"{OUTDIR}/RA13_effect_size_comparison.tsv", sep="\t", index=False)

        ok = m[m["allele_match"] & m["beta"].notna() & m["ftd_beta_wrt_alt"].notna()]
        if len(ok):
            n_same = int(ok["same_direction"].sum())
            print()
            print(f"comparable records            : {len(ok)}")
            print(f"same effect direction as FTD  : {n_same} ({100*n_same/len(ok):.0f}%)")
            print(f"median |dementia beta| / |FTD beta| : "
                  f"{ok['abs_ratio_dementia_over_ftd'].median():.2f}")
            print()
            if n_same == len(ok) and ok["abs_ratio_dementia_over_ftd"].median() > 1.5:
                print("  >> Same direction, larger effect in dementia GWAS. This is")
                print("     what contamination of an FTD cohort by misdiagnosed AD")
                print("     would look like. Report it as supporting Reviewer #1's")
                print("     hypothesis rather than resisting it.")
            elif n_same == len(ok):
                print("  >> Same direction, comparable magnitude. Shared mechanism and")
                print("     contamination cannot be distinguished with these data;")
                print("     say so explicitly.")
            else:
                print("  >> Directions differ across studies. Report per study and do")
                print("     not draw a single conclusion.")
    else:
        print("\n  Our variants are not credible-set leads in any dementia GWAS, so")
        print("  effect sizes are not directly comparable. Report the qualitative")
        print("  overlap from RA6c instead: rs429358 is the lead variant for")
        print("  Alzheimer's, Lewy body and vascular dementia credible sets.")

# ---------------------------------------------------------------
print()
print("=" * 76)
print("3. FOR THE MANUSCRIPT")
print("=" * 76)
print("Independent of the effect-size comparison, three facts are established and")
print("belong in the Signal 3 discussion:")
print("  * the chr19 lead rs429358 is the APOE e4-defining MISSENSE variant;")
print("  * it is the credible-set lead for Alzheimer's, Lewy body and vascular")
print("    dementia GWAS in Open Targets (RA6c);")
print("  * Open Targets assigns APOE an L2G score of 0.94 there, while this")
print("    pipeline nominates no APOE (RA6b).")
print("Together these argue for presenting Signal 3 with explicit caution rather")
print("than as an independent FTD regulatory finding.")

print(f"\n[RA13] done. Outputs in {OUTDIR}/")
