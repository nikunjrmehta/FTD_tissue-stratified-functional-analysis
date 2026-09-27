"""
RA10d — is the AlphaGenome score invariant to ref/alt orientation?

THE ONLY QUESTION LEFT
----------------------
278 chr19 variants have ref/alt reversed relative to GRCh38. Established: all
positions are correct, no strand flips, no consensus candidate affected, and the
swap rate declines smoothly with distance from the lead SNP rather than tracking
locus or allele composition.

What remains is whether it changes any number. Step 5 summarises AlphaGenome
output with a MAX OF ABSOLUTE VALUES. Reversing ref and alt negates the predicted
effect, and the absolute value undoes the negation -- so the score should be
unchanged. That is an argument, not evidence. This tests it directly by scoring
the same variant in both orientations.

This mirrors run_step5_alphagenome_FTD_spyder_compact.py exactly: the same
1 Mb interval construction (dummy variant at the locus index position, resized),
the same dna_client.Organism.HOMO_SAPIENS scorer set, the same
max-abs-over-all-dims summarisation. If it did not, a difference could reflect my
setup rather than allele orientation.

SETUP
-----
    pip install alphagenome
    setx ALPHAGENOME_API_KEY "your-rotated-key"     (then restart Spyder)

COST: N_SAMPLE scoring calls, each covering both orientations. Default 40.

Outputs -> work/step_RA10_allele_audit/
"""

import os

import numpy as np
import pandas as pd

from alphagenome.data import genome
from alphagenome.models import dna_client, variant_scorers

from ra_common import ensure_dir

OUTDIR = ensure_dir("work/step_RA10_allele_audit")
SWAPPED = f"{OUTDIR}/RA10b_swapped_for_rescoring.tsv"
MANIFEST = "work/step4_alphagenome_ftd/loci_manifest.tsv"

N_SAMPLE = 40
TOL = 1e-9
RNG = np.random.default_rng(20260808)

# SECURITY: the exposed key was re-added here as a default fallback. Removed --
# this file is destined for the public GitHub deposit that Reviewer #3 asked for.
# The key must still be rotated; it has been in plaintext in several scripts.
API_KEY = os.environ.get("ALPHAGENOME_API_KEY", "")
if not API_KEY:
    raise RuntimeError(
        "ALPHAGENOME_API_KEY is not set.\n"
        '  setx ALPHAGENOME_API_KEY "your-key"   then restart Spyder'
    )


# ---------------------------------------------------------------
# Helpers copied from run_step5_alphagenome_FTD_spyder_compact.py so the
# summarisation is identical
# ---------------------------------------------------------------
def normalize_chr(c):
    c = str(c)
    return c if c.startswith("chr") else f"chr{c}"


def _max_abs_over_all_dims(arr) -> np.ndarray:
    a = np.asarray(arr)
    if hasattr(a, "toarray"):
        a = a.toarray()
    a = np.asarray(a, dtype=float)
    if a.ndim == 0:
        return np.array([float(abs(a))])
    if a.ndim == 1:
        return np.array([float(np.max(np.abs(a)))])
    axes = tuple(range(1, a.ndim))
    return np.max(np.abs(a), axis=axes)


def build_model_interval_from_index_pos(chrom, index_pos_1based):
    dummy = genome.Variant(
        chromosome=normalize_chr(chrom),
        position=int(index_pos_1based),
        reference_bases="N",
        alternate_bases="N",
    )
    return dummy.reference_interval.resize(dna_client.SEQUENCE_LENGTH_1MB)


# ---------------------------------------------------------------
sw = pd.read_csv(SWAPPED, sep="\t")
man = pd.read_csv(MANIFEST, sep="\t")
index_pos = dict(zip(man["locus_id"], man["index_pos"]))

sample = sw.sample(n=min(N_SAMPLE, len(sw)),
                   random_state=int(RNG.integers(1e6))).reset_index(drop=True)
print(f"[RA10d] testing {len(sample)} of {len(sw)} swapped variants, "
      f"both orientations\n")

dna_model = dna_client.create(API_KEY)

# FIX: get_recommended_scorers() expects the PROTO enum value, not the Python
# enum member. Passing dna_client.Organism.HOMO_SAPIENS returns an empty list
# SILENTLY (the comparison `organism in SUPPORTED_ORGANISMS[...]` just never
# matches), which is why the first run reported "scorers: 0". Using .value gives
# the expected 19 scorers -- matching the scorer1..scorer19 columns from step 5.
scorers = variant_scorers.get_recommended_scorers(
    dna_client.Organism.HOMO_SAPIENS.value)
print(f"scorers: {len(scorers)}")
if len(scorers) == 0:
    raise SystemExit(
        "get_recommended_scorers returned 0 scorers. Every score would be NaN. "
        "Check the alphagenome version and the organism argument before "
        "proceeding."
    )
print()

rows = []
for i, r in enumerate(sample.itertuples(), 1):
    chrom = normalize_chr(r.chr)
    pos = int(r.pos)
    locus = r.locus_id
    if locus not in index_pos:
        print(f"  {r.variant_key}: locus {locus} not in manifest, skipping")
        continue

    interval = build_model_interval_from_index_pos(chrom, index_pos[locus])

    # both orientations of the SAME site, in one call
    vobjs = [
        genome.Variant(chromosome=chrom, position=pos,
                       reference_bases=str(r.ref), alternate_bases=str(r.alt)),
        genome.Variant(chromosome=chrom, position=pos,
                       reference_bases=str(r.ref_corrected),
                       alternate_bases=str(r.alt_corrected)),
    ]

    try:
        scores = dna_model.score_variants(
            intervals=interval,
            variants=vobjs,
            variant_scorers=scorers,
            organism=dna_client.Organism.HOMO_SAPIENS,
            progress_bar=False,
        )
    except Exception as e:
        print(f"  {r.variant_key}: FAILED {type(e).__name__}: {e}")
        continue

    # FIX: score_variants returns list[list[AnnData]] -- VARIANT-major, one inner
    # list of per-scorer AnnData per variant. The previous code assumed
    # scorer-major and hit "'list' object has no attribute 'X'".
    per_variant = []
    for variant_scores in scores:                    # one entry per variant
        vals = []
        for ad in variant_scores:                    # one entry per scorer
            v = _max_abs_over_all_dims(ad.X)
            vals.append(float(np.nanmax(v)) if len(v) else np.nan)
        vals = [v for v in vals if np.isfinite(v)]
        per_variant.append(max(vals) if vals else np.nan)

    if len(per_variant) != 2 or not np.isfinite(per_variant).all():
        print(f"  {r.variant_key}: incomplete scores, skipping")
        continue

    recorded, corrected = per_variant

    rows.append({
        "variant_key": r.variant_key,
        "locus_id": locus,
        "recorded_ref": r.ref, "recorded_alt": r.alt,
        "score_as_recorded": float(recorded),
        "score_as_corrected": float(corrected),
        "abs_difference": float(abs(recorded - corrected)),
        "identical": bool(abs(recorded - corrected) < TOL),
        "stored_alphagenome_max_abs": float(getattr(r, "alphagenome_max_abs", np.nan)),
    })
    if i % 5 == 0:
        print(f"  {i}/{len(sample)}", flush=True)
        pd.DataFrame(rows).to_csv(
            f"{OUTDIR}/RA10d_orientation_invariance.tsv", sep="\t", index=False)

res = pd.DataFrame(rows)
res.to_csv(f"{OUTDIR}/RA10d_orientation_invariance.tsv", sep="\t", index=False)

print()
print("=" * 74)
print("RESULT")
print("=" * 74)
if len(res) == 0:
    raise SystemExit("no variants scored -- check the API key and quota")

n_ident = int(res["identical"].sum())
print(f"variants tested               : {len(res)}")
print(f"identical in both orientations: {n_ident} "
      f"({100.0 * n_ident / len(res):.1f}%)")
print(f"max absolute difference       : {res['abs_difference'].max():.3e}")
print(f"median absolute difference    : {res['abs_difference'].median():.3e}")

d = (res["score_as_recorded"] - res["stored_alphagenome_max_abs"]).abs()
if d.notna().any():
    print(f"\nagreement with the values stored by step 5: max diff {d.max():.3e}")
    if d.max() > 1e-6:
        print("  !! These should match. A large difference means this script is not")
        print("     reproducing step 5 (interval, scorer set, or AlphaGenome version)")
        print("     -- resolve that BEFORE interpreting the invariance result.")
    else:
        print("  step 5 reproduced exactly.")

print()
if n_ident == len(res):
    print("INVARIANT. The max-absolute-value summarisation is unaffected by allele")
    print("orientation, so the 278 swapped variants already carry correct")
    print("AlphaGenome scores and NO result changes. Put this in the response letter")
    print("with the number of variants tested: it turns a potential objection into a")
    print("demonstrated non-issue.")
else:
    print("NOT INVARIANT. Some scorers are not antisymmetric under allele reversal.")
    print("All 278 must be re-scored with corrected orientation and steps 6-14")
    print("re-run. Check the magnitude of the differences before assuming rankings")
    print("move -- they may still be too small to matter.")

print(f"\n[RA10d] done. Outputs in {OUTDIR}/")
