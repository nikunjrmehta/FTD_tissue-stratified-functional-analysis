# -*- coding: utf-8 -*-
"""
Created on Sun Feb 15 22:01:56 2026

@author: nmehta22
"""

# run_step5_alphagenome_ftd_spyder.py
# Spyder/Windows-friendly AlphaGenome Step 5 (FTD only)

import os
from pathlib import Path
import pandas as pd

from alphagenome.data import genome
from alphagenome.models import dna_client
from alphagenome.models import variant_scorers


# =========================
# USER CONFIG (edit these)
# =========================
# Read from the environment so no credential is committed to the public repository
# required by Reviewer #3 (minor 4). Set it once before running, e.g.
#   Windows : setx ALPHAGENOME_API_KEY "your-key"      (then restart Spyder)
#   Linux   : export ALPHAGENOME_API_KEY="your-key"
# NOTE: the key previously hardcoded here has been exposed and must be rotated.
API_KEY = os.environ.get("ALPHAGENOME_API_KEY", "")
if not API_KEY:
    raise RuntimeError(
        "ALPHAGENOME_API_KEY is not set. Export it before running this script."
    )

# Point this to your Step 4 manifest (local path on your machine)
MANIFEST_TSV = r"work\step4_alphagenome_ftd\loci_manifest.tsv"

# Output directory for Step 5 results
OUTDIR = r"work\step5_alphagenome_ftd"

# Expected variants file inside each locus folder created in Step 4
# If your Step 4 wrote a different name, change this.
VARIANTS_FILENAME = "variants.tsv"

# Optional: limit loci for quick testing (e.g., 2). Set None to run all.
MAX_LOCI = None

# Optional: limit variants per locus for quick testing (e.g., 200). Set None for all.
MAX_VARIANTS_PER_LOCUS = None

# Parallelism inside the client (keep modest to avoid rate-limit pain)
MAX_WORKERS = 3


# =========================
# Helpers
# =========================
def require_columns(df: pd.DataFrame, cols, name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}. Found: {list(df.columns)}")


def normalize_chr(ch):
    ch = str(ch).strip()
    if ch.lower().startswith("chr"):
        return ch
    return f"chr{ch}"


def load_variants_table(path: Path) -> pd.DataFrame:
    """
    Load locus variants table. We try TSV first; fall back to CSV if needed.
    Required columns:
      - chr, pos, ref, alt
    Optional:
      - snpid / rsid / p / beta / se (kept if present)
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing locus variants file: {path}")

    # Try TSV then CSV
    try:
        df = pd.read_csv(path, sep="\t")
    except Exception:
        df = pd.read_csv(path)

    # Standardize column names (common variants)
    rename_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ["chrom", "chromosome"]:
            rename_map[c] = "chr"
        elif cl in ["bp", "position"]:
            rename_map[c] = "pos"
        elif cl == "a1":
            rename_map[c] = "alt"
        elif cl == "a2":
            rename_map[c] = "ref"
    if rename_map:
        df = df.rename(columns=rename_map)

    require_columns(df, ["chr", "pos", "ref", "alt"], name=str(path))

    df["chr"] = df["chr"].apply(normalize_chr)
    df["pos"] = pd.to_numeric(df["pos"], errors="raise").astype(int)
    df["ref"] = df["ref"].astype(str).str.upper()
    df["alt"] = df["alt"].astype(str).str.upper()
    
    # Keep only clean A/C/G/T alleles (drop anything ambiguous)
    df = df[df["ref"].isin(list("ACGT")) & df["alt"].isin(list("ACGT"))].copy()


    # Drop weird rows
    df = df[(df["ref"].str.len() >= 1) & (df["alt"].str.len() >= 1)].copy()
    return df


def build_model_interval_from_index_pos(chrom: str, index_pos_1based: int):
    """
    Safe way to build a model-compatible 1Mb interval:
    - make a dummy Variant at the index position (1-based)
    - take reference_interval (0-based)
    - resize to the model sequence length
    """
    dummy = genome.Variant(
        chromosome=chrom,
        position=int(index_pos_1based),   # 1-based
        reference_bases="N",
        alternate_bases="N",
    )
    interval = dummy.reference_interval.resize(dna_client.SEQUENCE_LENGTH_1MB)
    return interval


def variants_to_objects(df: pd.DataFrame):
    vars_ = []
    bad = 0
    for r in df.itertuples(index=False):
        ch = getattr(r, "chr")
        pos = int(getattr(r, "pos"))
        ref = str(getattr(r, "ref"))
        alt = str(getattr(r, "alt"))
        # quick sanity
        if ref not in "ACGT" or alt not in "ACGT" or pos <= 0:
            bad += 1
            continue
        vars_.append(
            genome.Variant(
                chromosome=ch,
                position=pos,          # 1-based
                reference_bases=ref,
                alternate_bases=alt,
            )
        )
    if bad:
        print(f"Note: skipped {bad} malformed variants")
    return vars_


# =========================
# Main
# =========================
def main():
    outdir = Path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(MANIFEST_TSV, sep="\t")
    require_columns(
        manifest,
        ["locus_id", "chr", "index_pos", "folder"],
        name="loci_manifest.tsv",
    )

    if MAX_LOCI is not None:
        manifest_run = manifest.head(int(MAX_LOCI)).copy()
    else:
        manifest_run = manifest.copy()

    # Create API client
    if not API_KEY or "PASTE_" in API_KEY:
        raise ValueError("Please set API_KEY at top of script.")

    dna_model = dna_client.create(API_KEY)

    # Recommended scorers for human (keeps things standardized)
    # See the long note in run_step5_alphagenome_FTD_spyder_compact.py:
    # get_recommended_scorers() needs the PROTO enum value. Passing the Python
    # enum member returns an empty list silently; the original run only worked
    # because score_variants() falls back to the server-side recommended set.
    scorers = variant_scorers.get_recommended_scorers(
        dna_client.Organism.HOMO_SAPIENS.value)
    if not scorers:
        raise RuntimeError(
            "get_recommended_scorers() returned 0 scorers - every score would be "
            "NaN. Check the alphagenome version and the organism argument."
        )
    print(f"AlphaGenome scorers: {len(scorers)}")

    all_locus_summaries = []
    all_scores_paths = []

    print(f"Running loci: {len(manifest_run)}")
    print(f"Writing to: {outdir}")

    for i, row in manifest_run.iterrows():
        locus_id = row["locus_id"]
        chrom = normalize_chr(row["chr"])
        index_pos = int(row["index_pos"])
        locus_folder = Path(str(row["folder"]))
        variants_path = locus_folder / VARIANTS_FILENAME

        print("\n" + "=" * 80)
        print(f"[{locus_id}] chr={chrom} index_pos={index_pos}")
        print(f"Variants file: {variants_path}")

        vdf = load_variants_table(variants_path)

        if MAX_VARIANTS_PER_LOCUS is not None:
            vdf = vdf.head(int(MAX_VARIANTS_PER_LOCUS)).copy()

        # Build 1Mb model interval centered at index variant
        interval = build_model_interval_from_index_pos(chrom, index_pos)

        # Create Variant objects
        vobjs = variants_to_objects(vdf)

        print(f"Variants loaded: {len(vobjs)}")
        print(f"Model interval: {interval} (width={interval.width})")

        # Score variants
        # NOTE: score_variants returns list[AnnData] (one per scorer)
        scores = dna_model.score_variants(
            intervals=interval,
            variants=vobjs,
            variant_scorers=scorers,
            organism=dna_client.Organism.HOMO_SAPIENS,
            progress_bar=True,
            max_workers=MAX_WORKERS,
        )

        # Convert to tidy long dataframe
        tidy = variant_scorers.tidy_scores(scores)

        # Add locus metadata + join back to original variant table (p/beta/etc if present)
        tidy["locus_id"] = locus_id

        # Save per-locus
        locus_out_tsv = outdir / f"{locus_id}_alphagenome_scores_tidy.tsv.gz"
        tidy.to_csv(locus_out_tsv, sep="\t", index=False, compression="gzip")
        print(f"Saved: {locus_out_tsv}")

        # Lightweight per-locus summary
        # (You can refine these rankings later; this is a first-pass “top effects” table.)
        # We try common columns produced by tidy_scores, but keep it robust.
        # Typical fields include: variant_id, raw_score/quantile_score, output_type, ontology_term, etc.
        score_col = None
        for cand in ["quantile_score", "raw_score", "score"]:
            if cand in tidy.columns:
                score_col = cand
                break

        if score_col is not None:
            top = (
                tidy.assign(abs_score=tidy[score_col].abs())
                    .sort_values("abs_score", ascending=False)
                    .head(50)
                    .copy()
            )
            top_out = outdir / f"{locus_id}_TOP50.tsv"
            top.to_csv(top_out, sep="\t", index=False)
            print(f"Saved: {top_out} (ranked by |{score_col}|)")
        else:
            print("Note: could not find a score column to rank (expected raw_score/quantile_score/score).")

        all_locus_summaries.append(
            {
                "locus_id": locus_id,
                "chr": chrom,
                "index_pos": index_pos,
                "n_variants_scored": len(vobjs),
                "tidy_rows": len(tidy),
                "tidy_path": str(locus_out_tsv),
            }
        )
        all_scores_paths.append(str(locus_out_tsv))

    # Save run manifest
    summary_df = pd.DataFrame(all_locus_summaries)
    summary_path = outdir / "step5_run_summary.tsv"
    summary_df.to_csv(summary_path, sep="\t", index=False)
    print("\n" + "=" * 80)
    print(f"Done. Summary: {summary_path}")
    print("Per-locus tidy score files:")
    for p in all_scores_paths:
        print(" -", p)


if __name__ == "__main__":
    main()
