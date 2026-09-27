# -*- coding: utf-8 -*-
"""
Created on Sun Feb 15 22:01:56 2026

@author: nmehta22
"""

# run_step5_alphagenome_ftd_spyder.py
# Spyder/Windows-friendly AlphaGenome Step 5 (FTD only) - COMPACT VERSION (memory safe)

import os
from pathlib import Path
import numpy as np
import pandas as pd

from alphagenome.data import genome
from alphagenome.models import dna_client
from alphagenome.models import variant_scorers


# =========================
# USER CONFIG (EDIT THESE)
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

# Step 4 manifest
MANIFEST_TSV = r"work\step4_alphagenome_ftd\loci_manifest.tsv"

# Output directory for Step 5 results
OUTDIR = r"work\step5_alphagenome_ftd"

# variants file inside each locus folder
VARIANTS_FILENAME = "variants.tsv"

# Optional: limit loci for quick testing (e.g., 2). Set None to run all.
MAX_LOCI = None

# Optional: limit variants per locus for quick testing (e.g., 200). Set None for all.
MAX_VARIANTS_PER_LOCUS = None

# Parallelism inside the client (keep modest)
MAX_WORKERS = 3
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
    Load locus variants table.
    Required columns: chr, pos, ref, alt
    Keeps any extra metadata columns (p/beta/se/rsid/etc) as-is.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing locus variants file: {path}")

    # TSV expected
    try:
        df = pd.read_csv(path, sep="\t")
    except Exception:
        df = pd.read_csv(path)

    # Standardize some common column name variants
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

    # Keep only clean A/C/G/T alleles and valid positions
    df = df[df["ref"].isin(list("ACGT")) & df["alt"].isin(list("ACGT"))].copy()
    df = df[df["pos"] > 0].copy()

    return df


def build_model_interval_from_index_pos(chrom: str, index_pos_1based: int):
    """
    Safe way to build a model-compatible 1Mb interval:
    - make a dummy Variant at the index position (1-based)
    - take reference_interval (0-based)
    - resize to the model sequence length
    """
    dummy = genome.Variant(
        chromosome=normalize_chr(chrom),
        position=int(index_pos_1based),   # 1-based
        reference_bases="N",
        alternate_bases="N",
    )
    interval = dummy.reference_interval.resize(dna_client.SEQUENCE_LENGTH_1MB)
    return interval


def variants_to_objects(df: pd.DataFrame):
    """
    Convert dataframe rows to AlphaGenome Variant objects.
    Enforces chr formatting and A/C/G/T alleles.
    """
    vars_ = []
    bad = 0
    for r in df.itertuples(index=False):
        ch = normalize_chr(getattr(r, "chr"))
        pos = int(getattr(r, "pos"))
        ref = str(getattr(r, "ref")).upper()
        alt = str(getattr(r, "alt")).upper()

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


def _max_abs_over_all_dims(arr) -> np.ndarray:
    """
    If arr is (n_variants, n_features...), return per-variant max(abs(.)).
    If arr is (n_features...) for a single variant, return scalar max(abs(.)) as shape (1,).
    """
    a = np.asarray(arr)
    if a.ndim == 0:
        return np.array([float(abs(a))])
    if a.ndim == 1:
        # one variant, many features -> scalar max
        return np.array([float(np.max(np.abs(a)))])
    # a.ndim >= 2
    # if first dim is variants, reduce everything else
    axes = tuple(range(1, a.ndim))
    return np.max(np.abs(a), axis=axes)

def _scalar_max_abs_X(X):
    """
    Return scalar max(abs(X)) for a single-variant AnnData.X.
    Handles dense, sparse, and empty matrices safely.
    """
    if X is None:
        return np.nan
    try:
        A = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    except Exception:
        return np.nan
    if A.size == 0:
        return np.nan
    return float(np.max(np.abs(A)))


def _scorer_name_from_anndata(ad, fallback):
    name = None
    if hasattr(ad, "uns") and isinstance(ad.uns, dict):
        name = ad.uns.get("scorer_name") or ad.uns.get("name")
    return name if name else fallback

def scores_to_compact_df(scores, vdf: pd.DataFrame) -> pd.DataFrame:
    """
    Robust conversion of score_variants output into compact per-variant scores.

    Handles BOTH AlphaGenome return shapes:
      A) scorer-major: list[AnnData] where each AnnData has X shaped (n_variants, ...)
      B) variant-major: list[list[AnnData]] where outer length = n_variants

    Returns per-variant compact DataFrame with per-scorer max_abs columns + alphagenome_max_abs.
    """
    out = vdf.reset_index(drop=True).copy()

    out["variant_key"] = (
        out["chr"].astype(str) + ":" +
        out["pos"].astype(str) + ":" +
        out["ref"].astype(str) + ":" +
        out["alt"].astype(str)
    )

    if scores is None or len(scores) == 0:
        out["alphagenome_max_abs"] = np.nan
        return out

    first = scores[0]

    # -------------------------
    # Case A: scorer-major (list[AnnData])
    # -------------------------
    if hasattr(first, "X"):
        per_scorer_cols = []
        for j, ad in enumerate(scores, start=1):
            scorer_name = _scorer_name_from_anndata(ad, f"scorer{j}")
            col = f"{scorer_name}_max_abs"

            # ad.X should be (n_variants, n_features...)
            vals = _max_abs_over_all_dims(ad.X)

            # if vals is shape (n_variants,), align
            if len(vals) != len(out):
                raise ValueError(
                    f"Score length mismatch for {col}: got {len(vals)} scores, expected {len(out)} variants."
                )

            # avoid collisions
            if col in out.columns:
                k = 2
                while f"{col}_{k}" in out.columns:
                    k += 1
                col = f"{col}_{k}"

            out[col] = vals
            per_scorer_cols.append(col)

        out["alphagenome_max_abs"] = out[per_scorer_cols].max(axis=1) if per_scorer_cols else np.nan
        return out

    # -------------------------
    # Case B: variant-major (list[list[AnnData]])
    # -------------------------
    if not isinstance(first, list):
        raise TypeError(f"Unexpected scores type: first element is {type(first)}")

    n_variants = len(scores)
    if n_variants != len(out):
        # Sometimes library may return fewer if it silently dropped invalid variants;
        # fail loudly so we can correct mapping.
        raise ValueError(
            f"Variant count mismatch: scores has {n_variants} variants, vdf has {len(out)} rows."
        )

    # Determine scorer names from first variant’s list
    scorer_names = []
    for j, ad in enumerate(scores[0], start=1):
        scorer_names.append(_scorer_name_from_anndata(ad, f"scorer{j}"))

    # Build per-scorer columns (variant-major, so fill row-by-row)
    per_scorer_cols = []
    for name in scorer_names:
        col = f"{name}_max_abs"
        if col in out.columns:
            k = 2
            while f"{col}_{k}" in out.columns:
                k += 1
            col = f"{col}_{k}"
        out[col] = np.nan
        per_scorer_cols.append(col)

    empty_ct = 0
    # Fill
    for i in range(n_variants):
        variant_scores = scores[i]  # list[AnnData] for this variant
        if not isinstance(variant_scores, list):
            raise TypeError(f"Expected list[AnnData] for variant {i}, got {type(variant_scores)}")

        # For each scorer, compute scalar max_abs
        row_vals = []
        for ad in variant_scores:
            v = _scalar_max_abs_X(ad.X)
            if pd.isna(v):
                empty_ct += 1
            row_vals.append(v)

        # If the scorer count differs, handle gracefully
        m = min(len(row_vals), len(per_scorer_cols))
        for j in range(m):
            out.at[i, per_scorer_cols[j]] = row_vals[j]
        
        if hasattr(ad.X, "shape"):
            if np.prod(ad.X.shape) == 0:
                print("DEBUG empty X detected:", ad.X.shape)
        
    if empty_ct:
        print(f"Note: filled {empty_ct} empty scorer outputs with NaN.")

    out["alphagenome_max_abs"] = out[per_scorer_cols].max(axis=1) if per_scorer_cols else np.nan
    return out


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

    # Recommended scorers for human.
    #
    # IMPORTANT (found during the revision audit): get_recommended_scorers()
    # expects the PROTO enum value, not the Python enum member. Passing
    # dna_client.Organism.HOMO_SAPIENS returns an EMPTY list silently in every
    # alphagenome release, because SUPPORTED_ORGANISMS is keyed by
    # dna_model_pb2.Organism.values() (integers) and the membership test never
    # matches.
    #
    # The original run was unaffected only because score_variants() falls back to
    # the server-side recommended set when variant_scorers is empty -- "If no
    # variant scorers are provided, the recommended variant scorers for the
    # organism will be used." So the correct 19 scorers were used, but by
    # accident. If a future release drops that fallback, this would silently
    # produce all-NaN scores.
    #
    # Fixed to pass .value, with a hard stop so it can never fail quietly again.
    scorers = variant_scorers.get_recommended_scorers(
        dna_client.Organism.HOMO_SAPIENS.value)
    if not scorers:
        raise RuntimeError(
            "get_recommended_scorers() returned 0 scorers. Every AlphaGenome "
            "score would be NaN. Check the alphagenome version and the organism "
            "argument before running."
        )
    print(f"AlphaGenome scorers: {len(scorers)}")
    try:
        import alphagenome as _ag
        print(f"alphagenome version: {getattr(_ag, '__version__', 'unknown')}")
    except Exception:
        pass

    all_locus_summaries = []
    all_compact_paths = []

    print(f"Running loci: {len(manifest_run)}")
    print(f"Writing to: {outdir}")

    for _, row in manifest_run.iterrows():
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

        vobjs = variants_to_objects(vdf)

        # Build 1Mb model interval centered at index variant
        interval = build_model_interval_from_index_pos(chrom, index_pos)

        print(f"Variants loaded: {len(vobjs)}")
        print(f"Model interval: {interval} (width={interval.width})")

        # Define output paths (needed for resume-mode)
        out_compact = outdir / f"{locus_id}_variant_compact.tsv.gz"
        top_out = outdir / f"{locus_id}_TOP50.tsv"

        # Resume mode: skip loci already done
        if out_compact.exists() and top_out.exists():
            print(f"Skipping {locus_id} (outputs already exist)")
            all_compact_paths.append(str(out_compact))
            all_locus_summaries.append(
                {
                    "locus_id": locus_id,
                    "chr": chrom,
                    "index_pos": index_pos,
                    "n_variants_scored": -1,
                    "compact_path": str(out_compact),
                    "top50_path": str(top_out),
                }
            )
            continue

        # Score variants (returns list[AnnData])
        scores = dna_model.score_variants(
            intervals=interval,
            variants=vobjs,
            variant_scorers=scorers,
            organism=dna_client.Organism.HOMO_SAPIENS,
            progress_bar=True,
            max_workers=MAX_WORKERS,
        )
        
        # COMPACT summary
        compact = scores_to_compact_df(scores, vdf)
        compact["locus_id"] = locus_id
        
        if locus_id == "FTD_001":
            print("DEBUG: type(scores[0]) =", type(scores[0]))
            if isinstance(scores[0], list) and len(scores[0]) > 0:
                print("DEBUG: type(scores[0][0]) =", type(scores[0][0]))

        # COMPACT summary to avoid tidy_scores() MemoryError
        #print("DEBUG _scalar_max_abs_X:", _scalar_max_abs_X)
        compact = scores_to_compact_df(scores, vdf)
        compact["locus_id"] = locus_id

        # Save per-locus compact table
        out_compact = outdir / f"{locus_id}_variant_compact.tsv.gz"
        compact.to_csv(out_compact, sep="\t", index=False, compression="gzip")
        print(f"Saved: {out_compact}")
        all_compact_paths.append(str(out_compact))

        # TOP50 by alphagenome_max_abs
        top = compact.sort_values("alphagenome_max_abs", ascending=False).head(50).copy()
        top_out = outdir / f"{locus_id}_TOP50.tsv"
        top.to_csv(top_out, sep="\t", index=False)
        print(f"Saved: {top_out} (ranked by alphagenome_max_abs)")

        all_locus_summaries.append(
            {
                "locus_id": locus_id,
                "chr": chrom,
                "index_pos": index_pos,
                "n_variants_scored": int(len(vobjs)),
                "compact_path": str(out_compact),
                "top50_path": str(top_out),
            }
        )

    # Save run manifest
    summary_df = pd.DataFrame(all_locus_summaries)
    summary_path = outdir / "step5_run_summary.tsv"
    summary_df.to_csv(summary_path, sep="\t", index=False)

    print("\n" + "=" * 80)
    print(f"Done. Summary: {summary_path}")
    print("Per-locus compact score files:")
    for p in all_compact_paths:
        print(" -", p)


if __name__ == "__main__":
    main()
