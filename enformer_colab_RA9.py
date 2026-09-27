"""
RA9 (Colab) — score the RA5b benchmark variants with Enformer.

WHAT THIS ANSWERS
-----------------
Reviewer #2, Comment 1: "...does exchanging one tool with an alternative (such as
replacing AlphaGenome with Enformer) cause a big difference in the identified
candidate signals?"

Rather than re-ranking thousands of variants and reporting a concordance
statistic, this scores the RA5b matched benchmark set and asks the sharper
question: does Enformer predict fine-mapped variants better or worse than
AlphaGenome? AlphaGenome achieves AUROC 0.58 (PIP>=0.5) and 0.60 (PIP>=0.3)
against this standard. Enformer is scored on exactly the same variants, with the
same negatives, and summarised the same way. Neither model contributed to the
positive set, so the comparison is clean.

HOW TO RUN
----------
1. Open https://colab.research.google.com  ->  Runtime -> Change runtime type -> T4 GPU
2. Upload  work/step_RA5b_benchmark/RA5b_benchmark_variants_for_enformer.tsv
3. Upload this file and run:  %run enformer_colab_RA9.py
   (or paste the sections below into cells)
4. Download  RA9_enformer_scores.tsv  and place it in
   work/step_RA5b_benchmark/  , then run run_RA9_enformer_compare.py locally.

Runtime: ~20-40 min on a T4 for ~700 variants (2 forward passes each).
Cost: free tier is sufficient. The job is unattended -- start it and leave it.
"""

# ============================================================
# 1. Setup
# ============================================================
# FIX: this was previously a commented-out `!pip install` line, which is notebook
# magic and does nothing when the file is executed with %run. Dependencies are now
# installed programmatically so the script works either way.
import importlib
import subprocess
import sys

for _mod, _pkg in [("pyfaidx", "pyfaidx"), ("tensorflow_hub", "tensorflow-hub")]:
    try:
        importlib.import_module(_mod)
    except ImportError:
        print(f"installing {_pkg} ...", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", _pkg], check=True)

import gzip
import os
import shutil
import time
import urllib.request

import numpy as np
import pandas as pd

SEQUENCE_LENGTH = 393_216          # Enformer input window
CENTER_BINS = 3                    # central output bins used for the variant effect
MODEL_URL = "https://tfhub.dev/deepmind/enformer/1"
VARIANT_TSV = "RA5b_benchmark_variants_for_enformer.tsv"
OUT_TSV = "RA9_enformer_scores.tsv"

variants = pd.read_csv(VARIANT_TSV, sep="\t")
print(f"variants to score: {len(variants):,}")
print(variants["chr"].value_counts().to_string())

# ============================================================
# 2. Reference genome — only the chromosomes we need
# ============================================================
# FIX: a single hard-coded UCSC URL failed with a DNS error ("Temporary failure in
# name resolution") even though pip worked, so it was that host specifically.
# Now tries several mirrors with retries. Ensembl names chromosomes "17" rather
# than "chr17", so the FASTA header is rewritten on the fly to keep one key
# convention throughout.
SOURCES = [
    ("UCSC-soe", "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/{c}.fa.gz"),
    ("UCSC-cse", "https://hgdownload.cse.ucsc.edu/goldenPath/hg38/chromosomes/{c}.fa.gz"),
    ("Ensembl", "https://ftp.ensembl.org/pub/release-110/fasta/homo_sapiens/dna/"
                "Homo_sapiens.GRCh38.dna.chromosome.{n}.fa.gz"),
]
N_RETRIES = 3


def _fetch(url, dest, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def download_chrom(c):
    """Write {c}.fa with a '>{c}' header, from whichever mirror responds."""
    fa, gz = f"{c}.fa", f"{c}.fa.gz"
    if os.path.exists(fa) and os.path.getsize(fa) > 1_000_000:
        print(f"{c}: already present, skipping")
        return
    if not (os.path.exists(gz) and os.path.getsize(gz) > 1_000_000):
        n = c.replace("chr", "")
        last = None
        for name, tmpl in SOURCES:
            url = tmpl.format(c=c, n=n)
            for attempt in range(1, N_RETRIES + 1):
                try:
                    print(f"{c}: {name} attempt {attempt} ...", flush=True)
                    _fetch(url, gz)
                    print(f"{c}: got {os.path.getsize(gz) / 1e6:.0f} MB from {name}")
                    last = None
                    break
                except Exception as e:
                    last = e
                    print(f"   failed: {type(e).__name__}: {e}", flush=True)
                    time.sleep(5 * attempt)
            if last is None:
                break
        if last is not None:
            raise SystemExit(
                f"\nCould not download {c} from any mirror. Last error: {last}\n\n"
                "Workaround: download it on your own machine from\n"
                f"  https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/{c}.fa.gz\n"
                f"and upload {c}.fa.gz to this Colab session, then re-run. The script\n"
                "picks up an existing .fa.gz automatically."
            )
    # gunzip, normalising the header to '>{c}'
    print(f"{c}: decompressing ...", flush=True)
    with gzip.open(gz, "rt") as fin, open(fa, "w") as fout:
        first = fin.readline()
        fout.write(f">{c}\n" if first.startswith(">") else first)
        shutil.copyfileobj(fin, fout)
    os.remove(gz)


needed = sorted(variants["chr"].astype(str).unique())
for c in needed:
    download_chrom(c)

from pyfaidx import Fasta

genomes = {c: Fasta(f"{c}.fa", as_raw=True, sequence_always_upper=True)
           for c in needed}

# ============================================================
# 3. Model
# ============================================================
import tensorflow as tf
import tensorflow_hub as hub

print("TensorFlow:", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
print("GPU:", gpus)
if not gpus:
    print("\n!! No GPU detected. Runtime -> Change runtime type -> T4 GPU.")
    print("   On CPU this will take many hours rather than ~40 minutes.\n")

try:
    model = hub.load(MODEL_URL).model
except Exception as e:
    raise SystemExit(
        f"Failed to load Enformer from TF-Hub: {e}\n\n"
        "If this is a TensorFlow version incompatibility, pin an older TF in a "
        "fresh runtime:\n"
        "    !pip install -q 'tensorflow==2.15.*' 'tensorflow-hub'\n"
        "then Runtime -> Restart session and re-run this script. The downloaded "
        "chromosome FASTA files persist across restarts, so nothing is lost."
    )
print("Enformer loaded.")

_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}


def one_hot(seq: str) -> np.ndarray:
    x = np.zeros((len(seq), 4), dtype=np.float32)
    for i, b in enumerate(seq):
        j = _MAP.get(b, -1)
        if j >= 0:
            x[i, j] = 1.0          # N and other IUPAC codes stay all-zero
    return x


def window(chrom: str, pos: int) -> tuple:
    """1-based pos -> (0-based start, end) of the centred Enformer window."""
    half = SEQUENCE_LENGTH // 2
    start = (pos - 1) - half
    return start, start + SEQUENCE_LENGTH


def get_sequences(chrom, pos, ref, alt):
    """Return (ref_seq, alt_seq, ref_matches_genome) or None if out of bounds."""
    g = genomes[chrom][chrom]
    chrom_len = len(g)
    start, end = window(chrom, pos)
    if start < 0 or end > chrom_len:
        return None
    seq = g[start:end]
    off = (pos - 1) - start                     # variant offset within the window
    observed = seq[off:off + len(ref)]
    matches = observed.upper() == ref.upper()
    alt_seq = seq[:off] + alt + seq[off + len(ref):]
    # keep length fixed for indels
    if len(alt_seq) > SEQUENCE_LENGTH:
        alt_seq = alt_seq[:SEQUENCE_LENGTH]
    elif len(alt_seq) < SEQUENCE_LENGTH:
        alt_seq = alt_seq + g[end:end + (SEQUENCE_LENGTH - len(alt_seq))]
    return seq, alt_seq, matches


@tf.function
def _predict(x):
    return model.predict_on_batch(x)["human"]


def predict(seq: str) -> np.ndarray:
    x = tf.convert_to_tensor(one_hot(seq)[np.newaxis], dtype=tf.float32)
    return _predict(x).numpy()[0]                # (896, 5313)


# ============================================================
# 4. Score
# ============================================================
mid = 896 // 2
lo, hi = mid - CENTER_BINS // 2, mid + CENTER_BINS // 2 + 1

rows = []
for i, r in enumerate(variants.itertuples(), 1):
    chrom, pos = str(r.chr), int(r.pos)
    got = get_sequences(chrom, pos, str(r.ref), str(r.alt))
    if got is None:
        rows.append({"variant_key": r.variant_key, "enformer_status": "out_of_bounds"})
        continue
    ref_seq, alt_seq, matches = got

    pr = predict(ref_seq)
    pa = predict(alt_seq)
    d = pa - pr                                   # (896, 5313)
    dc = d[lo:hi]                                 # central bins only

    rows.append({
        "variant_key": r.variant_key,
        "enformer_status": "ok" if matches else "ref_mismatch",
        # summarisations mirroring how AlphaGenome was summarised, so the
        # comparison is like-for-like rather than an artefact of post-processing
        "enformer_max_abs_center": float(np.abs(dc).max()),
        "enformer_mean_abs_center": float(np.abs(dc).mean()),
        "enformer_l2_center": float(np.sqrt((dc ** 2).sum())),
        "enformer_max_abs_all_bins": float(np.abs(d).max()),
        "enformer_mean_abs_all_bins": float(np.abs(d).mean()),
    })

    if i % 25 == 0:
        print(f"{i}/{len(variants)}", flush=True)
        pd.DataFrame(rows).to_csv(OUT_TSV, sep="\t", index=False)   # crash-safe

out = pd.DataFrame(rows)
out.to_csv(OUT_TSV, sep="\t", index=False)

print("\ndone.")
print(out["enformer_status"].value_counts().to_string())
print(f"\nwrote {OUT_TSV} — download it and run run_RA9_enformer_compare.py locally.")
print("\nNOTE: any 'ref_mismatch' rows mean the recorded ref allele does not match")
print("GRCh38 at that position. That would be a liftover problem worth knowing")
print("about independently of the Enformer comparison — check before excluding.")
