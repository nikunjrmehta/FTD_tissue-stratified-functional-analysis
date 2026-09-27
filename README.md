# Tissue-stratified functional interpretation of sporadic FTD risk loci

Analysis code for an integrative in-silico pipeline that interprets sporadic
frontotemporal dementia (sFTD) GWAS loci by combining association strength,
sequence-model variant-effect prediction, regulatory annotation and GTEx
molecular QTL evidence, and reports candidate variants in two evidence tiers.

## What is and is not in this repository

**Included:** all analysis code.

**Not included:** input data. None of it is ours to redistribute, and the GTEx
files are large. Obtain them from the original sources:

|Input|Source|
|-|-|
|sFTD GWAS summary statistics (discovery: 3,756 cases, 11,233 controls, GRCh37)|UCL Research Data Repository, as released with Manzoni et al. 2024|
|GTEx v11 eQTL/sQTL significant pairs, SuSiE fine-mapping, median TPM|GTEx portal|
|ENCODE candidate cis-regulatory elements (GRCh38)|ENCODE SCREEN registry|
|GENCODE v49 basic gene annotation|GENCODE|
|1000 Genomes GRCh38 reference panel|1000 Genomes Project|
|liftOver hg19→hg38 chain file|UCSC|

Place these under `data/` following the paths at the top of each script, which
are relative to the repository root.

\---

## Requirements

Python 3.10 or later, with:

```
pandas
numpy
scipy
scikit-learn
matplotlib
statsmodels
pyarrow
openpyxl
requests
alphagenome==0.5.1
```

External tools: **PLINK 2** (LD clumping) and **UCSC liftOver**. `run\_step3`
resolves the PLINK 2 binary from a candidate list and from `PATH`; edit that
list if yours is installed elsewhere.

The AlphaGenome version is pinned deliberately. Scores drift by roughly
0.2–0.3% between v0.5.1 and v0.7.0, which is small but enough to perturb
rankings among closely scored variants. The published results are v0.5.1.

Two analyses were run in Google Colab for GPU access and are provided as
standalone scripts rather than as part of the local pipeline:
`enformer\_colab\_RA9.py` and `allele\_check\_colab\_RA10.py`.

\---

## Reproducing the analysis

Set the working directory to the repository root before running anything; all
paths are relative to it.

### Stage A — primary pipeline

Run in numerical order. Each step writes to `work/step\*/` and reads the previous
step's output.

|Step|Script|Purpose|
|-|-|-|
|2|`run\_step2\_harmonize.py`|Parse summary statistics, liftOver to GRCh38|
|3|`run\_step3\_clump\_and\_loci\_spyder\_pchr.py`|LD clumping in PLINK 2, define 25 locus windows|
|4|`run\_step4\_prepare\_alphagenome\_inputs\_FTD\_spyder.py`|Build per-locus AlphaGenome intervals|
|5|`run\_step5\_alphagenome\_FTD\_spyder\_compact.py`|Score variants with AlphaGenome (needs API key)|
|6|`run\_step6\_prioritize\_ftd\_spyder.py`|Within-locus percentiles|
|7|`run\_step7\_annotate\_ftd\_spyder.py`|GENCODE proximity, ENCODE cCRE overlap|
|8|`run\_step8\_gtex\_ftd\_spyder.py`|GTEx eQTL/sQTL annotation|
|9|`run\_step9\_gtex\_sigpairs\_ftd\_spyder.py`|Significant variant–feature pairs|
|10|`run\_step10\_expression\_ftd\_spyder.py`|GTEx median expression context|
|11|`run\_step11\_susie\_ftd\_spyder.py`|SuSiE fine-mapping summaries|
|11.5|`run\_step11\_5\_susie\_groups\_ftd\_spyder.py`|Brain and blood/immune PIP aggregation|
|12|`run\_step12\_finalize\_ftd\_spyder.py`|Integrated score and rankings|
|13|`run\_step13\_ranking\_within\_GWAS.py`|Association-constrained rankings|
|14|`run\_step14\_signal\_level\_collapse.py`|Collapse 25 windows into 3 signals|
|14.5|`run\_step14\_5\_signal\_story\_tables\_spyder.py`|Signal-level tables|

### 

### Stage B analyses

`ra\_common.py` is a shared helper, not run directly. It defines the master-table
loader, the six component scores, the canonical signal ordering and labels, and
the `locus\_id → signal` mapping. Every downstream script uses it, which is what
keeps signal numbering consistent across text, tables and figures.

Two properties of `ra\_common.load\_final()` matter for interpreting any result:
it deduplicates to unique variants, because overlapping ±500 kb windows would
otherwise pseudo-replicate them; and it returns both the row count and the
unique-variant count, so both can be reported honestly.

|Script|Purpose|
|-|-|
|`run\_RA0b\_missingness\_impact.py`|Effect of the missing-value correction|
|`run\_RA0c\_layer\_influence.py`|Nominal versus realized layer influence|
|`run\_RA0d\_tier1\_stability.py`|Candidate stability under the Tier 1 constraint|
|`run\_RA0e\_consensus\_ranking.py`|Specification curve; consensus rank over 240 specifications|
|`run\_RA1\_redundancy.py`|Correlation, VIF, and PCA among component scores|
|`run\_RA2\_effective\_weight.py`|Weight attributable to each underlying data source|
|`run\_RA3\_weight\_sensitivity.py`|Sixteen weighting schemes and ablations|
|`run\_RA4\_gwas\_dynamic\_range.py`|Realized range of the GWAS layer|
|`run\_RA4b\_transform\_sensitivity.py`|Four p-value transforms|
|`run\_RA5b\_benchmark.py`|AUROC against fine-mapped variants, with circularity guards|
|`run\_RA6b\_opentargets\_compare.py`|Gene- and variant-level comparison with Open Targets|
|`run\_RA7\_mapt\_inversion.py`|17q21.31 LD extent and credible-set structure|
|`run\_RA7b\_h1h2\_tag\_check.py`|Association at the H1/H2 haplotype tags|
|`run\_RA8a\_brain\_blood.py`|Brain versus blood/immune separation|
|`run\_RA8b\_effect\_direction.py`|Risk allele versus direction of expression change|
|`run\_RA9\_enformer\_compare.py`|AlphaGenome versus Enformer on identical sets|
|`run\_RA10b\_allele\_impact.py`|Impact of reference-allele orientation|
|`run\_RA10d\_orientation\_invariance.py`|Empirical test of model orientation invariance|
|`run\_RA11\_annotation\_density.py`|Annotation-density control|
|`run\_RA12\_gene\_symbols.py`|Ensembl identifier to approved symbol|
|`run\_RA13\_apoe\_vs\_ad.py`|APOE effect size across dementia GWAS|
|`run\_RA15\_coding\_consequences.py`|Protein-level consequences|

Colab, run separately with GPU:

|Script|Purpose|
|-|-|
|`enformer\_colab\_RA9.py`|Enformer scoring of the benchmark variant sets|
|`allele\_check\_colab\_RA10.py`|Reference-allele audit against the genome|

\---

## Repository layout

```
ra\_common.py                     shared helper: loader, scores, signal ordering
run\_step\*.py                     primary pipeline, stages 2-14.5
run\_RA\*.py                       revision analyses
colab/                           scripts run in Google Colab
```

