# Retrieve biological-prior resources

From the repository root, in the environment with `requirements.txt` installed:

```bash
python -m preprocessing.biological_priors.prepare_all
```

This command retrieves **actual public data for all ten families**, prepares the
normalized tables, enriches mouse/protein identifier links, expands genomic FASTA,
pins model revisions and downloads both pretrained models. It writes
`preprocessing/results/priors.mouse.json`; no hand-filled example configuration is needed.
The JSON configs are generated locally under `preprocessing/results/`, which is
ignored by Git. They contain machine-local resource paths and should not be
committed. `prepare_all` creates both `priors.go.json` and `priors.mouse.json`
when absent. To generate just the GO config, run
`python -m preprocessing.biological_priors.prepare_go`.
It does not construct the large gene–gene matrices or run neural inference.

| Family | Retrieved/prepared inputs |
| --- | --- |
| GO | Existing official mouse GAF/OBO and full MGI/Ensembl identifier mapping; bootstrapped with `prepare_go` if absent. |
| Pathways/complexes | Reactome Ensembl mapping; native mouse Complex Portal participants normalized to memberships. |
| Interactions | STRING 12.0 full mouse links; BioGRID 5.0.262 mouse physical-assay records. |
| Regulation/motifs | Mouse TRRUST v2; JASPAR 2024 CORE vertebrate PFMs; matched GRCm38 genome/annotation. |
| Evolution | Paginated OMA mouse proteins, official strict-group flat file, normalized root HOG families/orthogroups, within-HOG mouse paralog families, and group genome-breadth descriptors. |
| Protein | MGI InterPro memberships; valid mapped Ensembl protein-coding isoforms; pinned ESM-2 t6/8M weights/tokenizer. |
| Genome | Ensembl release-102 GRCm38 primary-assembly FASTA and matched GTF; plain FASTA for indexed access. This explicit source assembly is not inferred from the sample. |
| Literature | Exact function comments from reviewed mouse UniProt entries, with accession references; pinned all-MiniLM-L6-v2 model assets. |
| Phenotypes | MGI nonconditional single-gene genotype/MP annotations, retaining allele, genotype, background and publications. |
| Perturbation | BioGRID mouse genetic interactions with an explicit assay allowlist. Quantitative fitness screens are optional and are not fabricated or required by this recipe. |

The default resource directory is
`preprocessing/results/biological_prior_resources/all/`. Downloads are reused and
verified against persistent SHA256 receipts in `downloads.json`. First-download
hashes are locally computed receipt hashes, not claimed upstream checksums.
Mutable provider endpoints retain their first downloaded snapshot. Use a new
`--resource-dir` for a fresh snapshot.

Each successful family is checkpointed into the configuration. Failed sources
remain explicit in `preparation-report.json`; `input-inventory.json` reports
missing files or model assets. Exit code 2 means preparation is incomplete.
Rerun the same command to reuse completed downloads and retry failures. Model
commits are pinned before downloading weights, so retries keep the same revision.

```bash
# Retrieve just the nine non-GO families, reusing the shared GO/MGI mapping.
python -m preprocessing.biological_priors.prepare_all \
  --families pathways interactions regulation evolution protein genomic literature phenotypes perturbation

# Optional data-only run; uncached model assets remain explicit in the inventory.
python -m preprocessing.biological_priors.prepare_all --skip-models

# Custom output/cache locations.
python -m preprocessing.biological_priors.prepare_all \
  --output-config preprocessing/results/priors.mouse.json \
  --resource-dir preprocessing/results/biological_prior_resources/all

# Offline inventory after retrieval.
python -m preprocessing.biological_priors.prepare_resources \
  --config preprocessing/results/priors.mouse.json
```

`--go-config` reuses a different existing GO config. An existing config generated
by `prepare_all` can be updated with `--families` without discarding its other
families. The nine non-GO notebooks now use `priors.mouse.json`; the GO notebook
keeps its existing `priors.go.json` workflow. Set `ENABLE_INFERENCE=True` in the
protein/literature notebooks to compute embeddings from the retrieved models.

Biopython handles FASTA and JASPAR reading; Pooch handles downloads/decompression;
Hugging Face Hub handles model retrieval; the existing prior adapters construct
features. The preparation adapters only normalize provider columns and declared
relationships. UniProt accession/entry-name links map Complex Portal and OMA
proteins to genes. Ensembl FASTA gene cross-references provide protein/STRING IDs.
Ambiguous mappings are retained rather than guessed.

Source-specific limitations are recorded in the generated audits: TRRUST's
four-column file does not identify every publication's experimental readout;
Complex Portal membership does not imply a causal interaction; OMA paralog-family
features are shared root HOGs containing multiple distinct mouse genes, not
pairwise duplication confidence estimates. OMA group genome breadth is a
sequence-homology descriptor, not alignment identity. Reviewed UniProt function
comments are used verbatim; no descriptions are generated. MGI phenotype
preparation excludes multigene genotypes rather than attributing joint effects
to each gene. Protein preparation removes terminal stops and reports discarded
invalid, unmapped or noncoding sequences; runtime selects the longest isoform
without expression ranking.

The existing example template is kept locally at
`preprocessing/results/priors.example.json`; it is not an input to this utility.
The preparation commands build the real configs without requiring that template. The new retrieval recipes have not been run end to
end; the full download and verification will be performed with the user.
