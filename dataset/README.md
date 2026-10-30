# Loading datasets

## Download all 11 recommended starting points

The Bash entry point follows the CPU Slurm/Conda setup in the neighboring
MitoMap scripts. It selects all 11 resources in the catalog's priority order and
all their available stages, saving to
`/insomnia001/depts/morpheus/projects/hackathon/Datasets` by default:

```bash
# Preview the resource list without network access or data writes.
bash dataset/data/download_starting_points.sh --dry-run

# Run interactively, or submit the same script to Slurm.
bash dataset/data/download_starting_points.sh
sbatch dataset/data/download_starting_points.sh
```

It activates the `hackathon` Conda environment by default. Set `CONDA_ENV` to
choose another environment, or `PYTHON_BIN` to use a Python executable directly.
Set `DATA_DIR` or pass `--data-dir` to change the destination. For a subset:

```bash
bash dataset/data/download_starting_points.sh --datasets seu-3d tome harland
```

The runner downloads processed expression files, annotations and lineage trees;
it omits sequencing reads, image bundles, Pijuan-Sala chimera experiments, mdCLA
pituitary/kidney add-ons, and MOSTA adult controls and later Spateo products.
MOSTA uses its original 53 section H5AD files. Harland's observed and imputed
H5ADs remain separate. MELA retains all 16 native H5TD files and their trees.
The Extended Mouse Atlas server lists a 25 GB archive; the runner extracts only
its H5AD to avoid redundant RDS/Zarr/count copies. Expect substantial disk use
for all stages, especially MOSTA and MELA; archives are retained after extraction.

Downloads show progress, retry network failures, resume `.part` files when the
server supports byte ranges, and reuse completed files and converted H5ADs.
Public file listings are cached as `download_sources.json` per resource; use
`--refresh-links` to rediscover them. A resource failure does not prevent later
resources from running. `starting_points_status.json` reports successes/failures;
any failure gives a nonzero exit status. Interrupted extraction resumes on rerun.

The batch script requests `--requeue` for preemption on `burst`. Restarted jobs
reuse the same persistent data directory: completed downloads/extracted files
are skipped, and partial downloads continue from their saved byte offset. If a
server rejects byte-range resume, only that unfinished file is downloaded again.
An interrupted RDS conversion restarts that sample; its H5AD is published only
after a complete write, so partial conversions are never treated as complete.
Logs append across restarts and include `SLURM_RESTART_COUNT`. Slurm controls
preemption requeueing; no signal trap resubmits duplicate jobs or retries ordinary
application errors. See the [Slurm requeue option](https://slurm.schedmd.com/sbatch.html#OPT_requeue).
For a job submitted before this change, enable requeueing with
`scontrol update JobId=YOUR_JOB_ID Requeue=1`; new submissions use the script's
directive automatically.

By default, downloaded Seurat RDS files are converted through the existing
Python loader in separate processes to release memory between samples. Use
`--download-only` to skip conversion. Unsupported RDS classes (for example,
SingleCellExperiment products) remain downloaded and their conversion failures
are reported; R is never installed or invoked for RDS. Tables/MTX remain in the
original published format and may need a loader manifest before analysis. This
runner does not claim that every downloaded product is immediately loadable by
`load_dataset`. See `--help` for the available options.

Source routes were checked against the linked official pages in `Datasets.md`
and the authors' Pijuan-Sala and Harland download instructions. File listings
are resolved when you execute the script; syntax and the offline preview were
checked, but the full data transfers have not been run here.

## Loading individual resources

Install the shared Python dependencies from the repository root (Python 3.11+):

```bash
pip install -r requirements.txt
```

Choose a download/cache root with `data_dir` or `--data-dir`:

```python
from dataset.load_dataset import load_dataset

samples = load_dataset("tome", timepoint="E7.5", data_dir="/path/to/datasets")
```

```bash
python dataset/load_dataset.py tome --timepoint E7.5 --data-dir /path/to/datasets
```

All timepoints are selected by default. To download and convert every TOME stage
into the shared dataset directory, run from the repository root:

```bash
python dataset/load_dataset.py tome --data-dir /insomnia001/depts/morpheus/projects/hackathon/Datasets
```

You can also specify `--timepoint all` explicitly. In Python,
`load_dataset("tome", timepoint="all", data_dir="/path/to/datasets")` selects all
stages; omitting `timepoint` has the same effect. Use `--timepoint E7.5` for one
stage. TOME has 20 files, including separate E8.5a and E8.5b files.

Progress messages show the selected timepoints, destination, each download with
size/progress updates, cached files, conversion method, saved H5AD paths, sample
dimensions, and completion. Use `--quiet` or Python's `verbose=False` to suppress
progress. The CLI processes samples one at a time; Python's `load_dataset`
returns a dictionary holding all selected samples. For sequential processing in
Python, use `iter_dataset`, which yields `(sample_name, data)` pairs.

This stores files in `/path/to/datasets/tome/`. The default root is
`dataset/data/`, and SEU-3D uses the subfolder `GSE278603_RAW`. Gene annotation
downloads use `<data_dir>/annotations/`. A `dataset.json` manifest belongs inside
the dataset's subfolder; see `python dataset/load_dataset.py --help` for formats.

Seurat `.rds` files are read using
[readseurat](https://github.com/harryhaller001/readseurat), a Python package that
converts them directly to AnnData without R. Each successful RDS conversion is
saved beside its source, e.g. `seurat_object_E7.5.rds` becomes
`seurat_object_E7.5.h5ad`. Later calls prefer that H5AD, including when a manifest
still names the RDS or the original RDS has been removed. Cached conversions are
reused regardless of source modification time; remove the H5AD to reconvert an
updated source. Counts, metadata, and embeddings preserved by the converter are
stored in the cache before loader-specific gene-name and metadata processing.
When counts are available, `X` contains counts and the converter's original `X`
(Seurat's normalized data) is retained as `layers["logcounts"]`.

RDS conversion uses Python only and never falls back to `Rscript`. The loader
handles older Seurat assays (including TOME) by preserving gene names from
the expression matrix's R `Dimnames`, which `readseurat` 0.1.0 otherwise drops.
Missing dependencies and unsupported RDS objects raise an error identifying the
Python conversion problem. Install dependencies using the same Python that runs
the loader: `python -m pip install -r requirements.txt`.

`.rdata`/`.rda` workspaces still use the separate R conversion path, which needs
`Rscript`, `SingleCellExperiment`, `zellkonverter`, and `Seurat` for Seurat objects.
