
# Fieldnote — Gene panel planner

A local research prototype for preparing a gene panel for the next measurement round of a spatial transcriptomics experiment. The interface validates AnnData files, reports exact overlap, selects a reproducible random panel, and runs the repository's E7.5 reconstruction benchmark for the local mouse-embryo preset.

## Run locally

Requirements: Python 3.11+ and Node.js 20+ with npm.

In one terminal, install and start the API:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

In a second terminal, install and start the React app:

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite URL (normally `http://127.0.0.1:5173`). The frontend API URL defaults to `http://127.0.0.1:8000`; set `VITE_API_URL` at build time to use another API origin.

Set `GENE_PANEL_MAX_UPLOAD_MB` to change the per-file upload limit (default 250 MB), `GENE_PANEL_DATA_DIR` to change temporary storage, and `GENE_PANEL_SESSION_TTL_SECONDS` to change the idle-session retention period (default 24 hours). Uploads are written to generated per-session directories, are not sent to external data services, and are removed after the session expires and a subsequent API request triggers cleanup, or when the session is explicitly deleted. Session IDs are unguessable capabilities; run the API on a trusted local network.

## Trying it after cloning

The repository does not include any datasets (`data/` is git-ignored), so a fresh clone can run the app in two ways:

- **Without data:** click **Try synthetic example**, or upload your own `.h5ad` files. Gene selection and manual entry work; the `eval_team` benchmark and dummy methods are unavailable because they need the E7.5 files.
- **With the E7.5 files:** place `GSM9046243_Embryo_E7.5_stereo_rep1.h5ad` and `GSM9046244_Embryo_E7.5_stereo_rep2.h5ad` (GEO series GSE278603) in `data/GSE278603_RAW/`, or set `MORPHO_DATA_DIR` to the folder that holds them, then click **Use E7.5 mouse embryo**.

The benchmark runs on the CPU and needs the packages in `backend/requirements.txt` (including `scikit-learn`); PyTorch is not required. The API keeps sessions in memory, so run a single worker and keep it on a trusted local network.

## Local E7.5 mouse embryo preset

The landing page offers **Use E7.5 mouse embryo** to load the local spatial dataset and reference directly, while retaining the random selector. By default, the API looks in `data/GSE278603_RAW` at the repository root for:

- `GSM9046243_Embryo_E7.5_stereo_rep1.h5ad` (spatial candidates)
- `GSM9046244_Embryo_E7.5_stereo_rep2.h5ad` (reference for overlap)

Set `MORPHO_DATA_DIR` to the directory containing both files if they are stored elsewhere. The preset validates the files and references them in place; it does not copy or delete the source datasets. It initializes the four-gene prior panel (`Fth1`, `Rpl8`, `Eif5a`, `Cox7c`), panel size 10, and seed 42. The spatial preview maps `x`, `y`, and `z` from `obs` and uses `celltype` when available. The selector remains behind the web selection adapter so it can be replaced when the adaptive method is ready.

## Input schema and mappings

Both inputs must be readable `.h5ad` files with AnnData's observations × genes orientation: rows are cells/spots, columns are genes, and `X` or the selected numeric layer contains nonempty finite expression values. Observation names and the selected gene identifier field must be nonempty and unique. Numeric normalized or transformed expression is accepted; integer counts are not required.

The spatial file needs finite 2D or 3D coordinates. The default is `obsm["spatial"]`; choose another compatible `obsm` entry or map two/three numeric columns from `obs` when needed. Three-dimensional coordinates are displayed as an X–Y projection. A tissue image, `uns["spatial"]`, and cell-type annotation are optional.

Gene IDs default to `var_names`; an existing `var` column can be selected for exact matching. A separate display-name field can be selected without changing stable IDs. The reference gene list is matched exactly against spatial identifiers; case, versions, and aliases are never normalized. Reference-only genes are not added to the candidate pool. Optional cell-type annotations are mapped from `obs`.

Previous measurements must be provided explicitly by pasting IDs or uploading a simple CSV/TXT file. They are not inferred from expression matrix columns. Duplicate entries are counted once; exact matches are excluded, and unmatched IDs are reported.

## Demo selection method

The method selector includes the web random panel and the `eval_team.dummy_method` marker, variable-gene, and random-expressed-gene baselines. The latter three run the repository's nearest-neighbour reconstruction against the local E7.5 pair and require gene counts in multiples of four. The `gene_recommender.py` adaptive expected-information-gain method is still a work in progress and is not used by the webpage. After choosing a method, use **I chose other genes** to enter a measured panel directly instead of accepting the recommended set.

The web random selector's row scores are deterministic synthetic ranking placeholders, not information gain or biological evidence. For every method, the local E7.5 benchmark reconstructs the embryo by matching each cell to its nearest cells in the reference embryo using all genes measured so far (previous rounds plus the new selection), so the reconstruction can improve as measured genes accumulate; measured genes keep their observed values. The eval_team dummy methods only choose genes. All methods use `eval_team.Evaluator` and a section-level experiment log so held-out scoring excludes measured genes. These are retrospective benchmark results on the local E7.5 data, not prospective performance guarantees. Uploaded and synthetic-example datasets do not receive fabricated evaluation scores.

The repo benchmark can also report Peng and atlas reference scores when those reference tables have been prepared. See [eval_team/README.md](eval_team/README.md#setup) for the setup commands; the atlas preparation downloads a large external dataset. If the reference tables are absent, the webpage still shows the eval_team metrics that do not depend on them and identifies those omitted external scores.

## Current limitations

- The interface is a prototype; it does not infer temporal biology. “Next round” means the next measurement round only.
- Validation scans expression values in bounded row chunks. Preview display is capped and sampled; the full matrix is never sent to the browser.
- Orientation is shown as observations × genes and is never inferred from shape alone or automatically transposed. If the example IDs suggest a transposed object, the user can explicitly opt into a temporary transposed copy in the mapping controls, review the resulting IDs/coordinates, and switch back to the original.
- Session metadata is held in API process memory; use one API worker for this prototype. Temporary files are cleaned on a later request after the configured idle TTL.
- The random selector intentionally does not use expression, spatial coordinates, or the reference data. Only the local E7.5 preset supports the repository benchmark; other datasets show evaluation as unavailable.

## Tests

```powershell
cd backend
pytest
```

---

# Research background
*Name suggestions: SeqSelect ? GeneGuru?* 

## Data Setup
> For each of N embryos at T timesteps, we can only sequence k genes out of K possible genes, where k is significantly smaller than K. We would like to know the expression of all genes for all embryos at all timesteps.

## Research questions: 
- How do we choose which k genes to sequence for each embryo at each timestep?
  - What assumptions about the temporal profile can help this estimation? Are we better off ignoring temporal associations?
- Can we define bounds or CI’s on the number of cells? 
  - How much data do we need to get ε-close to the truth? 
- Can we use data from other timerpoints/collections/domains?
- 
## Methods suggestions:

- **Manifold learning**
  - Can we find a low-dimensional manifold to make this easier? For example, use PCA, NMF, diffusion maps, autoencoders, etc.
- **Factor analysis**
  - What latent biological factors must be recoverable from the selected genes?
  - Which genes provide the most information about each latent factor?
  - Can factors learned from one time point or dataset improve gene selection in another? 
- **Tensor decomposition** 
  - Can the data be represented as a tensor across cells, genes, time points, individuals, or experimental conditions?
  - Can Tucker or CP decomposition identify genes that jointly preserve variation across these modes?
  - Which genes are necessary for recovering the tensor’s latent components?
- **Gaussian Process**
  - Maximise expected information gain

- **Adversarial attacks/robustness**
  - Which omitted or corrupted gene measurement causes the largest deterioration in reconstruction or downstream prediction?
  - Can we define gene importance through the worst-case effect of not measuring that gene?
  - What is the smallest set of genes whose removal makes the biological state unrecoverable?

## Datasets

- [Digital reconstruction of full embryos during early
mouse organogenesis](https://www.cell.com/action/showPdf?pii=S0092-8674%2825%2900622-1) 
    - Data at: [https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE278603](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE278603), at the bottom is the "Supplementary Files" section with the file `GSE278603_RAW.tar`. This tarball contains the 6 sample files.
    - The data is also on the Insomnia server if you have access to that, in the folder `/insomnia001/depts/morpheus/users/wg2492/GSE278603_RAW/`
    - A basic script to load the `.h5ad` files with AnnData is [mouse_embryo.py](mouse_embryo.py).
    - To score a reconstruction of the hidden E7.5 embryo, run `python mouse_embryo.py --evaluate`, or call `evaluate_reconstruction` from the `eval_team` package (`from eval_team import evaluate_reconstruction`). See [eval_team/README.md](eval_team/README.md) for what the scores mean.
