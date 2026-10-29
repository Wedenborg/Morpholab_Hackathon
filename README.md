
# Fieldnote — Gene panel planner

A local research prototype for preparing a gene panel for the next measurement round of a spatial transcriptomics experiment. The interface validates uploaded AnnData files, reports exact overlap, selects a reproducible random panel, and clearly separates synthetic demo evaluation from facts about uploaded data.

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

## Input schema and mappings

Both inputs must be readable `.h5ad` files with AnnData's observations × genes orientation: rows are cells/spots, columns are genes, and `X` or the selected numeric layer contains nonempty finite expression values. Observation names and the selected gene identifier field must be nonempty and unique. Numeric normalized or transformed expression is accepted; integer counts are not required.

The spatial file needs finite 2D or 3D coordinates. The default is `obsm["spatial"]`; choose another compatible `obsm` entry or map two/three numeric columns from `obs` when needed. Three-dimensional coordinates are displayed as an X–Y projection. A tissue image, `uns["spatial"]`, and cell-type annotation are optional.

Gene IDs default to `var_names`; an existing `var` column can be selected for exact matching. A separate display-name field can be selected without changing stable IDs. The reference gene list is matched exactly against spatial identifiers; case, versions, and aliases are never normalized. Reference-only genes are not added to the candidate pool. Optional cell-type annotations are mapped from `obs`.

Previous measurements must be provided explicitly by pasting IDs or uploading a simple CSV/TXT file. They are not inferred from expression matrix columns. Duplicate entries are counted once; exact matches are excluded, and unmatched IDs are reported.

## Replace the demo methods

- Replace `backend/app/selection.py::select_genes` with the research selector. Keep it independent of API and UI code. It receives the validated spatial and reference AnnData objects, previous IDs, a prepared candidate pool, `Ng`, and seed.
- Replace `backend/app/evaluation.py::evaluate_demo` with the evaluator. It receives the validated spatial and reference AnnData objects, previous IDs, `Ng`, seed, and selected genes.
- Keep response keys and provenance explicit, or update the Pydantic response models and frontend types together.

The random choice is reproducible for a fixed candidate set and seed. All “Demo score” values and the demo evaluation curve are deterministic synthetic placeholders; they are not information gain, calibrated uncertainty, probabilities, or evidence of biological usefulness. Dataset dimensions, identifier examples, coordinate previews, exact overlap, and exclusions are derived from uploaded data.

## Current limitations

- The interface is a prototype; it does not infer temporal biology. “Next round” means the next measurement round only.
- Validation scans expression values in bounded row chunks. Preview display is capped and sampled; the full matrix is never sent to the browser.
- Orientation is shown as observations × genes and is never inferred from shape alone or automatically transposed. If the example IDs suggest a transposed object, the user can explicitly opt into a temporary transposed copy in the mapping controls, review the resulting IDs/coordinates, and switch back to the original.
- Session metadata is held in API process memory; use one API worker for this prototype. Temporary files are cleaned on a later request after the configured idle TTL.
- The evaluation is synthetic, and the random selector intentionally does not use expression, spatial coordinates, or the reference data.

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
