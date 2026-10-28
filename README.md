
# Hackathon! 
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
