# Methods

Adaptive gene-selection and reconstruction methods for the E7.5 benchmark.

| Module | Class | Selection | Reconstruction |
|---|---|---|---|
| `methods/eddi.py` | `EDDIMethod` | nonlinear Partial VAE information gain | conditional Partial VAE |
| `methods/ppca.py` | `PPCAMethod` | linear-Gaussian information gain | PPCA posterior |
| `methods/linear.py` | `LinearImputationMethod` | delegated to a wrapped selector | ridge regression |

`LinearImputationMethod` takes another method as its selector.  It uses only the selector's
chosen genes and replaces the selector's imputation with ridge regression from the genes
measured in previous rounds.

## Contract

A method is a stateful callable matching
`eval_team.harness.run_and_evaluate` and `mouse_embryo.random_method`:

```python
choices, imputed = method(
    n_genes_per_round,
    gene_embeddings,
    gene_similarity,
    locations,
    gene_log_counts,
    sc_gene_log_counts,
)
```

`choices` contains gene indices for the next measurement round.  `imputed` is an
`(N, G)` tensor for every hidden cell and gene, on the same `log1p` scale as the input.

## Usage

```python
from eval_team import run_and_evaluate
from methods import EDDIMethod, PPCAMethod, LinearImputationMethod

summary = run_and_evaluate(EDDIMethod(seed=0, verbose=False))
summary = run_and_evaluate(PPCAMethod(seed=0, verbose=False))
summary = run_and_evaluate(
    LinearImputationMethod(selector=PPCAMethod(seed=0, verbose=False), seed=0, verbose=False)
)
```

From the repository root, the benchmark CLI selects these methods with
`python mouse_embryo.py --evaluate --method {eddi,ppca,linear}`.  The `scanpy` and `random`
baselines remain in `mouse_embryo.py` because they are non-adaptive reference methods.

## Tests

```bash
python -m unittest discover -s eval_team -t . -v
```

`eval_team/test_methods.py` checks all three methods against the benchmark contract on
synthetic data using the CPU.
