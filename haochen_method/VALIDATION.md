# Validation

Validated on 2026-10-30 against Morpholab_Hackathon commit
`04c65bc8f1735799bde72afad460ea4ae519e14c`.

- Standalone CPU core tests pass in the existing Python 3.9 LUNA environment.
  A copy isolated from the local project's other modules also passes.
- The optional integration test passes in Python 3.11.14 against the actual
  upstream `MeasurementOracle` and `Evaluator`, using its synthetic embryo and
  reference-table fixtures. No benchmark logic is patched or copied into this
  directory.
- The integration run uses 560 hidden cells and 30 genes, a separate rep1-like
  reference, initial four genes, and one additional four-gene acquisition.
  The seven sections produce seven baseline experiments and fourteen total
  experiments. The test verifies exact cell/gene order, a valid oracle log,
  the final eight-gene panel, all principal benchmark score keys, and per-round
  held-out score tables.
- Core tests additionally verify strict >50% detection, original HVG pool/rank,
  reproducible random selection, SC/reference resolution, log-count scale,
  measured zeros and float64 precision, gene-axis alignment, final-round fitting,
  unchanged predictions when hidden unmeasured expression changes, candidate
  exhaustion before measurement, and refusal to overwrite existing results.

Reproduce from the target repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m unittest discover -s haochen_method/tests -v
```

The optional full evaluator test requires Python >=3.10 and `eval_team` on the
import path. Run the complete integration in Python 3.11+ as documented by the
target repository. The Python 3.9 core-only run intentionally skips that test.

This validation uses small CPU tests, not a completed real-embryo benchmark or
a GPU training validation. Real-data runs should record their own environment,
reference coverage, budgets, and scores in the generated run outputs.
