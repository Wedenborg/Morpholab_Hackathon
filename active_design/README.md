# Experimental design for the E7.5 reconstruction task

Method-side code, scored by `eval_team`. Local work, not committed.

It never uses the benchmark's references: the Pijuan-Sala atlas, anything containing it such as TOME or the Extended Mouse Atlas, or the Peng Geo-seq data. The independent single-cell source is mdCLA E7.5, in `reference_data/mdcla/`.

- `theory.md`: the math of the Gaussian factor model (fit, spatial prior, likelihood, posterior, reconstruction).
- `model.py`: 16-factor model fitted on the visible embryo's neighbourhood averages (about 30 cells each), a spatial prior linking neighbourhoods within a section and across adjacent sections, and an exact Gaussian posterior updated after every experiment. The reconstruction is the posterior mean, corrected per cell by its own measurements, with measured values kept.
- `designs.py`: fixed panels, rotating marker sets, greedy information gain (log-det criterion), adaptive cell-type separation.
- `counts.py`: negative binomial / zero-inflated NB version of the model: factors fitted on counts, a Laplace posterior, and a greedy information design that depends on the measured values.
- `ceilings.py`: noise and information ceilings on the held-out score.
- `registration.py`: rigid registration of the hidden embryo onto the visible one, and prediction by copying matched visible cells.
- `celltype_model.py`: discrete cell states with a Potts spatial prior (mean field).
- `celltype_design.py`: predictive adaptive design on the cell-state model, on whole sections or quarter sections.
- `sources.py`: mdCLA cells, log-normalised and as raw counts.
- `run_sources.py`: compares gene weights from the visible embryo with weights from mdCLA (rescaled).
- `run_counts.py`: count model vs the Gaussian model.
- `run_step2.py`, `run_step3.py`: new models on the same measurements; designs on the cell-state model.
- `tune.py`: chooses the model's settings on the visible embryo only.
- `run_tests.py`: compares designs at equal numbers of experiments.

```
python -m active_design.tune --visible 1 --out tune_rep1.csv
python -m active_design.run_tests --hidden 2 --visible 1 --lam 3 --cross 1.0 --out results_rep2.csv
```

Results are in `results/`.
