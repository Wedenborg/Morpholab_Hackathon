"""E7.5 reconstruction benchmark (eval team). See README.md in this folder.

    from eval_team import evaluate_reconstruction, run_and_evaluate, load_e75_inputs, measure

The functions most users need are re-exported here; the modules hold the details:
    harness            glue for methods: loading data, measuring, scoring tensors
    evaluate           Evaluator: all scores for one hidden embryo
    region_benchmark   region / cell-type comparison with Peng 2019 and the atlas
    spatial_scores     held-out gene score and cell-type proportion score
    submission         MeasurementOracle and the experiment log
    dummy_method       a stand-in method that produces submissions
"""
from .evaluate import Evaluator
from .harness import evaluate_reconstruction, load_e75_inputs, load_embryo, measure, run_and_evaluate
from .paths import DATA_DIR, REF_DIR
from .submission import MeasurementOracle, load_log

__all__ = ["Evaluator", "MeasurementOracle", "load_log", "load_embryo", "load_e75_inputs", "measure",
           "evaluate_reconstruction", "run_and_evaluate", "DATA_DIR", "REF_DIR"]
