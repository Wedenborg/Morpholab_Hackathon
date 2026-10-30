"""Method implementations for the Morpholab E7.5 reconstruction pipeline.

Every method implements the callable contract used by
``eval_team.harness.run_and_evaluate``: it receives the round budget, gene metadata,
spatial locations, the measurements collected so far, and the visible reference, then
returns the genes to measure next and a reconstruction of all genes for the hidden cells.
"""
from .eddi import EDDIConfig, EDDIMethod, eddi_method
from .linear import LinearImputationConfig, LinearImputationMethod
from .ppca import PPCAConfig, PPCAMethod, ppca_method

__all__ = [
    "EDDIConfig",
    "EDDIMethod",
    "eddi_method",
    "LinearImputationConfig",
    "LinearImputationMethod",
    "PPCAConfig",
    "PPCAMethod",
    "ppca_method",
]
