"""Adapters to other codebases.

Each module here is optional and self-contained: importing ``panelprep`` never
imports an integration, so the package has no dependency on any host repo.
Import the one you need explicitly::

    import panelprep.integrations.morpholab as morpholab

Importing an integration module registers its scores and embedders (e.g.
``corr_mae``, ``factor_analysis``), which then appear in the comparison report
like any other entry.

MODIFY HERE: add one module per host codebase; keep all repo-specific paths,
column names and conventions inside it.
"""

from __future__ import annotations

__all__ = ["morpholab"]
