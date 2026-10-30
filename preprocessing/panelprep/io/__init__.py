"""Data access: synthetic fixtures, the dataset catalogue, and loaders."""
from .synthetic import make_synthetic_case, SyntheticCase  # noqa: F401
from .catalog import CATALOG, catalog_table, resolve_candidates  # noqa: F401
