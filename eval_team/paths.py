"""Where the data live. Both default to folders next to the repo; override with
MORPHO_DATA_DIR (the GSE278603 .h5ad files) and MORPHO_REF_DIR (reference tables)."""
import os
import pathlib

PACKAGE_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = pathlib.Path(os.environ.get("MORPHO_DATA_DIR", PACKAGE_DIR.parent.parent / "GSE278603_RAW"))
REF_DIR = pathlib.Path(os.environ.get("MORPHO_REF_DIR", PACKAGE_DIR.parent.parent / "reference_data"))
