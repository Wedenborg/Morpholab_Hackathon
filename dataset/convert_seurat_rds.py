"""Convert a Seurat v3 .rds to .h5ad without R, using the `rdata` parser (pip install rdata).

The output sits next to the input with the same stem, which load_dataset prefers over the
.rds. X holds the counts and layers["logcounts"] the assay's normalised data slot, the
layout load_dataset and preprocessing.dataset_selection expect for TOME.

Usage:
    python dataset/convert_seurat_rds.py dataset/data/tome/seurat_object_E7.5.rds
"""
import argparse
from pathlib import Path
import sys
import warnings

import anndata as ad
import numpy as np
import pandas as pd
from rdata.parser import parse_file
from scipy import sparse


def _attributes(node) -> dict:
    """An R object's attribute pairlist as a dict of name to node."""
    found = {}
    pair = node.attributes
    while pair is not None and pair.info.type.name == "LIST":
        tag = pair.tag.referenced_object if pair.tag.info.type.name == "REF" else pair.tag
        found[_text(tag.value)] = pair.value[0]
        pair = pair.value[1]
    return found


def _text(node) -> str:
    value = node.value
    return value.decode() if isinstance(value, bytes) else value


def _strings(node) -> list[str]:
    return [_text(item) for item in node.value]


def _column(node) -> np.ndarray | pd.Categorical:
    """A meta.data column. R factors are 1-indexed codes with NA stored as a large negative."""
    levels = _attributes(node).get("levels")
    if levels is None:
        return np.asarray(node.value)
    codes = np.asarray(node.value).astype(int) - 1
    return pd.Categorical.from_codes(np.where(codes < 0, -1, codes), categories=_strings(levels))


def _sparse_matrix(node) -> sparse.csc_matrix:
    slots = _attributes(node)
    n_rows, n_columns = np.asarray(slots["Dim"].value).astype(int)
    return sparse.csc_matrix(
        (np.asarray(slots["x"].value, dtype=np.float32),
         np.asarray(slots["i"].value).astype(np.int32),
         np.asarray(slots["p"].value).astype(np.int32)),
        shape=(n_rows, n_columns),
    )


def convert(path: Path) -> Path:
    sys.setrecursionlimit(100_000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        root = parse_file(path, extension="rds").object
    slots = _attributes(root)
    assay = _attributes(slots["assays"].value[0])
    counts, logcounts = _sparse_matrix(assay["counts"]), _sparse_matrix(assay["data"])
    genes, cells = (_strings(names) for names in _attributes(assay["counts"])["Dimnames"].value)

    meta = slots["meta.data"]
    meta_slots = _attributes(meta)
    obs = pd.DataFrame(
        {name: _column(column) for name, column in zip(_strings(meta_slots["names"]), meta.value)},
        index=pd.Index(cells, name="cell"),
    )
    adata = ad.AnnData(
        X=counts.T.tocsr(),
        obs=obs,
        var=pd.DataFrame(index=pd.Index(genes, name="gene")),
        layers={"logcounts": logcounts.T.tocsr()},
    )
    output = path.with_suffix(".h5ad")
    adata.write_h5ad(output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path)
    print(f"Wrote {convert(parser.parse_args().path)}")
