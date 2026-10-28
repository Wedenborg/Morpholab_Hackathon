import pathlib

import anndata as ad

# data location
folder = pathlib.Path("../GSE278603_RAW")
for fname in folder.iterdir():
    adata = ad.io.read_h5ad(fname)
    # print(adata.X)  # a compressed sparse matrix (n_obs x n_vars)
    # In this case, the observations are cells and the vars are genes.
    # For example, the first one is 20,935 x 18,455
    # These are all pandas dataframes, which can be converted to numpy arrays with .to_numpy()
    # This is the counts data after applying the log1p transform.

    # The observation and variable names
    # print(adata.obs_names)
    # print(adata.var_names)

    # Arrays with 1 value for each observation
    # print(adata.obs["orig.ident"])
    # print(adata.obs["nCount_RNA"])
    # print(adata.obs["nFeature_RNA"])
    # print(adata.obs["x"])
    # print(adata.obs["y"])
    # print(adata.obs["cell_name"])
    # print(adata.obs["chip"])
    # print(adata.obs["id"])
    # print(adata.obs["nCount_SCT"])
    # print(adata.obs["nFeature_SCT"])
    # print(adata.obs["image_x"])
    # print(adata.obs["image_y"])
    # print(adata.obs["germ_layer"])
    # print(adata.obs["celltype"])
    # print(adata.obs["z"])
    # print(adata.obs["predicted.id"])

    # Arrays with 1 value for each variable
    # print(adata.var["in_scaled_data"])
    # print(adata.var["highly_variable"])

    # These appear to be numpy arrays already? Shape (n_obs,2)
    # print(adata.obsm["X_image"].shape)
    # print(adata.obsm["X_sagittal"].shape)
    # print(adata.obsm["X_spatial"].shape)

    # These appear to be all nan
    # print(adata.varm["image"].shape)
    # print(adata.varm["sagittal"].shape)
    # print(adata.varm["spatial"].shape)

    # Each of these is a dict with a single entry 'variance' with a (2,) array
    # print(adata.uns["image"])
    # print(adata.uns["sagittal"])
    # print(adata.uns["spatial"])

    # The original counts data, a compressed sparse matrix
    # print(adata.layers["counts"])

    print(adata)
