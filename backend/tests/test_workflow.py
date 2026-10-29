from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from scipy import sparse

from app import api
from app.candidates import prepare_candidates
from app.data_io import validate_dataset
from app.data_io import transpose_h5ad
from app.models import MappingRequest
from app.selection import select_genes


def write_dataset(
    path: Path,
    *,
    kind: str = "spatial",
    genes: list[str] | None = None,
    matrix: object | None = None,
    with_coordinates: bool = True,
) -> Path:
    gene_ids = genes or ["gA", "gB", "gC", "gD"]
    values = matrix if matrix is not None else sparse.csr_matrix(np.arange(20, dtype=np.float32).reshape(5, 4))
    dataset = ad.AnnData(
        X=values,
        obs=pd.DataFrame(
            {"cell_type": ["a", "b", "a", "b", "a"]},
            index=[f"{kind}_{index}" for index in range(5)],
        ),
        var=pd.DataFrame(
            {"symbol": [f"symbol_{index}" for index in range(len(gene_ids))]},
            index=gene_ids,
        ),
    )
    if with_coordinates:
        dataset.obsm["spatial"] = np.column_stack([np.arange(5), np.arange(5) * 2]).astype(float)
    dataset.write_h5ad(path)
    return path


def test_valid_spatial_and_reference_support_sparse_matrices(tmp_path: Path) -> None:
    spatial = write_dataset(tmp_path / "spatial.h5ad")
    reference = write_dataset(tmp_path / "reference.h5ad", kind="ref", with_coordinates=False)

    spatial_summary, spatial_ids, spatial_display, coordinates = validate_dataset(
        spatial, "spatial", "spatial.h5ad", spatial.stat().st_size, MappingRequest(display_field="symbol")
    )
    reference_summary, reference_ids, _, reference_coordinates = validate_dataset(
        reference, "reference", "reference.h5ad", reference.stat().st_size, MappingRequest()
    )

    assert spatial_summary.valid
    assert spatial_summary.n_observations == 5
    assert spatial_summary.n_genes == 4
    assert spatial_ids == ["gA", "gB", "gC", "gD"]
    assert spatial_display == ["symbol_0", "symbol_1", "symbol_2", "symbol_3"]
    assert coordinates is not None and coordinates.shape == (5, 2)
    assert reference_summary.valid
    assert reference_ids == spatial_ids
    assert reference_coordinates is None


def test_selected_expression_layer_and_obs_coordinate_mapping(tmp_path: Path) -> None:
    path = write_dataset(tmp_path / "mapped.h5ad")
    dataset = ad.read_h5ad(path)
    dataset.layers["normalized"] = dataset.X.copy()
    dataset.obs["x_coord"] = np.arange(dataset.n_obs, dtype=float)
    dataset.obs["y_coord"] = np.arange(dataset.n_obs, dtype=float) * 2
    dataset.write_h5ad(path)

    summary, _, _, coordinates = validate_dataset(
        path,
        "spatial",
        path.name,
        path.stat().st_size,
        MappingRequest(
            matrix_layer="normalized",
            coordinate_source="obs",
            coordinate_columns=["x_coord", "y_coord"],
        ),
    )

    assert summary.valid
    assert summary.matrix == "layers['normalized']"
    assert coordinates is not None
    assert coordinates.shape == (5, 2)


def test_missing_spatial_coordinates_are_blocking(tmp_path: Path) -> None:
    path = write_dataset(tmp_path / "missing-coords.h5ad", with_coordinates=False)

    summary, _, _, coordinates = validate_dataset(
        path, "spatial", path.name, path.stat().st_size, MappingRequest()
    )

    assert not summary.valid
    assert coordinates is None
    assert any(issue.code == "missing_coordinates" for issue in summary.issues)


def test_explicit_transpose_swaps_observation_and_gene_axes(tmp_path: Path) -> None:
    source = write_dataset(tmp_path / "source.h5ad", kind="source")
    corrected = tmp_path / "corrected.h5ad"

    transpose_h5ad(source, corrected)
    summary, gene_ids, _, _ = validate_dataset(
        corrected, "reference", "corrected.h5ad", corrected.stat().st_size, MappingRequest()
    )

    assert summary.n_observations == 4
    assert summary.n_genes == 5
    assert gene_ids == [f"source_{index}" for index in range(5)]


def test_observation_identifiers_must_be_unique(tmp_path: Path) -> None:
    path = write_dataset(tmp_path / "duplicate-observations.h5ad")
    dataset = ad.read_h5ad(path)
    dataset.obs_names = ["duplicate", "duplicate", "spot-2", "spot-3", "spot-4"]
    dataset.write_h5ad(path)

    summary, _, _, _ = validate_dataset(
        path, "spatial", path.name, path.stat().st_size, MappingRequest()
    )

    assert not summary.valid
    assert any(issue.code == "duplicate_observation_identifiers" for issue in summary.issues)


def test_duplicate_gene_identifiers_are_blocking(tmp_path: Path) -> None:
    path = write_dataset(tmp_path / "duplicates.h5ad")
    dataset = ad.read_h5ad(path)
    dataset.var["symbol"] = ["dup", "dup", "gC", "gD"]
    dataset.write_h5ad(path)

    summary, _, _, _ = validate_dataset(
        path, "spatial", path.name, path.stat().st_size, MappingRequest(gene_field="symbol")
    )

    assert not summary.valid
    issue = next(issue for issue in summary.issues if issue.code == "duplicate_gene_identifiers")
    assert "dup" in issue.what
    assert issue.fix


@pytest.mark.parametrize(
    ("coordinates", "expected_issue"),
    [
        (np.ones((5, 1)), "coordinate_shape"),
        (np.asarray([[0.0, 0.0], [1.0, 1.0], [np.nan, 2.0], [3.0, 3.0], [4.0, 4.0]]), "coordinate_non_finite"),
    ],
)
def test_malformed_spatial_coordinates_are_blocking(
    tmp_path: Path, coordinates: np.ndarray, expected_issue: str
) -> None:
    path = write_dataset(tmp_path / "bad-coordinates.h5ad")
    dataset = ad.read_h5ad(path)
    dataset.obsm["spatial"] = coordinates
    dataset.write_h5ad(path)

    summary, _, _, _ = validate_dataset(
        path, "spatial", path.name, path.stat().st_size, MappingRequest()
    )

    assert not summary.valid
    assert any(issue.code == expected_issue for issue in summary.issues)


def test_non_finite_expression_values_are_blocking(tmp_path: Path) -> None:
    matrix = np.arange(20, dtype=np.float32).reshape(5, 4)
    matrix[2, 1] = np.nan
    path = write_dataset(tmp_path / "nan.h5ad", matrix=matrix)

    summary, _, _, _ = validate_dataset(
        path, "spatial", path.name, path.stat().st_size, MappingRequest()
    )

    assert not summary.valid
    assert any(issue.code == "non_finite_expression" for issue in summary.issues)


def test_previous_gene_exclusions_and_unmatched_ids() -> None:
    pool = prepare_candidates(
        ["gA", "gB", "gC", "gD"],
        ["gA", "ref-only"],
        ["gA", "gA", "not-present"],
    )

    assert pool.identifiers == ("gB", "gC", "gD")
    assert pool.previous_unique == ("gA", "not-present")
    assert pool.matched_previous == ("gA",)
    assert pool.unmatched_previous == ("not-present",)
    assert "ref-only" not in pool.identifiers


def test_selection_is_exact_and_reproducible() -> None:
    pool = prepare_candidates(["gA", "gB", "gC", "gD"], [], [])

    first = select_genes(None, None, [], pool, 3, 42)
    second = select_genes(None, None, [], pool, 3, 42)

    assert first == second
    assert len(first) == 3
    assert len({gene.gene_id for gene in first}) == 3
    assert [gene.rank for gene in first] == [1, 2, 3]
    assert all(0 <= gene.score <= 1 for gene in first)
    assert all("Synthetic" in gene.score_provenance for gene in first)


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(api, "DATA_ROOT", tmp_path / "sessions")
    api.sessions.clear()
    with TestClient(api.app) as test_client:
        yield test_client
    api.sessions.clear()


def upload(client: TestClient, session_id: str, kind: str, source: Path) -> None:
    with source.open("rb") as stream:
        response = client.post(
            f"/api/datasets/{kind}",
            data={"session_id": session_id},
            files={"file": (source.name, stream, "application/x-hdf5")},
        )
    assert response.status_code == 200, response.text


def test_api_rejects_invalid_panel_sizes_and_returns_exact_panel(
    client: TestClient, tmp_path: Path
) -> None:
    session_id = client.post("/api/sessions").json()["session_id"]
    spatial = write_dataset(tmp_path / "spatial-api.h5ad")
    reference = write_dataset(tmp_path / "reference-api.h5ad", kind="ref", with_coordinates=False)
    upload(client, session_id, "spatial", spatial)
    upload(client, session_id, "reference", reference)

    invalid = client.post(
        "/api/select",
        json={"session_id": session_id, "previous_genes": ["gA", "gB", "gC", "gD"], "ng": 1, "seed": 42},
    )
    assert invalid.status_code == 422
    assert "0 eligible" in invalid.json()["detail"]
    non_positive = client.post("/api/select", json={"session_id": session_id, "ng": 0})
    assert non_positive.status_code == 422

    request = {"session_id": session_id, "previous_genes": ["gA"], "ng": 2, "seed": 42}
    first = client.post("/api/select", json=request)
    second = client.post("/api/select", json=request)
    assert first.status_code == second.status_code == 200
    first_data = first.json()
    second_data = second.json()
    assert len(first_data["selected_genes"]) == 2
    assert first_data["selected_genes"] == second_data["selected_genes"]
    assert "gA" not in {gene["gene_id"] for gene in first_data["selected_genes"]}
    assert first_data["reference_overlap_count"] == 4
    assert first_data["evaluation"]["synthetic"] is True


def test_upload_errors_are_actionable(client: TestClient) -> None:
    session_id = client.post("/api/sessions").json()["session_id"]
    wrong_extension = client.post(
        "/api/datasets/spatial",
        data={"session_id": session_id},
        files={"file": ("sample.txt", b"not h5ad", "text/plain")},
    )
    malformed = client.post(
        "/api/datasets/spatial",
        data={"session_id": session_id},
        files={"file": ("broken.h5ad", b"not h5ad", "application/x-hdf5")},
    )

    assert wrong_extension.status_code == 415
    assert "Choose a file with the .h5ad extension." in wrong_extension.json()["detail"]
    assert malformed.status_code == 422
    assert "AnnData could not read" in malformed.json()["detail"]["message"]


def test_api_orientation_correction_is_explicit_and_reversible(
    client: TestClient, tmp_path: Path
) -> None:
    session_id = client.post("/api/sessions").json()["session_id"]
    reference = write_dataset(tmp_path / "orientation.h5ad", kind="ref")
    upload(client, session_id, "reference", reference)

    corrected = client.post(
        f"/api/datasets/reference/mapping?session_id={session_id}",
        json={"transpose": True},
    )
    restored = client.post(
        f"/api/datasets/reference/mapping?session_id={session_id}",
        json={"transpose": False},
    )

    assert corrected.status_code == 200
    assert corrected.json()["dataset"]["transpose_applied"] is True
    assert corrected.json()["dataset"]["n_observations"] == 4
    assert corrected.json()["dataset"]["n_genes"] == 5
    assert restored.status_code == 200
    assert restored.json()["dataset"]["transpose_applied"] is False
    assert restored.json()["dataset"]["n_observations"] == 5
    assert restored.json()["dataset"]["n_genes"] == 4


def test_example_data_runs_through_candidate_and_selection_apis(client: TestClient) -> None:
    example = client.post("/api/example")
    session_id = example.json()["session_id"]
    previous_genes = example.json()["previous_genes"].splitlines()
    candidates = client.post(
        "/api/candidates",
        json={"session_id": session_id, "previous_genes": previous_genes},
    )
    selection = client.post(
        "/api/select",
        json={"session_id": session_id, "previous_genes": previous_genes, "ng": 5, "seed": 42},
    )

    assert candidates.status_code == 200
    assert candidates.json()["candidate_count"] == 20
    assert candidates.json()["matched_previous_count"] == 3
    assert candidates.json()["eligible_count"] == 17
    assert candidates.json()["reference_overlap_count"] == 16
    assert selection.status_code == 200
    assert len(selection.json()["selected_genes"]) == 5
    assert example.json()["session_id"] in api.sessions


def test_expression_previews_use_real_values_and_report_missing_genes(client: TestClient) -> None:
    example = client.post("/api/example").json()
    session_id = example["session_id"]

    spatial = client.get(
        "/api/preview/spatial",
        params={"session_id": session_id, "gene_id": "Gene-01"},
    ).json()
    reference = client.get(
        "/api/preview/reference",
        params={"session_id": session_id, "gene_id": "Gene-01"},
    ).json()
    unavailable = client.get(
        "/api/preview/reference",
        params={"session_id": session_id, "gene_id": "not-in-reference"},
    ).json()

    assert spatial["available"] is True
    assert len(spatial["points"]) == 100
    assert len(spatial["color_values"]) == 100
    assert reference["available"] is True
    assert len(reference["color_values"]) == 80
    assert unavailable["available"] is False
    assert "not present" in unavailable["message"]
