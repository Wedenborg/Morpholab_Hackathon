from __future__ import annotations

import importlib
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from scipy import sparse

from app import api
from app import evaluation
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
    assert first_data["evaluation"]["available"] is False
    assert first_data["evaluation"]["synthetic"] is False
    assert first_data["method"] == "random-demo-v1"


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
    assert selection.json()["evaluation"]["available"] is False
    assert selection.json()["evaluation"]["synthetic"] is False
    assert example.json()["session_id"] in api.sessions


def test_mouse_embryo_preset_uses_local_files_and_random_selection(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "mouse-data"
    data_dir.mkdir()
    spatial_path = data_dir / "GSM9046243_Embryo_E7.5_stereo_rep1.h5ad"
    reference_path = data_dir / "GSM9046244_Embryo_E7.5_stereo_rep2.h5ad"
    genes = ["Fth1", "Rpl8", "Eif5a", "Cox7c", *[f"g{index}" for index in range(10)]]
    observations = pd.DataFrame(
        {
            "x": np.arange(5, dtype=float),
            "y": np.arange(5, dtype=float) * 2,
            "z": np.arange(5, dtype=float) * 3,
            "celltype": ["a", "b", "a", "b", "a"],
        },
        index=[f"spatial_{index}" for index in range(5)],
    )
    ad.AnnData(
        X=sparse.csr_matrix(np.arange(5 * len(genes), dtype=np.float32).reshape(5, len(genes))),
        obs=observations,
        var=pd.DataFrame(index=genes),
    ).write_h5ad(spatial_path)
    ad.AnnData(
        X=sparse.csr_matrix(np.arange(5 * len(genes), dtype=np.float32).reshape(5, len(genes))),
        obs=observations.drop(columns=["x", "y", "z"]).rename(
            index=lambda value: value.replace("spatial_", "reference_")
        ),
        var=pd.DataFrame(index=genes),
    ).write_h5ad(reference_path)
    monkeypatch.setattr(api, "MOUSE_EMBRYO_DATA_DIR", data_dir)
    evaluated: dict[str, object] = {}

    def fake_evaluation(path: Path, reference: Path, measured_genes: list[str]) -> dict[str, object]:
        evaluated["path"] = path
        evaluated["genes"] = measured_genes
        return {
            "label": "eval_team E7.5 reconstruction benchmark",
            "synthetic": False,
            "available": True,
            "notice": "mock benchmark result",
            "metrics": [],
        }

    monkeypatch.setattr(api, "evaluate_mouse_embryo_baseline", fake_evaluation)

    preset = client.post("/api/mouse-embryo")
    assert preset.status_code == 200, preset.text
    data = preset.json()
    session_id = data["session_id"]
    assert data["previous_genes"].splitlines() == ["Fth1", "Rpl8", "Eif5a", "Cox7c"]
    assert data["panel_size"] == 10
    assert data["seed"] == 42

    session = client.get(f"/api/sessions/{session_id}").json()
    spatial = session["datasets"]["spatial"]
    assert spatial["coordinate_source"] == "obs"
    assert spatial["coordinate_field"] == "x, y, z"
    assert spatial["cell_type_field"] == "celltype"
    stored_spatial = api.sessions[session_id].datasets["spatial"]
    assert stored_spatial.path == spatial_path
    assert np.array_equal(stored_spatial.coordinates, observations[["x", "y", "z"]].to_numpy())

    candidates = client.post(
        "/api/candidates",
        json={"session_id": session_id, "previous_genes": data["previous_genes"].splitlines()},
    )
    selection = client.post(
        "/api/select",
        json={
            "session_id": session_id,
            "previous_genes": data["previous_genes"].splitlines(),
            "ng": data["panel_size"],
            "seed": data["seed"],
        },
    )
    assert candidates.status_code == 200
    assert candidates.json()["eligible_count"] == 10
    assert selection.status_code == 200
    assert selection.json()["method"] == "random-demo-v1"
    assert len(selection.json()["selected_genes"]) == 10
    assert selection.json()["evaluation"]["available"] is True
    assert evaluated["path"] == spatial_path
    assert set(evaluated["genes"]) == {
        "Fth1",
        "Rpl8",
        "Eif5a",
        "Cox7c",
        *{gene["gene_id"] for gene in selection.json()["selected_genes"]},
    }

    def fake_dummy_method(
        spatial_file: Path,
        reference_file: Path,
        how: str,
        n_sets: int,
        prior_genes: list[str],
        seed: int,
    ) -> tuple[list[str], dict[str, object]]:
        assert spatial_file == spatial_path
        assert reference_file == reference_path
        assert how == "markers"
        assert n_sets == 2
        assert prior_genes == data["previous_genes"].splitlines()
        assert seed == 42
        return [f"g{index}" for index in range(8)], {
            "label": "eval_team E7.5 reconstruction benchmark",
            "synthetic": False,
            "available": True,
            "notice": "mock benchmark result",
            "metrics": [],
        }

    monkeypatch.setattr(api, "run_eval_team_dummy_method", fake_dummy_method)
    dummy_selection = client.post(
        "/api/select",
        json={
            "session_id": session_id,
            "previous_genes": data["previous_genes"].splitlines(),
            "method": "dummy_markers",
            "ng": 8,
            "seed": 42,
        },
    )
    assert dummy_selection.status_code == 200, dummy_selection.text
    assert dummy_selection.json()["method"] == "eval-team-dummy_markers"
    assert [gene["gene_id"] for gene in dummy_selection.json()["selected_genes"]] == [
        f"g{index}" for index in range(8)
    ]
    assert all(gene["score"] is None for gene in dummy_selection.json()["selected_genes"])

    deleted = client.delete(f"/api/sessions/{session_id}")
    assert deleted.json()["deleted"] is True
    assert spatial_path.is_file()
    assert reference_path.is_file()


def test_mouse_embryo_preset_reports_missing_local_files(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api, "MOUSE_EMBRYO_DATA_DIR", tmp_path / "missing")

    response = client.post("/api/mouse-embryo")

    assert response.status_code == 503
    assert "MORPHO_DATA_DIR" in response.json()["detail"]


def test_user_chosen_genes_are_returned_without_synthetic_scores(client: TestClient) -> None:
    example = client.post("/api/example").json()
    session_id = example["session_id"]

    selected = client.post(
        "/api/select",
        json={
            "session_id": session_id,
            "method": "manual",
            "chosen_genes": ["Gene-05", "Gene-02", "Gene-05"],
            "ng": 2,
            "seed": 42,
        },
    )
    unknown = client.post(
        "/api/select",
        json={
            "session_id": session_id,
            "method": "manual",
            "chosen_genes": ["not-a-gene"],
            "ng": 1,
        },
    )

    assert selected.status_code == 200
    assert selected.json()["method"] == "manual-genes"
    assert [gene["gene_id"] for gene in selected.json()["selected_genes"]] == ["Gene-05", "Gene-02"]
    assert all(gene["score"] is None for gene in selected.json()["selected_genes"])
    assert selected.json()["evaluation"]["available"] is False
    assert unknown.status_code == 422
    assert "not found in the spatial dataset" in unknown.json()["detail"]


def test_eval_team_baseline_scores_only_measured_genes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    truth_path = tmp_path / "truth.h5ad"
    genes = ["Fth1", "gA", "gB", "gC"]
    obs = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": [0.0, 1.0, 2.0, 3.0],
            "z": [0.0, 1.0, 2.0, 3.0],
            "celltype": ["a", "a", "b", "b"],
            "germ_layer": ["ecto", "ecto", "meso", "meso"],
        },
        index=[f"cell-{index}" for index in range(4)],
    )
    ad.AnnData(
        X=sparse.csr_matrix(np.arange(16, dtype=np.float32).reshape(4, 4)),
        obs=obs,
        var=pd.DataFrame(index=genes),
    ).write_h5ad(truth_path)
    captured: dict[str, object] = {}

    class FakeEvaluator:
        def __init__(self, truth: ad.AnnData) -> None:
            captured["truth"] = truth
            self.regions = type("Regions", (), {"refs": {"peng": object()}})()

        def evaluate(
            self, reconstruction: ad.AnnData, log: pd.DataFrame
        ) -> tuple[dict[str, float | int], dict[str, object]]:
            captured["reconstruction"] = reconstruction
            captured["log"] = log
            return {
                "truth_region_spearman": 0.25,
                "heldout_corr": 0.5,
                "celltype_tv": 0.2,
                "celltype_balanced_accuracy": 0.75,
            }, {}

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    eval_module = importlib.import_module("eval_team.evaluate")
    monkeypatch.setattr(eval_module, "Evaluator", FakeEvaluator)
    reference_path = tmp_path / "reference.h5ad"
    ad.AnnData(
        # Reference cells match target cells on Fth1/gA; gB/gC differ so predictions are identifiable.
        X=sparse.csr_matrix(np.array(
            [[0, 1, 100, 200], [4, 5, 300, 400], [8, 9, 500, 600], [12, 13, 700, 800]], dtype=np.float32
        )),
        obs=pd.DataFrame(index=[f"ref-{index}" for index in range(4)]),
        var=pd.DataFrame(index=genes),
    ).write_h5ad(reference_path)
    result = evaluation.evaluate_mouse_embryo_baseline(truth_path, reference_path, ["Fth1", "gA", "missing"])

    reconstruction = captured["reconstruction"]
    log = captured["log"]
    assert reconstruction.var_names.tolist() == genes
    dense = reconstruction.X.toarray()
    truth_values = np.arange(16, dtype=np.float32).reshape(4, 4)
    np.testing.assert_array_equal(dense[:, :2], truth_values[:, :2])
    # Only four reference cells exist, so each prediction is their mean (not zero).
    np.testing.assert_allclose(dense[:, 2], 400)
    np.testing.assert_allclose(dense[:, 3], 500)
    assert log["genes"].tolist() == ["Fth1;gA"] * 4
    assert result["available"] is True
    assert result["synthetic"] is False
    assert result["experiment_count"] == 4
    assert {metric["id"]: metric["value"] for metric in result["metrics"]} == {
        "truth_region_spearman": 0.25,
        "heldout_corr": 0.5,
        "celltype_tv": 0.2,
        "celltype_balanced_accuracy": 0.75,
    }


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


def _embryo(path: Path, rng: np.random.Generator, n_cells: int, genes: list[str], *, counts: bool = False,
            obs_extra: dict | None = None, coords: bool = True, loadings: np.ndarray | None = None) -> Path:
    """Three sections of cells whose expression follows two spatial gradients."""
    z = np.repeat([0.0, 10.0, 20.0], n_cells // 3)
    x, y = rng.uniform(-100, 100, len(z)), rng.uniform(-100, 100, len(z))
    loadings = rng.normal(size=(2, len(genes))) if loadings is None else loadings
    rate = np.exp(0.5 + np.c_[x / 100, y / 100] @ loadings * 0.8)
    values = rng.poisson(rate).astype(np.float32)
    X = values if counts else np.log1p(values)
    obs = {"cell_type": np.where(x > 0, "anterior", "posterior")}
    if coords:
        obs.update({"x": x, "y": y, "z": z})
    obs.update(obs_extra or {})
    adata = ad.AnnData(X=sparse.csr_matrix(X), obs=pd.DataFrame(obs, index=[f"{path.stem}_{i}" for i in range(len(z))]),
                       var=pd.DataFrame(index=genes))
    adata.obsm["spatial"] = np.c_[x, y, z]
    adata.write_h5ad(path)
    return path


@pytest.fixture()
def design_session(client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    rng = np.random.default_rng(0)
    genes = [f"g{i:02d}" for i in range(30)]
    target = _embryo(tmp_path / "target.h5ad", rng, 300, genes)
    visible = _embryo(tmp_path / "visible.h5ad", rng, 300, genes)
    mdcla = _embryo(tmp_path / "mdcla.h5ad", rng, 240, genes[:25], counts=True, coords=False,
                    obs_extra={"level4_subtypes": np.repeat(["a", "b", "c"], 80)})
    monkeypatch.setattr(api, "MDCLA_PATH", mdcla)
    session_id = client.post("/api/sessions").json()["session_id"]
    for kind, path in (("spatial", target), ("reference", visible)):
        with path.open("rb") as handle:
            response = client.post(
                f"/api/datasets/{kind}",
                data={"session_id": session_id, "cell_type_field": "cell_type"},
                files={"file": (path.name, handle, "application/octet-stream")},
            )
        assert response.status_code == 200, response.text
    return session_id


@pytest.mark.parametrize("method", ["gaussian_visible", "gaussian_mdcla", "rotating_visible", "rotating_mdcla"])
def test_design_methods_select_new_genes_and_score_every_round(client: TestClient, design_session: str, method: str) -> None:
    previous = ["g00", "g01", "g02"]
    request = {"session_id": design_session, "previous_genes": previous, "ng": 3, "seed": 1, "method": method, "rounds": 3}
    first = client.post("/api/select", json=request)
    second = client.post("/api/select", json=request)

    assert first.status_code == 200, first.text
    body = first.json()
    genes = [gene["gene_id"] for gene in body["selected_genes"]]
    assert len(genes) == len(set(genes)) == 3
    assert not set(genes) & set(previous)
    assert genes == [gene["gene_id"] for gene in second.json()["selected_genes"]]
    assert body["method"] == method
    steps = body["evaluation"]["steps"]
    assert [step["round"] for step in steps] == [0, 1, 2, 3]
    assert steps[1]["genes"] == genes
    assert [step["n_measured"] for step in steps] == [3, 6, 9, 12]
    expected = [step["expected_share"] for step in steps]
    assert expected == sorted(expected)
    if method.endswith("mdcla"):
        assert all(int(gene[1:]) < 25 for step in steps for gene in step["genes"])


def test_information_gain_scores_decrease_down_the_ranking(client: TestClient, design_session: str) -> None:
    body = client.post(
        "/api/select",
        json={"session_id": design_session, "ng": 6, "seed": 0, "method": "gaussian_visible"},
    ).json()
    scores = [gene["score"] for gene in body["selected_genes"]]
    assert all(score >= 0 for score in scores)
    assert scores == sorted(scores, reverse=True)


def test_gaussian_reconstruction_keeps_measured_values_and_predicts_the_rest(tmp_path: Path) -> None:
    from app import design_methods

    rng = np.random.default_rng(3)
    genes = [f"g{i:02d}" for i in range(30)]
    patterns = rng.normal(size=(2, len(genes)))
    target = ad.read_h5ad(_embryo(tmp_path / "target.h5ad", rng, 300, genes, loadings=patterns))
    visible = ad.read_h5ad(_embryo(tmp_path / "visible.h5ad", rng, 300, genes, loadings=patterns))
    data = design_methods.load_expression(visible, "X", genes, genes, "cell_type",
                                          visible.obs[["x", "y", "z"]].to_numpy())
    coords = target.obs[["x", "y", "z"]].to_numpy()
    model = design_methods.fit(data, coords, seed=0)
    measured = [0, 1, 2, 3]
    values = target.X[:, measured].toarray()
    recon = design_methods.reconstruct(model, measured, values)

    assert recon.shape == (target.n_obs, len(genes))
    assert np.allclose(recon[:, measured], values)
    assert (recon >= 0).all()
    truth = target.X.toarray()
    held_out = [np.corrcoef(recon[:, g], truth[:, g])[0, 1] for g in range(4, len(genes))]
    assert np.nanmean(held_out) > 0.2
