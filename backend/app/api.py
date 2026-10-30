from __future__ import annotations

import os
import secrets
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import anndata as ad
import numpy as np
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from . import data_io, design_methods
from .candidates import prepare_candidates
from .data_io import MAX_UPLOAD_BYTES, MAX_UPLOAD_MB
from .evaluation import (
    evaluation_unavailable,
    evaluate_mouse_embryo_baseline,
    mouse_embryo_round_scorer,
    run_eval_team_dummy_method,
)
from .models import (
    DatasetSummary,
    DatasetUploadResponse,
    CandidateRequest,
    CandidateSummary,
    MappingRequest,
    PreviewResponse,
    SelectedGene,
    SelectionRequest,
    SelectionResponse,
)
from .selection import METHOD_ID, METHOD_LABEL, select_genes


DATA_ROOT = Path(os.getenv("GENE_PANEL_DATA_DIR", Path(os.getenv("TEMP", "/tmp")) / "gene-panel-planner"))
SESSION_TTL_SECONDS = int(os.getenv("GENE_PANEL_SESSION_TTL_SECONDS", str(24 * 60 * 60)))
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MOUSE_EMBRYO_DATA_DIR = Path(
    os.getenv("MORPHO_DATA_DIR", PROJECT_ROOT / "data" / "GSE278603_RAW")
)
MOUSE_EMBRYO_PREVIOUS_GENES = ["Fth1", "Rpl8", "Eif5a", "Cox7c"]
MDCLA_PATH = Path(os.getenv("MDCLA_PATH", PROJECT_ROOT.parent / "reference_data" / "mdcla" / "mdCLA_E7.5.h5ad"))
_mdcla_cache: dict[tuple[str, int, int], design_methods.Expression] = {}
DUMMY_METHODS = {
    "dummy_markers": ("markers", "Dummy markers · nearest-neighbour reconstruction"),
    "dummy_variance": ("variance", "Dummy variable genes · nearest-neighbour reconstruction"),
    "dummy_random": ("random", "Dummy random genes · nearest-neighbour reconstruction"),
}


@dataclass
class DatasetFile:
    kind: Literal["spatial", "reference"]
    path: Path
    source_path: Path
    transposed: bool
    filename: str
    size_bytes: int
    mapping: MappingRequest
    summary: DatasetSummary
    gene_ids: list[str]
    display_names: list[str]
    coordinates: np.ndarray | None = None


@dataclass
class SessionState:
    session_id: str
    directory: Path
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    datasets: dict[str, DatasetFile] = field(default_factory=dict)
    e75_benchmark: bool = False


sessions: dict[str, SessionState] = {}
router = APIRouter(prefix="/api")


def _session(session_id: str) -> SessionState:
    _cleanup_sessions()
    state = sessions.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="This session has expired or does not exist. Start a new session.")
    state.last_used_at = time.time()
    return state


def _cleanup_sessions() -> None:
    now = time.time()
    for session_id, state in list(sessions.items()):
        if now - state.last_used_at > SESSION_TTL_SECONDS:
            shutil.rmtree(state.directory, ignore_errors=False)
            del sessions[session_id]
    if DATA_ROOT.exists():
        for directory in DATA_ROOT.iterdir():
            if (
                directory.is_dir()
                and directory.name not in sessions
                and _is_session_directory(directory)
                and now - directory.stat().st_mtime > SESSION_TTL_SECONDS
            ):
                shutil.rmtree(directory, ignore_errors=False)


def _new_session_state() -> SessionState:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    _cleanup_sessions()
    session_id = str(uuid.uuid4())
    directory = DATA_ROOT / session_id
    directory.mkdir(mode=0o700)
    return SessionState(session_id=session_id, directory=directory)


def _is_session_directory(path: Path) -> bool:
    try:
        uuid.UUID(path.name)
    except ValueError:
        return False
    return True


def _summary_dataset(state: SessionState, kind: str) -> DatasetFile:
    dataset = state.datasets.get(kind)
    if dataset is None:
        raise HTTPException(status_code=409, detail=f"Upload the {kind} dataset first.")
    if not dataset.summary.valid:
        raise HTTPException(
            status_code=422,
            detail=f"The {kind} dataset has blocking validation errors. Review its mapping or re-upload it.",
        )
    return dataset


def _mapping_for_upload(
    gene_field: str,
    display_field: str,
    matrix_layer: str,
    coordinate_source: Literal["obsm", "obs"],
    coordinate_field: str,
    coordinate_columns: str,
    cell_type_field: str,
    transpose: bool,
) -> MappingRequest:
    columns = [column.strip() for column in coordinate_columns.split(",") if column.strip()]
    return MappingRequest(
        gene_field=gene_field or "var_names",
        display_field=display_field or "var_names",
        matrix_layer=matrix_layer or "X",
        coordinate_source=coordinate_source,
        coordinate_field=coordinate_field or "spatial",
        coordinate_columns=columns,
        cell_type_field=cell_type_field or None,
        transpose=transpose,
    )


async def _store_upload(
    state: SessionState,
    kind: Literal["spatial", "reference"],
    file: UploadFile,
    mapping: MappingRequest,
) -> DatasetUploadResponse:
    filename = file.filename or ""
    if Path(filename).suffix.lower() != ".h5ad":
        raise HTTPException(status_code=415, detail="Choose a file with the .h5ad extension.")
    stored_path = state.directory / f"{uuid.uuid4().hex}.h5ad"
    active_path = stored_path
    size = 0
    try:
        with stored_path.open("wb") as target:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    target.close()
                    stored_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File is larger than the configured {MAX_UPLOAD_MB} MB upload limit.",
                    )
                target.write(chunk)
        if size == 0:
            stored_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="The selected file is empty.")
        try:
            if mapping.transpose:
                active_path = state.directory / f"{uuid.uuid4().hex}-corrected.h5ad"
                data_io.transpose_h5ad(stored_path, active_path)
            summary, gene_ids, display_names, coordinates = data_io.validate_dataset(
                active_path, kind, Path(filename).name, size, mapping
            )
            summary.transpose_applied = mapping.transpose
            options = data_io.mapping_options(active_path)
        except (OSError, ValueError, TypeError, ImportError) as exc:
            stored_path.unlink(missing_ok=True)
            if active_path != stored_path:
                active_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "AnnData could not read this .h5ad file. Confirm that it is complete and readable.",
                    "technical": str(exc),
                },
            ) from exc
        prior = state.datasets.get(kind)
        state.datasets[kind] = DatasetFile(
            kind=kind,
            path=active_path,
            source_path=stored_path,
            transposed=mapping.transpose,
            filename=Path(filename).name,
            size_bytes=size,
            mapping=mapping,
            summary=summary,
            gene_ids=gene_ids,
            display_names=display_names,
            coordinates=coordinates,
        )
        if prior and prior.path != stored_path:
            prior.path.unlink(missing_ok=True)
        if prior and prior.source_path != prior.path:
            prior.source_path.unlink(missing_ok=True)
        return DatasetUploadResponse(session_id=state.session_id, dataset=summary, mapping_options=options)
    finally:
        await file.close()


@router.post("/sessions")
def create_session() -> dict[str, str]:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    _cleanup_sessions()
    session_id = str(uuid.uuid4())
    directory = DATA_ROOT / session_id
    directory.mkdir(mode=0o700)
    state = SessionState(session_id=session_id, directory=directory)
    sessions[session_id] = state
    return {"session_id": session_id}


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str) -> dict[str, bool]:
    state = sessions.pop(session_id, None)
    if state:
        shutil.rmtree(state.directory, ignore_errors=False)
    return {"deleted": state is not None}


@router.post("/datasets/{kind}", response_model=DatasetUploadResponse)
async def upload_dataset(
    kind: Literal["spatial", "reference"],
    session_id: str = Form(...),
    file: UploadFile = File(...),
    gene_field: str = Form("var_names"),
    display_field: str = Form("var_names"),
    matrix_layer: str = Form("X"),
    coordinate_source: Literal["obsm", "obs"] = Form("obsm"),
    coordinate_field: str = Form("spatial"),
    coordinate_columns: str = Form(""),
    cell_type_field: str = Form(""),
    transpose: bool = Form(False),
) -> DatasetUploadResponse:
    state = _session(session_id)
    mapping = _mapping_for_upload(
        gene_field, display_field, matrix_layer, coordinate_source, coordinate_field, coordinate_columns,
        cell_type_field, transpose
    )
    response = await _store_upload(state, kind, file, mapping)
    state.e75_benchmark = False
    return response


@router.post("/datasets/{kind}/mapping", response_model=DatasetUploadResponse)
def update_mapping(
    kind: Literal["spatial", "reference"],
    session_id: str,
    request: MappingRequest,
) -> DatasetUploadResponse:
    state = _session(session_id)
    old = state.datasets.get(kind)
    if old is None:
        raise HTTPException(status_code=409, detail=f"Upload the {kind} dataset first.")
    active_path = old.path
    created_path: Path | None = None
    try:
        if request.transpose and not old.transposed:
            created_path = state.directory / f"{uuid.uuid4().hex}-corrected.h5ad"
            data_io.transpose_h5ad(old.source_path, created_path)
            active_path = created_path
        elif not request.transpose and old.transposed:
            active_path = old.source_path
        summary, gene_ids, display_names, coordinates = data_io.validate_dataset(
            active_path, kind, old.filename, old.size_bytes, request
        )
        summary.transpose_applied = request.transpose
        options = data_io.mapping_options(active_path)
    except (OSError, ValueError, TypeError, ImportError) as exc:
        if created_path is not None:
            created_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=422,
            detail={"message": "The mapping could not be validated against this AnnData file.", "technical": str(exc)},
        ) from exc
    if old.path != active_path and old.path != old.source_path:
        old.path.unlink(missing_ok=True)
    old.path = active_path
    old.transposed = request.transpose
    old.mapping = request
    old.summary = summary
    old.gene_ids = gene_ids
    old.display_names = display_names
    old.coordinates = coordinates
    return DatasetUploadResponse(session_id=state.session_id, dataset=summary, mapping_options=options)


@router.get("/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, object]:
    state = _session(session_id)
    return {"session_id": state.session_id, "datasets": {key: value.summary for key, value in state.datasets.items()}}


def _benchmark_ready(state: SessionState, spatial: DatasetFile) -> bool:
    return (
        state.e75_benchmark
        and spatial.mapping.gene_field == "var_names"
        and spatial.mapping.matrix_layer == "X"
        and not spatial.transposed
    )


def _visible_coordinates(adata: ad.AnnData, dataset: DatasetFile) -> np.ndarray:
    """Positions of the reference (visible embryo) cells: obs x, y (and z) when present,
    otherwise the reference's mapped coordinates."""
    columns = [c for c in ("x", "y", "z") if c in adata.obs]
    if {"x", "y"} <= set(columns):
        return adata.obs[columns].to_numpy(float)
    if dataset.coordinates is not None:
        return np.asarray(dataset.coordinates, float)
    if dataset.mapping.coordinate_field in adata.obsm:
        return np.asarray(adata.obsm[dataset.mapping.coordinate_field], float)[:, :3]
    raise HTTPException(
        status_code=422,
        detail="The Gaussian and rotating methods fit on the reference as a visible embryo, so the reference "
        "needs spatial coordinates (obs x/y[/z] or a mapped coordinate field).",
    )


def _mdcla(target_genes: list[str]) -> design_methods.Expression:
    if not MDCLA_PATH.is_file():
        raise HTTPException(status_code=503, detail=f"The mdCLA file is unavailable; set MDCLA_PATH. Missing: {MDCLA_PATH}")
    key = (str(MDCLA_PATH), len(target_genes), hash(tuple(target_genes)))
    if key not in _mdcla_cache:
        _mdcla_cache.clear()
        _mdcla_cache[key] = design_methods.load_mdcla(MDCLA_PATH, target_genes)
    return _mdcla_cache[key]


def _run_design_method(request: SelectionRequest, state: SessionState, spatial: DatasetFile,
                       reference: DatasetFile, pool: Any, seed: int) -> tuple[list[SelectedGene], dict[str, Any]]:
    if spatial.coordinates is None:
        raise HTTPException(status_code=422, detail="This method needs the spatial dataset's coordinates.")
    adata = ad.read_h5ad(reference.path)
    visible = design_methods.load_expression(
        adata, reference.mapping.matrix_layer, reference.gene_ids, spatial.gene_ids,
        reference.mapping.cell_type_field, _visible_coordinates(adata, reference),
    )
    del adata
    if not visible.found.any():
        raise HTTPException(status_code=422, detail="No spatial gene identifiers match the reference dataset.")
    mdcla = _mdcla(spatial.gene_ids) if design_methods.METHODS[request.method][1] == "mdcla" else None
    scorer = None
    if _benchmark_ready(state, spatial):
        # Each round is reconstructed with the method's own Gaussian posterior, then scored.
        scorer = mouse_embryo_round_scorer(
            spatial.source_path, reference.source_path,
            reconstruct=lambda columns, values, model: design_methods.reconstruct(model, columns, values),
        )
    try:
        selected, steps, facts = design_methods.run(
            request.method, visible, mdcla, np.asarray(spatial.coordinates, float), spatial.gene_ids,
            request.previous_genes, list(pool.identifiers), request.ng, request.rounds, seed,
            pool.display_names, scorer,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if scorer is not None:
        evaluation = dict(steps[1]["benchmark"])
        evaluation["notice"] = (
            "Computed with the repository's eval_team benchmark. The embryo is reconstructed with this method's "
            "Gaussian factor model: the posterior of each neighbourhood's factors given the genes measured so far "
            "(previous rounds plus this one), corrected per cell by its own measured values; measured genes keep "
            "their observed values, and the experiment log excludes them from held-out scoring. This is a "
            "retrospective benchmark on the local E7.5 spatial dataset, not a prospective performance guarantee."
        )
        for step in steps:
            step["benchmark"] = {metric["id"]: metric["value"] for metric in step["benchmark"]["metrics"]}
    else:
        evaluation = evaluation_unavailable(
            "The eval_team benchmark is available for the unmodified local E7.5 mouse-embryo preset only. "
            "The per-round table shows the model's expected share of variance predicted."
        )
    evaluation["steps"] = steps
    evaluation["model"] = facts
    return selected, evaluation


@router.post("/select", response_model=SelectionResponse)
def run_selection(request: SelectionRequest) -> SelectionResponse:
    state = _session(request.session_id)
    spatial = _summary_dataset(state, "spatial")
    reference = _summary_dataset(state, "reference")
    pool = prepare_candidates(spatial.gene_ids, reference.gene_ids, request.previous_genes, spatial.display_names)
    display_by_id = dict(zip(spatial.gene_ids, spatial.display_names))
    seed = request.seed if request.seed is not None else secrets.randbits(32)
    if request.method == "manual":
        chosen = list(dict.fromkeys(gene.strip() for gene in request.chosen_genes if gene.strip()))
        if not chosen:
            raise HTTPException(status_code=422, detail="Enter at least one gene you chose.")
        unknown = [gene for gene in chosen if gene not in spatial.gene_ids]
        if unknown:
            raise HTTPException(
                status_code=422,
                detail=f"These chosen gene identifiers were not found in the spatial dataset: {', '.join(unknown[:20])}",
            )
        selected = [
            SelectedGene(
                gene_id=gene,
                display_name=display_by_id[gene],
                rank=index,
                score=None,
                score_provenance="User-selected gene; no method score.",
            )
            for index, gene in enumerate(chosen, start=1)
        ]
    elif request.method in DUMMY_METHODS:
        if not state.e75_benchmark:
            raise HTTPException(
                status_code=422,
                detail="The eval_team dummy methods are only available with the local E7.5 mouse-embryo preset.",
            )
        if request.ng % 4:
            raise HTTPException(
                status_code=422,
                detail="The eval_team dummy methods measure genes in sets of four; choose a gene count divisible by 4.",
            )
        if (
            spatial.mapping.gene_field != "var_names"
            or spatial.mapping.matrix_layer != "X"
            or spatial.transposed
        ):
            raise HTTPException(
                status_code=422,
                detail="Restore the local preset's var_names, X matrix, and original orientation to run eval_team methods.",
            )
        how, method_label = DUMMY_METHODS[request.method]
        try:
            chosen_genes, evaluation = run_eval_team_dummy_method(
                spatial.source_path,
                reference.source_path,
                how,
                request.ng // 4,
                request.previous_genes,
                seed,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if len(chosen_genes) != request.ng:
            raise HTTPException(
                status_code=422,
                detail=f"The dummy method selected {len(chosen_genes)} genes; {request.ng} were requested. "
                "Reduce the number of genes or try another method.",
            )
        selected = [
            SelectedGene(
                gene_id=gene,
                display_name=display_by_id[gene],
                rank=index,
                score=None,
                score_provenance="Selected by eval_team dummy method; no gene-level score.",
            )
            for index, gene in enumerate(chosen_genes, start=1)
        ]
    elif request.method in design_methods.METHODS:
        if request.ng > len(pool.identifiers):
            raise HTTPException(
                status_code=422,
                detail=f"Requested {request.ng} genes, but only {len(pool.identifiers)} eligible genes remain.",
            )
        method_label = design_methods.METHODS[request.method][2]
        selected, evaluation = _run_design_method(request, state, spatial, reference, pool, seed)
    else:
        method_label = METHOD_LABEL

    if request.method == "random" and request.ng > len(pool.identifiers):
        raise HTTPException(
            status_code=422,
            detail=f"Requested {request.ng} genes, but only {len(pool.identifiers)} eligible genes remain.",
        )
    if request.method == "random":
        evaluation_spatial = data_io.open_annotated(spatial.path)
        try:
            evaluation_reference = data_io.open_annotated(reference.path)
            try:
                selected = select_genes(
                    evaluation_spatial,
                    evaluation_reference,
                    request.previous_genes,
                    pool,
                    request.ng,
                    seed,
                )
            finally:
                data_io.close_annotated(evaluation_reference)
        finally:
            data_io.close_annotated(evaluation_spatial)

    if request.method == "manual":
        if request.ng != len(selected):
            raise HTTPException(status_code=422, detail="The requested gene count does not match the chosen gene list.")
        method_id = "manual-genes"
        method_label = "User-selected genes"
        evaluation = (
            evaluate_mouse_embryo_baseline(spatial.source_path, reference.source_path, [*request.previous_genes, *(gene.gene_id for gene in selected)])
            if state.e75_benchmark
            and spatial.mapping.gene_field == "var_names"
            and spatial.mapping.matrix_layer == "X"
            and not spatial.transposed
            else evaluation_unavailable(
                "The eval_team benchmark is available for the unmodified local E7.5 mouse-embryo preset only. "
                "No synthetic evaluation is generated for uploaded or synthetic-example datasets."
            )
        )
    elif request.method in DUMMY_METHODS:
        method_id = f"eval-team-{request.method}"
    elif request.method in design_methods.METHODS:
        method_id = request.method
    else:
        method_id = METHOD_ID
        if (
            state.e75_benchmark
            and spatial.mapping.gene_field == "var_names"
            and spatial.mapping.matrix_layer == "X"
            and not spatial.transposed
        ):
            evaluation = evaluate_mouse_embryo_baseline(
                spatial.source_path,
                reference.source_path,
                [*request.previous_genes, *(gene.gene_id for gene in selected)],
            )
        else:
            evaluation = evaluation_unavailable(
                "The eval_team benchmark is available for the unmodified local E7.5 mouse-embryo preset only. "
                "No synthetic evaluation is generated for uploaded or synthetic-example datasets."
            )
    overlap = len(set(spatial.gene_ids).intersection(reference.gene_ids))
    run_summary = {
        "method": method_id,
        "method_label": method_label,
        "seed": seed,
        "requested_count": len(selected),
        "selection_method": request.method,
        "candidate_count": len(spatial.gene_ids),
        "previous_genes_unique_count": len(pool.previous_unique),
        "matched_previous_genes": list(pool.matched_previous),
        "unmatched_previous_genes": list(pool.unmatched_previous),
        "eligible_count": len(pool.identifiers),
        "reference_overlap_count": overlap,
        "mappings": {
            "spatial": {
                "gene_field": spatial.mapping.gene_field,
                "display_field": spatial.mapping.display_field,
                "matrix_layer": spatial.mapping.matrix_layer,
                "coordinate_source": spatial.mapping.coordinate_source,
                "coordinate_field": spatial.mapping.coordinate_field,
                "coordinate_columns": spatial.mapping.coordinate_columns,
                "transpose": spatial.transposed,
            },
            "reference": {
                "gene_field": reference.mapping.gene_field,
                "display_field": reference.mapping.display_field,
                "matrix_layer": reference.mapping.matrix_layer,
                "cell_type_field": reference.mapping.cell_type_field,
                "transpose": reference.transposed,
            },
        },
        "datasets": {"spatial": spatial.summary.model_dump(), "reference": reference.summary.model_dump()},
        "evaluation": evaluation,
    }
    return SelectionResponse(
        method=method_id,
        method_label=method_label,
        score_label=(
            design_methods.SCORE_LABELS[design_methods.METHODS[request.method][0]]
            if request.method in design_methods.METHODS else "Score"
        ),
        seed=seed,
        selected_genes=selected,
        candidate_count=len(spatial.gene_ids),
        previous_count=len(pool.previous_unique),
        matched_previous_count=len(pool.matched_previous),
        unmatched_previous_genes=list(pool.unmatched_previous),
        eligible_count=len(pool.identifiers),
        requested_count=len(selected),
        reference_overlap_count=overlap,
        evaluation=evaluation,
        run_summary=run_summary,
    )


@router.post("/candidates", response_model=CandidateSummary)
def candidate_summary(request: CandidateRequest) -> CandidateSummary:
    state = _session(request.session_id)
    spatial = _summary_dataset(state, "spatial")
    reference = _summary_dataset(state, "reference")
    pool = prepare_candidates(spatial.gene_ids, reference.gene_ids, request.previous_genes, spatial.display_names)
    return CandidateSummary(
        candidate_count=len(spatial.gene_ids),
        previous_count=len(pool.previous_unique),
        matched_previous_count=len(pool.matched_previous),
        unmatched_previous_genes=list(pool.unmatched_previous),
        eligible_count=len(pool.identifiers),
        reference_overlap_count=len(set(spatial.gene_ids).intersection(reference.gene_ids)),
    )


@router.get("/preview/{kind}", response_model=PreviewResponse)
def preview_dataset(
    kind: Literal["spatial", "reference"],
    session_id: str,
    gene_id: str | None = None,
    color_by: str | None = None,
) -> PreviewResponse:
    state = _session(session_id)
    dataset = _summary_dataset(state, kind)
    try:
        coordinates, color_values, color_label = data_io.read_preview_values(
            dataset.path, dataset.mapping, kind, gene_id, color_by
        )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=f"Could not create this preview: {exc}") from exc
    if kind == "spatial" and coordinates is None:
        return PreviewResponse(
            kind=kind,
            available=False,
            message="Spatial coordinates are not available. Map a compatible obsm entry or obs coordinate columns.",
        )
    count = dataset.summary.n_observations
    limit = data_io.preview_limit()
    indices = np.linspace(0, count - 1, num=min(count, limit), dtype=int) if count else np.array([], dtype=int)
    if gene_id and color_label == "Gene expression unavailable":
        return PreviewResponse(
            kind=kind,
            available=False,
            message=f"{gene_id} is not present in the {kind} dataset's selected gene identifiers.",
            total_observations=count,
            sampled=count > limit,
        )
    if not color_values and color_label == "Gene expression unavailable":
        return PreviewResponse(
            kind=kind,
            available=False,
            message=f"The selected gene is not available in the {kind} dataset.",
            total_observations=count,
        )
    points: list[dict[str, object]] = []
    if coordinates is not None:
        points = [
            {"x": float(coordinates[index, 0]), "y": float(coordinates[index, 1])}
            for index in indices
        ]
    sampled_colors = [color_values[index] for index in indices] if color_values else []
    return PreviewResponse(
        kind=kind,
        available=True,
        message=None,
        points=points,
        coordinate_dimensions=coordinates.shape[1] if coordinates is not None else None,
        sampled=count > limit,
        total_observations=count,
        color_label=color_label,
        color_values=sampled_colors,
    )


@router.post("/example")
def create_example() -> dict[str, str]:
    import pandas as pd
    from scipy import sparse

    state = _new_session_state()
    session_id = state.session_id
    directory = state.directory
    rng = np.random.default_rng(42)
    spatial_genes = [f"Gene-{index:02d}" for index in range(1, 21)]
    reference_genes = spatial_genes[:16] + [f"Ref-{index:02d}" for index in range(1, 5)]
    spatial_matrix = sparse.csr_matrix(rng.poisson(1.2, size=(100, len(spatial_genes))).astype(np.float32))
    reference_matrix = sparse.csr_matrix(rng.poisson(1.4, size=(80, len(reference_genes))).astype(np.float32))
    spatial_adata = ad.AnnData(
        X=spatial_matrix,
        obs=pd.DataFrame(
            {"cell_type": np.where(np.arange(100) % 2, "epithelial", "mesenchymal")},
            index=[f"spot-{index:03d}" for index in range(100)],
        ),
        var=pd.DataFrame(index=spatial_genes),
    )
    angles = np.linspace(0, 4 * np.pi, 100)
    spatial_adata.obsm["spatial"] = np.column_stack(
        [np.cos(angles) * (1 + 0.1 * rng.random(100)), np.sin(angles) * (1 + 0.1 * rng.random(100))]
    )
    reference_adata = ad.AnnData(
        X=reference_matrix,
        obs=pd.DataFrame(
            {"cell_type": np.where(np.arange(80) % 2, "progenitor", "differentiated")},
            index=[f"cell-{index:03d}" for index in range(80)],
        ),
        var=pd.DataFrame(index=reference_genes),
    )
    for kind, adata, mapping in (
        ("spatial", spatial_adata, MappingRequest(cell_type_field="cell_type")),
        ("reference", reference_adata, MappingRequest(cell_type_field="cell_type")),
    ):
        path = directory / f"{kind}-{uuid.uuid4().hex}.h5ad"
        adata.write_h5ad(path)
        summary, gene_ids, display_names, coordinates = data_io.validate_dataset(
            path, kind, f"example-{kind}.h5ad", path.stat().st_size, mapping
        )
        state.datasets[kind] = DatasetFile(
            kind=kind, path=path, filename=f"example-{kind}.h5ad",
            source_path=path, transposed=False,
            size_bytes=path.stat().st_size, mapping=mapping, summary=summary,
            gene_ids=gene_ids, coordinates=coordinates,
            display_names=display_names,
        )
    sessions[session_id] = state
    return {"session_id": session_id, "previous_genes": "Gene-01\nGene-02\nGene-03"}


@router.post("/mouse-embryo")
def create_mouse_embryo_session() -> dict[str, object]:
    spatial_path = MOUSE_EMBRYO_DATA_DIR / "GSM9046243_Embryo_E7.5_stereo_rep1.h5ad"
    reference_path = MOUSE_EMBRYO_DATA_DIR / "GSM9046244_Embryo_E7.5_stereo_rep2.h5ad"
    missing = [path for path in (spatial_path, reference_path) if not path.is_file()]
    if missing:
        raise HTTPException(
            status_code=503,
            detail=(
                "The local E7.5 mouse embryo files are unavailable. Set MORPHO_DATA_DIR "
                f"to a folder containing both required .h5ad files. Missing: {', '.join(map(str, missing))}"
            ),
        )

    state = _new_session_state()
    try:
        datasets = (
            (
                "spatial",
                spatial_path,
                MappingRequest(
                    coordinate_source="obs",
                    coordinate_columns=["x", "y", "z"],
                    cell_type_field="celltype",
                ),
            ),
            ("reference", reference_path, MappingRequest(cell_type_field="celltype")),
        )
        for kind, path, mapping in datasets:
            summary, gene_ids, display_names, coordinates = data_io.validate_dataset(
                path, kind, path.name, path.stat().st_size, mapping
            )
            if not summary.valid:
                errors = "; ".join(issue.what for issue in summary.issues if issue.severity == "error")
                raise HTTPException(
                    status_code=422,
                    detail=f"The local {kind} mouse embryo dataset did not pass validation: {errors}",
                )
            state.datasets[kind] = DatasetFile(
                kind=kind,
                path=path,
                source_path=path,
                transposed=False,
                filename=path.name,
                size_bytes=path.stat().st_size,
                mapping=mapping,
                summary=summary,
                gene_ids=gene_ids,
                display_names=display_names,
                coordinates=coordinates,
            )
    except HTTPException:
        shutil.rmtree(state.directory, ignore_errors=True)
        raise
    except (OSError, ValueError, TypeError, ImportError) as exc:
        shutil.rmtree(state.directory, ignore_errors=True)
        raise HTTPException(
            status_code=422,
            detail=f"Could not load the local mouse embryo datasets: {exc}",
        ) from exc

    sessions[state.session_id] = state
    state.e75_benchmark = True
    return {
        "session_id": state.session_id,
        "previous_genes": "\n".join(MOUSE_EMBRYO_PREVIOUS_GENES),
        "panel_size": 10,
        "seed": 42,
    }


app = FastAPI(title="Gene Panel Planner", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)
app.include_router(router)


@app.get("/api/health")
def health() -> dict[str, str | int]:
    return {"status": "ok", "max_upload_mb": MAX_UPLOAD_MB}
