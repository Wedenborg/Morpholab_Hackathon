"""Check/fetch every configured prior input without constructing feature matrices.

python -m preprocessing.biological_priors.prepare_resources --config CONFIG
Add --download for HTTPS/SHA256 files, --download-models for pinned HF assets.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import re

from .core import FAMILIES, audit, digest, write_json
from .resources import PACKAGES, REQUIRED_PATHS, materialize
from .sequence import require_model
from .sources import KIND_BASES, KIND_FAMILY


def _model_assets(spec, cache, download):
    """Fetch/check assets only; do not load model weights or run inference."""
    require_model(spec)
    revision = spec["model_revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Resource preparation requires a full 40-character HF commit in model_revision")
    from huggingface_hub import snapshot_download
    snapshot = Path(snapshot_download(repo_id=spec["model"], revision=revision,
                                      cache_dir=str(cache), local_files_only=not download,
                                      ignore_patterns=["*.h5", "*.msgpack", "*.onnx", "*.tflite", "*.ot", "*.pt"],
                                      max_workers=4))
    # Older HF Hub versions can return incomplete local snapshots. Verify the
    # minimum layout, including all shards listed by a weight index.
    transformer = snapshot
    if spec["kind"] == "descriptions":
        modules = json.loads((snapshot / "modules.json").read_text())
        transformer_modules = [m for m in modules if m.get("type", "").endswith(".Transformer")]
        if len(transformer_modules) != 1:
            raise ValueError("Text model must declare one Sentence Transformers Transformer module")
        transformer = snapshot / transformer_modules[0]["path"]
        for module in modules:
            # Normalize is stateless and saves no assets; Git/HF snapshots
            # cannot contain the resulting empty module directory.
            if module.get("type", "").endswith(".Normalize"):
                continue
            if not (snapshot / module["path"]).is_dir():
                raise FileNotFoundError(f"Missing text model module: {module['path']}")
    required = [transformer / "config.json", transformer / "tokenizer_config.json"]
    if not any((transformer / name).is_file() for name in ("tokenizer.json", "vocab.txt", "vocab.json", "spiece.model", "sentencepiece.bpe.model")):
        raise FileNotFoundError(f"Missing tokenizer vocabulary in {transformer}")
    weights = [transformer / name for name in ("model.safetensors", "pytorch_model.bin")]
    indices = [transformer / name for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json")]
    index = next((p for p in indices if p.is_file()), None)
    if index:
        required.append(index)
        required.extend(transformer / name for name in set(json.loads(index.read_text())["weight_map"].values()))
    elif not any(p.is_file() for p in weights):
        raise FileNotFoundError(f"Missing model weights in {transformer}")
    else:
        required.append(next(p for p in weights if p.is_file()))
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing/empty model asset: {path}")
    return dict(snapshot=str(snapshot), revision=revision,
                verification="Required asset layout and shard presence; no inference or weight-content validation")


def prepare_resources(config_path, *, download=False, download_models=False, families=None):
    config_path = Path(config_path).expanduser().resolve()
    original = json.loads(config_path.read_text())
    config = deepcopy(original)
    base = config_path.parent
    cache_value = config.get("cache_dir", ".prior_cache")
    if not isinstance(cache_value, str):
        raise ValueError("cache_dir must be a local directory path")
    cache = (base / Path(cache_value).expanduser()).resolve()
    config["cache_dir"] = str(cache)
    files, checks = [], []
    selected = [s for s in config.get("sources", []) if families is None or s.get("family") in families]
    config["sources"] = selected

    def check(scope, function):
        try:
            detail = function()
            checks.append(dict(scope=scope, status="ready", detail=detail))
        except Exception as error:
            checks.append(dict(scope=scope, status="blocked", reason=f"{type(error).__name__}: {error}"))

    def paths(spec, scope, required):
        for key in sorted(set(required) | {k for k in spec if k == "path" or k.endswith("_path")}):
            entry = dict(scope=scope, key=key, configured=spec.get(key))
            try:
                if not spec.get(key):
                    raise ValueError(f"Required input key missing: {key}")
                path = materialize(spec[key], base, cache, download=download)
                if not path.is_file() or path.stat().st_size == 0:
                    raise FileNotFoundError(f"Missing/empty file: {path}. Supply the normalized/local file or an HTTPS+SHA256 path descriptor.")
                entry.update(status="ready", path=str(path), sha256=digest(path), size_bytes=path.stat().st_size)
                spec[key] = str(path)
            except Exception as error:
                entry.update(status="blocked", reason=f"{type(error).__name__}: {error}")
            files.append(entry)
        if "transfer" in spec:
            paths(spec["transfer"], scope + ".transfer", ("mapping_path",))

    identifiers = config.setdefault("identifiers", {})
    paths(identifiers, "identifiers", ("path",))
    for key in ("version", "url", "license"):
        if not identifiers.get(key):
            checks.append(dict(scope="identifiers", status="blocked", reason=f"Missing identifier metadata: {key}"))
    package_names = {"numpy", "scipy", "scikit-learn"}
    for spec in selected:
        scope, kind = spec.get("name", "unnamed_source"), spec.get("kind")
        package_names.update(PACKAGES.get(kind, ()))

        def validate(spec=spec, kind=kind):
            audit(spec)
            if spec["family"] not in KIND_FAMILY.get(kind, ()) or spec["evidence_basis"] not in KIND_BASES.get(kind, ()):
                raise ValueError("Unsupported source kind/family/evidence combination")
            if kind in {"go_gaf", "reactome", "string", "genome", "motifs"} and str(spec["species"]) != "10090":
                raise ValueError("This adapter requires native mouse input")
            if kind in {"genome", "motifs"}:
                assemblies = [spec.get(key) for key in ("assembly", "gtf_assembly", "fasta_assembly")]
                if not all(assemblies) or len(set(assemblies)) != 1:
                    raise ValueError("Specify the same assembly, gtf_assembly and fasta_assembly")
            return "Source metadata and upstream audit declared; biological file contents validated during construction"

        check(scope, validate)
        paths(spec, scope, REQUIRED_PATHS.get(kind, ("path",)))
        if kind in {"protein_esm", "descriptions"}:
            check(scope + ".model", lambda spec=spec: _model_assets(spec, cache, download_models))
    if any(isinstance(f["configured"], dict) for f in files):
        package_names.add("pooch")
    dependencies = {}
    for package in sorted(package_names):
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            dependencies[package] = "not installed"
            checks.append(dict(scope="dependencies", status="blocked", reason=f"Install {package}: python -m pip install -r requirements.txt"))
    unconfigured = [family for family in (families or FAMILIES) if not any(s.get("family") == family for s in selected)]
    for family in unconfigured:
        checks.append(dict(scope=family, status="blocked", reason="No source configured for this family"))
    ready = all(entry["status"] == "ready" for entry in files + checks)
    return config, dict(config=str(config_path), ready=ready, families_requested=list(families or FAMILIES),
                        unconfigured_families=unconfigured, dependencies=dependencies, files=files, checks=checks,
                        readiness_scope="Input availability, required packages and declared audits; run construction for format/content validation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--families", nargs="+", choices=FAMILIES, help="Check only selected families; default checks all ten")
    parser.add_argument("--download", action="store_true", help="Fetch checksum-pinned descriptors and expand gzip descriptors")
    parser.add_argument("--download-models", action="store_true", help="Fetch pinned HF model assets; potentially large")
    parser.add_argument("--report", type=Path, help="Save a JSON inventory including every missing input")
    parser.add_argument("--output-config", type=Path, help="Write a resolved config only if all requested inputs are ready")
    args = parser.parse_args()
    if args.output_config and args.output_config.resolve() == args.config.resolve():
        parser.error("--output-config must differ from the input config")
    try:
        config, report = prepare_resources(args.config, download=args.download,
                                           download_models=args.download_models, families=args.families)
    except Exception as error:
        parser.exit(2, f"{type(error).__name__}: {error}\n")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.report, report)
    for entry in report["files"]:
        print(f"{entry['scope']}.{entry['key']}: {entry['status']} — {entry.get('path', entry.get('reason'))}")
    for entry in report["checks"]:
        if entry["status"] != "ready":
            print(f"{entry['scope']}: {entry['reason']}")
    if report["ready"] and args.output_config:
        args.output_config.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output_config, config)
    print("Inputs ready" if report["ready"] else "Inputs incomplete; see the entries above")
    raise SystemExit(0 if report["ready"] else 2)


if __name__ == "__main__":
    main()
