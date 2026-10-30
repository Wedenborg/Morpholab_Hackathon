"""Shared resource contracts and package-backed, checksum-verified downloads."""

from pathlib import Path
import re
from urllib.parse import urlparse

from .core import digest


# Every file consumed by an adapter, including secondary provenance inputs.
REQUIRED_PATHS = {
    "membership": ("path",), "go_gaf": ("path", "ontology_path"),
    "reactome": ("path",), "string": ("path",), "biogrid": ("path",),
    "trrust": ("path",), "regulatory_tsv": ("path",), "numeric": ("path",),
    "protein_embeddings": ("path", "inputs_path"),
    "text_embeddings": ("path", "inputs_path"),
    "protein_esm": ("path",), "descriptions": ("path",),
    "genome": ("gtf_path", "fasta_path"),
    "motifs": ("path", "gtf_path", "fasta_path"),
}

PACKAGES = {
    "go_gaf": ("goatools",),
    "protein_esm": ("biopython", "torch", "transformers", "huggingface-hub"),
    "descriptions": ("sentence-transformers", "huggingface-hub"),
    "genome": ("gffutils", "intervaltree", "pyfaidx", "biopython"),
    "motifs": ("gffutils", "intervaltree", "pyfaidx", "biopython"),
}


def materialize(value, base, cache, *, download=True):
    """Resolve a local path or an HTTPS+SHA256 descriptor with Pooch.

    Optional ``compression: gzip`` expands indexed genome FASTA in the cache.
    SHA256 always describes the downloaded bytes, before decompression.
    ``download=False`` never makes a network request or repairs corrupt caches.
    """
    base, cache = Path(base), Path(cache)
    if isinstance(value, str) and value:
        return (base / Path(value).expanduser()).resolve()
    if not isinstance(value, dict) or not {"url", "sha256"} <= value.keys():
        raise ValueError("Resource path must be a local path or {url, sha256}")
    url, checksum = value["url"], value["sha256"]
    if not isinstance(url, str) or urlparse(url).scheme != "https" or not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
        raise ValueError("Remote sources require HTTPS and an expected lowercase SHA256")
    compression = value.get("compression")
    if compression not in (None, "gzip"):
        raise ValueError("Supported resource compression: gzip (or omit to retain original bytes)")
    suffix = ".gz" if urlparse(url).path.endswith(".gz") or compression == "gzip" else ""
    destination = cache / (checksum + suffix)
    if destination.exists():
        if digest(destination) != checksum:
            raise ValueError(f"Cached resource checksum mismatch: {destination}")
    elif not download:
        raise FileNotFoundError(f"Not cached: {url}; run prepare_resources --download")
    import pooch
    if not destination.exists():
        cache.mkdir(parents=True, exist_ok=True)
        destination = Path(pooch.retrieve(url=url, known_hash="sha256:" + checksum,
                                        fname=destination.name, path=cache,
                                        downloader=pooch.HTTPDownloader(timeout=60)))
    if compression:
        expanded = Path(str(destination) + ".decomp")
        if not download:
            if not expanded.is_file():
                raise FileNotFoundError(f"Run prepare_resources --download to decompress {destination}")
            return expanded
        # Pooch maintains the decompressed file; runtime provenance hashes its bytes.
        return Path(pooch.Decompress(method="gzip")(str(destination), "fetch", None))
    return destination


def prepare(spec, base, cache):
    """Resolve every declared input, checking the adapter's required file keys."""
    for key in REQUIRED_PATHS.get(spec.get("kind"), ()):
        if not spec.get(key):
            raise ValueError(f"{spec['kind']} source requires {key}")
    resolved, artifacts = dict(spec), {}
    for key, value in spec.items():
        if key == "path" or key.endswith("_path"):
            path = materialize(value, base, cache)
            if not path.is_file() or not path.stat().st_size:
                raise FileNotFoundError(f"Required local resource unavailable or empty: {path}")
            resolved[key] = str(path)
            artifacts[key] = {"path": str(path), "sha256": digest(path)}
    if "transfer" in spec:
        transfer, extra = prepare(spec["transfer"], base, cache)
        resolved["transfer"], artifacts["transfer"] = transfer, extra
    resolved["cache_dir"] = str(cache)
    return resolved, artifacts
