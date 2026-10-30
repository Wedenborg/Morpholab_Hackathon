"""Persistent, verified provider downloads used by the preparation recipes."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from threading import RLock
from uuid import uuid4

import pooch

from .core import digest, write_json, write_table


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".partial")
    try:
        write_json(temporary, value)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class ResourceStore:
    """Pin a first-download snapshot, then verify and reuse it on every run.

    Provider snapshots without published SHA256 are explicitly marked as hashes
    computed after HTTPS retrieval. They are not advertised as upstream checksums.
    A receipt is written after each file, so failed runs are resumable.
    """

    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest = self.root / "downloads.json"
        self.records = json.loads(self.manifest.read_text()) if self.manifest.is_file() else {}
        self.lock = RLock()

    def record(self, key, value):
        with self.lock:
            self.records[key] = value
            atomic_json(self.manifest, self.records)

    def fetch(self, name, url):
        if not url.startswith("https://"):
            raise ValueError("Provider retrieval requires HTTPS")
        path = self.root / "raw" / name
        if not path.resolve().is_relative_to(self.root / "raw"):
            raise ValueError("Unsafe resource filename")
        with self.lock:
            previous = self.records.get(name)
        if previous and previous["url"] != url:
            raise ValueError(f"{name}: URL differs from the cached snapshot; use a new resource directory")
        expected = previous.get("sha256") if previous else None
        if path.is_file() and expected and digest(path) != expected:
            raise ValueError(f"{name}: cached resource checksum mismatch")
        if not path.is_file():
            print(f"Retrieving {name}: {url}", flush=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            path = Path(pooch.retrieve(url, known_hash="sha256:" + expected if expected else None,
                                      fname=path.name, path=path.parent,
                                      downloader=pooch.HTTPDownloader(timeout=(30, 180))))
        with path.open("rb") as handle:
            prefix = handle.read(512).lstrip().lower()
        if not path.stat().st_size or b"<html" in prefix or prefix.startswith(b"<!doctype html"):
            if previous is None:
                path.unlink(missing_ok=True)
            raise ValueError(f"{name}: provider returned an empty file or an HTML page instead of data")
        receipt = dict(url=url, path=str(path), sha256=digest(path), size_bytes=path.stat().st_size,
                       retrieved_utc=previous.get("retrieved_utc") if previous else datetime.now(timezone.utc).isoformat(),
                       checksum_basis="locally computed snapshot hash after HTTPS retrieval")
        self.record(name, receipt)
        return path

    def table(self, name, rows, columns):
        path = self.root / "prepared" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + "." + uuid4().hex + ".partial")
        count = 0

        def counted():
            nonlocal count
            for row in rows:
                count += 1
                yield row

        try:
            write_table(temporary, counted(), columns)
            if not count:
                raise ValueError(f"{name}: preparation produced no eligible records")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        self.record("prepared/" + name, dict(path=str(path), sha256=digest(path), rows=count,
                                           created_utc=datetime.now(timezone.utc).isoformat()))
        print(f"Prepared {name}: {count:,} records", flush=True)
        return path


MEMBERSHIP_COLUMNS = ("gene_id", "feature", "evidence", "association", "reference")


def source(name, family, kind, path, basis, url, version, components, filters, **options):
    """An audit of the specific retrieval/filtering recipe, not a whole database."""
    return dict(name=name, family=family, kind=kind, path=str(path), species="10090",
                version=version, url=url, license="Provider access/citation terms: " + url,
                evidence_basis=basis,
                audit=dict(expression_independent=True, upstream_components=components,
                           evidence_filters=filters,
                           justification="This recipe consumes only the declared sequence, curated annotation, text or genetic-interaction inputs; no expression matrices, expression-derived numerical features or expression-supervised gene models are used.",
                           scope="Specific downloaded files and the explicit transformations/filters in this recipe; not database-wide certification"),
                **options)
