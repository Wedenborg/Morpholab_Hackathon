"""Download the 11 catalog starting points; no scientific imports until conversion.

The Bash entry point supplies the cluster environment. Public directory listings
are resolved at runtime and cached; HTML/login pages are never accepted as data.
"""
import argparse
from datetime import datetime, timezone
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
from urllib.parse import unquote, urljoin, urlparse
from urllib.request import Request, urlopen
import zipfile


ROOT = Path(__file__).resolve().parent
# Order and names match "Best starting points for this project" in Datasets.md.
SOURCES = {
    "seu-3d": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE278603",
    "pijuan-sala": "https://github.com/MarioniLab/EmbryoTimecourse2018/blob/master/download/download.sh",
    "extended-mouse-atlas": "https://marionilab.github.io/ExtendedMouseAtlas/",
    "tome": "https://tome.gs.washington.edu/",
    "mdcla": "https://ngdc.cncb.ac.cn/omix/release/OMIX014646",
    "harland": "https://raw.githubusercontent.com/SydneyBioX/SpatiotemporalMouseAtlas/main/README.md",
    "mittnenzweig": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE169nnn/GSE169210/suppl/",
    "tyser-heart": "https://content.cruk.cam.ac.uk/jmlab/mouseEmbryonicHeartAtlas/",
    "mosta": "https://db.cngb.org/stomics/mosta/download/",
    "mela": "https://mela.wi.mit.edu/download/",
    "nextcell": "https://next-cell.org/",
}
DATA_SUFFIXES = (".h5ad", ".h5td", ".rds", ".rdata", ".rda", ".loom",
                 ".csv", ".csv.gz", ".tsv", ".tsv.gz", ".txt", ".txt.gz",
                 ".mtx", ".mtx.gz", ".tar", ".tar.gz", ".zip", ".nwk.gz")


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.urls.append(href)


def links(url):
    print(f"Resolving public download links: {url}", flush=True)
    request = Request(url, headers={"User-Agent": "Morpholab-dataset-downloader/1.0"})
    with urlopen(request, timeout=90) as response:
        parser = Links()
        parser.feed(response.read().decode("utf-8", errors="replace"))
    return list(dict.fromkeys(urljoin(url, link) for link in parser.urls))


def filename(url):
    return Path(unquote(urlparse(url).path)).name


def data_links(url):
    return [link for link in links(url) if urlparse(link).scheme in ("https", "http")
            and filename(link).lower().endswith(DATA_SUFFIXES)]


def resolve(name):
    if name == "seu-3d":
        return ["https://ftp.ncbi.nlm.nih.gov/geo/series/GSE278nnn/GSE278603/suppl/GSE278603_RAW.tar"]
    if name == "pijuan-sala":
        # Author script distinguishes atlas from WT/Tal1 chimeras and containers.
        return ["https://content.cruk.cam.ac.uk/jmlab/atlas_data.tar.gz"]
    if name == "extended-mouse-atlas":
        # Only a 25 GB bundle is publicly listed, rather than standalone H5AD.
        return ["https://bioinformatics.stemcells.cam.ac.uk/rlh60/Supplemental/ExtendedMouseAtlas/ExtendedMouseAtlas.tar.gz"]
    if name == "tome":
        stages = ("E3.5", "E4.5", "E5.25", "E5.5", "E6.25", "E6.5", "E6.75", "E7.0",
                  "E7.25", "E7.5", "E7.75", "E8", "E8.25", "E8.5a", "E8.5b", "E9.5",
                  "E10.5", "E11.5", "E12.5", "E13.5")
        base = "https://shendure-web.gs.washington.edu/content/members/cxqiu/public/nobackup/tome_summary_data/mm/"
        return [base + f"seurat_object_{stage}.rds" for stage in stages]
    if name == "mosta":
        # Original 53 section products; omit adult controls, Spateo, image/GEM
        # bundles, combined copies, and later cell-bin/brain-specific products.
        return [url for url in links(SOURCES[name])
                if re.fullmatch(r"E\d+\.\d+_E\d+S\d+\.MOSTA\.h5ad", filename(url))]
    if name == "mdcla":
        # Entries 01-38: metadata and whole-embryo stages. Entries 39-41 are
        # separate pituitary perturbations and kidney Xenium experiments.
        return [url for url in data_links(SOURCES[name])
                if urlparse(url).hostname == "download.cncb.ac.cn"
                and (match := re.search(r"OMIX014646-(\d+)", filename(url)))
                and 1 <= int(match[1]) <= 38]
    if name == "mela":
        return [url for url in data_links(SOURCES[name])
                if urlparse(url).hostname == "zenodo.org"
                and filename(url).lower().endswith((".h5td", ".csv", ".csv.gz"))]
    if name == "nextcell":
        geo = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE341nnn/GSE341627/suppl/"
        counts = data_links(geo)
        if not any(filename(url).endswith((".mtx.gz", ".mtx", ".tar", ".tar.gz", ".h5ad")) for url in counts):
            raise ValueError(f"No processed expression downloads found at {geo}")
        trees = [url for url in data_links(SOURCES[name]) if filename(url).endswith(".nwk.gz")]
        if len(trees) < 2:
            raise ValueError("NextCell no longer lists both dated lineage trees")
        return counts + trees + ["https://raw.githubusercontent.com/shendurelab/dtt-mouse/main/support_data/cell_metadata.v8.txt.gz"]
    if name == "harland":
        return [url for url in data_links(SOURCES[name]) if urlparse(url).hostname == "maths.usyd.edu.au"]
    return data_links(SOURCES[name])


def write_json(path, value):
    partial = path.with_name(path.name + ".part")
    partial.write_text(json.dumps(value, indent=2) + "\n")
    partial.replace(path)


def validate_file(path):
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Empty or missing download: {path}")
    with path.open("rb") as handle:
        head = handle.read(512)
    lower = head.lstrip().lower()
    if lower.startswith((b"<!doctype html", b"<html", b"<?xml")):
        raise ValueError(f"Server returned an HTML/XML page instead of data: {path}")
    if path.name.lower().endswith((".h5ad", ".h5td")) and not head.startswith(b"\x89HDF\r\n\x1a\n"):
        raise ValueError(f"Not an HDF5 file: {path}")


def download(url, destination):
    if destination.suffix.lower() == ".rds" and destination.with_suffix(".h5ad").is_file():
        validate_file(destination.with_suffix(".h5ad"))
        print(f"Reusing converted AnnData: {destination.with_suffix('.h5ad')}", flush=True)
        return
    if destination.exists():
        validate_file(destination)
        print(f"Reusing download: {destination}", flush=True)
        return
    partial = destination.with_name(destination.name + ".part")
    print(f"Downloading {url}\n  -> {destination}", flush=True)
    command = [
        "curl", "--fail", "--location", "--retry", "5", "--retry-delay", "5",
        "--connect-timeout", "30", "--speed-limit", "1024", "--speed-time", "180",
        "--continue-at", "-", "--progress-bar", "--output", str(partial), url,
    ]
    result = subprocess.run(command)
    if result.returncode == 33 and partial.exists():
        # CURLE_RANGE_ERROR: the server cannot resume. Overwrite only this
        # unfinished file, so it cannot permanently block every requeued run.
        print(f"Server does not support resuming {partial.name}; restarting this file", flush=True)
        offset = command.index("--continue-at")
        command = command[:offset] + command[offset + 2:]
        result = subprocess.run(command)
    result.check_returncode()
    # Check using the final suffix before publishing (the partial ends in .part).
    validate_file(partial)
    if destination.suffix.lower() in (".h5ad", ".h5td"):
        with partial.open("rb") as handle:
            if handle.read(8) != b"\x89HDF\r\n\x1a\n":
                raise ValueError(f"Not HDF5: {url}")
    partial.replace(destination)


def unpack(path, folder, h5ad_only=False):
    """Extract regular files atomically; never create links or escape the root."""
    marker = folder / ("." + path.name + ".extracted")
    if marker.exists():
        return
    def save(name, source):
        target = (folder / name).resolve()
        if not target.is_relative_to(folder.resolve()):
            raise ValueError(f"Unsafe archive path: {name}")
        if h5ad_only and target.suffix.lower() != ".h5ad":
            return
        if target.exists():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        with source() as reader, partial.open("wb") as writer:
            shutil.copyfileobj(reader, writer)
        partial.replace(target)
    print(f"Extracting {path.name}" + (" (H5AD only)" if h5ad_only else ""), flush=True)
    if path.name.lower().endswith(".zip"):
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                if not member.is_dir():
                    save(member.filename, lambda m=member: archive.open(m))
    else:
        with tarfile.open(path) as archive:
            for member in archive:
                if member.isfile():
                    save(member.name, lambda m=member: archive.extractfile(m))
    marker.touch()


def process(name, args):
    folder = args.data_dir / ("GSE278603_RAW" if name == "seu-3d" else name)
    folder.mkdir(parents=True, exist_ok=True)
    source_file = folder / "download_sources.json"
    if source_file.exists() and not args.refresh_links:
        urls = json.loads(source_file.read_text())["urls"]
    else:
        urls = list(dict.fromkeys(resolve(name)))
        if not urls:
            raise ValueError(f"No processed download links found; inspect {SOURCES[name]}")
        expected = {"mosta": 53, "mdcla": 38}
        if name in expected and len(urls) != expected[name]:
            raise ValueError(f"Expected {expected[name]} processed files for {name}, found {len(urls)}; inspect the source page")
        if name == "mela" and sum(filename(url).endswith(".h5td") for url in urls) != 16:
            raise ValueError("Expected 16 MELA embryo H5TD files; inspect the source page")
        if any(not filename(url) or Path(filename(url)).name in (".", "..") for url in urls):
            raise ValueError("Invalid remote filename")
        names = [filename(url) for url in urls]
        if len(names) != len(set(names)):
            raise ValueError("Different remote files share a filename; resolve names before downloading")
        write_json(source_file, {"dataset": name, "source": SOURCES[name], "urls": urls,
                                "resolved_utc": datetime.now(timezone.utc).isoformat()})
    print(f"{name}: {len(urls)} file(s), all available stages -> {folder}", flush=True)
    for index, url in enumerate(urls, 1):
        path = folder / filename(url)
        print(f"[{name} {index}/{len(urls)}] {path.name}", flush=True)
        download(url, path)
    seen = set()
    while True:
        archives = [path for path in folder.rglob("*") if path.is_file() and path not in seen
                    and path.name.lower().endswith((".tar", ".tar.gz", ".zip"))]
        if not archives:
            break
        for path in archives:
            unpack(path, folder, h5ad_only=name in ("seu-3d", "extended-mouse-atlas"))
            seen.add(path)
    if not args.download_only:
        failures = []
        for path in sorted(folder.rglob("*")):
            if path.suffix.lower() != ".rds" or path.with_suffix(".h5ad").exists():
                continue
            print(f"Converting RDS in a separate Python process: {path}", flush=True)
            result = subprocess.run([
                sys.executable, "-u", "-c",
                "from pathlib import Path; from dataset.load_dataset import _read_r; "
                "import sys; _read_r(Path(sys.argv[1]), {}, verbose=True)", str(path),
            ], cwd=ROOT.parent)
            if result.returncode:
                failures.append(str(path))
        if failures:
            raise RuntimeError(f"Downloaded, but Python conversion failed for: {failures}")
    return {"status": "downloaded", "folder": str(folder), "source_count": len(urls),
            "conversion": "skipped" if args.download_only else "RDS converted; other formats retained"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/insomnia001/depts/morpheus/projects/hackathon/Datasets"))
    parser.add_argument("--datasets", nargs="+", choices=list(SOURCES), default=list(SOURCES),
                        help="Default: all 11 resources in catalog priority order")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without network access or writes")
    parser.add_argument("--download-only", action="store_true", help="Download/extract processed files; skip RDS conversion")
    parser.add_argument("--refresh-links", action="store_true", help="Refresh cached public download listings")
    args = parser.parse_args()
    args.data_dir = args.data_dir.expanduser().resolve()
    if args.dry_run:
        for index, name in enumerate(args.datasets, 1):
            print(f"{index:2}. {name}: {SOURCES[name]}")
        print(f"Destination: {args.data_dir}; all stages; RDS conversion: {not args.download_only}")
        print("Processed expression/metadata/lineage files only. MOSTA and MELA are large.")
        return 0
    if shutil.which("curl") is None:
        parser.error("curl is required for resumable downloads")
    args.data_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for index, name in enumerate(args.datasets, 1):
        print(f"\n=== Resource {index}/{len(args.datasets)}: {name} ===", flush=True)
        try:
            summary[name] = process(name, args)
        except Exception as error:
            print(f"FAILED {name}: {error}", file=sys.stderr, flush=True)
            summary[name] = {"status": "failed", "error": str(error), "source": SOURCES[name]}
        write_json(args.data_dir / "starting_points_status.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return int(any(item["status"] == "failed" for item in summary.values()))


if __name__ == "__main__":
    raise SystemExit(main())
