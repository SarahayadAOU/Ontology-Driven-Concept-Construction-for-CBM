"""Download the ontology products used during the Gallstone mapping experiment.

For archival reproducibility, run this once for the paper release and commit the
resulting files (or archive them in Zenodo/OSF if repository size is a concern).
The script also writes SHA-256 checksums.
"""
from pathlib import Path
import hashlib
import urllib.request

ROOT = Path(__file__).resolve().parent
ONTOLOGIES = {
    "cmo.obo": "http://purl.obolibrary.org/obo/cmo.obo",
    "mondo.obo": "http://purl.obolibrary.org/obo/mondo.obo",
    "oba.obo": "http://purl.obolibrary.org/obo/oba.obo",
    "pato.obo": "http://purl.obolibrary.org/obo/pato.obo",
    "vt.owl": "http://purl.obolibrary.org/obo/vt.owl",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    lines = ["file\turl\tsha256"]
    for filename, url in ONTOLOGIES.items():
        dst = ROOT / filename
        print(f"Downloading {url} -> {dst}")
        urllib.request.urlretrieve(url, dst)
        lines.append(f"{filename}\t{url}\t{sha256(dst)}")
    (ROOT / "checksums.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote ontology/checksums.tsv")


if __name__ == "__main__":
    main()
