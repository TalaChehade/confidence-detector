"""Prepare the official Adaptive-RAG generated-label data for Eva training.

The official archive already contains generated silver labels and inductive-bias
labels. This script only normalizes its ``binary_silver/train.json`` and
``silver/valid.json`` files; it does not invent labels from query wording.
"""

import argparse
import json
import tarfile
import urllib.request
from pathlib import Path


OFFICIAL_ARCHIVE = "https://github.com/starsuzi/Adaptive-RAG/raw/main/data.tar.gz"
LABELS = {"A", "B", "C"}


def download_archive(destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(OFFICIAL_ARCHIVE, destination)
    return destination


def extract_archive(archive, destination):
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        root = destination.resolve()
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"Unsafe path in archive: {member.name}")
        tar.extractall(destination)


def find_single(root, suffix):
    matches = list(root.rglob(suffix))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one {suffix} below {root}; found {matches}")
    return matches[0]


def normalize_rows(path):
    with open(path, encoding="utf-8") as handle:
        rows = json.load(handle)
    normalized = []
    for row in rows:
        question, label = row.get("question"), str(row.get("answer", "")).upper()
        if not isinstance(question, str) or not question.strip() or label not in LABELS:
            raise ValueError(f"Invalid official Eva row in {path}: {row}")
        normalized.append({"question": question, "answer": label, "id": row.get("id"), "dataset_name": row.get("dataset_name")})
    return normalized


def main(data_root, output_path):
    data_root, output_path = Path(data_root), Path(output_path)
    train = normalize_rows(find_single(data_root, "binary_silver/train.json"))
    validation = normalize_rows(find_single(data_root, "silver/valid.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump({"train": train, "validation": validation}, handle, indent=2)
    print(f"Wrote {len(train)} train and {len(validation)} validation Eva examples to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare official Adaptive-RAG labels for Eva")
    parser.add_argument("--data-root", help="Directory where official data.tar.gz was extracted")
    parser.add_argument("--output", default="data/adaptive_rag_eva.json")
    parser.add_argument("--download-to", help="Download the official archive to this path and extract it")
    parser.add_argument("--extract-to", default="data/adaptive_rag_official")
    args = parser.parse_args()
    if args.download_to:
        archive = download_archive(Path(args.download_to))
        extract_archive(archive, Path(args.extract_to))
        data_root = Path(args.extract_to)
    elif args.data_root:
        data_root = Path(args.data_root)
    else:
        parser.error("provide --data-root, or --download-to (with optional --extract-to)")
    main(data_root, args.output)
