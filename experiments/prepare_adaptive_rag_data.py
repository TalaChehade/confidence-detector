"""Prepare the official Adaptive-RAG generated-label data for Eva training.

The official archive already contains generated silver labels and inductive-bias
labels. This script only normalizes its ``binary_silver/train.json`` and
``silver/valid.json`` files; it does not invent labels from query wording.
"""

import argparse
import json
import sys
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
        # ``filter`` is available in Python 3.12+. The explicit path check
        # above protects older Colab runtimes while avoiding their warning.
        if sys.version_info >= (3, 12):
            tar.extractall(destination, filter="data")
        else:
            tar.extractall(destination)


def find_label_file(root, label_source, split, kind):
    path = (
        root / "classifier" / "data" / "musique_hotpot_wiki2_nq_tqa_sqd"
        / label_source / kind / f"{split}.json"
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"Official Adaptive-RAG {kind}/{split}.json for {label_source!r} "
            f"not found at {path}"
        )
    return path


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


def main(data_root, output_path, label_source="flan_t5_xl"):
    data_root, output_path = Path(data_root), Path(output_path)
    train = normalize_rows(find_label_file(data_root, label_source, "train", "binary_silver"))
    validation = normalize_rows(find_label_file(data_root, label_source, "valid", "silver"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump({"train": train, "validation": validation}, handle, indent=2)
    print(
        f"Wrote {len(train)} train and {len(validation)} validation Eva examples "
        f"from {label_source} labels to {output_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare official Adaptive-RAG labels for Eva")
    parser.add_argument("--data-root", help="Directory where official data.tar.gz was extracted")
    parser.add_argument("--output", default="data/adaptive_rag_eva.json")
    parser.add_argument(
        "--label-source",
        choices=("flan_t5_xl", "flan_t5_xxl", "gpt"),
        default="flan_t5_xl",
        help="Official generator whose silver labels train Eva (default: flan_t5_xl).",
    )
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
    main(data_root, args.output, args.label_source)
